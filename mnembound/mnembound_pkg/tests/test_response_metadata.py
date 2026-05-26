"""
Response metadata tests.

Verifies the four reviewer-identified fixes:
1. response_to_dict() serializes value_view (and the other new metadata).
2. summarize_by_rung groups responses by rung correctly.
3. run_probe captures raw model output and parse errors.
4. RecallResponse carries rung/model/seed/tenant_visible metadata.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.agent_io import (
    run_probe,
    stub_agent_always_abstain,
    stub_agent_always_recall,
)
from mnembound.generator import generate_corpus
from mnembound.retrieval import ALL_RUNGS, Rung, retrieve
from mnembound.schema import ProbeType
from mnembound.scoring import (
    RecallResponse,
    ValueView,
    filter_responses,
    is_boundary_invalid,
    summarize_by_rung,
)
from mnembound.serialization import (
    read_jsonl,
    response_to_dict,
    write_responses_jsonl,
)


def _make_corpus():
    return generate_corpus(
        seed=42, n_tenants=5, records_per_tenant=20,
        n_supported_probes=5, n_cross_boundary_probes=10,
    )


# ----- Item 1: value_view in response_to_dict -----


def test_response_to_dict_includes_value_view() -> None:
    """The critical reviewer fix: value_view in serialized output."""
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
    response = RecallResponse(
        probe_id=p.probe_id, probe_type=p.probe_type,
        recall=p.recall, abstained=False, fragments=result.fragments,
        value_view=ValueView.TENANT_HIDDEN,
    )
    d = response_to_dict(response)
    assert "value_view" in d, (
        "response_to_dict MUST include value_view; without it, reloaded "
        "responses default to TENANT_VISIBLE and silently score wrong"
    )
    assert d["value_view"] == "tenant_hidden"
    print("[OK] response_to_dict includes value_view")


def test_response_to_dict_value_view_round_trips_through_jsonl() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
    responses = [
        RecallResponse(
            probe_id=p.probe_id, probe_type=p.probe_type,
            recall=p.recall, abstained=False, fragments=result.fragments,
            value_view=ValueView.TENANT_HIDDEN,
        ),
        RecallResponse(
            probe_id=p.probe_id + "-v", probe_type=p.probe_type,
            recall=p.recall, abstained=False, fragments=result.fragments,
            value_view=ValueView.TENANT_VISIBLE,
        ),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_responses_jsonl(responses, path)
        loaded = read_jsonl(path)
        assert loaded[0]["value_view"] == "tenant_hidden"
        assert loaded[1]["value_view"] == "tenant_visible"
    finally:
        Path(path).unlink(missing_ok=True)
    print("[OK] value_view round-trips through JSONL")


# ----- Item 2: per-rung scoring -----


def test_summarize_by_rung_groups_by_rung_field() -> None:
    """
    summarize_by_rung produces one MetricSummary per rung. The §13 pass
    gates require these per-rung values; a combined summarize() would
    average them and hide the rung-to-rung gap that is the paper claim.
    """
    corpus = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=200,
        n_cross_boundary_probes=20,
    )

    responses = []
    for rung in ALL_RUNGS:
        for p in corpus.probes:
            result = retrieve(rung, p, corpus)
            def stub(s, u, _p=p, _r=result):
                return stub_agent_always_recall(_p, _r, value_view=ValueView.TENANT_HIDDEN)
            responses.append(run_probe(p, result, stub, tenant_visible=False))

    by_rung = summarize_by_rung(responses, verifier_functions={
        "blind": lambda r: True,
        "oracle": lambda r: not (r.recall and is_boundary_invalid(r.recall, r.fragments, value_view=r.value_view)),
    })

    # Every rung should appear in the result.
    assert set(by_rung.keys()) == {r.value for r in ALL_RUNGS}, (
        f"summarize_by_rung missed rungs: {set(by_rung.keys())}"
    )
    # The central empirical claim, per-rung:
    assert by_rung["isolated"].boundary_invalid_recall_rate == 0.0
    assert by_rung["boundary_preserving_topk"].boundary_invalid_recall_rate == 0.0
    assert by_rung["shared_vector_index"].boundary_invalid_recall_rate == 1.0
    assert by_rung["summary_merge"].boundary_invalid_recall_rate == 1.0
    print(
        "[OK] summarize_by_rung: ISO=0.0, BP=0.0, SVI=1.0, SM=1.0 "
        "(the gate metrics)"
    )


def test_summarize_by_rung_buckets_no_rung_metadata_separately() -> None:
    """Responses without rung metadata go into a clearly-named bucket."""
    corpus = _make_corpus()
    p = corpus.probes[0]
    # Build a response by hand WITHOUT setting rung.
    response = RecallResponse(
        probe_id=p.probe_id, probe_type=p.probe_type,
        recall=p.recall, abstained=False, fragments=p.fragments,
    )
    by_rung = summarize_by_rung([response], verifier_functions={})
    assert "_no_rung_metadata" in by_rung, (
        "responses with rung=None should land in the _no_rung_metadata bucket, "
        "not be silently dropped"
    )
    print("[OK] responses without rung metadata are bucketed, not dropped")


def test_filter_responses_works_by_rung_model_seed() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result_iso = retrieve(Rung.ISOLATED, p, corpus)
    result_shr = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    def stub_a(s, u, _p=p, _r=result_iso):
        return stub_agent_always_recall(_p, _r, value_view=ValueView.TENANT_HIDDEN)
    def stub_b(s, u, _p=p, _r=result_shr):
        return stub_agent_always_recall(_p, _r, value_view=ValueView.TENANT_HIDDEN)

    responses = [
        run_probe(p, result_iso, stub_a, tenant_visible=False, model="modelA", seed=1),
        run_probe(p, result_shr, stub_b, tenant_visible=False, model="modelA", seed=1),
        run_probe(p, result_iso, stub_a, tenant_visible=False, model="modelB", seed=1),
        run_probe(p, result_iso, stub_a, tenant_visible=False, model="modelA", seed=2),
    ]

    assert len(filter_responses(responses, rung="isolated")) == 3
    assert len(filter_responses(responses, rung="shared_vector_index")) == 1
    assert len(filter_responses(responses, model="modelA")) == 3
    assert len(filter_responses(responses, model="modelB")) == 1
    assert len(filter_responses(responses, seed=1)) == 3
    assert len(filter_responses(responses, seed=2)) == 1
    assert len(filter_responses(responses, model="modelA", seed=1)) == 2
    print("[OK] filter_responses slices by rung/model/seed correctly")


# ----- Item 3: raw_text and parse_error captured -----


def test_run_probe_captures_raw_text_on_successful_parse() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    def stub(s, u):
        return stub_agent_always_recall(p, result, value_view=ValueView.TENANT_HIDDEN)

    response = run_probe(p, result, stub, tenant_visible=False)
    assert response.raw_text is not None
    assert "recall" in response.raw_text  # the JSON payload contains "recall"
    assert response.parse_error is None
    assert response.abstain_reason is None
    print("[OK] raw_text captured on successful parse")


def test_run_probe_captures_parse_error_on_malformed_output() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    def broken_agent(s, u):
        return "this is not json at all"

    response = run_probe(p, result, broken_agent, tenant_visible=False)
    assert response.abstained
    assert response.raw_text == "this is not json at all"
    assert response.parse_error is not None
    assert "json decode failed" in response.parse_error
    print("[OK] parse_error captured on malformed output")


def test_run_probe_captures_abstain_reason() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    stub = stub_agent_always_abstain("insufficient evidence")
    response = run_probe(p, result, stub, tenant_visible=False)
    assert response.abstained
    assert response.abstain_reason == "insufficient evidence"
    assert response.parse_error is None
    print("[OK] abstain_reason captured when agent abstains explicitly")


# ----- Item 4: rung/model/seed/tenant_visible metadata -----


def test_run_probe_populates_rung_from_retrieval_result() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    for rung in ALL_RUNGS:
        result = retrieve(rung, p, corpus)
        def stub(s, u, _p=p, _r=result):
            return stub_agent_always_recall(_p, _r, value_view=ValueView.TENANT_HIDDEN)
        response = run_probe(p, result, stub, tenant_visible=False)
        assert response.rung == rung.value, (
            f"expected rung={rung.value}, got {response.rung}"
        )
    print("[OK] run_probe populates rung from retrieval_result.rung")


def test_run_probe_populates_model_and_seed_from_kwargs() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    def stub(s, u):
        return stub_agent_always_recall(p, result, value_view=ValueView.TENANT_HIDDEN)

    response = run_probe(
        p, result, stub, tenant_visible=False,
        model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", seed=42,
    )
    assert response.model == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    assert response.seed == 42
    print("[OK] run_probe populates model and seed from kwargs")


def test_run_probe_populates_tenant_visible_flag() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    def stub_v(s, u):
        return stub_agent_always_recall(p, result, value_view=ValueView.TENANT_VISIBLE)
    def stub_h(s, u):
        return stub_agent_always_recall(p, result, value_view=ValueView.TENANT_HIDDEN)

    r_v = run_probe(p, result, stub_v, tenant_visible=True)
    r_h = run_probe(p, result, stub_h, tenant_visible=False)
    assert r_v.tenant_visible is True
    assert r_h.tenant_visible is False
    print("[OK] run_probe populates tenant_visible flag")


# ----- Combined: serialization carries the metadata -----


def test_metadata_round_trips_through_jsonl() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    def stub(s, u):
        return stub_agent_always_recall(p, result, value_view=ValueView.TENANT_HIDDEN)

    response = run_probe(
        p, result, stub, tenant_visible=False,
        model="Qwen-30B", seed=42,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_responses_jsonl([response], path)
        loaded = read_jsonl(path)[0]
        assert loaded["rung"] == "shared_vector_index"
        assert loaded["model"] == "Qwen-30B"
        assert loaded["seed"] == 42
        assert loaded["tenant_visible"] is False
        assert loaded["value_view"] == "tenant_hidden"
        assert "raw_text" in loaded
    finally:
        Path(path).unlink(missing_ok=True)
    print("[OK] all metadata fields round-trip through JSONL")


def main() -> int:
    tests = [
        test_response_to_dict_includes_value_view,
        test_response_to_dict_value_view_round_trips_through_jsonl,
        test_summarize_by_rung_groups_by_rung_field,
        test_summarize_by_rung_buckets_no_rung_metadata_separately,
        test_filter_responses_works_by_rung_model_seed,
        test_run_probe_captures_raw_text_on_successful_parse,
        test_run_probe_captures_parse_error_on_malformed_output,
        test_run_probe_captures_abstain_reason,
        test_run_probe_populates_rung_from_retrieval_result,
        test_run_probe_populates_model_and_seed_from_kwargs,
        test_run_probe_populates_tenant_visible_flag,
        test_metadata_round_trips_through_jsonl,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"=== {len(failures)} test(s) failed ===")
        return 1
    print(f"=== all {len(tests)} response-metadata tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
