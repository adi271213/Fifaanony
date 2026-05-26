"""
Serialization tests.

Verifies:
- Every artifact file is written and well-formed JSONL or JSON.
- The tenant-hidden fragments file contains ONLY the V_frag-safe keys.
- The store fragments file contains hidden metadata (boundary_id,
  event_id, tenant_id, ...).
- Probes and responses round-trip through JSON.
- Metrics JSON contains all expected fields.
- write_experiment_artifacts produces all 9 files.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.generator import generate_corpus
from mnembound.retrieval import ALL_RUNGS, Rung, retrieve, summarize_rung
from mnembound.scoring import (
    RecallResponse,
    is_boundary_invalid,
    summarize,
)
from mnembound.serialization import (
    fragment_to_store_dict,
    probe_to_dict,
    read_jsonl,
    record_to_dict,
    write_experiment_artifacts,
    write_fragments_store_jsonl,
    write_fragments_tenant_hidden_jsonl,
    write_fragments_tenant_visible_jsonl,
    write_metrics_json,
    write_probes_jsonl,
    write_records_jsonl,
    write_responses_jsonl,
    write_rung_stats_jsonl,
)


def _make_corpus_and_responses():
    c = generate_corpus(
        seed=42, n_tenants=5, records_per_tenant=20,
        n_supported_probes=5, n_cross_boundary_probes=10,
    )
    responses = []
    for p in c.probes:
        result = retrieve(Rung.SHARED_VECTOR_INDEX, p, c)
        responses.append(RecallResponse(
            probe_id=p.probe_id,
            probe_type=p.probe_type,
            recall=p.recall,
            abstained=False,
            fragments=result.fragments,
        ))
    return c, responses


# ----- Per-export tests -----


def test_records_jsonl_roundtrips() -> None:
    c, _ = _make_corpus_and_responses()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        n = write_records_jsonl(c.records, path)
        assert n == len(c.records)
        loaded = read_jsonl(path)
        assert len(loaded) == len(c.records)
        # Spot-check first record matches the dict form.
        assert loaded[0]["record_id"] == c.records[0].record_id
        assert loaded[0]["client_id"] == c.records[0].client_id
        assert loaded[0]["object"] == c.records[0].object_  # object_ -> object
    finally:
        Path(path).unlink(missing_ok=True)
    print(f"[OK] records.jsonl writes {n} records and round-trips")


def test_tenant_hidden_fragments_file_has_no_forbidden_keys() -> None:
    """The single most important serialization test: no V_frag leakage."""
    c, _ = _make_corpus_and_responses()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_fragments_tenant_hidden_jsonl(c, path)
        loaded = read_jsonl(path)
        allowed = {"role", "value", "text", "provenance"}
        forbidden = {
            "record_id", "boundary_id", "event_id", "tenant_id",
            "timestamp", "log_source", "incident_template", "entity_set",
            "salt", "resolver",
        }
        for view in loaded:
            keys = set(view.keys())
            assert keys == allowed, f"got keys {sorted(keys)}, expected {sorted(allowed)}"
            assert not (keys & forbidden), (
                f"tenant-hidden fragments file leaks: {keys & forbidden}"
            )
    finally:
        Path(path).unlink(missing_ok=True)
    print(f"[OK] fragments.tenant_hidden.jsonl has ONLY {sorted(allowed)}")


def test_tenant_hidden_fragments_file_has_no_tenant_strings() -> None:
    c, _ = _make_corpus_and_responses()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_fragments_tenant_hidden_jsonl(c, path)
        loaded = read_jsonl(path)
        for view in loaded:
            for tid in c.tenants():
                assert tid not in view["value"], (
                    f"tenant-hidden fragment leaks {tid} in value: {view['value']}"
                )
                assert tid not in view["text"], (
                    f"tenant-hidden fragment leaks {tid} in text: {view['text']}"
                )
    finally:
        Path(path).unlink(missing_ok=True)
    print("[OK] fragments.tenant_hidden.jsonl has no tenant strings")


def test_tenant_visible_fragments_file_has_only_allowed_keys() -> None:
    c, _ = _make_corpus_and_responses()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_fragments_tenant_visible_jsonl(c, path)
        loaded = read_jsonl(path)
        allowed = {"role", "value", "text", "provenance"}
        for view in loaded:
            assert set(view.keys()) == allowed
    finally:
        Path(path).unlink(missing_ok=True)
    print(f"[OK] fragments.tenant_visible.jsonl has ONLY {sorted(allowed)}")


def test_store_fragments_file_has_hidden_metadata() -> None:
    """Symmetric check: the store file MUST have the hidden fields."""
    c, _ = _make_corpus_and_responses()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_fragments_store_jsonl(c, path)
        loaded = read_jsonl(path)
        required = {
            "role", "value", "text", "public_provenance_id",
            "record_id", "boundary_id", "event_id", "tenant_id",
            "timestamp", "log_source", "incident_template",
        }
        for f in loaded:
            assert required <= set(f.keys()), (
                f"store file missing fields: {required - set(f.keys())}"
            )
    finally:
        Path(path).unlink(missing_ok=True)
    print(f"[OK] fragments.store.jsonl has all required hidden metadata fields")


def test_probes_jsonl_carries_ground_truth_labels() -> None:
    c, _ = _make_corpus_and_responses()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_probes_jsonl(c.probes, path)
        loaded = read_jsonl(path)
        # Spot-check fields.
        for p_dict in loaded:
            assert "probe_id" in p_dict
            assert "probe_type" in p_dict
            assert "recall" in p_dict
            assert "is_fragment_supported" in p_dict
            assert "is_event_supported" in p_dict
            assert "boundary_event_pairs" in p_dict
            assert "bridge_context" in p_dict
            assert len(p_dict["fragment_provenance_ids"]) >= 1
    finally:
        Path(path).unlink(missing_ok=True)
    print(f"[OK] probes.jsonl carries ground-truth labels and provenance IDs")


def test_responses_jsonl_roundtrips() -> None:
    c, responses = _make_corpus_and_responses()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_responses_jsonl(responses, path)
        loaded = read_jsonl(path)
        assert len(loaded) == len(responses)
        for r_dict in loaded:
            assert "probe_id" in r_dict
            assert "abstained" in r_dict
    finally:
        Path(path).unlink(missing_ok=True)
    print(f"[OK] responses.jsonl writes {len(responses)} responses")


def test_metrics_json_has_all_fields() -> None:
    c, responses = _make_corpus_and_responses()
    summary = summarize(responses, verifier_functions={
        "blind": lambda r: True,
        "oracle": lambda r: not (
            r.recall is not None and is_boundary_invalid(r.recall, r.fragments)
        ),
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        write_metrics_json(summary, path)
        loaded = json.loads(Path(path).read_text())
        required = {
            "boundary_invalid_recall_rate",
            "fragment_valid_false_recall_rate",
            "supported_recall_accuracy",
            "absent_refusal_rate",
            "partial_refusal_rate",
            "audit_invisibility_per_verifier",
            "n_responses",
            "n_violations",
        }
        assert required <= set(loaded.keys())
        assert loaded["audit_invisibility_per_verifier"]["blind"] == 1.0
        assert loaded["audit_invisibility_per_verifier"]["oracle"] == 0.0
    finally:
        Path(path).unlink(missing_ok=True)
    print("[OK] metrics.json has all required fields")


def test_rung_stats_jsonl_writes_one_per_rung() -> None:
    c, _ = _make_corpus_and_responses()
    stats = [summarize_rung(r, list(c.probes), c) for r in ALL_RUNGS]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        n = write_rung_stats_jsonl(stats, path)
        assert n == len(ALL_RUNGS)
        loaded = read_jsonl(path)
        rung_names = {s["rung"] for s in loaded}
        assert rung_names == {r.value for r in ALL_RUNGS}
    finally:
        Path(path).unlink(missing_ok=True)
    print(f"[OK] rung_stats.jsonl writes {n} entries (one per rung)")


# ----- Full experiment artifact write -----


def test_write_experiment_artifacts_creates_all_files() -> None:
    c, responses = _make_corpus_and_responses()
    summary = summarize(responses, verifier_functions={
        "blind": lambda r: True,
    })
    rung_stats = [summarize_rung(r, list(c.probes), c) for r in ALL_RUNGS]

    with tempfile.TemporaryDirectory() as out_dir:
        paths = write_experiment_artifacts(c, responses, summary, rung_stats, out_dir)
        expected_keys = {
            "corpus_metadata", "records",
            "fragments_tenant_visible", "fragments_tenant_hidden",
            "fragments_store", "probes", "responses",
            "metrics", "rung_stats",
        }
        assert set(paths.keys()) == expected_keys
        for k, p in paths.items():
            assert Path(p).exists(), f"{k} file was not written: {p}"
            assert Path(p).stat().st_size > 0, f"{k} file is empty"
    print(f"[OK] write_experiment_artifacts creates all 9 expected files")


def main() -> int:
    tests = [
        test_records_jsonl_roundtrips,
        test_tenant_hidden_fragments_file_has_no_forbidden_keys,
        test_tenant_hidden_fragments_file_has_no_tenant_strings,
        test_tenant_visible_fragments_file_has_only_allowed_keys,
        test_store_fragments_file_has_hidden_metadata,
        test_probes_jsonl_carries_ground_truth_labels,
        test_responses_jsonl_roundtrips,
        test_metrics_json_has_all_fields,
        test_rung_stats_jsonl_writes_one_per_rung,
        test_write_experiment_artifacts_creates_all_files,
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
    print(f"=== all {len(tests)} serialization tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
