"""
Agent I/O tests.

Verifies:
- Prompt construction includes the right evidence block for each rung.
- Tenant-hidden prompts contain sanitized values, not tenant-revealing ones.
- The JSON parser is forgiving (fenced, prefixed, abstention-with/without-reason).
- The parser returns parse_error on malformed JSON.
- run_probe end-to-end produces a RecallResponse and handles parse errors
  by abstaining.
- run_probe requires tenant_visible as a keyword (no default).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.agent_io import (
    build_prompt,
    parse_agent_response,
    render_fragments_as_evidence,
    run_probe,
    stub_agent_always_abstain,
    stub_agent_always_recall,
)
from mnembound.generator import generate_corpus
from mnembound.retrieval import Rung, retrieve
from mnembound.schema import ProbeType
from mnembound.scoring import RecallResponse, ValueView


def _make_corpus():
    return generate_corpus(
        seed=42, n_tenants=5, records_per_tenant=20,
        n_supported_probes=5, n_cross_boundary_probes=10,
    )


# ----- Prompt construction -----


def test_build_prompt_for_shared_rung_lists_fragments() -> None:
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    p = traps[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
    prompt = build_prompt(p, result, tenant_visible=False)
    # The user prompt mentions the evidence block.
    assert "Evidence:" in prompt.user
    # Each of the 5 fragments is referenced by an indexed line.
    for i in range(1, 6):
        assert f"[{i} prov=" in prompt.user, f"missing fragment marker [{i}]"
    print("[OK] SHARED_VECTOR_INDEX prompt lists 5 indexed fragments")


def test_build_prompt_for_summary_merge_uses_summary_text() -> None:
    """
    SUMMARY_MERGE default (COLLAPSED) emits a single 'Merged memory summary:'
    block. The per-fragment indexed markers must NOT appear (the rung
    deliberately strips per-fragment provenance).
    """
    from mnembound.retrieval import SummaryMergeMode

    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    p = traps[0]
    result = retrieve(Rung.SUMMARY_MERGE, p, corpus)

    # Default: COLLAPSED
    prompt = build_prompt(p, result, tenant_visible=False)
    assert "Merged memory summary:" in prompt.user
    # Individual fragment markers should NOT appear.
    assert "[1 prov=" not in prompt.user
    # The collapsed prompt has no per-client framing.
    assert "Summary of recent activity for this client" not in prompt.user

    # Per-client mode: includes the per-tenant block headers.
    prompt_pcb = build_prompt(
        p, result, tenant_visible=False,
        summary_merge_mode=SummaryMergeMode.PER_CLIENT_BLOCKS,
    )
    assert "Summary of recent activity for this client" in prompt_pcb.user
    assert "[1 prov=" not in prompt_pcb.user

    print(
        "[OK] SUMMARY_MERGE prompt: COLLAPSED has no per-client framing; "
        "PER_CLIENT_BLOCKS does"
    )


def test_tenant_hidden_prompt_has_no_tenant_strings() -> None:
    corpus = _make_corpus()
    for p in corpus.probes:
        result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
        prompt = build_prompt(p, result, tenant_visible=False)
        for tid in corpus.tenants():
            assert tid not in prompt.user, (
                f"tenant-hidden prompt leaks {tid} in user text"
            )
    print("[OK] tenant-hidden prompts contain no clientA/clientB/... strings")


def test_tenant_visible_prompt_does_contain_tenant_strings() -> None:
    """Symmetric sanity check: visible prompts SHOULD contain tenant labels."""
    corpus = _make_corpus()
    saw_tenant_string = False
    for p in corpus.probes:
        result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
        prompt = build_prompt(p, result, tenant_visible=True)
        if any(tid in prompt.user for tid in corpus.tenants()):
            saw_tenant_string = True
            break
    assert saw_tenant_string, (
        "no tenant-visible prompt contained any tenant string; the test setup "
        "is probably broken"
    )
    print("[OK] tenant-visible prompts do contain tenant strings (as expected)")


# ----- JSON parser -----


def test_parser_extracts_clean_json() -> None:
    text = '{"abstain": true, "reason": "insufficient evidence"}'
    p = parse_agent_response(text)
    assert p.recall is None
    assert p.abstain_reason == "insufficient evidence"
    assert p.parse_error is None
    print("[OK] parser handles clean abstention JSON")


def test_parser_extracts_recall_json() -> None:
    text = '{"recall": {"subject": "s", "action": "a", "object": "o", "outcome": "x", "time": "t"}}'
    p = parse_agent_response(text)
    assert p.recall is not None
    assert p.recall.subject == "s"
    assert p.recall.action == "a"
    assert p.recall.object_ == "o"
    print("[OK] parser handles clean recall JSON with all 5 components")


def test_parser_strips_json_fences() -> None:
    text = '```json\n{"abstain": true, "reason": "nope"}\n```'
    p = parse_agent_response(text)
    assert p.recall is None
    assert p.abstain_reason == "nope"
    print("[OK] parser strips ```json fences")


def test_parser_strips_leading_commentary() -> None:
    text = 'Here is my answer: {"abstain": true, "reason": "x"} done.'
    p = parse_agent_response(text)
    assert p.abstain_reason == "x"
    print("[OK] parser handles prefixed/suffixed commentary")


def test_parser_handles_abstention_without_reason() -> None:
    p = parse_agent_response('{"abstain": true}')
    assert p.abstain_reason == "(no reason given)"
    print("[OK] parser handles abstention with no reason field")


def test_parser_returns_error_on_malformed_json() -> None:
    p = parse_agent_response("this is not json at all")
    assert p.parse_error is not None
    assert p.recall is None
    assert p.abstain_reason is None
    print("[OK] parser returns parse_error on malformed input")


def test_parser_returns_error_on_missing_recall_fields() -> None:
    text = '{"recall": {"subject": "s", "action": "a"}}'
    p = parse_agent_response(text)
    assert p.parse_error is not None
    assert "object" in p.parse_error or "missing" in p.parse_error
    print("[OK] parser returns parse_error on incomplete recall object")


def test_parser_returns_error_on_non_string_field() -> None:
    text = '{"recall": {"subject": 42, "action": "a", "object": "o", "outcome": "x", "time": "t"}}'
    p = parse_agent_response(text)
    assert p.parse_error is not None
    print("[OK] parser returns parse_error when a recall field is not a string")


# ----- Full run_probe pipeline -----


def test_run_probe_with_always_recall_stub_produces_recall() -> None:
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    p = traps[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    # Test tenant-visible path: stub returns visible recall verbatim.
    def stub_visible(system: str, user: str) -> str:
        return stub_agent_always_recall(p, result, value_view=ValueView.TENANT_VISIBLE)

    response = run_probe(p, result, stub_visible, tenant_visible=True)
    assert isinstance(response, RecallResponse)
    assert response.probe_id == p.probe_id
    assert not response.abstained
    assert response.recall is not None
    assert response.recall.subject == p.recall.subject

    # Test tenant-hidden path: stub returns sanitized recall; scoring under
    # response.value_view=TENANT_HIDDEN sees these as fragment-supported
    # (because they match f.sanitized_value).
    def stub_hidden(system: str, user: str) -> str:
        return stub_agent_always_recall(p, result, value_view=ValueView.TENANT_HIDDEN)

    response_h = run_probe(p, result, stub_hidden, tenant_visible=False)
    assert not response_h.abstained
    assert response_h.recall is not None
    # The hidden recall's subject is the SANITIZED value, not the visible one.
    subject_frag = next(f for f in p.fragments if f.role.value == "subject")
    assert response_h.recall.subject == subject_frag.sanitized_value
    assert response_h.value_view == ValueView.TENANT_HIDDEN

    print("[OK] run_probe with always-recall stub respects value_view")


def test_run_probe_with_abstain_stub_produces_abstention() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
    stub = stub_agent_always_abstain("insufficient evidence")
    response = run_probe(p, result, stub, tenant_visible=False)
    assert response.abstained
    assert response.recall is None
    print("[OK] run_probe with always-abstain stub produces abstention")


def test_run_probe_treats_parse_error_as_abstention() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    def broken_agent(system: str, user: str) -> str:
        return "completely unparseable output with no json"

    response = run_probe(p, result, broken_agent, tenant_visible=False)
    assert response.abstained
    assert response.recall is None
    print("[OK] run_probe treats unparseable output as abstention (no data loss)")


def test_run_probe_requires_tenant_visible_keyword() -> None:
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
    stub = stub_agent_always_abstain()
    # Missing tenant_visible should raise.
    try:
        run_probe(p, result, stub)  # type: ignore[call-arg]
        raise AssertionError("run_probe should require tenant_visible keyword")
    except TypeError:
        pass
    print("[OK] run_probe requires explicit tenant_visible keyword")


def main() -> int:
    tests = [
        test_build_prompt_for_shared_rung_lists_fragments,
        test_build_prompt_for_summary_merge_uses_summary_text,
        test_tenant_hidden_prompt_has_no_tenant_strings,
        test_tenant_visible_prompt_does_contain_tenant_strings,
        test_parser_extracts_clean_json,
        test_parser_extracts_recall_json,
        test_parser_strips_json_fences,
        test_parser_strips_leading_commentary,
        test_parser_handles_abstention_without_reason,
        test_parser_returns_error_on_malformed_json,
        test_parser_returns_error_on_missing_recall_fields,
        test_parser_returns_error_on_non_string_field,
        test_run_probe_with_always_recall_stub_produces_recall,
        test_run_probe_with_abstain_stub_produces_abstention,
        test_run_probe_treats_parse_error_as_abstention,
        test_run_probe_requires_tenant_visible_keyword,
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
    print(f"=== all {len(tests)} agent_io tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
