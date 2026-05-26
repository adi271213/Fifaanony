"""
Scoring tests.

Verifies the predicate and rate definitions in `mnembound.scoring`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.generator import generate_corpus
from mnembound.probes import (
    generate_absent_probes,
    generate_cross_boundary_traps,
    generate_supported_probes,
)
from mnembound.schema import ProbeType, Recall
from mnembound.scoring import (
    MetricSummary,
    RecallResponse,
    abstention_rate,
    audit_invisibility_rate,
    boundary_invalid_recall_rate,
    fragment_valid_false_recall_rate,
    is_boundary_invalid,
    is_event_supported,
    is_fragment_supported,
    summarize,
    supported_recall_accuracy,
)


def _make_corpus():
    return generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)


def test_predicates_on_supported_probe() -> None:
    """A SUPPORTED probe satisfies fragment- AND event-supported."""
    corpus = _make_corpus()
    probe = generate_supported_probes(corpus, n_probes=1, seed=1)[0]
    assert is_fragment_supported(probe.recall, probe.fragments)
    assert is_event_supported(probe.recall, probe.fragments)
    assert not is_boundary_invalid(probe.recall, probe.fragments)
    print("[OK] SUPPORTED probe predicates: fragment=True, event=True, invalid=False")


def test_predicates_on_cross_boundary_trap() -> None:
    """A CROSS_BOUNDARY trap is fragment-supported but NOT event-supported."""
    corpus = _make_corpus()
    probes = generate_cross_boundary_traps(corpus, n_probes=10, seed=2)
    for p in probes:
        assert is_fragment_supported(p.recall, p.fragments)
        assert not is_event_supported(p.recall, p.fragments)
        assert is_boundary_invalid(p.recall, p.fragments)
    print(f"[OK] {len(probes)} cross-boundary trap predicates: fragment=True, event=False, invalid=True")


def test_predicates_on_absent_probe() -> None:
    corpus = _make_corpus()
    probes = generate_absent_probes(corpus, n_probes=10, seed=3)
    for p in probes:
        assert not is_fragment_supported(p.recall, p.fragments)
        assert not is_event_supported(p.recall, p.fragments)
        assert not is_boundary_invalid(p.recall, p.fragments)
    print(f"[OK] {len(probes)} absent probe predicates: all False")


def test_event_supported_rejects_fragments_across_events() -> None:
    """
    Explicit check: a recall whose components come from different events is
    fragment-supported but NOT event-supported (the §4.2 corner case).
    """
    corpus = _make_corpus()
    trap = generate_cross_boundary_traps(corpus, n_probes=1, seed=42)[0]

    # The trap's fragments cover all 5 components, so fragment-supported.
    assert is_fragment_supported(trap.recall, trap.fragments)
    # But they span multiple (boundary, event) pairs, so no single
    # authorized support set covers the whole recall.
    boundary_event_pairs = {(f.boundary_id, f.event_id) for f in trap.fragments}
    assert len(boundary_event_pairs) >= 2
    assert not is_event_supported(trap.recall, trap.fragments)
    print("[OK] event-supported correctly rejects fragments spread across events")


def _make_responses_from_probes(probes) -> list[RecallResponse]:
    """Simulate a perfectly-truthful agent: always returns probe.recall."""
    return [
        RecallResponse(
            probe_id=p.probe_id,
            probe_type=p.probe_type,
            recall=p.recall,
            abstained=False,
            fragments=p.fragments,
        )
        for p in probes
    ]


def test_boundary_invalid_recall_rate_on_trap_responses() -> None:
    """
    When the agent (or trivially the construction itself) emits the probe's
    own recall on every CROSS_BOUNDARY trap, the boundary-invalid rate is
    1.0 on cross-boundary probes.
    """
    corpus = _make_corpus()
    traps = generate_cross_boundary_traps(corpus, n_probes=30, seed=5)
    responses = _make_responses_from_probes(traps)
    rate = boundary_invalid_recall_rate(responses)
    assert rate == 1.0, f"expected 1.0, got {rate}"
    print(f"[OK] boundary_invalid_recall_rate=1.0 on {len(traps)} trap responses")


def test_boundary_invalid_recall_rate_on_supported_responses() -> None:
    """On SUPPORTED probes, a truthful agent has rate 0.0."""
    corpus = _make_corpus()
    supp = generate_supported_probes(corpus, n_probes=20, seed=6)
    responses = _make_responses_from_probes(supp)
    rate = boundary_invalid_recall_rate(responses, restrict_to=None)
    assert rate == 0.0, f"expected 0.0, got {rate}"
    print(f"[OK] boundary_invalid_recall_rate=0.0 on supported-only responses")


def test_fragment_valid_false_recall_rate() -> None:
    """
    On a trap-only response set, every violation is fragment-supported
    (the very definition of a CROSS_BOUNDARY trap), so the rate is 1.0.
    """
    corpus = _make_corpus()
    traps = generate_cross_boundary_traps(corpus, n_probes=20, seed=7)
    responses = _make_responses_from_probes(traps)
    rate = fragment_valid_false_recall_rate(responses)
    assert rate == 1.0, f"expected 1.0, got {rate}"
    print("[OK] fragment_valid_false_recall_rate=1.0 on trap responses")


def test_audit_invisibility_with_blind_verifier() -> None:
    """
    An audit-blind verifier that always admits gives audit_invisibility_rate
    1.0 on a trap-only response set.
    """
    corpus = _make_corpus()
    traps = generate_cross_boundary_traps(corpus, n_probes=20, seed=8)
    responses = _make_responses_from_probes(traps)
    rate = audit_invisibility_rate(responses, lambda r: True)
    assert rate == 1.0
    print("[OK] audit_invisibility_rate=1.0 with always-admit verifier")


def test_audit_invisibility_with_oracle_verifier() -> None:
    """
    An oracle verifier that always rejects boundary-invalid recalls (using
    the hidden (boundary, event) metadata) gives audit_invisibility_rate
    0.0. This is the upper bound a fragment-level baseline cannot match
    without exiting V_frag.
    """
    corpus = _make_corpus()
    traps = generate_cross_boundary_traps(corpus, n_probes=20, seed=9)
    responses = _make_responses_from_probes(traps)

    def oracle(resp: RecallResponse) -> bool:
        # admits = NOT a boundary violation
        if resp.recall is None:
            return True
        return not is_boundary_invalid(resp.recall, resp.fragments)

    rate = audit_invisibility_rate(responses, oracle)
    assert rate == 0.0
    print("[OK] audit_invisibility_rate=0.0 with oracle verifier")


def test_abstention_rate() -> None:
    corpus = _make_corpus()
    traps = generate_cross_boundary_traps(corpus, n_probes=10, seed=10)
    responses = [
        RecallResponse(
            probe_id=p.probe_id,
            probe_type=p.probe_type,
            recall=None,
            abstained=True,
            fragments=p.fragments,
        )
        for p in traps
    ]
    assert abstention_rate(responses) == 1.0
    # And boundary_invalid_recall_rate is 0.0 because abstained responses
    # don't count as violations.
    assert boundary_invalid_recall_rate(responses) == 0.0
    print("[OK] abstention shifts boundary_invalid rate to 0.0")


def test_summarize_returns_metric_summary() -> None:
    corpus = _make_corpus()
    supp = generate_supported_probes(corpus, n_probes=10, seed=11)
    traps = generate_cross_boundary_traps(corpus, n_probes=20, seed=12)
    responses = _make_responses_from_probes(supp) + _make_responses_from_probes(traps)
    summary = summarize(
        responses,
        verifier_functions={
            "blind": lambda r: True,
            "oracle": lambda r: not (
                r.recall is not None and is_boundary_invalid(r.recall, r.fragments)
            ),
        },
    )
    assert isinstance(summary, MetricSummary)
    # 20 traps in 30 responses -> CROSS_BOUNDARY restricted rate = 1.0
    assert summary.boundary_invalid_recall_rate == 1.0
    assert summary.supported_recall_accuracy == 1.0
    assert summary.audit_invisibility_per_verifier["blind"] == 1.0
    assert summary.audit_invisibility_per_verifier["oracle"] == 0.0
    assert summary.n_responses == 30
    assert summary.n_violations == 20
    print("[OK] summarize() returns a populated MetricSummary")


def main() -> int:
    tests = [
        test_predicates_on_supported_probe,
        test_predicates_on_cross_boundary_trap,
        test_predicates_on_absent_probe,
        test_event_supported_rejects_fragments_across_events,
        test_boundary_invalid_recall_rate_on_trap_responses,
        test_boundary_invalid_recall_rate_on_supported_responses,
        test_fragment_valid_false_recall_rate,
        test_audit_invisibility_with_blind_verifier,
        test_audit_invisibility_with_oracle_verifier,
        test_abstention_rate,
        test_summarize_returns_metric_summary,
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
    print(f"=== all {len(tests)} scoring tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
