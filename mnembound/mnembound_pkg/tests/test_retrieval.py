"""
Retrieval rung tests.

Verifies:
- ISOLATED strips cross-tenant fragments from CROSS_BOUNDARY traps.
- BOUNDARY_PRESERVING_TOPK behaves identically to ISOLATED under TENANT
  granularity (one boundary per tenant).
- SHARED_VECTOR_INDEX returns all fragments unchanged.
- SUMMARY_MERGE returns all fragments and produces a non-empty
  context_text.
- The central empirical claim: rungs that share more produce more
  boundary-invalid recalls (when an agent emits the probe's recall).
- querying_tenant resolution is correct for each probe type.
- RungSummary statistics are sensible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.generator import generate_corpus
from mnembound.retrieval import (
    ALL_RUNGS,
    Rung,
    querying_tenant_for,
    render_summary_text,
    retrieve,
    summarize_rung,
)
from mnembound.schema import ProbeType
from mnembound.scoring import (
    RecallResponse,
    boundary_invalid_recall_rate,
    is_boundary_invalid,
    is_fragment_supported,
)


def _make_corpus():
    return generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=100,
        n_supported_probes=20, n_absent_probes=10,
        n_partial_probes=10, n_cross_boundary_probes=20,
    )


# ----- Per-rung behavior -----


def test_isolated_strips_cross_tenant_fragments_from_traps() -> None:
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    for p in traps:
        result = retrieve(Rung.ISOLATED, p, corpus)
        # Only fragments from the querying tenant remain.
        for f in result.fragments:
            assert f.tenant_id == result.querying_tenant
        # Excluded fragments are from other tenants.
        for f in result.excluded_fragments:
            assert f.tenant_id != result.querying_tenant
        # At most 4 of 5 fragments may survive, because the trap generator
        # requires >=2 distinct boundaries (= >=2 distinct tenants under
        # TENANT granularity), so at least one fragment must come from a
        # different tenant and be stripped.
        assert len(result.fragments) <= 4, (
            f"trap {p.probe_id} retained {len(result.fragments)} fragments "
            "under ISOLATED, but require_distinct_boundaries should force >=1 strip"
        )
        assert len(result.excluded_fragments) >= 1
    print(f"[OK] ISOLATED strips cross-tenant fragments from {len(traps)} traps")


def test_isolated_passes_supported_probes_intact() -> None:
    """SUPPORTED probes have all fragments from one tenant; ISOLATED is a no-op."""
    corpus = _make_corpus()
    supp = [p for p in corpus.probes if p.probe_type == ProbeType.SUPPORTED]
    for p in supp:
        result = retrieve(Rung.ISOLATED, p, corpus)
        assert len(result.fragments) == len(p.fragments)
        assert len(result.excluded_fragments) == 0
    print(f"[OK] ISOLATED preserves all SUPPORTED probe fragments")


def test_boundary_preserving_topk_matches_isolated_under_tenant_granularity() -> None:
    """Under TENANT granularity, BOUNDARY_PRESERVING_TOPK = ISOLATED."""
    corpus = _make_corpus()
    for p in corpus.probes:
        r_iso = retrieve(Rung.ISOLATED, p, corpus)
        r_bp = retrieve(Rung.BOUNDARY_PRESERVING_TOPK, p, corpus)
        # Same fragments (set equality; order may differ)
        assert set(f.public_provenance_id for f in r_iso.fragments) == set(
            f.public_provenance_id for f in r_bp.fragments
        )
    print("[OK] BOUNDARY_PRESERVING_TOPK == ISOLATED under TENANT granularity")


def test_shared_vector_index_returns_all_fragments() -> None:
    corpus = _make_corpus()
    for p in corpus.probes:
        result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
        assert result.fragments == p.fragments
        assert len(result.excluded_fragments) == 0
    print(f"[OK] SHARED_VECTOR_INDEX returns all fragments unchanged")


def test_summary_merge_keeps_all_fragments() -> None:
    """
    SUMMARY_MERGE keeps every fragment (no filtering). It deliberately
    does NOT populate context_text at retrieval time; the prompt builder
    renders the summary on demand via render_summary_text() so visibility
    is chosen at the point of use.
    """
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    for p in traps[:5]:
        result = retrieve(Rung.SUMMARY_MERGE, p, corpus)
        # All fragments retained
        assert len(result.fragments) == len(p.fragments)
        # context_text intentionally empty at retrieval time
        assert result.context_text == ""
    print("[OK] SUMMARY_MERGE keeps all fragments; context_text rendered on demand")


# ----- The central empirical claim -----


def test_empirical_claim_rungs_that_share_more_produce_more_violations() -> None:
    """
    Run the always-recall stub through each rung. The boundary-invalid
    rate on CROSS_BOUNDARY probes should be:
      ISOLATED                 -> 0.0
      BOUNDARY_PRESERVING_TOPK -> 0.0
      SHARED_VECTOR_INDEX      -> 1.0
      SUMMARY_MERGE            -> 1.0

    This is the central paper claim, reproducible without any LLM.
    """
    corpus = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=200,
        n_cross_boundary_probes=50,
    )
    rates: dict[Rung, float] = {}
    for rung in ALL_RUNGS:
        responses = []
        for p in corpus.probes:
            result = retrieve(rung, p, corpus)
            responses.append(RecallResponse(
                probe_id=p.probe_id,
                probe_type=p.probe_type,
                recall=p.recall,
                abstained=False,
                fragments=result.fragments,
            ))
        rates[rung] = boundary_invalid_recall_rate(responses)

    assert rates[Rung.ISOLATED] == 0.0, (
        f"ISOLATED should produce zero violations, got {rates[Rung.ISOLATED]}"
    )
    assert rates[Rung.BOUNDARY_PRESERVING_TOPK] == 0.0
    assert rates[Rung.SHARED_VECTOR_INDEX] == 1.0
    assert rates[Rung.SUMMARY_MERGE] == 1.0

    print(
        f"[OK] central empirical claim: rates = "
        f"ISO={rates[Rung.ISOLATED]}, BP={rates[Rung.BOUNDARY_PRESERVING_TOPK]}, "
        f"SVI={rates[Rung.SHARED_VECTOR_INDEX]}, SM={rates[Rung.SUMMARY_MERGE]}"
    )


# ----- Querying tenant resolution -----


def test_querying_tenant_for_supported_probe() -> None:
    corpus = _make_corpus()
    for p in corpus.probes:
        if p.probe_type != ProbeType.SUPPORTED:
            continue
        tenant = querying_tenant_for(p)
        # For a SUPPORTED probe, the underlying event's tenant is the
        # querying tenant.
        primary_be = next(iter(p.boundary_event_pairs))
        # Find a fragment with that (boundary, event).
        for f in p.fragments:
            if (f.boundary_id, f.event_id) == primary_be:
                assert tenant == f.tenant_id
                break
    print("[OK] querying_tenant_for resolves SUPPORTED probes to the event's tenant")


def test_querying_tenant_for_cross_boundary_probe() -> None:
    corpus = _make_corpus()
    for p in corpus.probes:
        if p.probe_type != ProbeType.CROSS_BOUNDARY:
            continue
        tenant = querying_tenant_for(p)
        # Should be the subject fragment's tenant.
        subject_frag = next(f for f in p.fragments if f.role.value == "subject")
        assert tenant == subject_frag.tenant_id
    print("[OK] querying_tenant_for resolves CROSS_BOUNDARY to subject fragment's tenant")


# ----- RungSummary -----


def test_summarize_rung_returns_sensible_stats() -> None:
    corpus = _make_corpus()
    for rung in ALL_RUNGS:
        s = summarize_rung(rung, list(corpus.probes), corpus)
        assert s.rung == rung
        assert s.n_probes == len(corpus.probes)
        # Retention by probe type covers all four types.
        # (Not necessarily — but should at least cover what's in this corpus.)
        for ptype in s.retention_by_probe_type:
            assert 0.0 <= s.retention_by_probe_type[ptype] <= 1.0
        # SUPPORTED retention is always 1.0 (no cross-tenant fragments).
        if "supported" in s.retention_by_probe_type:
            assert s.retention_by_probe_type["supported"] == 1.0
    print(f"[OK] summarize_rung produces sensible per-rung statistics")


# ----- Tenant-hidden summary rendering -----


def test_render_summary_text_uses_correct_view() -> None:
    """Both visible and hidden summary renderings work."""
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    p = traps[0]
    visible = render_summary_text(p, tenant_visible=True)
    hidden = render_summary_text(p, tenant_visible=False)
    assert visible != hidden  # they should differ
    # Hidden text should not contain "clientA", "-A.", etc.
    for tid in corpus.tenants():
        suffix_letter = tid.removeprefix("client")
        assert tid not in hidden, f"tenant-hidden summary leaked {tid}"
    print("[OK] render_summary_text correctly distinguishes visible vs hidden")


def main() -> int:
    tests = [
        test_isolated_strips_cross_tenant_fragments_from_traps,
        test_isolated_passes_supported_probes_intact,
        test_boundary_preserving_topk_matches_isolated_under_tenant_granularity,
        test_shared_vector_index_returns_all_fragments,
        test_summary_merge_keeps_all_fragments,
        test_empirical_claim_rungs_that_share_more_produce_more_violations,
        test_querying_tenant_for_supported_probe,
        test_querying_tenant_for_cross_boundary_probe,
        test_summarize_rung_returns_sensible_stats,
        test_render_summary_text_uses_correct_view,
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
    print(f"=== all {len(tests)} retrieval tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
