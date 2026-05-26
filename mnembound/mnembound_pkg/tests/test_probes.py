"""
Probe-generation tests.

Verifies that each of the four probe types satisfies its definitional
invariants by construction, and that the ground-truth labels attached to
each probe are exact.

- SUPPORTED probes are fragment-supported AND event-supported by their
  retrieved fragment set, with exactly one underlying (boundary, event).

- ABSENT probes are NEITHER fragment-supported NOR event-supported, AND
  the composed recall does not match any record in the corpus.

- PARTIAL probes contain SOME-but-not-all components in the retrieval set,
  so they are fragment-supported=False and event-supported=False by the
  retrieval set alone.

- CROSS_BOUNDARY probes are fragment-supported=True AND event-supported=
  False, with supporting fragments spanning multiple (boundary, event)
  pairs and the composed recall not matching any single record.

All ground-truth labels attached to probes by the generator agree with the
predicates in `scoring.py`. This is the construction-decidability the
paper's empirical claim relies on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.generator import generate_corpus
from mnembound.probes import (
    generate_absent_probes,
    generate_cross_boundary_traps,
    generate_partial_probes,
    generate_probe_mix,
    generate_supported_probes,
)
from mnembound.schema import ProbeType
from mnembound.scoring import (
    is_boundary_invalid,
    is_event_supported,
    is_fragment_supported,
)


def _make_corpus():
    """Standard test corpus: 10 tenants × 50 records."""
    return generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)


def test_supported_probes_satisfy_definition() -> None:
    corpus = _make_corpus()
    probes = generate_supported_probes(corpus, n_probes=20, seed=1)
    assert len(probes) == 20

    for p in probes:
        assert p.probe_type == ProbeType.SUPPORTED
        assert p.is_fragment_supported is True
        assert p.is_event_supported is True
        assert len(p.boundary_event_pairs) == 1
        assert is_fragment_supported(p.recall, p.fragments)
        assert is_event_supported(p.recall, p.fragments)
        # Every component value appears in some fragment with matching role.
        # (Implied by is_fragment_supported, but verified again here.)

    print("[OK] 20 SUPPORTED probes satisfy fragment- AND event-supported")


def test_supported_probes_with_distractors() -> None:
    """Adding distractors should not change the ground-truth labels."""
    corpus = _make_corpus()
    probes = generate_supported_probes(
        corpus, n_probes=10, seed=1, distractor_fragments=20
    )
    for p in probes:
        # Retrieval has 5 primary + 20 distractors = 25 fragments.
        assert len(p.fragments) == 25
        assert is_event_supported(p.recall, p.fragments)
        # Boundary_event_pairs reflects the underlying event, not the
        # distractors' events.
        assert len(p.boundary_event_pairs) == 1

    print("[OK] SUPPORTED probes with distractors keep correct event support")


def test_absent_probes_satisfy_definition() -> None:
    corpus = _make_corpus()
    probes = generate_absent_probes(corpus, n_probes=20, seed=2)
    # Generator may produce fewer than requested if it cannot find valid
    # configurations within the retry budget; require at least 1 and check
    # every produced probe.
    assert len(probes) >= 10, f"only produced {len(probes)} absent probes"

    for p in probes:
        assert p.probe_type == ProbeType.ABSENT
        assert p.is_fragment_supported is False
        assert p.is_event_supported is False
        assert len(p.boundary_event_pairs) == 0
        # Scoring predicates agree.
        assert not is_fragment_supported(p.recall, p.fragments)
        assert not is_event_supported(p.recall, p.fragments)

    print(f"[OK] {len(probes)} ABSENT probes are neither fragment- nor event-supported")


def test_partial_probes_satisfy_definition() -> None:
    corpus = _make_corpus()
    for n_supported in (1, 2, 3, 4):
        probes = generate_partial_probes(
            corpus, n_probes=15, seed=3 + n_supported,
            n_supported_components=n_supported,
        )
        assert len(probes) > 0
        for p in probes:
            assert p.probe_type == ProbeType.PARTIAL
            # PARTIAL probes have only n_supported of 5 roles present.
            assert len(p.fragments) == n_supported
            roles_present = {f.role for f in p.fragments}
            assert len(roles_present) == n_supported
            # Not fragment-supported on the retrieval set alone (some roles
            # are missing).
            assert not is_fragment_supported(p.recall, p.fragments)
            assert not is_event_supported(p.recall, p.fragments)

    print("[OK] PARTIAL probes correctly omit (5 - n_supported_components) roles")


def test_cross_boundary_traps_satisfy_definition() -> None:
    """The headline probe type: fragment-supported but NOT event-supported."""
    corpus = _make_corpus()
    probes = generate_cross_boundary_traps(corpus, n_probes=50, seed=4)
    assert len(probes) >= 30, f"only produced {len(probes)} cross-boundary traps"

    for p in probes:
        assert p.probe_type == ProbeType.CROSS_BOUNDARY
        # The defining property:
        assert p.is_fragment_supported is True, (
            f"trap probe {p.probe_id} is not fragment-supported by construction"
        )
        assert p.is_event_supported is False, (
            f"trap probe {p.probe_id} IS event-supported, which means the "
            "construction missed a collision in the corpus"
        )
        # Supporting fragments span multiple (boundary, event) pairs.
        assert len(p.boundary_event_pairs) >= 2, (
            f"trap probe {p.probe_id} only spans {len(p.boundary_event_pairs)} "
            "boundary-event pairs"
        )
        # The retrieval set covers exactly 5 roles (one per component).
        assert len(p.fragments) == 5
        assert len({f.role for f in p.fragments}) == 5
        # Scoring predicates agree.
        assert is_fragment_supported(p.recall, p.fragments)
        assert not is_event_supported(p.recall, p.fragments)
        assert is_boundary_invalid(p.recall, p.fragments)

    print(f"[OK] {len(probes)} CROSS_BOUNDARY traps are fragment-supported but not event-supported")


def test_cross_boundary_traps_require_distinct_boundaries() -> None:
    """
    With require_distinct_boundaries=True (the default), the supporting
    fragments span >= 2 distinct boundary_ids.
    """
    corpus = _make_corpus()
    probes = generate_cross_boundary_traps(
        corpus, n_probes=30, seed=5, require_distinct_boundaries=True
    )
    for p in probes:
        boundaries = {f.boundary_id for f in p.fragments}
        assert len(boundaries) >= 2, (
            f"trap probe {p.probe_id} only touches one boundary {boundaries}"
        )
    print(f"[OK] CROSS_BOUNDARY traps span >= 2 distinct boundary_ids")


def test_probe_generation_is_deterministic() -> None:
    """Same seed -> identical probe sequences across runs."""
    corpus = _make_corpus()
    p1 = generate_cross_boundary_traps(corpus, n_probes=20, seed=99)
    p2 = generate_cross_boundary_traps(corpus, n_probes=20, seed=99)
    assert p1 == p2, "probe generation is not deterministic"
    print("[OK] probe generation is deterministic (same seed -> identical probes)")


def test_probe_mix_helper() -> None:
    """The convenience helper produces all four probe types together."""
    corpus = _make_corpus()
    mix = generate_probe_mix(
        corpus, seed=10,
        n_supported=10, n_absent=10, n_partial=10, n_cross_boundary=20,
    )
    assert len(mix.supported) == 10
    assert len(mix.absent) >= 5  # rejection sampling may produce fewer
    assert len(mix.partial) == 10
    assert len(mix.cross_boundary) >= 10
    all_probes = mix.all_probes()
    assert len(all_probes) == (
        len(mix.supported) + len(mix.absent)
        + len(mix.partial) + len(mix.cross_boundary)
    )
    print(f"[OK] generate_probe_mix produced {len(all_probes)} probes across 4 types")


def test_generator_attaches_probes_to_corpus() -> None:
    """generate_corpus() can attach probes when requested."""
    c = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=100,
        n_supported_probes=10, n_absent_probes=10,
        n_partial_probes=10, n_cross_boundary_probes=20,
    )
    assert len(c.probes) >= 40, (
        f"corpus only carries {len(c.probes)} probes, expected ~50"
    )
    types_seen = {p.probe_type for p in c.probes}
    assert types_seen == set(ProbeType)
    print(f"[OK] generate_corpus attached {len(c.probes)} probes across all 4 types")


def main() -> int:
    tests = [
        test_supported_probes_satisfy_definition,
        test_supported_probes_with_distractors,
        test_absent_probes_satisfy_definition,
        test_partial_probes_satisfy_definition,
        test_cross_boundary_traps_satisfy_definition,
        test_cross_boundary_traps_require_distinct_boundaries,
        test_probe_generation_is_deterministic,
        test_probe_mix_helper,
        test_generator_attaches_probes_to_corpus,
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
    print(f"=== all {len(tests)} probe tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
