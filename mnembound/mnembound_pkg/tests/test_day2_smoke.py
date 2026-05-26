"""
Smoke test for the corpus generator.

Verifies:
1. Generator runs end-to-end at 10 tenants × 100 records without error.
2. Default boundary granularity is TENANT (b-A, not b-A-0000).
3. Disjoint-pool invariants pass.
4. Determinism: same seed -> identical output.
5. Per-tenant record counts are correct.
6. Each record produces exactly 5 fragments.
7. Fragments expose tenant-visible AND tenant-hidden views correctly.
8. ENGAGEMENT and EVENT boundary granularities still work.
9. tenant_prefix() handles indices beyond 25 cleanly.
10. Validation correctly catches an injected disjointness violation.
11. Reasonable performance.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.generator import generate_corpus
from mnembound.identifiers import BoundaryGranularity, tenant_prefix
from mnembound.schema import ALL_ROLES
from mnembound.validation import ValidationError


def test_generator_runs_clean() -> None:
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=100)

    assert len(corpus.records) == 1000, f"got {len(corpus.records)} records"
    assert len(corpus.fragments) == 5000, f"got {len(corpus.fragments)} fragments"
    assert len(corpus.tenants()) == 10

    for tid in corpus.tenants():
        n = len(corpus.records_for_tenant(tid))
        assert n == 100, f"tenant {tid} has {n} records"

    assert corpus.resolver is not None, "resolver was not built"
    assert corpus.salt.startswith("mnembound:"), "salt malformed"
    assert len(corpus.entity_sanitizer) > 0, "entity sanitizer was not built"

    print("[OK] generator produced 1000 records / 5000 fragments / 10 tenants")


def test_default_boundary_is_tenant_level() -> None:
    """
    With BoundaryGranularity.TENANT (the default), all records within a
    tenant share one boundary_id matching `b-<tenant_prefix>`.
    """
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=100)

    boundaries_by_tenant: dict[str, set[str]] = {}
    for r in corpus.records:
        boundaries_by_tenant.setdefault(r.client_id, set()).add(r.boundary_id)

    for tid, boundaries in boundaries_by_tenant.items():
        assert len(boundaries) == 1, (
            f"tenant {tid} has {len(boundaries)} boundaries under TENANT "
            f"granularity, expected exactly 1: {boundaries}"
        )
        boundary = next(iter(boundaries))
        prefix_letter = tid.removeprefix("client")
        expected = f"b-{prefix_letter}"
        assert boundary == expected, (
            f"tenant {tid} boundary {boundary!r} != expected {expected!r}"
        )

    print("[OK] default boundary granularity is TENANT (one boundary per tenant)")


def test_engagement_boundary_granularity() -> None:
    """ENGAGEMENT granularity gives multiple boundaries per tenant."""
    corpus = generate_corpus(
        seed=42,
        n_tenants=3,
        records_per_tenant=50,
        boundary_granularity=BoundaryGranularity.ENGAGEMENT,
        n_engagements_per_tenant=5,
    )

    boundaries_by_tenant: dict[str, set[str]] = {}
    for r in corpus.records:
        boundaries_by_tenant.setdefault(r.client_id, set()).add(r.boundary_id)

    for tid, boundaries in boundaries_by_tenant.items():
        assert len(boundaries) == 5, (
            f"tenant {tid} has {len(boundaries)} engagement boundaries, "
            f"expected 5: {sorted(boundaries)}"
        )

    print("[OK] ENGAGEMENT granularity produces 5 boundaries per tenant")


def test_event_boundary_granularity_still_works() -> None:
    """EVENT granularity is still supported for stress testing."""
    corpus = generate_corpus(
        seed=42,
        n_tenants=3,
        records_per_tenant=30,
        boundary_granularity=BoundaryGranularity.EVENT,
    )

    boundaries_by_tenant: dict[str, set[str]] = {}
    for r in corpus.records:
        boundaries_by_tenant.setdefault(r.client_id, set()).add(r.boundary_id)

    for tid, boundaries in boundaries_by_tenant.items():
        # 30 records per tenant -> 30 unique boundaries under EVENT
        # granularity.
        assert len(boundaries) == 30, (
            f"tenant {tid} has {len(boundaries)} boundaries under EVENT, "
            "expected 30"
        )

    print("[OK] EVENT granularity preserves one-boundary-per-record")


def test_determinism() -> None:
    c1 = generate_corpus(seed=7, n_tenants=10, records_per_tenant=50)
    c2 = generate_corpus(seed=7, n_tenants=10, records_per_tenant=50)

    assert c1.records == c2.records
    assert c1.fragments == c2.fragments
    assert c1.salt == c2.salt
    assert dict(c1.tenant_entity_pools) == dict(c2.tenant_entity_pools)
    assert dict(c1.entity_sanitizer) == dict(c2.entity_sanitizer), (
        "entity sanitizer differs across runs with same seed"
    )

    c3 = generate_corpus(seed=8, n_tenants=10, records_per_tenant=50)
    assert c1.records != c3.records
    assert c1.salt != c3.salt

    print("[OK] determinism verified across seeds and includes the sanitizer")


def test_fragment_record_consistency() -> None:
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=100)
    record_index = {r.record_id: r for r in corpus.records}

    frags_by_record: dict[str, list] = {}
    for f in corpus.fragments:
        frags_by_record.setdefault(f.record_id, []).append(f)

    for rid, frags in frags_by_record.items():
        r = record_index[rid]
        roles_seen = {f.role for f in frags}
        assert roles_seen == set(ALL_ROLES), (
            f"record {rid} has fragment roles {roles_seen}, expected all 5"
        )
        for f in frags:
            assert f.boundary_id == r.boundary_id
            assert f.event_id == r.event_id
            assert f.tenant_id == r.client_id
            assert f.value == r.component(f.role)
            assert f.text == r.text_for(f.role), (
                f"fragment text drift: fragment has {f.text!r}, "
                f"record has {r.text_for(f.role)!r}"
            )

    print("[OK] 5000 fragments are consistent with their parent records")


def test_tenant_hidden_view_strips_tenant_identity() -> None:
    """
    The tenant-hidden view of every fragment must contain no tenant prefix
    in either `value` or `text`.
    """
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=100)

    tenant_revealing_substrings = set()
    for tid in corpus.tenants():
        prefix_letter = tid.removeprefix("client")
        # 'clientA', '-A.', '-A@', '/A/', '_A' all reveal tenant identity in
        # entity strings.
        tenant_revealing_substrings.add(tid)
        tenant_revealing_substrings.add(f"-{prefix_letter}.")
        tenant_revealing_substrings.add(f"-{prefix_letter}@")
        tenant_revealing_substrings.add(f"/{prefix_letter}/")
        tenant_revealing_substrings.add(f"_{prefix_letter}")

    for f in corpus.fragments:
        hidden_view = f.view_for_v_frag(tenant_visible=False)
        for forbidden in tenant_revealing_substrings:
            assert forbidden not in hidden_view["value"], (
                f"hidden value {hidden_view['value']!r} contains tenant marker "
                f"{forbidden!r}"
            )
            assert forbidden not in hidden_view["text"], (
                f"hidden text {hidden_view['text']!r} contains tenant marker "
                f"{forbidden!r}"
            )

    print("[OK] tenant-hidden view strips every tenant-revealing substring")


def test_tenant_visible_view_preserves_full_strings() -> None:
    """
    The tenant-visible view of a fragment matches the original record's
    component value and text exactly.
    """
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=100)
    record_index = {r.record_id: r for r in corpus.records}

    for f in corpus.fragments:
        visible_view = f.view_for_v_frag(tenant_visible=True)
        r = record_index[f.record_id]
        assert visible_view["value"] == r.component(f.role)
        assert visible_view["text"] == r.text_for(f.role)

    print("[OK] tenant-visible view matches the underlying record verbatim")


def test_disjoint_pools_are_actually_disjoint() -> None:
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=100)
    pools = corpus.tenant_entity_pools
    tids = sorted(pools.keys())
    for i, ti in enumerate(tids):
        for tj in tids[i + 1 :]:
            inter = pools[ti] & pools[tj]
            assert not inter, (
                f"tenants {ti} and {tj} share entities: {sorted(inter)[:3]}"
            )

    for r in corpus.records:
        union = pools[r.client_id]
        assert r.subject in union
        assert r.object_ in union

    print("[OK] all 10 tenant pools are pairwise disjoint")


def test_event_ids_are_globally_unique() -> None:
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=100)
    event_ids = [r.event_id for r in corpus.records]
    assert len(event_ids) == len(set(event_ids))
    print(f"[OK] {len(event_ids)} event_ids are globally unique")


def test_boundary_event_pairs_are_unique() -> None:
    """
    Even though many records share a boundary under TENANT granularity, the
    (boundary_id, event_id) pair must be unique across the entire corpus.
    """
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=100)
    pairs = [(r.boundary_id, r.event_id) for r in corpus.records]
    assert len(pairs) == len(set(pairs)), "duplicate (boundary, event) pairs"
    print(f"[OK] {len(pairs)} (boundary_id, event_id) pairs are unique")


def test_validation_catches_synthetic_violation() -> None:
    from dataclasses import replace
    from types import MappingProxyType

    from mnembound.validation import BridgePools, validate_corpus

    corpus = generate_corpus(seed=42, n_tenants=2, records_per_tenant=10)

    tids = sorted(corpus.tenant_entity_pools.keys())
    leaked_entity = next(iter(corpus.tenant_entity_pools[tids[0]]))
    corrupted_pools = {
        tids[0]: corpus.tenant_entity_pools[tids[0]],
        tids[1]: corpus.tenant_entity_pools[tids[1]] | {leaked_entity},
    }
    corrupted_corpus = replace(
        corpus, tenant_entity_pools=MappingProxyType(corrupted_pools)
    )

    try:
        validate_corpus(corrupted_corpus, bridges=BridgePools.empty())
        raise AssertionError("validation should have failed on a leaked entity")
    except ValidationError as e:
        assert "disjointness" in str(e), f"unexpected error message: {e}"

    print("[OK] validation correctly catches an injected disjointness violation")


def test_tenant_prefix_edge_cases() -> None:
    assert tenant_prefix(0) == "A"
    assert tenant_prefix(25) == "Z"
    assert tenant_prefix(26) == "AA"
    assert tenant_prefix(27) == "AB"
    assert tenant_prefix(51) == "AZ"
    assert tenant_prefix(52) == "BA"
    assert tenant_prefix(701) == "ZZ"
    assert tenant_prefix(702) == "AAA"
    assert tenant_prefix(703) == "AAB"

    prefixes = [tenant_prefix(i) for i in range(1000)]
    assert len(set(prefixes)) == 1000

    print("[OK] tenant_prefix handles indices 0..1000 correctly and uniquely")


def test_performance() -> None:
    start = time.perf_counter()
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=100)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"generation took {elapsed:.2f}s, expected < 2s"
    assert len(corpus.records) == 1000
    print(f"[OK] smoke config (1000 records) generated in {elapsed:.2f}s")


def test_scale_500_per_tenant() -> None:
    start = time.perf_counter()
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=500)
    elapsed = time.perf_counter() - start
    assert len(corpus.records) == 5000
    assert len(corpus.fragments) == 25000
    assert elapsed < 5.0, f"full config took {elapsed:.2f}s, expected < 5s"
    print(
        f"[OK] full config (5000 records / 25000 fragments) generated in {elapsed:.2f}s"
    )


def main() -> int:
    tests = [
        test_generator_runs_clean,
        test_default_boundary_is_tenant_level,
        test_engagement_boundary_granularity,
        test_event_boundary_granularity_still_works,
        test_determinism,
        test_fragment_record_consistency,
        test_tenant_hidden_view_strips_tenant_identity,
        test_tenant_visible_view_preserves_full_strings,
        test_disjoint_pools_are_actually_disjoint,
        test_event_ids_are_globally_unique,
        test_boundary_event_pairs_are_unique,
        test_validation_catches_synthetic_violation,
        test_tenant_prefix_edge_cases,
        test_performance,
        test_scale_500_per_tenant,
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
    print(f"=== all {len(tests)} tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
