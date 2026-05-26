"""
Cross-tenant bridge entity tests.

Verifies:
- BridgePools are built deterministically.
- BridgeDensitySpec validates and caps densities at 0.15.
- assign_bridges_to_records produces the right counts at the right
  densities, deterministically.
- generate_corpus accepts the bridge parameters and produces corpora that
  validate against the populated BridgePools.
- Slot-bridges actually appear in subject/object slots; component_text is
  updated coherently.
- Extra-bridges appear in Record.extra['bridge'].
- Tenant-hidden view does NOT sanitize bridge entities (they remain
  visible as ip:.../hash:.../cve:.../vendor:.../actor:... in both view
  modes; this is the §5 realism story).
- Cross-boundary trap preference: prefer_bridge_linked=True produces
  traps with non-empty bridge_context when bridges exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.bridges import (
    BridgeAssignment,
    BridgeDensitySpec,
    DEFAULT_BRIDGE_POOL_SIZES,
    SLOT_ELIGIBLE_BRIDGE_TYPES,
    assign_bridges_to_records,
    build_bridge_pools,
)
from mnembound.generator import generate_corpus
from mnembound.probes import generate_cross_boundary_traps
from mnembound.schema import ProbeType


# ----- BridgePools and BridgeDensitySpec -----


def test_bridge_pools_have_expected_sizes() -> None:
    bp = build_bridge_pools(seed=1)
    assert len(bp.shared_ips) == DEFAULT_BRIDGE_POOL_SIZES["shared_ip"]
    assert len(bp.shared_hashes) == DEFAULT_BRIDGE_POOL_SIZES["malware_hash"]
    assert len(bp.shared_cves) == DEFAULT_BRIDGE_POOL_SIZES["cve"]
    assert len(bp.shared_vendors) == DEFAULT_BRIDGE_POOL_SIZES["vendor"]
    assert len(bp.shared_threat_actors) == DEFAULT_BRIDGE_POOL_SIZES["threat_actor"]
    print(
        f"[OK] bridge pools have default sizes: "
        f"{DEFAULT_BRIDGE_POOL_SIZES['shared_ip']} IPs, "
        f"{DEFAULT_BRIDGE_POOL_SIZES['malware_hash']} hashes, "
        f"{DEFAULT_BRIDGE_POOL_SIZES['cve']} CVEs, "
        f"{DEFAULT_BRIDGE_POOL_SIZES['vendor']} vendors, "
        f"{DEFAULT_BRIDGE_POOL_SIZES['threat_actor']} actors"
    )


def test_bridge_pools_are_deterministic() -> None:
    bp1 = build_bridge_pools(seed=42)
    bp2 = build_bridge_pools(seed=42)
    assert bp1.shared_ips == bp2.shared_ips
    assert bp1.shared_hashes == bp2.shared_hashes
    assert bp1.shared_cves == bp2.shared_cves
    bp3 = build_bridge_pools(seed=43)
    assert bp1.shared_hashes != bp3.shared_hashes  # different seed -> different hashes
    print("[OK] bridge pools are deterministic across seeds")


def test_bridge_density_spec_validates() -> None:
    # Default is fine
    spec = BridgeDensitySpec()
    assert spec.extra == 0.05
    assert spec.slot == 0.0

    # Negative values rejected
    try:
        BridgeDensitySpec(extra=-0.01)
        raise AssertionError("negative density not rejected")
    except ValueError:
        pass

    # > 1.0 rejected
    try:
        BridgeDensitySpec(extra=1.5)
        raise AssertionError("density > 1.0 not rejected")
    except ValueError:
        pass

    # > realism cap (0.15) rejected
    try:
        BridgeDensitySpec(extra=0.2)
        raise AssertionError("density > cap not rejected")
    except ValueError:
        pass

    # Cap of exactly 0.15 accepted
    BridgeDensitySpec(extra=0.15, slot=0.15)
    print("[OK] BridgeDensitySpec validates [0, 0.15] and rejects values outside")


# ----- assign_bridges_to_records -----


def test_assign_bridges_produces_correct_counts() -> None:
    record_ids = [f"r-A-{i:06d}" for i in range(1000)]
    bridges = build_bridge_pools(seed=1)

    assignment = assign_bridges_to_records(
        record_ids=record_ids,
        bridges=bridges,
        density=BridgeDensitySpec(extra=0.05, slot=0.10),
        seed=42,
    )

    # 5% extra: 50 records.
    assert len(assignment.extra_bridges_by_record_id) == 50
    # 10% slot: 100 records.
    assert len(assignment.slot_bridges_by_record_id) == 100

    # Every slot assignment is on an eligible category.
    for rid, (slot_name, entity) in assignment.slot_bridges_by_record_id.items():
        assert slot_name in ("subject", "object_")
        prefix = entity.split(":", 1)[0]
        # The prefix appears in SLOT_ELIGIBLE_BRIDGE_TYPES via the type letter
        # mapping (h_/ip:, hash:, etc.).
        assert prefix in ("ip", "hash", "cve", "vendor", "actor"), (
            f"slot entity {entity!r} has unexpected prefix"
        )

    print(
        f"[OK] assign_bridges produced {len(assignment.extra_bridges_by_record_id)} "
        f"extra + {len(assignment.slot_bridges_by_record_id)} slot assignments"
    )


def test_assign_bridges_is_deterministic() -> None:
    record_ids = [f"r-A-{i:06d}" for i in range(500)]
    bridges = build_bridge_pools(seed=1)
    a1 = assign_bridges_to_records(
        record_ids, bridges, BridgeDensitySpec(extra=0.10), seed=99
    )
    a2 = assign_bridges_to_records(
        record_ids, bridges, BridgeDensitySpec(extra=0.10), seed=99
    )
    assert a1.extra_bridges_by_record_id == a2.extra_bridges_by_record_id
    assert a1.slot_bridges_by_record_id == a2.slot_bridges_by_record_id
    print("[OK] assign_bridges_to_records is deterministic")


# ----- Generator integration -----


def test_generator_default_is_no_bridges() -> None:
    """Default generator behavior preserves the pre-bridge baseline (no bridges)."""
    c = generate_corpus(seed=42, n_tenants=5, records_per_tenant=50)
    n_extra = sum(1 for r in c.records if r.extra.get("bridge"))
    assert n_extra == 0
    # No slot-bridge entities either.
    for r in c.records:
        for slot_val in (r.subject, r.object_):
            prefix = slot_val.split(":", 1)[0]
            assert prefix not in ("ip", "hash", "cve", "vendor", "actor"), (
                f"unexpected bridge-typed slot in default generation: {slot_val}"
            )
    print("[OK] default generator produces no bridges (backwards-compat)")


def test_generator_extra_bridges_appear_at_target_density() -> None:
    c = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=200,
        bridge_density=BridgeDensitySpec(extra=0.05),
    )
    n_with_extra = sum(1 for r in c.records if r.extra.get("bridge"))
    expected = round(0.05 * len(c.records))
    assert n_with_extra == expected, (
        f"got {n_with_extra} extra bridges, expected {expected}"
    )
    print(f"[OK] generator produced {n_with_extra}/{len(c.records)} extra bridges (5%)")


def test_generator_slot_bridges_appear_at_target_density() -> None:
    c = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=200,
        bridge_density=BridgeDensitySpec(extra=0.0, slot=0.10),
    )
    n_with_slot = sum(
        1 for r in c.records
        if any(
            (slot or "").split(":", 1)[0] in ("ip", "hash", "cve", "vendor", "actor")
            for slot in (r.subject, r.object_)
        )
    )
    expected = round(0.10 * len(c.records))
    assert n_with_slot == expected, (
        f"got {n_with_slot} slot-bridges, expected {expected}"
    )
    print(f"[OK] generator produced {n_with_slot}/{len(c.records)} slot-bridges (10%)")


def test_generator_with_bridges_validates() -> None:
    """Generator should validate corpora at max bridge density."""
    c = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=100,
        bridge_density=BridgeDensitySpec(extra=0.15, slot=0.15),
    )
    # If validation passes (it must — generate_corpus runs validate_corpus
    # before returning), we made it here.
    assert c.bridge_pools is not None
    print(f"[OK] corpus with 15% extra + 15% slot validates")


def test_slot_bridge_component_text_updates() -> None:
    """When a slot is replaced by a bridge entity, component_text reflects it."""
    c = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=200,
        bridge_density=BridgeDensitySpec(extra=0.0, slot=0.10),
    )
    n_checked = 0
    for r in c.records:
        if r.object_.startswith(("ip:", "hash:", "cve:", "vendor:")):
            # The object text should mention the bridge entity.
            for role, text in r.component_text.items():
                if role.value == "object":
                    assert r.object_ in text, (
                        f"component_text[object] for record {r.record_id} "
                        f"does not contain bridge entity {r.object_}: {text!r}"
                    )
            n_checked += 1
            if n_checked >= 10:
                break
    assert n_checked >= 5, f"only found {n_checked} slot-bridged records to check"
    print(f"[OK] slot-bridged records have updated component_text ({n_checked} checked)")


# ----- Tenant-hidden view treatment of bridges -----


def test_bridge_entities_are_not_sanitized() -> None:
    """
    Bridge entities are tenant-neutral by construction and should remain
    VISIBLE in both view modes. The §5 realism story depends on this:
    an aggregator must see that two tenants share the same IP.
    """
    c = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=100,
        bridge_density=BridgeDensitySpec(extra=0.10),
    )
    bridge_values = {r.extra.get("bridge") for r in c.records if r.extra.get("bridge")}
    sanitizer = c.entity_sanitizer
    for bridge in bridge_values:
        assert bridge not in sanitizer, (
            f"bridge entity {bridge!r} was put in the entity sanitizer; "
            "bridges must remain tenant-visible by design"
        )
    print(f"[OK] {len(bridge_values)} bridge entities are NOT sanitized")


# ----- Cross-boundary trap bridge preference -----


def test_bridge_preferred_traps_have_bridge_context() -> None:
    """When prefer_bridge_linked=True, traps carry non-empty bridge_context."""
    c = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=200,
        bridge_density=BridgeDensitySpec(extra=0.10, slot=0.05),
    )
    traps = generate_cross_boundary_traps(
        c, n_probes=30, seed=1, prefer_bridge_linked=True
    )
    n_with_bridge = sum(1 for t in traps if t.is_bridge_linked())
    assert n_with_bridge >= 20, (
        f"only {n_with_bridge}/{len(traps)} bridge-preferred traps have "
        "non-empty bridge_context"
    )
    print(
        f"[OK] {n_with_bridge}/{len(traps)} bridge-preferred traps carry "
        "shared-bridge evidence"
    )


def test_random_traps_have_empty_bridge_context_typically() -> None:
    """
    Pure-random traps usually have empty bridge_context — they sample
    source events without regard to bridges, so accidental bridge-sharing
    is rare at 5% density.
    """
    c = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=200,
        bridge_density=BridgeDensitySpec(extra=0.05),
    )
    traps = generate_cross_boundary_traps(c, n_probes=30, seed=1)
    n_with_bridge = sum(1 for t in traps if t.is_bridge_linked())
    # At 5% extra density across 5 source events, the expected number of
    # accidental bridge-shared trap pairs is low. We allow up to ~30% as a
    # generous statistical bound (not strictly enforced; the test is about
    # the qualitative difference vs prefer_bridge_linked=True).
    assert n_with_bridge < len(traps) // 2, (
        f"{n_with_bridge}/{len(traps)} random traps have bridges, expected fewer"
    )
    print(
        f"[OK] only {n_with_bridge}/{len(traps)} random traps incidentally "
        "share bridges (expected: rare)"
    )


def test_bridge_preferred_traps_remain_boundary_invalid() -> None:
    """Bridge preference does not relax the boundary-invalidity invariant."""
    c = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=200,
        bridge_density=BridgeDensitySpec(extra=0.10),
    )
    traps = generate_cross_boundary_traps(
        c, n_probes=20, seed=2, prefer_bridge_linked=True
    )
    for t in traps:
        assert t.is_fragment_supported
        assert not t.is_event_supported
        assert t.is_mnemonic_boundary_violation()
        assert len(t.boundary_event_pairs) >= 2
    print(
        f"[OK] all {len(traps)} bridge-preferred traps remain boundary-invalid"
    )


def main() -> int:
    tests = [
        test_bridge_pools_have_expected_sizes,
        test_bridge_pools_are_deterministic,
        test_bridge_density_spec_validates,
        test_assign_bridges_produces_correct_counts,
        test_assign_bridges_is_deterministic,
        test_generator_default_is_no_bridges,
        test_generator_extra_bridges_appear_at_target_density,
        test_generator_slot_bridges_appear_at_target_density,
        test_generator_with_bridges_validates,
        test_slot_bridge_component_text_updates,
        test_bridge_entities_are_not_sanitized,
        test_bridge_preferred_traps_have_bridge_context,
        test_random_traps_have_empty_bridge_context_typically,
        test_bridge_preferred_traps_remain_boundary_invalid,
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
    print(f"=== all {len(tests)} bridge tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
