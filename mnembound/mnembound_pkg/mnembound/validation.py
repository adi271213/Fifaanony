"""
mnembound.validation — Invariant checks for generated corpora.

These checks enforce the construction-decidable ground truth that the
paper's empirical claim depends on. A violation here means the generator
has a bug AND the ground-truth labels for some probes will be wrong.
Every violation raises ValidationError.

- Removed the global `allow_bridges=True` bypass that silently skipped
  entity-membership checks for both subject and object_.
- Replaced with an explicit `BridgePools` whitelist that names exactly which
  cross-tenant entities are allowed by design. Day 3 populates this; Day 2
  passes an empty BridgePools, which enforces strict per-tenant membership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from mnembound.identifiers import TenantEntityPool
from mnembound.schema import Corpus, Fragment, Record


class ValidationError(AssertionError):
    """A corpus invariant was violated. The corpus is unusable as-is."""


@dataclass(frozen=True)
class BridgePools:
    """
    Explicit whitelist of cross-tenant bridge entities.

    Each frozenset names the SHARED entities of a given type that are
    permitted to appear in records across tenants. A record entity that
    is not in its tenant's per-tenant pool MUST be in the corresponding
    bridge pool, otherwise the disjoint-pool invariant is violated.

    Day 2 uses BridgePools.empty(). Day 3 will populate these with the
    actual shared IPs, hashes, CVEs, vendors, and threat actors that the
    bridge-insertion step creates.
    """

    shared_ips: frozenset[str] = field(default_factory=frozenset)
    shared_hashes: frozenset[str] = field(default_factory=frozenset)
    shared_cves: frozenset[str] = field(default_factory=frozenset)
    shared_vendors: frozenset[str] = field(default_factory=frozenset)
    shared_threat_actors: frozenset[str] = field(default_factory=frozenset)

    @staticmethod
    def empty() -> "BridgePools":
        return BridgePools()

    def all_bridges(self) -> frozenset[str]:
        """Union of every bridge entity across all bridge categories."""
        return (
            self.shared_ips
            | self.shared_hashes
            | self.shared_cves
            | self.shared_vendors
            | self.shared_threat_actors
        )

    def is_empty(self) -> bool:
        return not self.all_bridges()


def assert_disjoint_entity_pools(pools: Mapping[str, TenantEntityPool]) -> None:
    """Every entity string belongs to exactly one tenant's per-tenant pool."""
    seen: dict[str, str] = {}
    for tid, pool in pools.items():
        for entity in pool.all_entities():
            if entity in seen and seen[entity] != tid:
                raise ValidationError(
                    f"entity {entity!r} appears in both {seen[entity]!r} "
                    f"and {tid!r}; disjoint-pool invariant violated"
                )
            seen[entity] = tid


def assert_disjoint_boundaries(records: tuple[Record, ...]) -> None:
    """Every boundary_id belongs to exactly one tenant."""
    seen: dict[str, str] = {}
    for r in records:
        if r.boundary_id in seen and seen[r.boundary_id] != r.client_id:
            raise ValidationError(
                f"boundary_id {r.boundary_id!r} appears in both "
                f"{seen[r.boundary_id]!r} and {r.client_id!r}"
            )
        seen[r.boundary_id] = r.client_id


def assert_disjoint_events(records: tuple[Record, ...]) -> None:
    """(boundary_id, event_id) pairs are unique; event_ids alone are also unique."""
    seen_pairs: set[tuple[str, str]] = set()
    seen_events: set[str] = set()
    for r in records:
        key = (r.boundary_id, r.event_id)
        if key in seen_pairs:
            raise ValidationError(
                f"(boundary_id={r.boundary_id!r}, event_id={r.event_id!r}) "
                f"appears in multiple records"
            )
        seen_pairs.add(key)
        if r.event_id in seen_events:
            raise ValidationError(
                f"event_id {r.event_id!r} appears in multiple records "
                "across boundaries; namespacing convention violated"
            )
        seen_events.add(r.event_id)


def assert_unique_record_ids(records: tuple[Record, ...]) -> None:
    """No two records share a record_id."""
    seen: set[str] = set()
    for r in records:
        if r.record_id in seen:
            raise ValidationError(f"duplicate record_id {r.record_id!r}")
        seen.add(r.record_id)


def assert_record_entities_in_pool_or_bridges(
    records: tuple[Record, ...],
    tenant_entity_unions: Mapping[str, frozenset[str]],
    bridges: BridgePools,
) -> None:
    """
    Each record's subject and object_ are either in the tenant's per-tenant
    pool OR in an explicit bridge pool.

    This replaces the earlier `allow_bridges=True` bypass. When
    bridges are absent (Day 2), this reduces to strict per-tenant
    membership. When bridges are present (Day 3+), a value is permitted
    if and only if it is explicitly in the bridge whitelist.
    """
    bridge_union = bridges.all_bridges()
    for r in records:
        if r.client_id not in tenant_entity_unions:
            raise ValidationError(
                f"record {r.record_id} client_id {r.client_id!r} has no "
                "corresponding entity pool"
            )
        pool_union = tenant_entity_unions[r.client_id]

        for role_name, value in (("subject", r.subject), ("object_", r.object_)):
            if value in pool_union:
                continue
            if value in bridge_union:
                continue
            raise ValidationError(
                f"record {r.record_id} {role_name} {value!r} is neither in "
                f"tenant {r.client_id}'s per-tenant pool nor in the bridge "
                "whitelist; disjoint-pool invariant violated"
            )


def assert_fragments_consistent_with_records(
    fragments: tuple[Fragment, ...], records: tuple[Record, ...]
) -> None:
    """Every fragment's hidden metadata matches its parent record exactly."""
    record_index: dict[str, Record] = {r.record_id: r for r in records}
    for f in fragments:
        if f.record_id not in record_index:
            raise ValidationError(
                f"fragment with public id {f.public_provenance_id} references "
                f"unknown record_id {f.record_id!r}"
            )
        r = record_index[f.record_id]
        if f.boundary_id != r.boundary_id:
            raise ValidationError(
                f"fragment {f.public_provenance_id} boundary_id={f.boundary_id!r} "
                f"!= record {r.record_id} boundary_id={r.boundary_id!r}"
            )
        if f.event_id != r.event_id:
            raise ValidationError(
                f"fragment {f.public_provenance_id} event_id={f.event_id!r} "
                f"!= record {r.record_id} event_id={r.event_id!r}"
            )
        if f.tenant_id != r.client_id:
            raise ValidationError(
                f"fragment {f.public_provenance_id} tenant_id={f.tenant_id!r} "
                f"!= record {r.record_id} client_id={r.client_id!r}"
            )


def assert_pool_bridge_disjoint(
    tenant_entity_unions: Mapping[str, frozenset[str]],
    bridges: BridgePools,
) -> None:
    """
    No bridge entity appears in any per-tenant pool.

    If a bridge entity were also in a tenant pool, the "is this a bridge?"
    classification would be ambiguous and the disjoint-pool guarantee
    would be partially undermined.
    """
    bridge_union = bridges.all_bridges()
    if not bridge_union:
        return
    for tid, pool_union in tenant_entity_unions.items():
        overlap = pool_union & bridge_union
        if overlap:
            raise ValidationError(
                f"bridge pool overlaps tenant {tid!r} per-tenant pool on "
                f"{sorted(overlap)[:3]}; bridges must be distinct from "
                "per-tenant entities"
            )


def validate_corpus(corpus: Corpus, bridges: BridgePools | None = None) -> None:
    """
    Run every invariant check on a generated corpus.

    Args:
        corpus: The Corpus to validate.
        bridges: Explicit bridge whitelist. Pass BridgePools.empty() (or
            None, which is normalized to empty) for Day 2; pass the
            populated BridgePools used by the generator on Day 3+.

    Raises ValidationError on first failure with a precise message.
    """
    if not corpus.tenant_entity_pools:
        raise ValidationError(
            "corpus has no tenant_entity_pools; cannot validate disjointness"
        )

    bridges = bridges if bridges is not None else BridgePools.empty()

    assert_pool_bridge_disjoint(corpus.tenant_entity_pools, bridges)
    assert_disjoint_boundaries(corpus.records)
    assert_disjoint_events(corpus.records)
    assert_unique_record_ids(corpus.records)
    assert_record_entities_in_pool_or_bridges(
        corpus.records, corpus.tenant_entity_pools, bridges
    )
    assert_fragments_consistent_with_records(corpus.fragments, corpus.records)

    # Final cross-tenant entity disjointness check on the stored unions.
    seen: dict[str, str] = {}
    for tid, entity_union in corpus.tenant_entity_pools.items():
        for entity in entity_union:
            if entity in seen and seen[entity] != tid:
                raise ValidationError(
                    f"entity {entity!r} appears in both {seen[entity]!r} "
                    f"and {tid!r}; corpus-level disjointness violated"
                )
            seen[entity] = tid
