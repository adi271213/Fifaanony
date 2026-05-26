"""
mnembound.bridges — Cross-tenant bridge entities.

A "bridge" is an entity that legitimately appears in records belonging to
more than one tenant. SOC examples:

- A shared IP that participated in events at multiple MSSP clients
  (e.g., a phishing C2 server that hit clientA on Monday and clientB on
  Tuesday).
- A malware family observed at multiple sites.
- A CVE referenced by alerts at multiple sites.
- A vendor product (Vendor-X EDR) mentioned in alerts at multiple sites.
- A named threat actor attributed to incidents at multiple sites.
- A near-timestamp correlation (not a bridge entity per se, but a
  proximity signal that an aggregator might exploit).

Bridges are the realism layer of the benchmark. Without them, a cross-
boundary trap probe looks like an arbitrary stitched recall ("user A from
clientA, action B from clientB, ..."). With bridges, the trap looks like
the kind of confusion a real MSSP aggregator might produce: "a host scan
from 198.51.100.42 was followed by an authentication failure on the same
IP at clientB ten minutes later." The composed recall is still
event-false (no single event has all five components), but it now has a
plausible-looking thread of bridge evidence.

Design:

1. Two density knobs:
   - `bridge_density_extra` (default 5%): fraction of records that carry a
     bridge entity in their `Record.extra` field (as auxiliary context).
   - `bridge_density_slot` (default 0%): fraction of records that have a
     bridge entity in their subject or object slot. Off by default
     because it changes record semantics; on for realism-heavy
     experiments.

2. Five bridge categories (no near_timestamp; that is handled as a
   proximity signal at probe-construction time, not as a bridge entity):
   - shared_ip
   - malware_hash
   - cve
   - vendor
   - threat_actor

3. Each bridge category has a configurable pool size. Bridges are
   sampled deterministically from these pools.

4. Validation: every bridge must appear in the BridgePools whitelist.
   The existing `assert_record_entities_in_pool_or_bridges` check covers
   slot-bridges; a new `assert_record_extra_bridges_are_whitelisted`
   check covers extra-bridges.

5. Whitelist generation: `build_bridge_pools(seed, sizes)` returns a
   BridgePools whose entities are deterministic given the seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mnembound.validation import BridgePools


# Default bridge pool sizes. Calibrated so that at default density (5%
# extra) and a 10-tenant × 500-record corpus, every bridge entity is
# reused across ~2-5 records on average, which is the SOC-realistic
# range. (Lower reuse and bridges look incidental; higher reuse and they
# look like a single attacker dominating the dataset.)
DEFAULT_BRIDGE_POOL_SIZES: dict[str, int] = {
    "shared_ip": 50,
    "malware_hash": 30,
    "cve": 20,
    "vendor": 15,
    "threat_actor": 10,
}


# Default density knobs (fractions of records carrying a bridge).
DEFAULT_BRIDGE_DENSITY_EXTRA: float = 0.05
DEFAULT_BRIDGE_DENSITY_SLOT: float = 0.0

# When a record gets a slot-bridge, which slot does it go in? object_
# by default because subjects already carry tenant-specific user/process
# identifiers. shared_ip is the most natural object-slot bridge.
_BRIDGE_TO_SLOT: dict[str, str] = {
    "shared_ip": "object_",
    "malware_hash": "object_",
    "cve": "object_",
    "vendor": "object_",
    "threat_actor": "subject",
}


# Categories of bridges that may appear in a slot. We restrict slot-
# bridges to categories that naturally fill a subject or object position
# in a SOC event. Currently all five qualify; this list exists so future
# bridge categories (e.g., near_timestamp) can be added as extra-only.
SLOT_ELIGIBLE_BRIDGE_TYPES: frozenset[str] = frozenset({
    "shared_ip", "malware_hash", "cve", "vendor", "threat_actor",
})


# ----- Bridge entity factories -----


def _shared_ip(i: int) -> str:
    """RFC-5737 documentation range, padded to make 50 distinct entries."""
    # 198.51.100.0/24 has 256 usable addresses.
    return f"ip:198.51.100.{i % 256}"


def _malware_hash(i: int, rng: random.Random) -> str:
    """A 64-char hex string that looks like SHA-256."""
    return f"hash:" + "".join(rng.choice("0123456789abcdef") for _ in range(64))


def _cve(year: int, i: int) -> str:
    return f"cve:CVE-{year}-{10000 + i:05d}"


def _vendor(i: int) -> str:
    names = [
        "Sentinel", "Talos", "Lighthouse", "Aegis", "Nimbus", "Falcon",
        "Beacon", "Cipher", "Cobalt", "Drake", "Echelon", "Forge",
        "Glacier", "Harbor", "Ironwood",
    ]
    return f"vendor:{names[i % len(names)]}-{i // len(names):02d}"


def _threat_actor(i: int) -> str:
    actors = [
        "Lynx", "Hydra", "Specter", "Ghost", "Vulture", "Crow", "Serpent",
        "Wolf", "Cobra", "Mantis",
    ]
    return f"actor:APT-{actors[i % len(actors)]}-{i // len(actors):02d}"


# ----- Bridge pool construction -----


@dataclass(frozen=True)
class BridgeDensitySpec:
    """
    Bridge density specification.

    Two independent knobs:
    - extra:  fraction of records that carry a bridge in Record.extra
              (default 0.05).
    - slot:   fraction of records that carry a bridge in subject or
              object_ (default 0.0).

    A record may have both an extra-bridge AND a slot-bridge; the two
    knobs are independent. With both at 5%, ~0.25% of records have both
    (uncorrelated).
    """

    extra: float = DEFAULT_BRIDGE_DENSITY_EXTRA
    slot: float = DEFAULT_BRIDGE_DENSITY_SLOT

    def __post_init__(self) -> None:
        if not (0.0 <= self.extra <= 1.0):
            raise ValueError(f"extra density must be 0..1, got {self.extra}")
        if not (0.0 <= self.slot <= 1.0):
            raise ValueError(f"slot density must be 0..1, got {self.slot}")
        # Cap each density at 0.15 (the documented ablation range). Higher
        # values are technically allowed but produce unrealistic corpora.
        if self.extra > 0.15:
            raise ValueError(
                f"extra density {self.extra} exceeds the realism cap 0.15; "
                "if you want a higher density, edit bridges.py and document why"
            )
        if self.slot > 0.15:
            raise ValueError(
                f"slot density {self.slot} exceeds the realism cap 0.15"
            )

    @staticmethod
    def disabled() -> "BridgeDensitySpec":
        return BridgeDensitySpec(extra=0.0, slot=0.0)


def build_bridge_pools(
    seed: int, sizes: dict[str, int] | None = None
) -> BridgePools:
    """
    Build deterministic bridge pools.

    Pools are independent of the corpus's per-tenant pools — bridges live
    in a separate identifier namespace (ip:, hash:, cve:, vendor:, actor:)
    and never collide with the per-tenant entity prefixes (host:, user:,
    file:, proc:, malware:).

    Returns a BridgePools that the validation module will treat as the
    cross-tenant whitelist.
    """
    sizes = sizes or DEFAULT_BRIDGE_POOL_SIZES
    rng = random.Random(seed)

    n_ip = sizes.get("shared_ip", DEFAULT_BRIDGE_POOL_SIZES["shared_ip"])
    n_hash = sizes.get("malware_hash", DEFAULT_BRIDGE_POOL_SIZES["malware_hash"])
    n_cve = sizes.get("cve", DEFAULT_BRIDGE_POOL_SIZES["cve"])
    n_vendor = sizes.get("vendor", DEFAULT_BRIDGE_POOL_SIZES["vendor"])
    n_actor = sizes.get("threat_actor", DEFAULT_BRIDGE_POOL_SIZES["threat_actor"])

    ips = frozenset(_shared_ip(i) for i in range(n_ip))
    hashes = frozenset(_malware_hash(i, rng) for i in range(n_hash))
    # CVE years span 2022-2025 for a mix of realism.
    cves = frozenset(_cve(2022 + (i % 4), i) for i in range(n_cve))
    vendors = frozenset(_vendor(i) for i in range(n_vendor))
    actors = frozenset(_threat_actor(i) for i in range(n_actor))

    return BridgePools(
        shared_ips=ips,
        shared_hashes=hashes,
        shared_cves=cves,
        shared_vendors=vendors,
        shared_threat_actors=actors,
    )


# ----- Bridge sampling for record annotation -----


@dataclass(frozen=True)
class BridgeAssignment:
    """
    Per-record bridge assignment, produced by `assign_bridges_to_records`.

    Fields:
        extra_bridges_by_record_id: { record_id -> bridge_entity }.
            Records present in this dict carry that bridge entity in
            Record.extra under the key 'bridge'. Records absent from the
            dict carry no extra bridge.
        slot_bridges_by_record_id: { record_id -> (slot_name, bridge_entity) }.
            Records present in this dict have their `slot_name` field
            (one of 'subject' / 'object_') replaced with the bridge entity
            during generation. Records absent from the dict have their
            slot filled from the per-tenant pool as usual.
        bridges: the BridgePools used. Carried here so the generator can
            pass it into validation without recomputing.
    """

    extra_bridges_by_record_id: dict[str, str]
    slot_bridges_by_record_id: dict[str, tuple[str, str]]
    bridges: BridgePools


def assign_bridges_to_records(
    record_ids: list[str],
    bridges: BridgePools,
    density: BridgeDensitySpec,
    seed: int,
) -> BridgeAssignment:
    """
    Deterministically assign bridge entities to a subset of records.

    Args:
        record_ids: All record IDs in the corpus, in generator order.
        bridges: The BridgePools to draw from.
        density: How many records receive each type of bridge annotation.
        seed: For deterministic sampling.

    Returns:
        BridgeAssignment with two dicts keyed by record_id.

    Sampling:
    - We compute how many records get an extra-bridge:
        n_extra = round(density.extra * len(record_ids))
      and select those records uniformly at random. Each selected record
      gets a bridge entity sampled from a category, with category weights
      proportional to pool sizes (so larger pools get more usage).
    - Same procedure for slot-bridges, with the additional constraint
      that the bridge category must be in SLOT_ELIGIBLE_BRIDGE_TYPES.
    - The extra-bridge and slot-bridge samples are INDEPENDENT — a record
      can have both, neither, or one.
    """
    if not record_ids:
        return BridgeAssignment({}, {}, bridges)

    rng = random.Random(seed)
    record_ids_sorted = sorted(record_ids)  # deterministic input order

    # ---- Extra-bridges ----
    n_extra = round(density.extra * len(record_ids_sorted))
    extra_assignments: dict[str, str] = {}
    if n_extra > 0:
        selected_extra = rng.sample(record_ids_sorted, n_extra)
        for rid in selected_extra:
            category = _sample_category(rng, bridges)
            entity = _sample_bridge_entity(rng, bridges, category)
            if entity is not None:
                extra_assignments[rid] = entity

    # ---- Slot-bridges ----
    n_slot = round(density.slot * len(record_ids_sorted))
    slot_assignments: dict[str, tuple[str, str]] = {}
    if n_slot > 0:
        selected_slot = rng.sample(record_ids_sorted, n_slot)
        for rid in selected_slot:
            # Restrict to slot-eligible categories.
            category = _sample_category(rng, bridges, restrict_to_slot=True)
            entity = _sample_bridge_entity(rng, bridges, category)
            if entity is not None:
                slot_name = _BRIDGE_TO_SLOT[category]
                slot_assignments[rid] = (slot_name, entity)

    return BridgeAssignment(
        extra_bridges_by_record_id=extra_assignments,
        slot_bridges_by_record_id=slot_assignments,
        bridges=bridges,
    )


def _sample_category(
    rng: random.Random, bridges: BridgePools, restrict_to_slot: bool = False
) -> str:
    """Sample a bridge category weighted by pool size."""
    candidates: list[tuple[str, int]] = []
    if bridges.shared_ips:
        candidates.append(("shared_ip", len(bridges.shared_ips)))
    if bridges.shared_hashes:
        candidates.append(("malware_hash", len(bridges.shared_hashes)))
    if bridges.shared_cves:
        candidates.append(("cve", len(bridges.shared_cves)))
    if bridges.shared_vendors:
        candidates.append(("vendor", len(bridges.shared_vendors)))
    if bridges.shared_threat_actors:
        candidates.append(("threat_actor", len(bridges.shared_threat_actors)))

    if restrict_to_slot:
        candidates = [
            (cat, w) for cat, w in candidates
            if cat in SLOT_ELIGIBLE_BRIDGE_TYPES
        ]

    if not candidates:
        return "shared_ip"  # safe fallback; assign_bridges will get None and skip
    names = [c for c, _ in candidates]
    weights = [w for _, w in candidates]
    return rng.choices(names, weights=weights, k=1)[0]


def _sample_bridge_entity(
    rng: random.Random, bridges: BridgePools, category: str
) -> str | None:
    """Sample one entity from the specified bridge category."""
    pool: frozenset[str]
    if category == "shared_ip":
        pool = bridges.shared_ips
    elif category == "malware_hash":
        pool = bridges.shared_hashes
    elif category == "cve":
        pool = bridges.shared_cves
    elif category == "vendor":
        pool = bridges.shared_vendors
    elif category == "threat_actor":
        pool = bridges.shared_threat_actors
    else:
        return None
    if not pool:
        return None
    return rng.choice(sorted(pool))
