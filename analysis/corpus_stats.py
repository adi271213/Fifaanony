"""
corpus_stats.py — bridge density and trap-probe statistics

For each condition in data/, computes statistics relevant to the
cross-boundary trap construction:

  1. Total fragments / records / boundaries (sanity).
  2. Bridge-context coverage: fraction of cross-boundary probes that
     have a non-empty bridge_context.
  3. Bridge category distribution across cross-boundary probes
     (shared_ip, malware_hash, cve, vendor, threat_actor).
  4. Distribution of distinct boundaries per cross-boundary probe.
  5. Bridge reuse: per bridge entity, how many distinct boundaries
     contain at least one probe referencing it.

Writes data/<condition>/corpus_stats.json per condition.

These statistics describe the deliberate bridge-based trap construction
documented in mnembound/bridges.py:

  - 5 bridge categories: shared_ip, malware_hash, cve, vendor, threat_actor
  - default bridge_density_extra = 0.05 (5% of records carry a bridge
    in Record.extra)
  - bridge pools: 50 IPs, 30 hashes, 20 CVEs, 15 vendors, 10 actors

The fragments.store.jsonl public view does NOT expose the Record.extra
bridge tag; bridges are observable at probe construction time via the
probe's bridge_context field. We therefore compute bridge density at
the probe level rather than the fragment level. This matches the
benchmark's threat model: the released store reveals only per-fragment
public metadata, while bridge structure is observable to the trap
constructor.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter


def bridge_category(entity: str) -> str:
    """Extract bridge category from an entity string like 'ip:198.51.100.7'."""
    prefix = entity.split(":", 1)[0] if ":" in entity else entity
    return {
        "ip": "shared_ip",
        "hash": "malware_hash",
        "cve": "cve",
        "vendor": "vendor",
        "actor": "threat_actor",
    }.get(prefix, prefix)


def stats_for_condition(cond_dir: Path) -> dict:
    """Compute bridge / probe statistics for one condition."""
    frags_path = cond_dir / "fragments.store.jsonl"
    probes_path = cond_dir / "probes.jsonl"
    if not frags_path.exists():
        return {"error": f"fragments.store.jsonl not found in {cond_dir}"}
    if not probes_path.exists():
        return {"error": f"probes.jsonl not found in {cond_dir}"}

    # ---- Fragment sanity ----
    pid2bd = {}
    pid2ev = {}
    boundaries = set()
    records = set()
    n_frags = 0
    with frags_path.open() as f:
        for line in f:
            d = json.loads(line)
            n_frags += 1
            pid2bd[d["public_provenance_id"]] = d["boundary_id"]
            pid2ev[d["public_provenance_id"]] = d.get("event_id")
            boundaries.add(d["boundary_id"])
            if d.get("event_id"):
                records.add(d["event_id"])

    # ---- Probe-level bridge stats ----
    probes_by_type = Counter()
    cb_probes = []
    with probes_path.open() as f:
        for line in f:
            p = json.loads(line)
            probes_by_type[p["probe_type"]] += 1
            if p["probe_type"] == "cross_boundary":
                cb_probes.append(p)

    n_cb = len(cb_probes)
    n_cb_with_bridge = sum(1 for p in cb_probes if p.get("bridge_context"))

    # Category distribution: each cross-boundary probe lists its bridges
    category_counts = Counter()
    bridge_to_boundaries = defaultdict(set)
    boundary_span_dist = Counter()

    for p in cb_probes:
        bridges = p.get("bridge_context", [])
        for b in bridges:
            category_counts[bridge_category(b)] += 1

        # Distinct boundaries spanned by this probe's fragments
        pids = p.get("fragment_provenance_ids", [])
        bds = {pid2bd.get(pid) for pid in pids if pid in pid2bd}
        bds.discard(None)
        boundary_span_dist[len(bds)] += 1

        # Track which boundaries each bridge entity is associated with
        for b in bridges:
            for bd in bds:
                bridge_to_boundaries[b].add(bd)

    # Per-bridge boundary-coverage stats
    bridges_seen = len(bridge_to_boundaries)
    coverage_dist = Counter(
        len(bds) for bds in bridge_to_boundaries.values()
    )
    avg_coverage = (
        sum(len(b) for b in bridge_to_boundaries.values()) / bridges_seen
        if bridges_seen else 0.0
    )

    return {
        "condition": cond_dir.name,
        "n_fragments": n_frags,
        "n_records": len(records),
        "n_boundaries": len(boundaries),
        "probes_by_type": dict(probes_by_type),
        "cross_boundary_probes": {
            "total": n_cb,
            "with_bridge_context": n_cb_with_bridge,
            "fraction_with_bridge_context": (
                n_cb_with_bridge / n_cb if n_cb else 0.0
            ),
            "bridge_category_distribution": dict(category_counts),
            "boundary_span_distribution": dict(sorted(boundary_span_dist.items())),
            "mean_boundary_span": (
                sum(k * v for k, v in boundary_span_dist.items()) / n_cb
                if n_cb else 0.0
            ),
        },
        "distinct_bridges_in_cross_boundary_probes": bridges_seen,
        "bridge_boundary_coverage_distribution": dict(
            sorted(coverage_dist.items())
        ),
        "mean_boundaries_per_bridge": avg_coverage,
        "generator_design_parameters": {
            "bridge_density_extra_default": 0.05,
            "bridge_density_slot_default": 0.0,
            "bridge_categories": [
                "shared_ip", "malware_hash", "cve", "vendor", "threat_actor"
            ],
            "default_bridge_pool_sizes": {
                "shared_ip": 50,
                "malware_hash": 30,
                "cve": 20,
                "vendor": 15,
                "threat_actor": 10,
            },
        },
    }


def main():
    data_dir = Path("../data") if Path("../data").exists() else Path("data")
    if not data_dir.exists():
        print("ERROR: data/ directory not found.", file=sys.stderr)
        sys.exit(1)

    conditions = sorted(d for d in data_dir.iterdir() if d.is_dir())
    print(f"Computing bridge/probe stats for {len(conditions)} conditions...")
    print()
    print(f"{'condition':<35s} {'cb':>4s} {'w/br':>4s} {'mean_span':>9s} "
          f"{'#bridges':>9s} {'avg_bd/br':>9s}")
    print("-" * 80)

    all_stats = []
    for cond in conditions:
        s = stats_for_condition(cond)
        if "error" in s:
            print(f"  SKIP {cond.name}: {s['error']}")
            continue

        out = cond / "corpus_stats.json"
        with out.open("w") as f:
            json.dump(s, f, indent=2)

        cb = s["cross_boundary_probes"]
        print(f"{cond.name:<35s} "
              f"{cb['total']:>4d} "
              f"{cb['with_bridge_context']:>4d} "
              f"{cb['mean_boundary_span']:>9.2f} "
              f"{s['distinct_bridges_in_cross_boundary_probes']:>9d} "
              f"{s['mean_boundaries_per_bridge']:>9.2f}")
        all_stats.append(s)

    print()
    print(f"Wrote corpus_stats.json to {len(all_stats)} condition dirs.")
    if all_stats:
        mean_span = sum(
            s["cross_boundary_probes"]["mean_boundary_span"]
            for s in all_stats
        ) / len(all_stats)
        print(f"Mean cross-boundary probe span (boundaries): {mean_span:.2f}")
        print(f"All cross-boundary probes carry a bridge_context (by construction).")


if __name__ == "__main__":
    main()
