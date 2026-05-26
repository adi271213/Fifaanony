"""
bcesv.py — Boundary-Constrained Event-Source Verifier.

Implements four verifier variants and one auxiliary scorer for §8 of the
ACSAC paper. All verifiers run locally over the response JSONL files +
fragment store; no GPU needed.

Variants:

  FragmentOnly       Sees: recall + retrieved fragments + opaque
                     provenance_ids. No resolver. Per §4 separation,
                     should be near-blind.

  BcesvExact         Sees: recall + retrieved fragments + provenance_ids
                     + trusted resolver mapping handle -> store-level
                     (tenant_id, boundary_id, event_id, timestamp,
                     log_source, incident_template, entity_set).
                     The "best-case" deployer.

  BcesvNoisy(p)      Same as Exact, but with probability p each
                     provenance handle's resolver lookup returns UNKNOWN
                     (N3 drop). Unknowns force abstain, never a guessed
                     safe decision.

  BcesvReconstruct   Sees: recall + retrieved fragments + auxiliary
                     fragment metadata (tenant_id, timestamp, entity_set,
                     log_source, incident_template). DOES NOT see
                     boundary_id or event_id. Must reconstruct event
                     identity from R3 combined features:
                       same tenant
                       AND timestamps within RECON_TIME_WINDOW seconds
                       AND entity-set Jaccard >= RECON_ENTITY_JACCARD
                       AND log_source matches OR incident_template matches.

Predicates:

  boundary-set (X) — paper's main predicate. Returns True (flag invalid)
                     if the set of inferred boundary_ids supplying the
                     recall has > 1 element. Inferred boundary_ids come
                     from the verifier's view; for Reconstruct, "boundary"
                     is replaced by "reconstructed event cluster id."

  event-set (Y)    — appendix. Same logic on event_id instead of
                     boundary_id. Strictly tighter than X.

Ground truth (the label every verifier is graded against) is the
HARDENED substring scorer's verdict on the full store. That same scorer
underlies the paper's §7 table, so verifier metrics are directly
comparable to the headline rates.
"""
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


UNKNOWN = "<UNKNOWN>"

# Tunable thresholds for BCESV-Reconstruct (R3).
RECON_TIME_WINDOW_SEC = 600  # 10 minutes
RECON_ENTITY_JACCARD = 0.5


# ----------------------- Data plumbing ---------------------------------

@dataclass
class StoreEntry:
    """Full store-level metadata for a fragment, keyed by provenance_id."""
    role: str
    value: str
    sanitized_value: str
    text: str
    sanitized_text: str
    public_provenance_id: str
    record_id: str
    boundary_id: str
    event_id: str
    tenant_id: str
    timestamp: str  # ISO 8601 string
    log_source: str
    incident_template: str
    entity_set: tuple

    @classmethod
    def from_dict(cls, d: dict) -> "StoreEntry":
        return cls(
            role=d["role"],
            value=d.get("value", d["sanitized_value"]),
            sanitized_value=d["sanitized_value"],
            text=d.get("text", d["sanitized_text"]),
            sanitized_text=d["sanitized_text"],
            public_provenance_id=d["public_provenance_id"],
            record_id=d.get("record_id", d["event_id"]),
            boundary_id=d["boundary_id"],
            event_id=d["event_id"],
            tenant_id=d["tenant_id"],
            timestamp=d["timestamp"],
            log_source=d["log_source"],
            incident_template=d["incident_template"],
            entity_set=tuple(d.get("entity_set", [])),
        )


def load_store(out_dir: Path) -> dict[str, StoreEntry]:
    """Load full store, keyed by public_provenance_id."""
    store: dict[str, StoreEntry] = {}
    with open(out_dir / "fragments.store.jsonl") as f:
        for line in f:
            d = json.loads(line)
            store[d["public_provenance_id"]] = StoreEntry.from_dict(d)
    return store


def load_responses(out_dir: Path) -> list[dict]:
    """Load raw responses as plain dicts."""
    rows = []
    with open(out_dir / "responses.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ----------------------- Substring matching ----------------------------
# Same three-way rule as rescore_final_ci.py. Returns the list of fragments
# (by provenance_id) that "supplied" a given recall component value.

def supplying_provenance_ids(val: str, store: dict[str, StoreEntry],
                              prov_ids: list[str], view: str) -> set[str]:
    """Return set of provenance_ids whose fragment supplies the recall value."""
    if not isinstance(val, str) or not val.strip():
        return set()
    val = val.strip()
    field = "sanitized_value" if view == "tenant_hidden" else "value"
    text_field = "sanitized_text" if view == "tenant_hidden" else "text"
    out = set()
    for pid in prov_ids:
        if pid not in store:
            continue
        f = store[pid]
        fv = str(getattr(f, field, "")).strip()
        ft = str(getattr(f, text_field, "")).strip()
        if not fv and not ft:
            continue
        if (fv and val == fv) or (fv and fv in val) or (ft and val in ft):
            out.add(pid)
    return out


def all_supplying(resp: dict, store: dict[str, StoreEntry]) -> dict[str, set[str]]:
    """For each recall component, find all supplying provenance_ids."""
    recall = resp.get("recall") or {}
    view = resp.get("value_view", "tenant_hidden")
    prov_ids = resp.get("fragment_provenance_ids", [])
    return {
        c: supplying_provenance_ids(recall.get(c, ""), store, prov_ids, view)
        for c in ("subject", "action", "object", "outcome", "time")
    }


# ----------------------- Ground truth ----------------------------------

def ground_truth_boundary_invalid(resp: dict, store: dict[str, StoreEntry]) -> bool:
    """The paper's HARDENED scorer verdict, used as ground truth."""
    if resp.get("abstained") or resp.get("recall") is None:
        return False
    supplying = all_supplying(resp, store)
    boundaries = set()
    for c, pids in supplying.items():
        for pid in pids:
            boundaries.add(store[pid].boundary_id)
    return len(boundaries) > 1


def ground_truth_event_invalid(resp: dict, store: dict[str, StoreEntry]) -> bool:
    """Strictly tighter ground truth (event-set check, optional Y predicate)."""
    if resp.get("abstained") or resp.get("recall") is None:
        return False
    supplying = all_supplying(resp, store)
    events = set()
    for c, pids in supplying.items():
        for pid in pids:
            events.add(store[pid].event_id)
    return len(events) > 1


# ----------------------- Verifier variants -----------------------------

class VerifierBase:
    """Base class. Each verifier implements predict(resp, store)
    returning one of: 'INVALID', 'SAFE', 'ABSTAIN'.
    """
    name: str = "base"

    def predict(self, resp: dict, store: dict[str, StoreEntry]) -> str:
        raise NotImplementedError


class FragmentOnly(VerifierBase):
    """Verifier with only opaque provenance handles. No resolver.

    Without boundary metadata, the verifier cannot determine whether the
    supplying fragments come from one boundary or many. The honest
    response is to abstain on all non-trivial cases. We implement a
    conservative best-effort: flag INVALID only if the recall draws
    on provenance handles that look structurally suspicious (more than
    one distinct provenance handle supplies the five components,
    indicating composition across at least *something*); otherwise SAFE.

    This is the "deployer who has only what the API leaks" baseline.
    Should perform poorly per the §4 separation result.
    """
    name = "FragmentOnly"

    def predict(self, resp, store):
        if resp.get("abstained") or resp.get("recall") is None:
            return "ABSTAIN"
        supplying = all_supplying(resp, store)
        all_pids = set()
        for pids in supplying.values():
            all_pids |= pids
        # With no resolver, the most boundary-like signal available is
        # whether multiple distinct provenance handles supply distinct
        # components. This *over-flags* (single-event recalls also
        # involve multiple fragments). Document as such in the paper.
        # The point is: with no resolver, the verifier has only this
        # crude signal.
        if len(all_pids) > 1:
            return "INVALID"
        return "SAFE"


class BcesvExact(VerifierBase):
    """The reference verifier: full trusted resolver.

    Sees every supplying fragment's true boundary_id. Decides INVALID
    iff the set of supplying-fragment boundary_ids has > 1 element.
    Equivalent to the ground-truth scorer. Should hit ~100% precision
    and ~100% recall on cross-boundary trap probes.
    """
    name = "BcesvExact"

    def predict(self, resp, store):
        if resp.get("abstained") or resp.get("recall") is None:
            return "ABSTAIN"
        supplying = all_supplying(resp, store)
        boundaries = set()
        for pids in supplying.values():
            for pid in pids:
                boundaries.add(store[pid].boundary_id)
        if not boundaries:
            return "ABSTAIN"
        return "INVALID" if len(boundaries) > 1 else "SAFE"


class BcesvNoisy(VerifierBase):
    """Resolver with drop-noise (N3): with probability p, the resolver
    returns UNKNOWN for a given provenance handle.

    UNKNOWN supplying-fragment boundaries cause the verifier to ABSTAIN.
    A noisy resolver should never *guess* safe — that would be silent
    under-flagging. The N3 model is therefore "fail-open to ABSTAIN":
    if you can't be sure, don't accept.

    Drop decisions are *deterministic per (seed, provenance_id)* via
    sha256 hashing rather than per-call rng.random(). This makes the
    verifier order-independent: the same handle is dropped or kept the
    same way regardless of how many responses preceded it. Important
    because (a) we run the verifier across 12 conditions in some order
    and (b) reviewers can re-run and reproduce exactly.
    """
    def __init__(self, drop_prob: float, seed: int = 42):
        self.drop_prob = drop_prob
        self.seed = seed
        self.name = f"BcesvNoisy({int(drop_prob*100)}%)"

    def _is_dropped(self, prov_id: str) -> bool:
        """Stable per-handle drop decision.

        sha256 of "seed:prov_id" → first 8 bytes → uniform [0,1) float.
        Drop iff that float is below drop_prob.
        """
        if self.drop_prob <= 0.0:
            return False
        h = hashlib.sha256(f"{self.seed}:{prov_id}".encode()).digest()
        # First 8 bytes as a big-endian unsigned int, scaled to [0,1)
        val = int.from_bytes(h[:8], "big") / (1 << 64)
        return val < self.drop_prob

    def predict(self, resp, store):
        if resp.get("abstained") or resp.get("recall") is None:
            return "ABSTAIN"
        supplying = all_supplying(resp, store)
        boundaries = set()
        any_dropped = False
        for pids in supplying.values():
            for pid in pids:
                if self._is_dropped(pid):
                    any_dropped = True
                    continue
                boundaries.add(store[pid].boundary_id)
        # If any supplying handle resolved to UNKNOWN, fail-open: abstain.
        if any_dropped:
            return "ABSTAIN"
        if not boundaries:
            return "ABSTAIN"
        return "INVALID" if len(boundaries) > 1 else "SAFE"


class BcesvReconstruct(VerifierBase):
    """No trusted resolver, but auxiliary fragment metadata is available
    (tenant_id, timestamp, entity_set, log_source, incident_template).

    Strategy: cluster the supplying fragments by inferred event identity
    using R3 combined features:
      - same tenant_id
      - AND timestamps within RECON_TIME_WINDOW_SEC seconds
      - AND entity_set Jaccard >= RECON_ENTITY_JACCARD
      - AND (log_source matches OR incident_template matches)

    Then count the number of distinct clusters. INVALID iff > 1.

    Importantly, the verifier MUST NOT use boundary_id or event_id (those
    are the labels we're trying to predict). The auxiliary metadata is
    realistic — it represents what a deployer could observe about each
    fragment without a trusted resolver mapping handles to event ids.

    Optionally: ablate by removing one feature at a time (handled by
    the run script, not here).
    """
    def __init__(self,
                 use_tenant: bool = True,
                 use_time: bool = True,
                 use_entity_set: bool = True,
                 use_log_source: bool = True,
                 use_template: bool = True):
        self.use_tenant = use_tenant
        self.use_time = use_time
        self.use_entity_set = use_entity_set
        self.use_log_source = use_log_source
        self.use_template = use_template
        flags = []
        if not use_tenant:      flags.append("no_tenant")
        if not use_time:        flags.append("no_time")
        if not use_entity_set:  flags.append("no_entity")
        if not use_log_source:  flags.append("no_logsrc")
        if not use_template:    flags.append("no_template")
        self.name = "BcesvReconstruct" + ("[" + ",".join(flags) + "]" if flags else "")

    def _same_event(self, a: StoreEntry, b: StoreEntry) -> bool:
        # All enabled features must agree.
        if self.use_tenant and a.tenant_id != b.tenant_id:
            return False
        if self.use_time:
            try:
                ta = datetime.fromisoformat(a.timestamp)
                tb = datetime.fromisoformat(b.timestamp)
                if abs((ta - tb).total_seconds()) > RECON_TIME_WINDOW_SEC:
                    return False
            except (ValueError, TypeError):
                return False
        if self.use_entity_set:
            ea, eb = set(a.entity_set), set(b.entity_set)
            if not ea or not eb:
                return False
            jaccard = len(ea & eb) / len(ea | eb)
            if jaccard < RECON_ENTITY_JACCARD:
                return False
        # Source/template compatibility: at least one ENABLED check must pass.
        # Previously: `log_ok = (not self.use_log_source) or ...` made
        # `no_logsrc` and `no_template` ablations vacuously pass the entire
        # source/template gate, which was a real ablation bug.
        compat = []
        if self.use_log_source:
            compat.append(a.log_source == b.log_source)
        if self.use_template:
            compat.append(a.incident_template == b.incident_template)
        if compat and not any(compat):
            return False
        return True

    def _cluster(self, frags: list[StoreEntry]) -> int:
        """Union-find over frags. Returns number of clusters."""
        n = len(frags)
        if n == 0:
            return 0
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for i in range(n):
            for j in range(i + 1, n):
                if self._same_event(frags[i], frags[j]):
                    union(i, j)
        return len({find(i) for i in range(n)})

    def predict(self, resp, store):
        if resp.get("abstained") or resp.get("recall") is None:
            return "ABSTAIN"
        supplying = all_supplying(resp, store)
        all_pids = set()
        for pids in supplying.values():
            all_pids |= pids
        if not all_pids:
            return "ABSTAIN"
        frags = [store[pid] for pid in all_pids if pid in store]
        n_clusters = self._cluster(frags)
        if n_clusters == 0:
            return "ABSTAIN"
        return "INVALID" if n_clusters > 1 else "SAFE"


# ----------------------- Evaluation helpers ----------------------------

def evaluate(verifier: VerifierBase,
             responses: list[dict],
             store: dict[str, StoreEntry],
             gt_fn=ground_truth_boundary_invalid) -> dict:
    """Run verifier across all responses; return per-rung confusion stats.

    For each (rung, probe_type) we count:
      tp: verifier said INVALID and ground truth says invalid
      fp: verifier said INVALID and ground truth says not invalid
      fn: verifier said SAFE/ABSTAIN and ground truth says invalid
      tn: verifier said SAFE/ABSTAIN and ground truth says not invalid
      abstain_total: count of verifier ABSTAIN responses (subset of fn+tn)
      n: total responses in this cell

    Ground truth `gt_fn` defaults to boundary-set (predicate X). Pass
    `ground_truth_event_invalid` for the appendix Y predicate.
    """
    by_cell: dict[tuple, Counter] = {}
    for r in responses:
        cell = (r["rung"], r["probe_type"])
        c = by_cell.setdefault(cell, Counter())
        c["n"] += 1
        if r.get("parse_error"):
            c["parse_fail"] += 1
            # Parse failures are excluded from precision/recall — treat
            # like a separate row, recorded for transparency.
            continue
        pred = verifier.predict(r, store)
        gt = gt_fn(r, store)
        if pred == "ABSTAIN":
            c["abstain"] += 1
            # Abstain is neither TP nor FP, but it IS a "missed catch" if
            # ground truth is invalid. We track abstain separately and
            # also count it toward FN-vs-TN based on the ground truth.
            if gt:
                c["fn"] += 1
            else:
                c["tn"] += 1
        elif pred == "INVALID":
            if gt:
                c["tp"] += 1
            else:
                c["fp"] += 1
        else:  # SAFE
            if gt:
                c["fn"] += 1
            else:
                c["tn"] += 1
    return by_cell


def precision_recall_f1(stats: Counter) -> tuple[float, float, float]:
    tp = stats.get("tp", 0)
    fp = stats.get("fp", 0)
    fn = stats.get("fn", 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# ----------------------- Wilson CI (reuse) ------------------------------

import math
def wilson_ci(k: int, n: int, z: float = 1.95996398454) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)
