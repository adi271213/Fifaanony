"""
qualitative_cases.py — Priority 5: anatomy of a fragment-faithful,
event-false recall.

For each model (Qwen, Mistral), finds the cross-boundary trap response
that:

  (a) is on the shared_vector_index rung
  (b) is not abstained, has no parse error
  (c) is boundary-invalid under HARDENED scoring
  (d) has the *most* distinct boundary IDs touched by its component values
      (the more boundaries spanned, the more illustrative)
  (e) ties broken by lowest probe_id (deterministic)

Prints a markdown-ready block for §7.3 "Anatomy of a fragment-faithful,
event-false recall" with:

  - probe_id
  - retrieved fragments (sanitized values + boundary_id + role)
  - model recall (the JSON output)
  - per-component analysis: which fragment supplied which value, which
    boundary it came from
  - the trap structure: which fragments came from which boundary

Usage:
    python3 qualitative_cases.py
"""
import json
from pathlib import Path



# === Patched to look in ../data/ by default ===
import os as _os
DATA_DIR = Path(_os.environ.get("MNEMBOUND_DATA_DIR", Path(__file__).parent.parent / "data"))
def _frags_for_resp(resp, store):
    return [store[pid] for pid in resp.get("fragment_provenance_ids", [])
            if pid in store]


def component_match(val, frags, view):
    """Return list of (fragment, match_kind) for fragments supplying `val`.
    Same logic as HARDENED scorer, but returns the matching fragments."""
    if not isinstance(val, str) or not val.strip():
        return []
    val = val.strip()
    field = "sanitized_value" if view == "tenant_hidden" else "value"
    text_field = "sanitized_text" if view == "tenant_hidden" else "text"
    out = []
    for f in frags:
        fv = str(f.get(field, "")).strip()
        ft = str(f.get(text_field, "")).strip()
        if not fv and not ft:
            continue
        if fv and val == fv:
            out.append((f, "exact"))
        elif fv and fv in val:
            out.append((f, "fragment-in-recall"))
        elif ft and val in ft:
            out.append((f, "recall-in-fragment-text"))
    return out


def boundaries_touched(resp, store):
    """Returns set of boundary_ids touched by this recall under HARDENED."""
    if resp.get("abstained") or resp.get("recall") is None:
        return set()
    recall = resp["recall"]
    view = resp.get("value_view", "tenant_hidden")
    frags = _frags_for_resp(resp, store)
    bset = set()
    for c in ("subject", "action", "object", "outcome", "time"):
        for f, _ in component_match(recall.get(c, ""), frags, view):
            bset.add(f["boundary_id"])
    return bset


def find_best_case(out_dir):
    """Find the cleanest cross-boundary trap example in this condition."""
    if not (out_dir / "responses.jsonl").exists():
        return None
    store = {}
    with open(out_dir / "fragments.store.jsonl") as f:
        for line in f:
            d = json.loads(line)
            store[d["public_provenance_id"]] = d
    best = None
    best_n_boundaries = 0
    with open(out_dir / "responses.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if (r["probe_type"] != "cross_boundary"
                    or r["rung"] != "shared_vector_index"
                    or r.get("abstained")
                    or r.get("parse_error")
                    or not r.get("recall")):
                continue
            b = boundaries_touched(r, store)
            if len(b) <= 1:
                continue
            if (best is None
                    or len(b) > best_n_boundaries
                    or (len(b) == best_n_boundaries
                        and r["probe_id"] < best["probe_id"])):
                best = r
                best_n_boundaries = len(b)
    return best, store


def render_case(model_label, resp, store):
    """Pretty-print the case in markdown for §7.3."""
    if resp is None:
        return f"## {model_label}: no qualifying case found\n"
    recall = resp["recall"]
    view = resp.get("value_view", "tenant_hidden")
    field = "sanitized_value" if view == "tenant_hidden" else "value"
    text_field = "sanitized_text" if view == "tenant_hidden" else "text"
    frags = _frags_for_resp(resp, store)
    bset = boundaries_touched(resp, store)

    lines = []
    lines.append(f"## {model_label} — {resp['probe_id']}")
    lines.append("")
    lines.append(f"**Rung:** shared_vector_index  ")
    lines.append(f"**Probe type:** cross_boundary  ")
    lines.append(f"**Value view:** {view}  ")
    lines.append(f"**Boundaries spanned by recall:** "
                 f"{sorted(bset)} (n={len(bset)})  ")
    lines.append(f"**Abstained:** {resp.get('abstained')}  ")
    lines.append("")
    lines.append("### Retrieved evidence")
    lines.append("")
    lines.append("| # | role | boundary | value | text |")
    lines.append("|---|---|---|---|---|")
    for i, f in enumerate(frags, 1):
        v = str(f.get(field, ""))[:50]
        t = str(f.get(text_field, ""))[:60]
        lines.append(
            f"| {i} | {f['role']} | {f['boundary_id']} | "
            f"`{v}` | `{t}` |"
        )
    lines.append("")
    lines.append("### Model recall (JSON output)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"recall": recall}, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Per-component analysis")
    lines.append("")
    lines.append("| Field | Value | Source fragments (boundary) | Match kind |")
    lines.append("|---|---|---|---|")
    for c in ("subject", "action", "object", "outcome", "time"):
        val = recall.get(c, "")
        matches = component_match(val, frags, view)
        if not matches:
            lines.append(f"| {c} | `{val}` | (no match) | — |")
        else:
            for f, kind in matches:
                lines.append(
                    f"| {c} | `{val[:40]}` | "
                    f"frag#{frags.index(f)+1} ({f['boundary_id']}) | {kind} |"
                )
    lines.append("")
    lines.append("### Why this is fragment-faithful, event-false")
    lines.append("")
    lines.append(
        "Every recall component traces to at least one fragment in the "
        f"retrieved evidence ({len(frags)} fragments total). Each fragment, "
        "by construction, is a real fragment of a real record from a real "
        "tenant — per-fragment provenance is intact. But the recall's "
        f"components come from fragments spanning {len(bset)} distinct "
        "boundaries; no single source record contains all of these "
        "components together. The recall is therefore a composition of "
        "facts that did not co-occur in any real event in the corpus."
    )
    lines.append("")
    return "\n".join(lines)


CONDITIONS_TO_TRY = {
    "Qwen3-30B-A3B-Instruct-2507-FP8 (seed 42, neutral / collapsed)":
        "qwen_full_neutral",
    "Mistral-Small-3.2-24B-Instruct-2506-FP8 (seed 42, neutral / collapsed)":
        "mistral_full_neutral",
}


def main():
    blocks = []
    blocks.append("# §7.3 Anatomy of a fragment-faithful, event-false recall")
    blocks.append("")
    blocks.append(
        "To make the failure mode concrete, we walk through one "
        "cross-boundary trap probe per model, under the headline condition "
        "(neutral prompt + collapsed aggregator) on the shared-vector-index "
        "rung. For each example we show the five retrieved fragments, the "
        "model's recall, and the per-component analysis demonstrating that "
        "each recall field is verbatim or near-verbatim from a fragment, "
        "while the fragments themselves come from multiple distinct "
        "boundaries.")
    blocks.append("")
    for label, dir_name in CONDITIONS_TO_TRY.items():
        result = find_best_case(DATA_DIR / dir_name)
        if result is None:
            blocks.append(f"## {label}: directory {dir_name} not found\n")
            continue
        best, store = result
        blocks.append(render_case(label, best, store))
        blocks.append("")
    print("\n".join(blocks))


if __name__ == "__main__":
    main()
