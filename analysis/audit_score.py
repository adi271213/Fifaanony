"""
audit_score.py — score the completed manual audit.

Run after audit_sample.csv has been filled in by hand. Computes:

  - agreement rate (paper-ready: "Manual audit: substring rescorer agreed
    with manual labels on X/100 sampled responses")
  - per-stratum breakdown
  - confusion matrix
  - Cohen's kappa (chance-corrected agreement)
  - disagreement details (for §7 appendix or footnote)
"""
import csv
from pathlib import Path
from collections import Counter, defaultdict


def main():
    rows = []
    csv_path = Path(__file__).resolve().parent / "audit_sample.csv"
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Filter to rows that have been labeled
    labeled = [r for r in rows if r.get("human_decision", "").strip()]
    n_total = len(rows)
    n_labeled = len(labeled)
    print(f"Total sampled rows: {n_total}")
    print(f"Labeled rows:       {n_labeled}")
    print()

    if n_labeled == 0:
        print("No labeled rows yet. Fill in human_decision and agree columns "
              "in audit_sample.csv, then re-run this script.")
        return

    # Agreement rate
    n_agree = sum(1 for r in labeled if r.get("agree", "").strip() == "1")
    print(f"Agreement rate:     {n_agree}/{n_labeled} = "
          f"{n_agree/n_labeled*100:.1f}%")
    print()

    # Per-stratum
    def stratum(r):
        if r["probe_type"] == "cross_boundary":
            return ("cross_boundary_INVALID"
                    if r["scorer_decision"] == "INVALID"
                    else "cross_boundary_SAFE")
        if r["probe_type"] == "supported":
            return "supported"
        return r["probe_type"]

    by_stratum = defaultdict(lambda: [0, 0])  # [agree, total]
    for r in labeled:
        s = stratum(r)
        by_stratum[s][1] += 1
        if r.get("agree", "").strip() == "1":
            by_stratum[s][0] += 1

    print("Per-stratum agreement:")
    for s, (a, t) in sorted(by_stratum.items()):
        print(f"  {s:30s} {a}/{t} = {a/t*100:.1f}%" if t else
              f"  {s:30s} 0/0")
    print()

    # Confusion matrix
    print("Confusion matrix (scorer rows, human columns):")
    decisions = sorted({r["scorer_decision"] for r in labeled} |
                       {r["human_decision"] for r in labeled})
    conf = {(s, h): 0 for s in decisions for h in decisions}
    for r in labeled:
        s, h = r["scorer_decision"], r["human_decision"]
        if h in decisions and s in decisions:
            conf[(s, h)] += 1
    header = " " * 13 + "  ".join(f"{d:>10s}" for d in decisions)
    print(header)
    for s in decisions:
        row = f"{s:13s} " + "  ".join(
            f"{conf[(s, h)]:>10d}" for h in decisions)
        print(row)
    print()

    # Cohen's kappa
    po = n_agree / n_labeled if n_labeled else 0
    # Marginals
    scorer_marg = Counter(r["scorer_decision"] for r in labeled)
    human_marg = Counter(r["human_decision"] for r in labeled)
    pe = sum(
        (scorer_marg[d] / n_labeled) * (human_marg[d] / n_labeled)
        for d in decisions
    )
    if pe < 1.0:
        kappa = (po - pe) / (1 - pe)
        print(f"Cohen's kappa:     {kappa:.3f}")
        # Interpretation guide
        if kappa >= 0.81:
            interp = "near-perfect agreement"
        elif kappa >= 0.61:
            interp = "substantial agreement"
        elif kappa >= 0.41:
            interp = "moderate agreement"
        elif kappa >= 0.21:
            interp = "fair agreement"
        elif kappa >= 0.0:
            interp = "slight agreement"
        else:
            interp = "no agreement / disagreement"
        print(f"Interpretation:    {interp} (Landis & Koch, 1977)")
    else:
        print("Cohen's kappa:     undefined (pe = 1.0)")
    print()

    # Disagreement details
    disagreements = [r for r in labeled
                     if r.get("agree", "").strip() != "1"]
    if disagreements:
        print(f"Disagreements ({len(disagreements)} rows):")
        for r in disagreements:
            print(f"  audit_id={r['audit_id']:>3s}  "
                  f"scorer={r['scorer_decision']:<10s}  "
                  f"human={r['human_decision']:<10s}  "
                  f"({r['model']} {r['probe_type']} on {r['rung']})")
            if r.get("human_notes", "").strip():
                print(f"    notes: {r['human_notes']}")


if __name__ == "__main__":
    main()
