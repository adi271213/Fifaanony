"""
bcesv_run_all.py — run all verifier variants across all 12 conditions,
plus ablations on BcesvReconstruct.

Outputs:
  - Stdout: paper-ready table with precision/recall/F1 and 95% CIs on
    cross-boundary trap probes.
  - bcesv_results.csv: machine-readable per-cell metrics.
  - bcesv_summary.csv: per-variant pooled metrics across all conditions.

The table answers the question §8 asks:
  "Does the post-hoc verifier catch the failure mode? Under what input
   conditions?"
"""
import csv
import sys
from pathlib import Path
from collections import Counter

from bcesv import (
    StoreEntry, load_store, load_responses,
    FragmentOnly, BcesvExact, BcesvNoisy, BcesvReconstruct,
    evaluate, precision_recall_f1, wilson_ci,
    ground_truth_boundary_invalid, ground_truth_event_invalid,
)


CONDITIONS = [
    ("qwen_full_neutral",         "Qwen    s42 N/C"),
    ("qwen_full_cautious",        "Qwen    s42 C/C"),
    ("qwen_full_pcb",             "Qwen    s42 N/P"),
    ("qwen_full_cautious_pcb",    "Qwen    s42 C/P"),
    ("qwen_seed43_neutral",       "Qwen    s43 N/C"),
    ("qwen_seed44_neutral",       "Qwen    s44 N/C"),
    ("mistral_full_neutral",      "Mistral s42 N/C"),
    ("mistral_full_cautious",     "Mistral s42 C/C"),
    ("mistral_full_pcb",          "Mistral s42 N/P"),
    ("mistral_full_cautious_pcb", "Mistral s42 C/P"),
    ("mistral_seed43_neutral",    "Mistral s43 N/C"),
    ("mistral_seed44_neutral",    "Mistral s44 N/C"),
]

RUNGS = ["isolated", "boundary_preserving_topk",
         "shared_vector_index", "summary_merge"]

# Build the variant list. Each entry is (display_name, verifier_instance).
def build_variants() -> list:
    variants = [
        ("FragmentOnly",        FragmentOnly()),
        ("BcesvExact",          BcesvExact()),
        ("BcesvNoisy(10%)",     BcesvNoisy(0.10, seed=42)),
        ("BcesvNoisy(20%)",     BcesvNoisy(0.20, seed=42)),
        ("BcesvNoisy(30%)",     BcesvNoisy(0.30, seed=42)),
        ("BcesvReconstruct",    BcesvReconstruct()),
    ]
    # Ablations on Reconstruct (R3 components dropped one at a time).
    variants.extend([
        ("BcesvRecon[no_tenant]",   BcesvReconstruct(use_tenant=False)),
        ("BcesvRecon[no_time]",     BcesvReconstruct(use_time=False)),
        ("BcesvRecon[no_entity]",   BcesvReconstruct(use_entity_set=False)),
        ("BcesvRecon[no_logsrc]",   BcesvReconstruct(use_log_source=False)),
        ("BcesvRecon[no_template]", BcesvReconstruct(use_template=False)),
    ])
    return variants


def fmt_rate_ci(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson_ci(k, n)
    return f"{k/n*100:5.1f}% [{lo*100:5.1f},{hi*100:5.1f}]"


def main(base_dir: Path):
    variants = build_variants()

    # Aggregate stats per (variant, rung) across all conditions, pooled.
    # Also keep per-(variant, condition, rung) for the CSV.
    pooled: dict = {v_name: {r: Counter() for r in RUNGS} for v_name, _ in variants}
    per_cell_rows = []

    for cond_dir, cond_label in CONDITIONS:
        cond_path = base_dir / cond_dir
        if not (cond_path / "responses.jsonl").exists():
            print(f"  (skipping {cond_label}: {cond_path} missing)", file=sys.stderr)
            continue
        print(f"  loading {cond_label} from {cond_path}...", file=sys.stderr)
        store = load_store(cond_path)
        responses = load_responses(cond_path)

        for v_name, verifier in variants:
            by_cell = evaluate(verifier, responses, store,
                                gt_fn=ground_truth_boundary_invalid)
            for (rung, pt), c in by_cell.items():
                # We only pool/report on cross_boundary probes (the
                # verifier's primary job) and supported probes (its FP rate).
                if pt not in ("cross_boundary", "supported"):
                    continue
                key = (rung, pt)
                # Pool
                pooled[v_name][rung]["tp_" + pt] += c.get("tp", 0)
                pooled[v_name][rung]["fp_" + pt] += c.get("fp", 0)
                pooled[v_name][rung]["fn_" + pt] += c.get("fn", 0)
                pooled[v_name][rung]["tn_" + pt] += c.get("tn", 0)
                pooled[v_name][rung]["abstain_" + pt] += c.get("abstain", 0)
                pooled[v_name][rung]["n_" + pt] += c.get("n", 0)
                pooled[v_name][rung]["parse_" + pt] += c.get("parse_fail", 0)
                # Per-cell row for CSV
                per_cell_rows.append({
                    "variant": v_name,
                    "condition": cond_label,
                    "rung": rung,
                    "probe_type": pt,
                    "n": c.get("n", 0),
                    "tp": c.get("tp", 0),
                    "fp": c.get("fp", 0),
                    "fn": c.get("fn", 0),
                    "tn": c.get("tn", 0),
                    "abstain": c.get("abstain", 0),
                    "parse_fail": c.get("parse_fail", 0),
                })

    # ---------------- Paper table (pooled, cross-boundary, 4 rungs) ----------
    # The interesting metrics:
    #   Recall on cross-boundary traps: TP / (TP + FN). High = catches the failure.
    #   Precision: TP / (TP + FP). High = doesn't false-alarm.
    #   Abstain rate: how often the verifier punted.
    #   On supported probes: FP rate (how often verifier wrongly flags
    #     a legitimate single-event recall).

    print()
    print("=" * 110)
    print("BCESV table (pooled across all 12 conditions, cross-boundary probes)")
    print("=" * 110)
    print()
    print(f"{'variant':24s} {'rung':28s} "
          f"{'recall (sensitivity)':>26s} "
          f"{'precision':>26s} "
          f"{'abstain':>14s}")
    print("-" * 124)

    for v_name, _ in variants:
        for rung in RUNGS:
            c = pooled[v_name][rung]
            tp = c.get("tp_cross_boundary", 0)
            fp = c.get("fp_cross_boundary", 0)
            fn = c.get("fn_cross_boundary", 0)
            tn = c.get("tn_cross_boundary", 0)
            ab = c.get("abstain_cross_boundary", 0)
            n_cb = c.get("n_cross_boundary", 0) - c.get("parse_cross_boundary", 0)

            # Recall = TP / (TP+FN); positives = ground-truth invalids
            pos = tp + fn
            # Precision = TP / (TP+FP); flagged = TP + FP
            flagged = tp + fp

            recall_str = fmt_rate_ci(tp, pos) if pos else "n/a"
            prec_str = fmt_rate_ci(tp, flagged) if flagged else "n/a"
            abstain_str = fmt_rate_ci(ab, n_cb) if n_cb else "n/a"

            print(f"{v_name:24s} {rung:28s} "
                  f"{recall_str:>26s} "
                  f"{prec_str:>26s} "
                  f"{abstain_str:>14s}")
        print()

    # ---------------- Supported-probe FP rate --------------------------
    print("=" * 110)
    print("Verifier false-positive rate on SUPPORTED probes (pooled)")
    print("=" * 110)
    print("Lower is better. A high FP rate means the verifier flags")
    print("legitimate single-event recalls as boundary-invalid.")
    print()
    print(f"{'variant':24s} {'rung':28s} {'FP/(FP+TN)':>26s}")
    print("-" * 82)
    for v_name, _ in variants:
        for rung in RUNGS:
            c = pooled[v_name][rung]
            fp = c.get("fp_supported", 0)
            tn = c.get("tn_supported", 0)
            denom = fp + tn
            fp_str = fmt_rate_ci(fp, denom) if denom else "n/a"
            print(f"{v_name:24s} {rung:28s} {fp_str:>26s}")
        print()

    # ---------------- CSV output ---------------------------------------
    csv_path = Path(__file__).resolve().parent / "bcesv_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "condition", "rung", "probe_type",
            "n", "tp", "fp", "fn", "tn", "abstain", "parse_fail",
        ])
        w.writeheader()
        for row in per_cell_rows:
            w.writerow(row)
    print(f"Wrote per-cell CSV to {csv_path}")

    # ---------------- Pooled summary CSV --------------------------------
    pooled_csv = Path(__file__).resolve().parent / "bcesv_summary.csv"
    with open(pooled_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "variant", "rung", "probe_type",
            "tp", "fp", "fn", "tn", "abstain", "n_minus_parse",
            "recall", "recall_lo", "recall_hi",
            "precision", "precision_lo", "precision_hi",
            "abstain_rate", "abstain_lo", "abstain_hi",
        ])
        for v_name, _ in variants:
            for rung in RUNGS:
                c = pooled[v_name][rung]
                for pt in ("cross_boundary", "supported"):
                    tp = c.get(f"tp_{pt}", 0)
                    fp = c.get(f"fp_{pt}", 0)
                    fn = c.get(f"fn_{pt}", 0)
                    tn = c.get(f"tn_{pt}", 0)
                    ab = c.get(f"abstain_{pt}", 0)
                    n_eff = c.get(f"n_{pt}", 0) - c.get(f"parse_{pt}", 0)
                    pos = tp + fn
                    flagged = tp + fp
                    rec_lo, rec_hi = wilson_ci(tp, pos) if pos else (0, 1)
                    prec_lo, prec_hi = wilson_ci(tp, flagged) if flagged else (0, 1)
                    ab_lo, ab_hi = wilson_ci(ab, n_eff) if n_eff else (0, 1)
                    w.writerow([
                        v_name, rung, pt,
                        tp, fp, fn, tn, ab, n_eff,
                        tp / pos if pos else 0, rec_lo, rec_hi,
                        tp / flagged if flagged else 0, prec_lo, prec_hi,
                        ab / n_eff if n_eff else 0, ab_lo, ab_hi,
                    ])
    print(f"Wrote pooled summary to {pooled_csv}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base = Path(sys.argv[1])
    else:
        # Default: ../data relative to this script's location
        base = Path(__file__).resolve().parent.parent / "data"
        print(f"No path given. Defaulting to {base}", flush=True)
    if not base.exists():
        raise SystemExit(
            f"ERROR: data path does not exist: {base}\n"
            f"Usage: python3 bcesv_run_all.py <path-to-data-dir>\n"
            f"From the bcesv/ directory: python3 bcesv_run_all.py ../data"
        )
    # Count condition dirs (must have responses.jsonl)
    cond_dirs = [d for d in base.iterdir()
                 if d.is_dir() and (d / "responses.jsonl").exists()]
    if not cond_dirs:
        raise SystemExit(
            f"ERROR: no condition directories found under {base}\n"
            f"Expected: <data>/<condition>/responses.jsonl\n"
            f"Usage: python3 bcesv_run_all.py <path-to-data-dir>\n"
            f"From the bcesv/ directory: python3 bcesv_run_all.py ../data"
        )
    print(f"Found {len(cond_dirs)} condition directories.", flush=True)
    main(base)
