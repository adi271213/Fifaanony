"""
audit_labeler.py — interactive labeling tool for audit_sample.csv

For each unlabeled row, displays the probe context (model recall + fragment
summary + scorer decision) and prompts the auditor to type their own
independent label. Writes back to audit_sample.csv incrementally so you
can quit and resume at any time.

The tool deliberately does NOT suggest decisions. The auditor must read
each row and choose.

USAGE:
    cd analysis
    python3 audit_labeler.py

KEYS:
    i  INVALID       (cross-boundary trap: model composed across boundaries)
    s  SAFE          (cross-boundary trap: model stayed in one boundary)
    a  ABSTAIN       (model declined to recall)
    p  PARSE_FAIL    (model output was malformed)
    r  RECALL        (supported/absent/partial: model produced a recall)
    n  NO_MATCH      (model recall has components not matching any fragment)
    ?  show this row again
    b  back (re-label previous row)
    q  quit and save
"""
import csv
import sys
from pathlib import Path
from collections import Counter

DECISIONS = {
    "i": "INVALID",
    "s": "SAFE",
    "a": "ABSTAIN",
    "p": "PARSE_FAIL",
    "r": "RECALL",
    "n": "NO_MATCH",
}

CSV_PATH = Path("audit_sample.csv")
FIELDNAMES_REQUIRED = [
    "audit_id", "model", "condition", "probe_id", "probe_type", "rung",
    "scorer_decision", "recall_subject", "recall_action", "recall_object",
    "recall_outcome", "recall_time", "fragment_summary",
    "human_decision", "human_notes", "agree",
]


def show_row(row, idx, total, n_labeled):
    """Show a row with NO decision suggestion. Just the facts."""
    print("\n" + "=" * 78)
    print(f"  Row {idx + 1} / {total}    "
          f"(labeled so far: {n_labeled})    "
          f"audit_id: {row['audit_id']}")
    print("=" * 78)
    print(f"  model:       {row['model']}")
    print(f"  condition:   {row['condition']}")
    print(f"  probe_type:  {row['probe_type']}")
    print(f"  rung:        {row['rung']}")
    print(f"  probe_id:    {row['probe_id']}")
    print()
    print("  --- MODEL RECALL ---")
    print(f"    subject: {row.get('recall_subject', '')}")
    print(f"    action:  {row.get('recall_action', '')}")
    print(f"    object:  {row.get('recall_object', '')}")
    print(f"    outcome: {row.get('recall_outcome', '')}")
    print(f"    time:    {row.get('recall_time', '')}")
    print()
    print("  --- FRAGMENT SUMMARY (what was retrieved) ---")
    print(f"    {row.get('fragment_summary', '')}")
    print()
    print(f"  Scorer decided: {row['scorer_decision']}")
    print()


def save_csv(rows, fieldnames):
    """Atomic write: write to .tmp, then rename."""
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    tmp.replace(CSV_PATH)


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run from analysis/ directory.",
              file=sys.stderr)
        sys.exit(1)

    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or FIELDNAMES_REQUIRED
        rows = list(reader)

    for needed in ("human_decision", "agree", "human_notes"):
        if needed not in fieldnames:
            fieldnames.append(needed)

    n_total = len(rows)
    print(f"Loaded {n_total} rows from {CSV_PATH}")
    print(f"Already labeled: "
          f"{sum(1 for r in rows if r.get('human_decision', '').strip())}")
    print()
    print("Labels:")
    for k, v in DECISIONS.items():
        print(f"  [{k}] {v}")
    print("  [?] show this row again    [b] back    [q] quit and save")
    print()
    input("Press ENTER to begin...")

    i = 0
    try:
        while i < n_total:
            row = rows[i]
            # Skip already-labeled unless going back
            if row.get("human_decision", "").strip():
                i += 1
                continue

            n_labeled = sum(1 for r in rows
                            if r.get("human_decision", "").strip())
            show_row(row, i, n_total, n_labeled)

            while True:
                resp = input(f"decision (i/s/a/p/r/n/?/b/q) > ").strip().lower()
                if resp == "q":
                    raise KeyboardInterrupt
                if resp == "?":
                    show_row(row, i, n_total, n_labeled)
                    continue
                if resp == "b":
                    # Find previous labeled row, clear it, redo
                    for j in range(i - 1, -1, -1):
                        if rows[j].get("human_decision", "").strip():
                            print(f"  Clearing row {j + 1} for relabeling.")
                            rows[j]["human_decision"] = ""
                            rows[j]["agree"] = ""
                            rows[j]["human_notes"] = ""
                            save_csv(rows, fieldnames)
                            i = j
                            break
                    else:
                        print("  No previous labeled row to revisit.")
                        continue
                    break
                if resp in DECISIONS:
                    decision = DECISIONS[resp]
                    row["human_decision"] = decision
                    row["agree"] = (
                        "1" if decision == row.get("scorer_decision") else "0"
                    )
                    notes = input("notes (ENTER to skip) > ").strip()
                    row["human_notes"] = notes
                    save_csv(rows, fieldnames)
                    if row["agree"] == "0":
                        print(f"  Recorded: human={decision} "
                              f"vs scorer={row['scorer_decision']} "
                              f"(disagreement)")
                    else:
                        print(f"  Recorded: {decision} (agrees with scorer)")
                    i += 1
                    break
                print(f"  Unknown key '{resp}'. Try one of: "
                      f"{','.join(DECISIONS)},?,b,q")
    except KeyboardInterrupt:
        print("\n\nSaving progress and exiting...")

    save_csv(rows, fieldnames)
    n_labeled_final = sum(1 for r in rows
                          if r.get("human_decision", "").strip())
    print(f"\nWrote {CSV_PATH}")
    print(f"Labeled: {n_labeled_final} / {n_total}")
    if n_labeled_final == n_total:
        print("\nAll rows labeled. Run: python3 audit_score.py")
    else:
        remaining = n_total - n_labeled_final
        print(f"\n{remaining} rows still unlabeled. Re-run audit_labeler.py "
              f"to continue.")


if __name__ == "__main__":
    main()
