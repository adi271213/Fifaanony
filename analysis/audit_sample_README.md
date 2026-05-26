# Manual Scorer Audit — Instructions

## Goal

Validate the substring rescorer (HARDENED) against manual judgment on a stratified sample of 100 responses.

## How to fill in

For each row in `audit_sample.csv`, examine the recall fields and fragment_summary, then decide:

**human_decision**: one of
  - `INVALID` (cross-boundary trap, model composed across boundaries)
  - `SAFE` (cross-boundary trap, model did not compose across boundaries)
  - `ABSTAIN` (model declined to recall)
  - `PARSE_FAIL` (model output was malformed)
  - `RECALL` (supported/absent/partial probe, model produced a recall)
  - `NO_MATCH` (model recall has components that don't match any fragment)

**agree**: 1 if `human_decision` matches `scorer_decision`, 0 otherwise.

**human_notes**: optional, any disagreement reasoning.

## After labeling

Run `python3 audit_score.py` to compute the agreement rate.
