# BCESV — Boundary-Constrained Event-Source Verifier

BCESV is a post-hoc verifier that checks whether the components of a
model's recall response actually trace to a coherent set of fragments
from a single boundary, or whether they were stitched across multiple
boundaries (the cross-boundary trap pattern documented in §7).

## Variants

| Variant | Description |
|---|---|
| FragmentOnly | Opaque provenance handles only (no resolver). Baseline. |
| BcesvExact | Trusted resolver, exact-match verification. Upper bound. |
| BcesvNoisy@10/20/30 | Drop noise simulation; fails-open to ABSTAIN. |
| BcesvReconstruct | R3 feature clustering, no trusted resolver. |
| (five single-feature ablations) | per-feature contribution analysis |

## How to run

```bash
cd bcesv
python3 test_bcesv.py                              # smoke test
python3 bcesv_run_all.py ../data | tee bcesv_run_all.log
```

The `../data` argument is required. The script reads each
`data/<condition>/{fragments.store.jsonl, probes.jsonl, responses.jsonl}`
and writes:

- `bcesv_results.csv` — 1,057 rows (11 variants × 12 conditions × rungs)
- `bcesv_summary.csv` — 89 rows (variant × rung pooled)

Reproducibility: the released CSVs have MD5 checksums:

| File | MD5 |
|---|---|
| `bcesv_results.csv` | `313d17510c142ccea8917f9666447e8e` |
| `bcesv_summary.csv` | `861b8ab75fc941eb7a8a37ab761d7bb0` |

A fresh run of `bcesv_run_all.py ../data` produces bit-for-bit identical
CSVs (matching MD5s).

## Headline numbers (paper Table 2, SV rung)

| Variant | Recall | Precision | FPR (supported) |
|---|---|---|---|
| FragmentOnly | 100.0% | 99.7% | 100.0% |
| BcesvExact | 100.0% | 100.0% | 0.0% |
| BcesvNoisy(30%) SV | 19.0% | 100.0% | 0.0% |
| BcesvNoisy(30%) SM | 23.3% | 100.0% | 0.0% |

FragmentOnly flags every supported recall as positive (FPR = 100%),
confirming Theorem 1 empirically: opaque-handle-only verifiers cannot
achieve nontrivial precision and recall simultaneously.

## Files

| File | Purpose |
|---|---|
| `bcesv.py` | Verifier library (StoreEntry, BCESV classes, variants) |
| `bcesv_run_all.py` | Driver — loads all 12 conditions, writes both CSVs |
| `test_bcesv.py` | Smoke test on synthetic mini-corpus |
| `bcesv_results.csv` | Precomputed per-condition results |
| `bcesv_summary.csv` | Precomputed variant×rung summary |
