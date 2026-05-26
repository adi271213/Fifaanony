# MnemBound — Fragment-Faithful, Event-False Recall in Multi-Tenant Agent Memory

**Anonymous submission to ACSAC 2026.**

This repository contains the complete reproducibility artifact for the paper
*MnemBound: Fragment-Faithful, Event-False Recall in Multi-Tenant Agent
Memory.* It includes the benchmark generator, 14,400 LLM responses across
12 experimental conditions, the BCESV post-hoc verifier, all analysis
scripts, and the paper LaTeX source.

## Quick start

```bash
# Verify the repository
./verify.sh

# Reproduce the headline table (~5 seconds)
cd analysis
python3 rescore_final_ci.py
```

For full reproduction instructions, see `REPRODUCE.md`.

## Repository structure
mnembound_repo/
├── README.md             this file
├── REPRODUCE.md          step-by-step reproduction guide
├── LICENSE               Apache-2.0
├── requirements.txt      Python dependencies (stdlib only)
├── verify.sh             one-command sanity check
├── MANIFEST.sha256       cryptographic integrity manifest
├── data/                 12 conditions × responses + fragments + probes
│   ├── qwen_full_neutral/         (Qwen, seed 42, neutral prompt, collapsed)
│   ├── qwen_full_cautious/        (Qwen, seed 42, cautious prompt, collapsed)
│   ├── qwen_full_pcb/             (Qwen, seed 42, neutral, per-client-blocks)
│   ├── qwen_full_cautious_pcb/    (Qwen, seed 42, cautious, per-client-blocks)
│   ├── qwen_seed43_neutral/       (Qwen, seed 43, neutral, collapsed)
│   ├── qwen_seed44_neutral/       (Qwen, seed 44, neutral, collapsed)
│   ├── mistral_full_neutral/      (Mistral × 4 prompt-aggregator combinations)
│   ├── mistral_full_cautious/
│   ├── mistral_full_pcb/
│   ├── mistral_full_cautious_pcb/
│   ├── mistral_seed43_neutral/
│   └── mistral_seed44_neutral/
├── analysis/             6 paper analysis scripts
│   ├── rescore_final_ci.py        Wilson 95% CIs across all cells
│   ├── scorer_sensitivity.py      EXACT / HARDENED / LOOSE scorers
│   ├── parse_fail_sensitivity.py  hostile parse-failure handling
│   ├── qualitative_cases.py       case studies for §7.3
│   ├── manual_scorer_audit.py     builds the stratified audit sample
│   ├── audit_score.py             computes agreement rate + Cohen's κ
│   ├── audit_sample.csv           the 100-row labeled audit sample
│   └── audit_sample_README.md
├── bcesv/                Post-hoc verifier (§8)
│   ├── bcesv.py                   verifier implementation
│   ├── bcesv_run_all.py           driver: 11 variants × 12 conditions × 4 rungs
│   ├── test_bcesv.py              smoke test
│   ├── BCESV_README.md
│   ├── bcesv_results.csv          1057-row per-cell results
│   ├── bcesv_summary.csv          89-row summary by variant × rung
│   └── bcesv_run_all.log          execution log
├── mnembound/            corpus generator + retrieval harness 
│   └── mnembound_pkg/             137-test Python package
└── paper/                LaTeX source + compiled PDF
├── main.tex
├── references.bib
└── main.pdf                   compiled paper preview

## What's in each condition directory

Each `data/<condition>/` folder contains:

| File | Lines | Purpose |
|---|---|---|
| `responses.jsonl` | 1{,}200 | 300 probes × 4 retrieval rungs; the LLM's structured recall + metadata |
| `fragments.store.jsonl` | 25{,}000 | Trusted store metadata: `boundary_id`, sanitized values, tenant, event, timestamp |
| `probes.jsonl` | 300 | Probe definitions (100 supported, 100 cross-boundary, 50 absent, 50 partial) |
| `corpus.json` | 1 | Corpus generator parameters (seed, tenants, density, etc.) |
| `metrics.json` | 1 | Generator-time validation metrics |
| `rung_stats.jsonl` | 4 | Per-rung aggregate statistics |

Records and the un-sanitized raw fragment fields are dropped from the public
repo. They reproduce the corpus, not the experiment, and contain raw tenant
identifiers that are intentionally hidden from the model and the verifier.
The corpus generator in `mnembound/` reproduces them from a seed.

## Data hygiene notes

The `fragments.store.jsonl` files contain `tenant_id` and raw `entity_set`
values. These are **store-internal** metadata that the deployed system has
access to but never exposes to the agent or the public-handle layer. They
are needed by BCESV-Reconstruct's R3 rule (tenant + entity-Jaccard cohesion
clustering) and by the HARDENED scorer's trusted-resolver call. The agent's
view (which is what produced `responses.jsonl`) uses only the `sanitized_*`
fields. See §4 and §8 of the paper for the formal model.

## Reproducibility scope

Running `python3 rescore_final_ci.py` regenerates Table 1 of the paper
deterministically from the saved responses. No GPU is required. Total
analysis runtime: under 30 seconds on a laptop.

Running the corpus generator (`mnembound/mnembound_pkg`) end-to-end requires
deterministic seeded execution and reproduces `fragments.store.jsonl` and
`probes.jsonl` from `corpus.json`. The LLM-evaluation step (regenerating
`responses.jsonl`) requires an Nvidia H100 (or equivalent FP8-capable GPU)
running vLLM 0.21 with the two open-weights models named in §5.4. Approximate
cost to fully regenerate the 14{,}400 responses: ~3 GPU-hours, ~\$10.

## Licence

Apache-2.0. See `LICENSE`.
