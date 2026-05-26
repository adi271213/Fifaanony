# MnemBound

A controlled SOC benchmark generator for evaluating event-level provenance in multi-tenant agent memory.

## Overview

MnemBound generates synthetic Security Operations Center (SOC) corpora with construction-decidable ground truth for event-supported recall. It is designed to study whether retrieval-augmented agents preserve confidentiality boundaries when composing recalls from multiple fragments.

The package implements:

- **Corpus generation**: deterministic synthetic SOC records with explicit tenant boundaries and bridge entities (shared IPs, malware hashes, CVEs, vendors, threat actors) that legitimately appear across multiple tenants.
- **Probe construction**: four probe types (supported, absent, partial, cross-boundary) with known ground-truth boundary structure.
- **Retrieval rungs**: four retrieval configurations from boundary-preserving to deliberately cross-boundary (isolated, boundary-preserving top-k, shared-vector-index, summary-merge).
- **Agent I/O**: prompt construction and response parsing for vLLM-served instruction-tuned models.
- **Scoring**: substring-based boundary-invalid detection with three strictness levels (EXACT, HARDENED, LOOSE).

## Installation

```bash
pip install -e .
```

## Running the tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The test suite covers corpus generation, retrieval, probe construction, prompt rendering, bridge density invariants, scoring correctness, response serialization, and the smoke-runner defaults.

## End-to-end evaluation

End-to-end runs (which call vLLM on a GPU) are driven by `run_smoke.py`:

```bash
python run_smoke.py --prompt-mode neutral --summary-mode collapsed
```

Default flags reproduce the paper's headline neutral/collapsed condition. Other prompt and summary modes are documented in the `--help` output.

The runner reports two diagnostics:

- **Unsafe-rung abstention**: an over-suppression signal on `shared_vector_index` and `summary_merge` rungs.
- **Safe-rung abstention**: a sanity check on `isolated` and `boundary_preserving_topk` rungs; expected near 100% under proper isolation.

## Hardware requirements

End-to-end LLM evaluation requires a single Nvidia H100 SXM 80GB GPU with FP8 kernels and is calibrated for vLLM 0.21 with Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3. Approximate runtime: 3 GPU-hours for the full 12-condition × 1,200-response sweep.

The released artifact ships precomputed responses in `data/<condition>/responses.jsonl`. Reproducing the headline results from saved responses is CPU-only and takes under one minute (see top-level `REPRODUCE.md`).

## Package layout
mnembound_pkg/
├── README.md                 this file
├── pyproject.toml            package metadata
├── run_smoke.py              end-to-end runner (requires GPU)
├── mnembound/                importable Python package
│   ├── agent_io.py           prompt construction, response parsing
│   ├── bridges.py            cross-tenant bridge entity generation
│   ├── generator.py          corpus generation pipeline
│   ├── identifiers.py        deterministic ID generation
│   ├── probes.py             probe construction (four types)
│   ├── prompt_modes.py       prompt-rendering variants
│   ├── retrieval.py          retrieval rungs and summary modes
│   ├── scoring.py            boundary-invalid predicate
│   ├── serialization.py      JSONL I/O
│   ├── tenants.py            tenant boundary definitions
│   └── validation.py         pool and bridge invariants
└── tests/                    unit + integration tests
