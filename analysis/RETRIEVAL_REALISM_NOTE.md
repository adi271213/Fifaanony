# Retrieval-realism sanity check — artifact-only

This note accompanies `retrieval_realism_sanity.py` and
`retrieval_realism_results.csv` in this directory. The check is
released as part of the artifact only; it is not used to compute any
headline number in the paper, and the paper's empirical results
should be read independently of it.

## What the script does

For each of the 12 (model, seed, condition) cells in `data/`:

1. Index every fragment in that cell's `fragments.store.jsonl`
   using two off-the-shelf retrievers:
    - **BM25** over fragment `sanitized_text`
    - **MiniLM** (sentence-transformers `all-MiniLM-L6-v2`) embeddings
2. Issue two query sets:
    - **Probe-derived queries**: for each cross-boundary trap probe,
      a natural-language search built by concatenating the target
      recall's `subject action object outcome time`. This is the
      query an analyst would type when investigating the event the
      probe is designed to test.
    - **Generic threat-hunting queries**: 40 hand-written SOC
      queries embedded in the script, written independently of the
      probe templates.
3. Retrieve top-10 per query.
4. Score each query on two metrics:
    - **Role-complete cross-boundary**: do the retrieved fragments
      collectively cover all five role slots AND span at least two
      distinct boundaries?
    - **Supplying-fragment recall** (probe queries only): of the five
      ground-truth supplying fragments the construction-attached
      unsafe rung surfaces for this probe, how many appear in the
      retriever's top-10?

## Pooled result (1,200 cross-boundary probe queries × 12 conditions)

|                                       | BM25         | MiniLM       |
|---------------------------------------|--------------|--------------|
| Role-complete cross-boundary          | 106/1200 = 8.8% | 0/1200 = 0.0% |
| Probes with ≥1 supplying hit in top-10| 90/1200 = 7.5%  | 52/1200 = 4.3% |
| Probes with ALL supplying hits        | 0/1200 = 0.0%   | 0/1200 = 0.0%  |
| Pooled supplying-fragment recall      | 90/6000 = 1.5%  | 52/6000 = 0.9% |
| Generic role-complete cross-boundary  | 0/480 = 0.0%    | 0/480 = 0.0%   |

## What this means

Off-the-shelf retrievers do **not** naturally surface the
role-complete cross-boundary fragment combinations that our
construction-attached unsafe rungs surface by construction. Across
1,200 trap queries and 6,000 supplying fragments, neither BM25 nor
MiniLM lands all five supplying fragments in top-10 even once, and
the role-complete cross-boundary co-occurrence rate is at most 8.8%.

This is consistent with the structure of the failure mode. Each
ground-truth supplying fragment contains only one of the five role
components of the probe's target recall, so a similarity-based
retriever scoring against the full query string will, in
expectation, favour fragments that contain *multiple* of those
tokens (typical of fragments belonging to a single event that
genuinely covers the query) over five disjoint single-component
fragments drawn from five different events.

## Implications for the paper

The paper's headline empirical results use construction-attached
retrieval rungs, and the §7.1 "Scope of the empirical claim"
paragraph already states that the contribution is conditional on
exposure: MnemBound shows that *when* role-complete cross-boundary
evidence is exposed to the agent, current agent-memory interfaces
lack the information needed to refuse boundary-invalid composition.
The rate at which such exposure arises in any particular deployed
system is a property of that system's retrieval pipeline, not of
the failure mode itself.

The sanity check here is consistent with that framing in two ways:

1. **It motivates the use of construction-attached rungs as a
   controlled stress test.** The failure mode is hard to elicit
   under off-the-shelf BM25 or MiniLM retrieval, which is precisely
   why a benchmark is needed: to isolate the boundary-integrity
   question from confounders introduced by ANN search noise,
   embedding quality, reranker behaviour, or query rewriting.

2. **It does not undermine the security implication.** The
   reachability of the failure mode in a given deployed system is a
   function of that system's retrieval design (shared-vector
   indexing across tenants, downstream summary-merge aggregators,
   threat-hunting workflows that intentionally surface multi-tenant
   evidence, query rewriting that broadens recall, agentic
   tool-use that issues multiple retrievals per turn). Off-the-shelf
   BM25 and MiniLM with literal recall-derived queries represent
   one point in this space, and not the worst-case one.

## Reproduction

```bash
pip install rank-bm25 sentence-transformers numpy
cd mnembound_repo
python analysis/retrieval_realism_sanity.py \
    --data-root data \
    --output-csv analysis/retrieval_realism_results.csv
```

Smoke test on one condition:

```bash
python analysis/retrieval_realism_sanity.py --quick
```

Total runtime: 5–10 minutes on CPU. Deterministic given fixed
fragment text; no GPU, no model weights beyond MiniLM (≈90 MB).