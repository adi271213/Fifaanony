#!/usr/bin/env bash
# verify.sh — sanity check the repo structure
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

echo "==> Verifying MnemBound repository structure"
echo ""

# 1. Top-level
for f in README.md REPRODUCE.md LICENSE requirements.txt MANIFEST.sha256; do
    if [[ -f "$f" ]]; then
        echo "  [ok] $f"
    else
        echo "  [MISSING] $f"
    fi
done

# 2. Directories
for d in data analysis bcesv mnembound paper; do
    if [[ -d "$d" ]]; then
        echo "  [ok] $d/"
    else
        echo "  [MISSING] $d/"
    fi
done

# 3. Conditions
echo ""
echo "==> Conditions:"
n_cond=$(ls -d data/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "    found $n_cond (expected 12)"
for cond in data/*/; do
    cond_name=$(basename "$cond")
    if [[ -f "$cond/responses.jsonl" && -f "$cond/fragments.store.jsonl" && -f "$cond/probes.jsonl" ]]; then
        n_resp=$(wc -l < "$cond/responses.jsonl" | tr -d ' ')
        echo "  [ok] $cond_name (responses=$n_resp)"
    else
        echo "  [INCOMPLETE] $cond_name"
    fi
done

# 4. Analysis scripts
echo ""
echo "==> Analysis scripts:"
for s in rescore_final_ci.py scorer_sensitivity.py parse_fail_sensitivity.py \
         qualitative_cases.py manual_scorer_audit.py audit_score.py; do
    if [[ -f "analysis/$s" ]]; then
        echo "  [ok] analysis/$s"
    else
        echo "  [MISSING] analysis/$s"
    fi
done

# 5. BCESV
echo ""
echo "==> BCESV:"
for s in bcesv.py bcesv_run_all.py test_bcesv.py bcesv_results.csv bcesv_summary.csv; do
    if [[ -f "bcesv/$s" ]]; then
        echo "  [ok] bcesv/$s"
    else
        echo "  [MISSING] bcesv/$s"
    fi
done

# 6. Paper
echo ""
echo "==> Paper:"
for f in main.tex references.bib main.pdf; do
    if [[ -f "paper/$f" ]]; then
        echo "  [ok] paper/$f"
    else
        echo "  [MISSING] paper/$f"
    fi
done

# 7. Smoke test: load one condition's data
echo ""
echo "==> Smoke test: load one condition..."
python3 -c "
import json
store = {}
with open('data/qwen_full_neutral/fragments.store.jsonl') as f:
    for line in f:
        d = json.loads(line)
        store[d['public_provenance_id']] = d
print(f'  [ok] loaded {len(store)} fragments')

with open('data/qwen_full_neutral/responses.jsonl') as f:
    responses = [json.loads(line) for line in f]
print(f'  [ok] loaded {len(responses)} responses')

required_fragment_fields = ['boundary_id', 'sanitized_value', 'sanitized_text']
sample = next(iter(store.values()))
missing = [k for k in required_fragment_fields if k not in sample]
if missing:
    print(f'  [WARN] missing fragment fields: {missing}')
else:
    print(f'  [ok] all required fragment fields present')

required_response_fields = ['recall', 'fragment_provenance_ids', 'probe_type', 'rung']
missing = [k for k in required_response_fields if k not in responses[0]]
if missing:
    print(f'  [WARN] missing response fields: {missing}')
else:
    print(f'  [ok] all required response fields present')
"

echo ""
echo "==> Verification complete."
