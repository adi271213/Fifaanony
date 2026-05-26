#!/usr/bin/env bash
# generate_manifest.sh — produce MANIFEST.sha256 listing every file's hash
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
echo "==> Generating MANIFEST.sha256"
find . -type f \
    ! -name "MANIFEST.sha256" \
    ! -name ".DS_Store" \
    ! -path "./.git/*" \
    ! -path "*/__pycache__/*" \
    ! -name "*.pyc" \
    ! -name "._*" \
    -print0 | sort -z | xargs -0 shasum -a 256 > MANIFEST.sha256
echo "  wrote $(wc -l < MANIFEST.sha256 | tr -d ' ') file hashes"
