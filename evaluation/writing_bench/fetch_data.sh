#!/usr/bin/env bash
# Fetch the WritingBench data files (benchmark_query/) from the upstream
# repository. The benchmark JSONL is ~40MB and contains long reference
# materials, so it is NOT committed here — run this once after cloning.
#
# Idempotent: if benchmark_query/benchmark_all.jsonl already exists, it just
# prints OK and exits.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/benchmark_query"
TARGET="${DEST}/benchmark_all.jsonl"

if [[ -f "${TARGET}" ]]; then
    echo "[fetch_data] ${TARGET} already exists — nothing to do."
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "[fetch_data] Cloning upstream WritingBench repo (shallow)..."
git clone --depth=1 https://github.com/X-PLUG/WritingBench.git "${TMP}/wb"

echo "[fetch_data] Copying benchmark_query/ -> ${DEST}"
mkdir -p "${DEST}"
cp -R "${TMP}/wb/benchmark_query/." "${DEST}/"

echo "[fetch_data] Done. Files now in ${DEST}:"
ls -la "${DEST}"
