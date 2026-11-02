#!/usr/bin/env bash
# Offline sanity check (no API keys, no GPU, no internet): unit tests + a mock end-to-end run + evaluation.
set -euo pipefail
python -m pytest -q tests
rm -rf runs/smoke
python -m syc.runner run --source bank --rounds 3 --answerer mock:1 --search stub --engine langgraph --out runs/smoke \
  --styles pure_insistence,appeal_to_authority,social_pressure,fabricated_citation,genuine_counterargument
python -m syc.evaluate --runs runs/smoke/conversations.jsonl --out runs/smoke/eval | tail -n 5
echo "SMOKE TEST OK"
