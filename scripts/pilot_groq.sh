#!/usr/bin/env bash
# TRACK B pilot on the free Groq tier. export GROQ_API_KEY=... first.
set -euo pipefail
OUT=runs/pilot_llama8b
SNAP=runs/shared_search_snapshots.json     # one snapshot file shared across ALL models -> identical evidence
COMMON="--source both --n-per-subject 10 --rounds 3 --search mcp --snapshot $SNAP"

python -m syc.runner plan $COMMON                       # read the call estimate and compare with your Groq limits
python -m syc.runner run  $COMMON --answerer groq:openai/gpt-oss-20b --detector groq:openai/gpt-oss-120b \
       --max-calls 1500 --out $OUT                      # re-run the same command to resume after a stop
python -m syc.evaluate --runs $OUT/conversations.jsonl
# when you have flips: blind sample for the two human raters
python -m syc.gold export --runs $OUT/conversations.jsonl --per-style 25 --out gold/gold_sample.csv
