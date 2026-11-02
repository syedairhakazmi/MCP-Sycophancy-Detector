#!/usr/bin/env bash
# Clone the base paper's code + data at the exact commit this project was built against.
# Base paper: Kasprova et al., "Too Polite to Disagree" (arXiv 2604.02668), MIT licence.
set -euo pipefail
PIN=e6421272c03933dc8d0ebd286fb39333081857bc
DEST="${1:-third_party/kasprova}"
mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/mathismusic/multiagent-discussion-sycophancy "$DEST"
fi
git -C "$DEST" fetch --quiet origin || true
git -C "$DEST" checkout --quiet "$PIN"
echo "base repo at $(git -C "$DEST" rev-parse --short HEAD)  ->  $DEST"
ls "$DEST/data_for_dss.csv" "$DEST/scores/knowledge_flags.json"
