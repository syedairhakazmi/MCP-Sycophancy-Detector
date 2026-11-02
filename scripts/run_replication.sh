#!/usr/bin/env bash
# TRACK A -- reproduce the base paper's experiments with THEIR code, data and models (needs a GPU).
# Same CLI arguments as their run_all_experiments.sh, but sequential and on one GPU, with a model subset.
#
#   MODELS="llama3b llama8b qwen3b qwen7b" bash scripts/run_replication.sh          # free Colab/Kaggle-sized
#   MODELS="llama3b llama8b qwen3b qwen7b qwen14b qwen32b" EXPERIMENTS="baseline bss dss" bash scripts/run_replication.sh
#
# Their data + precomputed scores ship with the repo, so their steps 0a-0c are not needed.
set -euo pipefail
BASE="${BASE:-third_party/kasprova}"
MODELS="${MODELS:-llama3b llama8b qwen3b qwen7b}"
EXPERIMENTS="${EXPERIMENTS:-baseline bss accuracy_bss random_bss dss binary_bss warning_only random_binary dbss}"
cd "$BASE"
DATA=data_for_dss.csv; FLAGS=scores/knowledge_flags.json
BSS=scores/bss_scores_final.json; ACC=scores/accuracy_scores.json; RND=scores/random_scores.json; DBSS=scores/bss_scores_post_debate.json
for f in "$DATA" "$FLAGS" "$BSS"; do [ -f "$f" ] || { echo "missing $f (run scripts/setup_base.sh)"; exit 1; }; done

extra() {
  case "$1" in
    baseline)       echo "";;
    bss)            echo "--use_bss_scores --bss_scores_path $BSS";;
    accuracy_bss)   echo "--use_bss_scores --bss_scores_path $ACC";;
    random_bss)     echo "--use_bss_scores --bss_scores_path $RND";;
    dss)            echo "--use_dss_scores --bss_scores_path $BSS --alpha 0.2 --beta 0.0";;
    binary_bss)     echo "--use_binary_syco_flags --bss_scores_path $BSS";;
    warning_only)   echo "--use_warning_only";;
    random_binary)  echo "--use_random_binary_flags";;
    dbss)           echo "--use_bss_scores --bss_scores_path $DBSS";;
    *) echo "unknown experiment $1" >&2; exit 1;;
  esac
}
for exp in $EXPERIMENTS; do
  mkdir -p "logs/$exp"
  if [ -f "logs/$exp/log.jsonl" ]; then echo "($exp) already complete, skipping"; continue; fi
  echo "=== $exp  [models: $MODELS] ==="
  # shellcheck disable=SC2046
  python multiagent-debate.py -e "$exp" -m $MODELS --data_csv "$DATA" --knowledge_flags_path "$FLAGS" $(extra "$exp") \
      2>&1 | tee "logs/$exp/stdout.log"
done
python evaluate.py --all
echo "Their evaluation output: $BASE/logs/comparison.csv  (compare with Tables 2-3 of the paper)"
