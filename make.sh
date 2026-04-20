#!/usr/bin/env bash
# Pipeline runner — each target corresponds to a notebook section.
# Usage: ./make.sh <target> [options]
#
# Targets:
#   fetch           Download data_files/ from Google Drive (or --local PATH)
#   generate        Step 3-1: generate LLM solutions (--n N to limit problems)
#   prepare         Step 3-2: build SFT / DPO / RFT datasets
#   human_dist      Step 3-3: pre-compute human reference distribution
#   sft             Step 4:   SFT warm-start fine-tune
#   dpo             Step 5:   DPO preference fine-tune (--base-model <id> optional)
#   rft             Step 6:   RFT/PPO fine-tune       (--base-model <id> optional)
#   benchmark       Step 7:   evaluate all models      (--n N problems, default 50)
#   all             Run generate → prepare → human_dist → sft → dpo → rft → benchmark

set -euo pipefail
cd "$(dirname "$0")"

CONDA_ENV="mft"

# Activate conda env if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || { echo "conda not found"; exit 1; }
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

TARGET="${1:-help}"
shift || true          # remaining args forwarded to the script

case "$TARGET" in
  fetch)
    python scripts/fetch_data.py "$@"
    ;;

  generate)
    python scripts/generate_solutions.py "$@"
    ;;

  prepare)
    python scripts/prepare_datasets.py "$@"
    ;;

  human_dist)
    python scripts/verified_rewards.py \
      --solutions data_files/llm_solutions.jsonl \
      --out       data_files/human_dist.json \
      "$@"
    ;;

  sft)
    python scripts/sft_train.py "$@"
    ;;

  dpo)
    python scripts/dpo_train.py "$@"
    ;;

  rft)
    python scripts/rft_train.py "$@"
    ;;

  benchmark)
    python scripts/benchmark.py "$@"
    ;;

  all)
    bash "$0" generate "$@"
    bash "$0" prepare
    bash "$0" human_dist
    bash "$0" sft
    bash "$0" dpo
    bash "$0" rft
    bash "$0" benchmark
    ;;

  help|--help|-h|*)
    echo "Usage: ./make.sh <target> [options]"
    echo ""
    echo "Targets:"
    echo "  fetch       [--local PATH]       Download data_files/ from Google Drive (or local dir)"
    echo "  generate    [--n N]              Generate LLM solutions from EvalPlus"
    echo "  prepare                          Build SFT / DPO / RFT datasets"
    echo "  human_dist                       Pre-compute human reference distribution"
    echo "  sft         [--no-wait] [--no-wandb] [--project P] [--experiment N]"
    echo "  dpo         [--base-model ID] [--no-wandb] [--project P] [--experiment N]"
    echo "  rft         [--base-model ID] [--gated-style] [--no-wandb] [--project P] [--experiment N]"
    echo "  benchmark   [--n N]              Evaluate all models (default: 50 problems)"
    echo "  all         [--n N]              Run full pipeline end-to-end"
    ;;
esac
