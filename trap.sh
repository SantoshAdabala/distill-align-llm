#!/bin/bash
# Run the trap set through base/sft/dpo and report refusal vs fabrication per stage.
# Defaults to the Llama-3.1-8B paths. Override for other models, e.g.:
#   bash trap.sh --tag mistral7b --base_model mistralai/Mistral-7B-Instruct-v0.3 \
#       --sft_adapter outputs/mistral_7b/sft_adapter --dpo_base outputs/mistral_merged \
#       --dpo_adapter outputs/mistral_7b/dpo_adapter
# Stages whose adapter folder is absent are skipped (base always runs).
cd "$(dirname "$0")"
PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
"$PY" scripts/trap_eval.py "$@"
