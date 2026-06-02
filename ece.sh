#!/bin/bash
# Expected Calibration Error for one stage (defaults to the DPO Llama-3.1-8B, which already
# has responses + gpt-4o correctness). For base/sft, generate and judge those stages first,
# then pass --responses_csv / --correct_csv for that stage.
cd "$(dirname "$0")"
PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
"$PY" scripts/ece.py "$@"
