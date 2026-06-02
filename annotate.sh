#!/bin/bash
# Launcher for the human annotation session (kappa validation of the LLM judge).
# Run from the project root with:  bash annotate.sh
cd "$(dirname "$0")"
.venv/bin/python scripts/human_annotation.py \
  --results_csv outputs/human_annotation/annotation_input.csv \
  --n_samples 100 \
  --output_dir outputs/human_annotation
