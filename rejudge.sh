#!/bin/bash
# Re-judge the 500 stored responses with a stronger judge.
# Usage:  bash rejudge.sh           (defaults to gpt-4o)
#         bash rejudge.sh o4-mini   (or any OpenAI model id)
cd "$(dirname "$0")"
MODEL="${1:-gpt-4o}"
if [ -z "$OPENAI_API_KEY" ]; then
  read -rsp "Paste your OpenAI API key (hidden), then press Enter: " OPENAI_API_KEY
  echo
  export OPENAI_API_KEY
fi
echo "Judging 500 responses with: $MODEL"
.venv/bin/python scripts/rejudge.py --model "$MODEL"
