#!/bin/bash
# Audit the 500 gold reference answers for factual errors.
# Usage:  bash audit.sh           (defaults to gpt-4o)
#         bash audit.sh o4-mini
cd "$(dirname "$0")"
MODEL="${1:-gpt-4o}"
if [ -z "$OPENAI_API_KEY" ]; then
  read -rsp "Paste your OpenAI API key (hidden), then press Enter: " OPENAI_API_KEY
  echo
  export OPENAI_API_KEY
fi
echo "Auditing 500 reference answers with: $MODEL"
.venv/bin/python scripts/audit_references.py --model "$MODEL"
