#!/bin/bash
# One-word environment check for the GPU box:  bash check.sh
cd "$(dirname "$0")"
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
echo "--- python: $PY ---"
"$PY" - <<'EOF'
try:
    import torch, transformers, peft, bitsandbytes, accelerate
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    print("transformers", transformers.__version__, "peft", peft.__version__)
except Exception as e:
    print("DEPS MISSING:", e)
try:
    from huggingface_hub import HfFolder
    print("HF token:", "ok" if HfFolder.get_token() else "MISSING")
except Exception as e:
    print("hf check error:", e)
EOF
echo "--- model folders present in outputs/ ---"
ls -d outputs/*/ 2>/dev/null | grep -iE "merged|sft|dpo|8b|3b|mistral|llama" || echo "(none)"
echo "--- what trap.sh / ece.sh look for ---"
for d in outputs/sft/final_adapter outputs/dpo/dpo_adapter outputs/sft_merged outputs/dpo_8b_merged; do
  [ -d "$d" ] && echo "FOUND   $d" || echo "missing $d"
done
