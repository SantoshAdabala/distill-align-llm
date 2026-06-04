#!/bin/bash
# ──────────────────────────────────────────────────────────────
# GPU Box Setup Script
# ──────────────────────────────────────────────────────────────
# Run this ONCE when you first open a fresh GPU box.
# After setup, run the training commands below.
#
# USAGE:
#   1. Provision a GPU box (A100 recommended)
#   2. Open a terminal on the box
#   3. Run: bash scripts/setup_gpu.sh
#   4. Then run training (see bottom of this script)
# ──────────────────────────────────────────────────────────────

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════"
echo "  Setting up distill-align-llm on the GPU box"
echo "═══════════════════════════════════════════════════════════"

# Clone repo (skip if already cloned)
if [ ! -d "distill-align-llm" ]; then
    echo "[1/5] Cloning repository..."
    git clone https://github.com/SantoshAdabala/distill-align-llm.git
    cd distill-align-llm
else
    echo "[1/5] Repository already exists, pulling latest..."
    cd distill-align-llm
    git pull
fi

# Install dependencies
echo "[2/5] Installing Python dependencies..."
pip install -q torch torchvision torchaudio
pip install -q transformers accelerate peft datasets bitsandbytes trl
pip install -q sentence-transformers
pip install -q -e .

# Login to HuggingFace (needed for Llama and Mistral gated models)
echo "[3/5] HuggingFace login..."
echo "  You need a HuggingFace token with access to:"
echo "    - meta-llama/Llama-3.1-8B-Instruct"
echo "    - meta-llama/Llama-3.2-3B-Instruct"
echo "    - mistralai/Mistral-7B-Instruct-v0.3"
echo ""
echo "  Get your token at: https://huggingface.co/settings/tokens"
echo ""
huggingface-cli login

# Check GPU
echo "[4/5] Checking GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

# Verify setup
echo "[5/5] Verifying installation..."
python -c "from distill_align.config import ConfigManager; print('✅ distill_align package OK')"
python -c "from transformers import AutoTokenizer; print('✅ transformers OK')"
python -c "from trl import SFTTrainer, DPOTrainer; print('✅ TRL OK')"
python -c "from peft import LoraConfig; print('✅ PEFT OK')"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Setup complete! Now run training:"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  # Option 1: Full pipeline (recommended — runs SFT+DPO+Eval)"
echo "  nohup python scripts/run_full_pipeline.py --config configs/mistral_7b.yaml > mistral_log.txt 2>&1 &"
echo "  nohup python scripts/run_full_pipeline.py --config configs/llama_3b.yaml > llama3b_log.txt 2>&1 &"
echo ""
echo "  # Option 2: Run sequentially (watch progress)"
echo "  python scripts/run_full_pipeline.py --config configs/mistral_7b.yaml"
echo ""
echo "  # Monitor progress:"
echo "  tail -f mistral_log.txt"
echo ""
echo "  # Estimated times (A100):"
echo "    Mistral-7B:   ~1.5 hours"
echo "    Llama-3.2-3B: ~50 minutes"
echo "═══════════════════════════════════════════════════════════"
