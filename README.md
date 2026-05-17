# AlignLLM — End-to-End LLM Alignment Pipeline

Fine-tuning and aligning large language models using **SFT → DPO → RLHF (GRPO)** with parameter-efficient methods (LoRA/QLoRA). Trained Llama-3.1-8B on a single RTX 3090 for $1.01 total.

---

## What This Does

Takes a pre-trained LLM through three alignment stages:

1. **SFT** (Supervised Fine-Tuning) — Teaches the model to follow instructions
2. **DPO** (Direct Preference Optimization) — Aligns responses with human preferences
3. **RLHF/GRPO** (Reinforcement Learning from Human Feedback) — Policy optimization with reward signals

All training uses QLoRA (4-bit quantization + LoRA adapters), making it possible to fine-tune 8B parameter models on a single consumer GPU.

---

## Training Results

| Stage | Loss | Steps | Duration | GPU |
|-------|------|-------|----------|-----|
| **SFT** | 1.3879 | 297 | 13.6 min | RTX 3090 |
| **DPO** | 0.6976 | 1,187 | 71.5 min | RTX 3090 |

- **Model**: meta-llama/Llama-3.1-8B
- **Quantization**: QLoRA (4-bit NF4, double quantization, bf16 compute)
- **Trainable params**: 13.6M (0.30% of 4.55B total)
- **Platform**: RunPod.io
- **Total cost**: $1.01

See [docs/RESULTS.md](docs/RESULTS.md) for full training curves and analysis.

---

## Repository Structure

```
distill-align-llm/
├── configs/local_small.yaml       # Training hyperparameters
├── src/distill_align/
│   ├── config/                    # YAML + Pydantic config system
│   ├── data/processor.py          # Dataset loading, validation, tokenization
│   ├── models/loader.py           # Model loading with QLoRA + LoRA
│   ├── training/
│   │   ├── sft.py                 # SFT trainer (TRL SFTTrainer)
│   │   ├── dpo.py                 # DPO trainer (TRL DPOTrainer)
│   │   └── rlhf.py               # RLHF trainer (TRL GRPOTrainer)
│   ├── serving/
│   │   ├── engine.py              # vLLM inference engine
│   │   └── api.py                 # FastAPI REST gateway
│   └── monitoring/service.py      # Prometheus metrics
├── scripts/
│   ├── run_sft.py                 # SFT entry point
│   ├── run_dpo.py                 # DPO entry point
│   └── compare_models.py         # Base vs SFT vs DPO response comparison
├── dashboard/app.py               # Streamlit results dashboard
├── tests/                         # 44 passing tests
├── Makefile
└── pyproject.toml
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/SantoshAdabala/distill-align-llm.git
cd distill-align-llm
make install

# Run tests
make test

# Run SFT training (requires GPU)
python scripts/run_sft.py --config configs/local_small.yaml

# Run DPO alignment (requires SFT adapter)
python scripts/run_dpo.py --config configs/local_small.yaml --sft-adapter ./outputs/sft/final_adapter

# Compare model responses
python scripts/compare_models.py \
    --base-model meta-llama/Llama-3.1-8B \
    --sft-adapter ./outputs/sft/final_adapter \
    --dpo-adapter ./outputs/dpo/dpo_adapter

# Launch dashboard
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Training** | PyTorch, HuggingFace Transformers, TRL, PEFT, bitsandbytes |
| **Data** | HuggingFace Datasets |
| **Serving** | vLLM, FastAPI |
| **Monitoring** | Prometheus |
| **Dashboard** | Streamlit, Plotly |
| **Testing** | pytest, ruff |

---

## Model Configuration

```yaml
model:
  model_id: "Qwen/Qwen2.5-1.5B"    # or meta-llama/Llama-3.1-8B
  family: "qwen2.5"
  quantization:
    mode: "int4_nf4"                 # 4-bit QLoRA
    use_double_quant: true
  lora:
    rank: 16
    alpha: 32
    target_modules: [q_proj, k_proj, v_proj, o_proj]

sft:
  learning_rate: 0.0002
  batch_size: 2
  gradient_accumulation_steps: 8

dpo:
  beta: 0.1
  learning_rate: 0.00001
  eval_steps: 100
```

---

## License

MIT

---

*Built by [Santosh Adabala](https://github.com/SantoshAdabala)*
