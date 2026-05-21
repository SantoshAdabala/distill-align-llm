# AlignLLM — When Alignment Metrics Look Good but Factuality Does Not

End-to-end LLM alignment pipeline (**SFT → DPO**) with QLoRA, investigating how preference optimization interacts with domain-specific knowledge.

**Key finding:** DPO achieves 80% reward accuracy while factuality drops to 5.9% — but the base model itself only scores 9.8%. The bottleneck is insufficient SFT data, not DPO interference.

🔗 **[Live Dashboard](https://distill-align-llm-aembgrswzfay6bjupbnjpp.streamlit.app)**

---

## Results (v4 — Latest)

| Stage | Metric | Value |
|-------|--------|-------|
| **SFT** | Eval Loss | 0.825 |
| **SFT** | Token Accuracy | 78.3% |
| **DPO** | Reward Accuracy | 80% (peak 83%) |
| **DPO** | Loss | 0.54 |
| **Factuality** | Base / SFT / DPO | 9.8% / 7.8% / 5.9% |

### Version Progression

| Version | GPU | DPO Config | Peak Reward Acc | Key Change |
|---------|-----|-----------|-----------------|------------|
| v1 | RTX 3090 | Stacked, β=0.1, LR=5e-5 | 50% | Baseline (weak) |
| v2 | RTX A5000 | Stacked, β=0.1, LR=1e-5 | 75% | Lower LR + cleaned data |
| v3 | A100 SXM | Stacked, β=0.1 | 68% | Added technical SFT data |
| v4 | RTX A6000 | **Merged-SFT, β=0.05** | **83%** | Merge adapter + lower β |

**Total cost: ~$27** on RunPod.io

---

## The Metric-Factuality Mismatch

| Metric | Score | Looks Good? |
|--------|-------|-------------|
| DPO Reward Accuracy | 80% | ✅ |
| SFT Token Accuracy | 78% | ✅ |
| SFT Eval Loss | 0.825 | ✅ |
| Domain Factuality (51 prompts) | 5.9% | ❌ |

Standard alignment metrics don't capture factual degradation. The model learns to *sound helpful* without *being correct* on niche ML terminology.

---

## Factuality Evaluation

Tested Base vs SFT vs DPO on 51 technical ML prompts (strict keyword matching, temperature=0):

| Model Stage | Passed | Accuracy |
|-------------|--------|----------|
| Base (Llama-3.1-8B-Instruct) | 5/51 | 9.8% |
| SFT (OpenHermes + 875 technical) | 4/51 | 7.8% |
| DPO (Merged-SFT, β=0.05) | 3/51 | 5.9% |

**Interpretation:**
- Base model doesn't know niche ML terms (GRPO, PagedAttention, NF4, etc.)
- 875 SFT examples / 1 epoch is insufficient to teach factual recall
- DPO's contribution to factuality loss is secondary (~4pp)
- Primary bottleneck: SFT data quantity, not DPO interference

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE (RunPod.io)                          │
│                                                                                 │
│  ┌───────────┐    ┌────────────┐    ┌────────────┐    ┌────────────────┐        │
│  │ HuggingFace│    │ OpenHermes │    │UltraFeedback│    │ Factual DPO   │        │
│  │ Model Hub  │    │ + Technical│    │  Cleaned    │    │ Pairs (20%)   │        │
│  └─────┬─────┘    └─────┬──────┘    └─────┬──────┘    └───────┬────────┘        │
│        │                 │                 │                   │                 │
│        ▼                 ▼                 │                   │                 │
│  ┌───────────┐    ┌────────────┐           │                   │                 │
│  │Llama-3.1  │    │    SFT     │           │                   │                 │
│  │8B-Instruct│───▶│  (QLoRA)   │           │                   │                 │
│  │  4-bit    │    │ r=16, α=32 │           │                   │                 │
│  └───────────┘    └─────┬──────┘           │                   │                 │
│                         │                  │                   │                 │
│                         ▼                  │                   │                 │
│                   ┌────────────┐           │                   │                 │
│                   │   MERGE    │           │                   │                 │
│                   │ LoRA → Base │           │                   │                 │
│                   │  (bf16)    │           │                   │                 │
│                   └─────┬──────┘           │                   │                 │
│                         │                  │                   │                 │
│                         ▼                  ▼                   ▼                 │
│                   ┌─────────────────────────────────────────────┐                │
│                   │              DPO (Fresh QLoRA)               │                │
│                   │         β=0.05, LR=1e-5, 782 steps          │                │
│                   └──────────────────────┬──────────────────────┘                │
│                                          │                                       │
└──────────────────────────────────────────┼───────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      ▼                      │
                    │  ┌────────────────────────────────────────┐ │
                    │  │         EVALUATION & SERVING            │ │
                    │  ├────────────────────────────────────────┤ │
                    │  │                                        │ │
                    │  │  ┌──────────┐  ┌──────────┐  ┌──────┐ │ │
                    │  │  │Factuality│  │ Response │  │ vLLM │ │ │
                    │  │  │  Eval    │  │ Compare  │  │Serving│ │ │
                    │  │  │(51 prompts) │(Base/SFT/│  │+ API  │ │ │
                    │  │  │ temp=0)  │  │  DPO)    │  │       │ │ │
                    │  │  └──────────┘  └──────────┘  └──────┘ │ │
                    │  │                                        │ │
                    │  └────────────────────┬───────────────────┘ │
                    │                       │                     │
                    └───────────────────────┼─────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │              STREAMLIT DASHBOARD               │
                    │                                               │
                    │  • Training curves (loss, reward accuracy)    │
                    │  • Factuality comparison (Base vs SFT vs DPO) │
                    │  • Metric-factuality mismatch visualization   │
                    │  • Version comparison (v1–v4)                 │
                    │  • Response examples                          │
                    └───────────────────────────────────────────────┘
```

**Key design decisions:**
- **Merged-SFT strategy** — SFT adapter is merged into base weights before DPO, preventing adapter competition
- **QLoRA throughout** — 4-bit NF4 quantization enables 8B model training on consumer GPUs (24–48GB)
- **20% factual DPO pairs** — Upsampled domain-specific preferences to counterbalance generic helpfulness signal
- **Deterministic eval** — Temperature=0 with strict keyword matching for reproducible factuality measurement

---

## Repository Structure

```
distill-align-llm/
├── configs/local_small.yaml       # Training hyperparameters
├── src/distill_align/
│   ├── config/                    # YAML + Pydantic config system
│   ├── data/processor.py          # Dataset loading & tokenization
│   ├── models/loader.py           # Model loading with QLoRA + LoRA
│   ├── training/
│   │   ├── sft.py                 # SFT trainer (TRL SFTTrainer)
│   │   ├── dpo.py                 # DPO trainer (TRL DPOTrainer)
│   │   └── rlhf.py               # GRPO trainer (TRL GRPOTrainer)
│   └── serving/                    # Experimental serving module (not yet deployed)
│       ├── engine.py              # vLLM inference engine
│       └── api.py                 # FastAPI REST gateway
├── scripts/
│   ├── run_sft.py                 # SFT entry point
│   ├── run_dpo.py                 # DPO entry point (supports --merge-sft)
│   ├── eval_factuality_all.py     # Base vs SFT vs DPO factuality eval
│   └── compare_models.py          # Response comparison
├── data/
│   ├── technical_instructions.jsonl  # 875 domain-specific SFT examples
│   ├── factual_dpo_pairs.jsonl       # Factual preference pairs
│   ├── eval_factuality.jsonl         # 51 factuality test prompts
│   └── uncertainty_examples.jsonl    # "I don't know" training examples
├── dashboard/app.py               # Streamlit results dashboard
├── docs/RESULTS.md                # Detailed training logs
├── tests/                         # 44 passing tests
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

# Launch dashboard
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

### Training on RunPod

```bash
# Deploy GPU pod (RTX A6000 48GB recommended), then:
git clone https://github.com/SantoshAdabala/distill-align-llm.git
cd distill-align-llm
pip install transformers accelerate peft datasets bitsandbytes trl
pip install -e .
huggingface-cli login

# SFT (~12 min)
nohup python scripts/run_sft.py --config configs/local_small.yaml > sft_log.txt 2>&1 &

# DPO with merged-SFT (~70 min)
nohup python scripts/run_dpo.py --config configs/local_small.yaml \
    --sft-adapter ./outputs/sft/final_adapter --merge-sft > dpo_log.txt 2>&1 &

# Factuality evaluation (v4 merged-SFT DPO)
python scripts/eval_factuality_all.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --sft-adapter ./outputs/sft/final_adapter \
    --dpo-adapter ./outputs/dpo/dpo_adapter \
    --dpo-base ./outputs/sft_merged \
    --save-responses
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Training** | PyTorch, HuggingFace Transformers, TRL, PEFT, bitsandbytes |
| **Data** | HuggingFace Datasets, UltraFeedback, OpenHermes-2.5 |
| **Serving** | vLLM, FastAPI |
| **Dashboard** | Streamlit, Plotly |
| **Testing** | pytest, ruff |
| **Infrastructure** | RunPod.io (RTX 3090 / A5000 / A6000 / A100 SXM) |

---

## Configuration

```yaml
model:
  model_id: "meta-llama/Llama-3.1-8B-Instruct"
  family: "llama-3.1"
  quantization:
    mode: "int4_nf4"
    use_double_quant: true
  lora:
    rank: 16
    alpha: 32
    target_modules: [q_proj, k_proj, v_proj, o_proj]

dpo:
  beta: 0.05          # worked best empirically in this setup
  learning_rate: 1e-5
  # Uses merged-SFT strategy (--merge-sft flag)
```

---

## Next Experiments

- [ ] SFT with 3/5 epochs (same 875 examples)
- [ ] SFT with 2,500–5,000 technical examples
- [ ] Semantic/LLM-judge factuality eval (not just keyword matching)
- [ ] Token probability analysis (does the model *know* but not *generate*?)

---

## License

MIT

---


