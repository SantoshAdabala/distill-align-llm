# AlignLLM — When Alignment Metrics Look Good but Factuality Does Not

End-to-end LLM alignment pipeline (**SFT → DPO**) with QLoRA, investigating how preference optimization interacts with domain-specific knowledge.

**Key finding:** With 3 epochs of domain-specific SFT, factuality improves from 9% to 26% on TechFact-100, and DPO preserves this gain (25%). DPO achieves 82% reward accuracy — an Alignment-Factuality Gap (AFG$_{\text{exact}}$) of 57 points.

🔗 **[Live Dashboard](https://distill-align-llm-aembgrswzfay6bjupbnjpp.streamlit.app)**

---

## Results (v5 — Latest)

| Stage | Metric | Value |
|-------|--------|-------|
| **SFT** | Config | 875 examples × 3 epochs |
| **SFT** | Loss | 1.41 |
| **DPO** | Reward Accuracy | 82% (peak 88%) |
| **DPO** | Loss | 0.52 |
| **Factuality** | Base → SFT → DPO | 9.8% → 15.7% → 17.6% |
| **AFG** | Alignment-Factuality Gap | 57 points (exact) |

### Version Progression

| Version | GPU | DPO Config | Peak Reward Acc | Key Change |
|---------|-----|-----------|-----------------|------------|
| v1 | RTX 3090 | Stacked, β=0.1, LR=5e-5 | 50% | Baseline (weak) |
| v2 | RTX A5000 | Stacked, β=0.1, LR=1e-5 | 75% | Lower LR + cleaned data |
| v3 | A100 SXM | Stacked, β=0.1 | 68% | Added technical SFT data |
| v4 | RTX A6000 | Merged-SFT, β=0.05 | 83% | Merge adapter + lower β |
| v5 | A100 SXM | Merged-SFT, β=0.05 | **88%** | **3-epoch SFT (scaling study)** |

**Total cost: ~$27** on RunPod.io

---

## The Alignment-Factuality Gap (AFG)

| Metric | Score | |
|--------|-------|---|
| DPO Reward Accuracy | 82% | ✅ |
| SFT Token Accuracy | 78% | ✅ |
| Domain Factuality (DPO) | 17.6% | ⚠️ |
| **AFG** | **57 points** | |

Reward accuracy (82%) far exceeds factuality (17.6%). These metrics measure fundamentally different capabilities.

---

## Factuality Evaluation

Tested Base vs SFT vs DPO on 51 technical ML prompts (strict keyword matching, temperature=0):

| Model Stage | Passed | Accuracy |
|-------------|--------|----------|
| Base (Llama-3.1-8B-Instruct) | 5/51 | 9.8% |
| SFT (875 examples × 3 epochs) | 8/51 | 15.7% |
| DPO (Merged-SFT, β=0.05) | 9/51 | 17.6% |

### SFT Scaling Study

| Config | Factuality | Δ vs Base |
|--------|-----------|-----------|
| 875×1ep | 7.8% | -2.0pp |
| 875×3ep | **15.7%** | **+5.9pp** |
| 875×5ep | 15.7% | +5.9pp |
| 2.5K×1ep | 9.8% | 0.0pp |
| 2.5K×3ep | **15.7%** | **+5.9pp** |
| 5K×3ep | 7.8% | -2.0pp |
| 10K×1ep | 9.8% | 0.0pp |

**Key insight:** Within our tested configurations, repeated exposure (epochs) was more predictive of factual gains than data volume. 3 epochs is the threshold. More generic data can dilute technical knowledge.

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
                    │  │  ┌──────────┐  ┌──────────┐         │ │
                    │  │  │Factuality│  │ Response │         │ │
                    │  │  │  Eval    │  │ Compare  │         │ │
                    │  │  │(51 prompts) │(Base/SFT/│         │ │
                    │  │  │ temp=0)  │  │  DPO)    │         │ │
                    │  │  └──────────┘  └──────────┘         │ │
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
| **Evaluation** | sentence-transformers, TechFact-100 |
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

## Next Steps

- [ ] Expand benchmark to 500 prompts with category-level analysis
- [ ] Semantic/LLM-judge factuality eval (not just keyword matching)
- [ ] Token probability analysis (does the model *know* but not *generate*?)
- [ ] Test on larger models (70B) to see if epoch-sensitivity persists

---

## License

MIT

---


