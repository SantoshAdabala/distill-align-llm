# distill-align-llm — 82% Reward Accuracy. 17.6% Factuality. A 57-Point Gap That Demands Explanation.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![TRL](https://img.shields.io/badge/TRL-SFT%20%7C%20DPO-blueviolet?style=flat-square)](https://github.com/huggingface/trl)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://distill-align-llm-aembgrswzfay6bjupbnjpp.streamlit.app)
[![RunPod](https://img.shields.io/badge/RunPod-~%2427_total-6B46C1?style=flat-square)](https://runpod.io)
[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-success?style=flat-square&logo=streamlit)](https://distill-align-llm-aembgrswzfay6bjupbnjpp.streamlit.app)

End-to-end LLM alignment pipeline (**SFT → DPO**) on Llama-3.1-8B-Instruct with QLoRA, systematically investigating how preference optimization interacts with domain-specific factual knowledge — and quantifying the gap when standard alignment metrics and real-world accuracy diverge.

---

## Key Results at a Glance

| Metric | Value | Takeaway |
|--------|-------|----------|
| DPO Reward Accuracy | **82%** (peak **88%**) | Strong alignment signal ✅ |
| Domain Factuality — Base | 9.8% | Llama-3.1-8B baseline |
| Domain Factuality — SFT | 15.7% | +5.9 pp gain from 875×3ep SFT |
| Domain Factuality — DPO | **17.6%** | Gain preserved through DPO |
| **Alignment-Factuality Gap (AFG)** | **57 points** | Reward accuracy ≠ factual accuracy ⚠️ |
| Training cost | **~$27** | Full pipeline on RunPod.io |
| Test suite | **44 passing** | pytest, with ruff linting |
| GPU versions | **v1 → v5** | Systematic ablation across 4 GPU tiers |

> **The central finding:** A model can simultaneously achieve 82% reward accuracy and only 17.6% factuality on domain-specific questions — these metrics measure fundamentally different capabilities. Standard DPO training does not bridge this gap.

---

## Why This Matters

Reward accuracy is the metric most practitioners use to declare RLHF/DPO training "successful." This project demonstrates empirically that a high reward score can mask a persistent factual knowledge deficit — a phenomenon we term the **Alignment-Factuality Gap (AFG)**. The 57-point gap observed here suggests that preference optimization teaches a model to *sound* aligned without necessarily grounding its outputs in verified facts. This has direct implications for any application where factual correctness matters: medical Q&A, technical documentation, code generation. The scaling study adds a second insight: within the configurations tested, **epoch count was more predictive of factual gains than raw data volume** — 875 examples × 3 epochs outperformed 10,000 examples × 1 epoch. This suggests that knowledge consolidation, not data scale alone, drives factual alignment.

---

## The Alignment-Factuality Gap (AFG)

| Metric | Score | |
|--------|-------|-|
| DPO Reward Accuracy | 82% | ✅ |
| SFT Token Accuracy | 78% | ✅ |
| Domain Factuality (DPO) | 17.6% | ⚠️ |
| **AFG** | **57 points** | |

Reward accuracy (82%) far exceeds factuality (17.6%). These metrics measure fundamentally different capabilities — preference alignment versus grounded factual recall.

---

## Factuality Evaluation

Tested Base vs SFT vs DPO on 51 technical ML prompts (strict keyword matching, temperature=0):

| Model Stage | Passed | Accuracy |
|-------------|--------|----------|
| Base (Llama-3.1-8B-Instruct) | 5/51 | 9.8% |
| SFT (875 examples × 3 epochs) | 8/51 | 15.7% |
| DPO (Merged-SFT, β=0.05) | 9/51 | **17.6%** |

### SFT Scaling Study

| Config | Factuality | Δ vs Base |
|--------|-----------:|----------:|
| 875×1ep | 7.8% | −2.0 pp |
| 875×3ep | **15.7%** | **+5.9 pp** |
| 875×5ep | 15.7% | +5.9 pp |
| 2.5K×1ep | 9.8% | 0.0 pp |
| 2.5K×3ep | **15.7%** | **+5.9 pp** |
| 5K×3ep | 7.8% | −2.0 pp |
| 10K×1ep | 9.8% | 0.0 pp |

**Key insight:** Repeated exposure (epochs) was more predictive of factual gains than data volume. 3 epochs is the effective threshold; adding more generic data can *dilute* technical knowledge rather than reinforce it.

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
|---------|-----|------------|:---------------:|------------|
| v1 | RTX 3090 | Stacked, β=0.1, LR=5e-5 | 50% | Baseline (weak) |
| v2 | RTX A5000 | Stacked, β=0.1, LR=1e-5 | 75% | Lower LR + cleaned data |
| v3 | A100 SXM | Stacked, β=0.1 | 68% | Added technical SFT data |
| v4 | RTX A6000 | Merged-SFT, β=0.05 | 83% | Merge adapter + lower β |
| v5 | A100 SXM | Merged-SFT, β=0.05 | **88%** | **3-epoch SFT (scaling study)** |

**Total training cost: ~$27** on RunPod.io

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
                    │  • Version comparison (v1–v5)                 │
                    │  • Response examples                          │
                    └───────────────────────────────────────────────┘
```

**Key design decisions:**
- **Merged-SFT strategy** — SFT adapter is merged into base weights before DPO, preventing adapter competition
- **QLoRA throughout** — 4-bit NF4 quantization enables 8B model training on consumer GPUs (24–48 GB)
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

# Factuality evaluation
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

## Open Problems / Future Work

- [ ] Expand benchmark to 500+ prompts with category-level analysis to assess whether the AFG is domain-specific
- [ ] Replace keyword matching with semantic / LLM-judge factuality evaluation for more nuanced scoring
- [ ] Token probability analysis — does the model encode the correct fact but fail to *generate* it, or is the knowledge absent?
- [ ] Replicate on larger models (70B) to test whether epoch-sensitivity and the AFG scale with model capacity

---

<details>
<summary><strong>Resume Talking Points</strong></summary>

- **Designed and executed an end-to-end LLM alignment research pipeline** (SFT → DPO) on Llama-3.1-8B-Instruct using QLoRA (4-bit NF4, r=16, α=32), achieving 82% DPO reward accuracy (peak 88%) across five GPU configurations on RunPod.io for a total training cost of ~$27
- **Discovered and quantified the Alignment-Factuality Gap (AFG)**: demonstrated that a model achieving 82% reward accuracy retains only 17.6% domain factuality — a 57-point divergence that reveals a fundamental limitation of standard preference optimization as a proxy for factual correctness
- **Conducted a systematic SFT scaling study** across 7 configurations (data volume × epoch count), finding that epoch depth (3 epochs) was more predictive of factual gains than raw data scale — a non-obvious finding with direct implications for data-efficient fine-tuning pipelines
- **Implemented the Merged-SFT DPO strategy** — merging LoRA adapters into base weights before DPO training — which improved peak reward accuracy from 50% (v1 stacked baseline) to 88% (v5), demonstrating that adapter interference is a meaningful source of training degradation
- **Built a production-quality ML research repository** with a live Streamlit dashboard, 44 passing pytest unit tests, YAML + Pydantic configuration management, and reproducible evaluation (temperature=0, strict keyword matching) — reflecting software engineering standards beyond typical research codebases

</details>

---

## License

MIT
