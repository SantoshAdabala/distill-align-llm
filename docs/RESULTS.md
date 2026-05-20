# Results — AlignLLM Pipeline

## v4: Merged-SFT DPO (Latest)

### Training Configuration

| Component | Details |
|-----------|---------|
| **Base Model** | meta-llama/Llama-3.1-8B-Instruct |
| **SFT Data** | OpenHermes-2.5 (3K) + Technical Instructions (875) + Uncertainty (15) = 3,890 |
| **DPO Data** | UltraFeedback Cleaned (5K) + Factual DPO Pairs (20% upsampled) |
| **DPO Strategy** | Merged-SFT (adapter merged into base before DPO) |
| **DPO Beta** | 0.05 |
| **GPU** | RTX A6000 (48 GB), RunPod.io |
| **Quantization** | QLoRA (4-bit NF4, double quantization, bf16 compute) |

### SFT Results (v4)

| Metric | Value |
|--------|-------|
| **Train Loss** | 1.050 |
| **Eval Loss** | 0.825 |
| **Token Accuracy** | 78.3% |
| **Steps** | 219 |
| **Duration** | 12.5 min |

### DPO Results (v4)

| Metric | Value |
|--------|-------|
| **Loss** | 0.54 |
| **Peak Reward Accuracy** | 83% |
| **Avg Reward Accuracy (2nd half)** | 75–81% |
| **Reward Margin (peak)** | 0.74 |
| **Steps** | 782 |
| **Duration** | 67.6 min |

### Factuality Evaluation (v4)

| Model Stage | Passed | Accuracy | 95% CI |
|-------------|--------|----------|--------|
| Base (Llama-3.1-8B-Instruct) | 5/51 | 9.8% | [4.3%, 21.0%] |
| SFT (OpenHermes + Technical) | 4/51 | 7.8% | [3.1%, 18.5%] |
| DPO (Merged-SFT, β=0.05) | 3/51 | 5.9% | [2.0%, 15.9%] |

**Key finding:** All model stages perform poorly on strict factual recall of niche ML terminology. The base model itself doesn't reliably know these terms. SFT did not measurably improve factuality, and DPO's contribution to factuality loss is modest (~4pp) and within the noise range of a 51-prompt benchmark.

**Interpretation:** Low-resource SFT (875 examples, 1 epoch) does not reliably encode domain knowledge. The factuality bottleneck is SFT data quantity, not DPO interference.

---

## Version Comparison

| Metric | v1 | v2 | v3 | v4 |
|--------|----|----|-----|-----|
| **GPU** | RTX 3090 | RTX A5000 | A100 SXM | RTX A6000 |
| **SFT Data** | Alpaca 1K | Alpaca 1K | OpenHermes+Tech 3.9K | OpenHermes+Tech 3.9K |
| **DPO Config** | Stacked, β=0.1 | Stacked, β=0.1 | Stacked, β=0.1 | Merged, β=0.05 |
| **Peak Reward Acc** | 50% | 75% | 68% | **83%** |
| **DPO Loss** | 0.70 | 0.76 | 0.77 | **0.54** |
| **Factuality** | — | — | 9.8% (DPO only) | Base 9.8% / SFT 7.8% / DPO 5.9% |

---

## What Fixed DPO Alignment (v1 → v4)

| Change | v1 | v4 | Impact |
|--------|----|----|--------|
| Learning rate | 5e-5 | 1e-5 | Prevented overshooting |
| Dataset | UltraFeedback raw | UltraFeedback cleaned + 20% factual | Cleaner signal |
| Base model | Llama-3.1-8B (base) | Llama-3.1-8B-Instruct | Better starting point |
| Adapter strategy | Stacked | Merged-SFT | No adapter competition |
| Beta | 0.1 | 0.05 | More stable training |

---

## How to Reproduce

```bash
# Deploy RunPod pod (RTX A6000 48GB recommended)
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

# Factuality evaluation (use --dpo-base for merged-SFT DPO)
python scripts/eval_factuality_all.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --sft-adapter ./outputs/sft/final_adapter \
    --dpo-adapter ./outputs/dpo/dpo_adapter \
    --dpo-base ./outputs/sft_merged
```

---

## Next Experiments

- [ ] SFT scaling: 875/2.5K/5K/10K examples × 1/3/5 epochs
- [ ] Semantic/LLM-judge factuality eval (not just keyword matching)
- [ ] Token probability analysis (does the model know but not generate?)
- [ ] Expand benchmark to 500 prompts
