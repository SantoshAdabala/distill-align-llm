# Results — AlignLLM Pipeline

## Latest: SFT Scaling Study + DPO (v5)

### Key Finding

**Epochs matter more than data volume for factual recall.** 3 epochs on 875 technical examples improves factuality from 9.8% → 15.7%. DPO preserves and modestly improves it to 17.6%.

### Best Configuration Results

| Stage | Metric | Value |
|-------|--------|-------|
| **SFT** | Config | 875 examples × 3 epochs |
| **SFT** | Loss | 1.41 |
| **DPO** | Reward Accuracy | 81.9% |
| **DPO** | Loss | 0.52 |
| **Factuality** | Base → SFT → DPO | 9.8% → 15.7% → 17.6% |
| **AFG** | Alignment-Factuality Gap | 64.3 points |

### SFT Scaling Matrix

| Config | Loss | Factuality | Δ vs Base |
|--------|------|-----------|-----------|
| 875×1ep | 2.16 | 7.8% | -2.0pp |
| 875×3ep | 1.41 | **15.7%** | **+5.9pp** |
| 875×5ep | 1.11 | 15.7% | +5.9pp |
| 2.5K×1ep | 1.31 | 9.8% | 0.0pp |
| 2.5K×3ep | 1.10 | **15.7%** | **+5.9pp** |
| 5K×3ep | 1.02 | 7.8% | -2.0pp |
| 10K×1ep | 1.09 | 9.8% | 0.0pp |
| *Base* | — | 9.8% | — |

**Observations:**
- 3 epochs is the threshold — factuality jumps from ~9% to 15.7%
- More epochs beyond 3 doesn't help further (875×5ep = same as 875×3ep)
- More data with 1 epoch doesn't help (10K×1ep = same as base)
- Too much generic data hurts (5K×3ep drops to 7.8%)

### DPO on Best SFT (875×3ep)

| Metric | Value |
|--------|-------|
| Loss | 0.517 |
| Reward Accuracy (training avg) | 81.9% |
| Peak Reward Accuracy | 87.5% |
| Steps | 782 |
| Duration | 72.6 min |
| Factuality (DPO) | 17.6% (9/51) |

**DPO did NOT degrade factuality.** It improved from 15.7% → 17.6%.

---

## Cost

| Run | GPU | Cost |
|-----|-----|------|
| v1 (SFT+DPO) | RTX 3090 | $1.00 |
| v2 (SFT+DPO) | RTX A5000 | $2.51 |
| v3 (SFT+DPO) | A100 SXM | $6.51 |
| Scaling (8 configs) | A100 SXM | $15.78 |
| DPO + Eval | A100 SXM | $0.68 |
| **Total** | | **~$27** |

---

## Version History

| Metric | v1 | v2 | v3 | v4 | v5 (latest) |
|--------|----|----|-----|-----|-------------|
| **GPU** | RTX 3090 | RTX A5000 | A100 SXM | RTX A6000 | A100 SXM |
| **SFT Config** | Alpaca 1K×1ep | Alpaca 1K×1ep | Tech 3.9K×1ep | Tech 3.9K×1ep | Tech 875×3ep |
| **DPO Config** | Stacked, β=0.1 | Stacked, β=0.1 | Stacked, β=0.1 | Merged, β=0.05 | Merged, β=0.05 |
| **Peak Reward Acc** | 50% | 75% | 68% | 83% | **88%** |
| **Factuality** | — | — | 9.8% | 5.9% | **17.6%** |

---

## How to Reproduce

```bash
# Deploy RunPod pod (A100 SXM or RTX A6000 recommended)
git clone https://github.com/SantoshAdabala/distill-align-llm.git
cd distill-align-llm
pip install transformers accelerate peft datasets bitsandbytes trl
pip install -e .
hf auth login

# SFT scaling (full matrix ~3 hours)
nohup python scripts/run_sft_scaling.py --config configs/local_small.yaml --run-all > scaling_log.txt 2>&1 &

# Or single best config (~10 min)
python scripts/run_sft_scaling.py --config configs/local_small.yaml --num-examples 875 --epochs 3

# DPO on best SFT (~70 min)
nohup python scripts/run_dpo.py --config configs/local_small.yaml \
    --sft-adapter outputs/scaling/sft_875ex_3ep/final_adapter --merge-sft > dpo_log.txt 2>&1 &

# Factuality evaluation
python scripts/eval_factuality_all.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --sft-adapter outputs/scaling/sft_875ex_3ep/final_adapter \
    --dpo-adapter outputs/dpo/dpo_adapter \
    --dpo-base outputs/sft_merged
```
