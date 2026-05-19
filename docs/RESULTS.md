# Results — AlignLLM Pipeline

## v3: Technical SFT + DPO with Factual Pairs (Latest)

### Training Configuration

| Component | Details |
|-----------|---------|
| **Base Model** | meta-llama/Llama-3.1-8B-Instruct |
| **SFT Data** | OpenHermes-2.5 (3K) + Technical Instructions (875) + Uncertainty Examples (15) = 3,890 total |
| **DPO Data** | UltraFeedback Cleaned (5K) + Factual DPO Pairs (20) |
| **GPU** | RTX A6000 (48 GB), RunPod.io |
| **Quantization** | QLoRA (4-bit NF4, double quantization, bf16 compute) |

### SFT Results (v3)

| Metric | Value |
|--------|-------|
| **Train Loss** | 1.050 |
| **Eval Loss** | 0.825 |
| **Token Accuracy** | 78.3% |
| **Steps** | 219 |
| **Duration** | 12.5 min |

Loss curve: 1.93 → 0.83 over 219 steps. Strong convergence with technical data.

### DPO Results (v3)

| Metric | Training Logs | Post-Training Eval |
|--------|--------------|-------------------|
| **Loss** | 0.772 | — |
| **Reward Accuracy** | 60-68% (2nd half) | 0% (invalid — OOM during eval) |
| **Reward Margin** | +0.30 to +0.59 | — |
| **Steps** | 628 | — |
| **Duration** | 57 min | — |

Note: The 0% post-training eval is invalid due to OOM corruption. Training logs show healthy 60-68% accuracy.

### DPO Training Curve (v3)

| Step | Loss | Reward Accuracy | Margin |
|------|------|----------------|--------|
| 20 | 1.173 | 42.5% | -0.455 |
| 60 | 0.997 | 46.3% | -0.152 |
| 100 | 0.819 | 51.3% | +0.095 |
| 160 | 0.845 | 48.1% | +0.033 |
| 220 | 0.711 | 61.9% | +0.297 |
| 240 | 0.673 | 65.0% | +0.399 |
| 260 | 0.722 | 65.0% | +0.358 |
| 300 | 0.749 | 61.3% | +0.319 |
| 340 | 0.671 | 59.4% | +0.483 |
| 400 | 0.705 | 66.3% | +0.435 |
| 440 | 0.687 | 60.6% | +0.455 |
| 480 | 0.685 | 65.0% | +0.429 |
| 500 | 0.620 | 66.3% | +0.593 |
| 520 | 0.620 | 64.4% | +0.562 |
| 540 | 0.635 | 68.1% | +0.551 |
| 580 | 0.676 | 63.8% | +0.521 |
| 620 | 0.646 | 65.0% | +0.535 |

### Factuality Evaluation (v3)

**Result: 5/51 passed (9.8%)**

The DPO model fails on domain-specific technical terms. Examples of hallucinations:
- "DPO = Days Past Order" (should be Direct Preference Optimization)
- "QLoRA = Quantum LoiR Accelerator" (should be Quantized Low-Rank Adaptation)
- "RLHF = Rule-Based Human Evaluation and Feedback" (should be Reinforcement Learning from Human Feedback)
- "vLLM = Very Large Language Model" (should be a serving engine)

Passed correctly: LoRA, mixed precision, weight decay, attention_mask, RMSNorm.

### Response Comparison (v3)

| Prompt | Base | SFT | DPO |
|--------|------|-----|-----|
| Gradient checkpointing | Generic, verbose | Concise, mentions memory savings | Generic, verbose |
| Reduce GPU costs | Lists AWS instances | Practical tips (batch size, mixed precision) | Lists AWS instances |
| Safety refusal | Refuses correctly | Refuses correctly | Refuses correctly (most concise) |
| LoRA vs full FT | Correct explanation | Correct, concise | Correct explanation |

**Key observation:** SFT model gives more concise, technically-focused answers. DPO model reverts to base model's verbose, generic style on technical topics.

---

## Analysis: Why DPO Doesn't Preserve Technical Knowledge

### Hypothesis

DPO shifted the output distribution toward generic helpfulness patterns (rewarded by UltraFeedback), suppressing the technical response selection that SFT learned. The knowledge likely still exists in the model's representations — DPO just made it less likely to surface during generation.

**This is NOT knowledge overwriting.** It's distribution shift + preference optimization selecting for a different output style.

### Evidence

1. SFT model correctly answers LoRA questions (prompt 4 in comparison)
2. DPO training data: 5000 generic preference pairs vs 20 factual pairs — generic dominates
3. UltraFeedback rewards helpfulness/tone/style, not technical accuracy
4. The model learned: "be helpful > be technical" instead of "be correct > be helpful"

### Research Question

> Under what conditions does DPO preserve vs suppress domain-specific knowledge acquired during SFT?

### Planned Experiments

1. **Evaluate Base vs SFT vs DPO** on same factuality prompts (isolate where knowledge is lost)
2. **Increase factual DPO pairs to 20%** of training data (not 0.4%)
3. **Merge SFT adapter before DPO** (instead of stacked adapters)
4. **Lower DPO beta to 0.05** (less aggressive preference shift)
5. **Deterministic generation** (temperature=0) to rule out sampling artifacts

---

## v2: Improved DPO (Previous Best)

### Results

| Stage | Loss | Reward Accuracy | Duration |
|-------|------|----------------|----------|
| **SFT** | 1.763 (eval: 1.127) | 72% token accuracy | 3.1 min |
| **DPO** | 0.758 | **75%** (final step) | 125 min |

**Total cost:** ~$0.58 (RunPod RTX A5000 @ $0.27/hr)

### What Fixed DPO Alignment (v1 → v2)

| Change | v1 | v2 |
|--------|----|----|
| Learning rate | 5e-5 | 1e-5 |
| Dataset | UltraFeedback raw | UltraFeedback cleaned |
| Base model | Llama-3.1-8B (base) | Llama-3.1-8B-Instruct |
| Alignment quality | Weak (50%) | Strong (75%) |

---

## Version Comparison

| Metric | v1 | v2 | v3 |
|--------|----|----|-----|
| **SFT Eval Loss** | 1.127 | 1.127 | **0.825** |
| **SFT Token Accuracy** | 72% | 72% | **78.3%** |
| **DPO Reward Accuracy** | 50% | 75% | 60-68% |
| **SFT Data** | Alpaca 1K | Alpaca 1K | OpenHermes + Technical 3.9K |
| **Factuality** | Not tested | Not tested | 9.8% (DPO only) |
| **Technical Knowledge** | No | No | Yes (SFT), No (DPO) |

---

## How to Reproduce

```bash
# Deploy RunPod pod (RTX A6000 48GB recommended)
git clone https://github.com/SantoshAdabala/distill-align-llm.git
cd distill-align-llm
pip install transformers accelerate peft datasets bitsandbytes trl
pip install -e .
hf auth login

# SFT (~12 min)
python scripts/run_sft.py --config configs/local_small.yaml

# DPO (~57 min)
python scripts/run_dpo.py --config configs/local_small.yaml --sft-adapter ./outputs/sft/final_adapter

# Factuality evaluation
python scripts/eval_factuality.py --model-path ./outputs/dpo/dpo_adapter

# Response comparison
python scripts/compare_models.py --base-model meta-llama/Llama-3.1-8B-Instruct --sft-adapter ./outputs/sft/final_adapter --dpo-adapter ./outputs/dpo/dpo_adapter --max-prompts 4
```

---

## Next Steps

1. ~~SFT with technical data~~ ✅
2. ~~DPO with factual pairs~~ ✅
3. ~~Factuality evaluation~~ ✅ (baseline established: 9.8%)
4. Evaluate SFT adapter separately on factuality (isolate knowledge retention)
5. Increase factual DPO pairs to 20% of training data
6. Merge SFT before DPO (Experiment 3)
7. Lower DPO beta to 0.05
8. Add RAG for grounded technical responses
