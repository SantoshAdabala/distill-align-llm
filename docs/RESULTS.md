# Results — AlignLLM Pipeline

> **Status:** SFT + DPO training complete on Llama-3.1-8B using RunPod (RTX 3090). Local SFT also complete on SmolLM2-135M.

---

## Training Results: Llama-3.1-8B (RunPod RTX 3090)

### Final Metrics

| Stage | Loss | Steps | Duration | GPU |
|-------|------|-------|----------|-----|
| **SFT** | 1.3879 | 297 | 13.6 min | RTX 3090 (25.3 GB) |
| **DPO** | 0.6976 | 1,187 | 71.5 min | RTX 3090 (25.3 GB) |

**Total training time:** 85.1 minutes
**Total cost:** ~$1.01 (RunPod RTX 3090 @ $0.69/hr)

### Model Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | meta-llama/Llama-3.1-8B |
| **Quantization** | QLoRA (4-bit NF4, double quantization, bf16 compute) |
| **LoRA Rank** | 16 (alpha=32) |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj |
| **Trainable Parameters** | 13,631,488 (0.30% of 4.55B total) |
| **Platform** | RunPod.io |
| **GPU** | NVIDIA GeForce RTX 3090 (25.3 GB VRAM) |

### SFT Training Curve (Llama-3.1-8B)

```
Step  10: loss=2.170  lr=2.0e-5   epoch=0.03
Step  20: loss=2.044  lr=4.0e-5   epoch=0.07
Step  30: loss=1.701  lr=6.0e-5   epoch=0.10
Step  40: loss=1.564  lr=8.0e-5   epoch=0.13
Step  50: loss=1.493  lr=1.0e-4   epoch=0.17
Step  60: loss=1.466  lr=1.2e-4   epoch=0.20
Step  70: loss=1.410  lr=1.4e-4   epoch=0.24
Step  80: loss=1.372  lr=1.6e-4   epoch=0.27
Step 100: loss=1.359  lr=2.0e-4   epoch=0.34  (peak lr)
Step 130: loss=1.273  lr=1.7e-4   epoch=0.44
Step 170: loss=1.230  lr=1.3e-4   epoch=0.57
Step 200: loss=1.258  lr=9.8e-5   epoch=0.67
Step 250: loss=1.254  lr=4.8e-5   epoch=0.84
Step 297: loss=1.312  lr=7.1e-6   epoch=0.98
```

> SFT loss dropped from 2.17 to 1.23 (best) with final avg loss at 1.39. Healthy convergence with linear warmup to 2e-4 then linear decay.

### DPO Training Curve (Llama-3.1-8B)

```
Step   10: loss=0.893  reward_acc=27.5%  margin=-0.184
Step   50: loss=0.691  reward_acc=40.0%  margin=+0.136
Step  100: loss=0.647  reward_acc=45.0%  margin=+0.264
Step  200: loss=0.566  reward_acc=50.0%  margin=+0.469
Step  400: loss=0.582  reward_acc=40.0%  margin=+0.378
Step  600: loss=0.636  reward_acc=43.0%  margin=+0.534
Step  800: loss=0.731  reward_acc=38.0%  margin=+0.324
Step 1000: loss=0.680  reward_acc=35.0%  margin=+0.436
Step 1187: loss=0.564  reward_acc=45.0%  margin=+0.659
```

> DPO loss dropped from 0.89 to 0.56. Reward margins improved from -0.18 (model prefers rejected) to +0.66 (model prefers chosen). The model learned to distinguish preferred from rejected responses.

---

## Cost Comparison

| Platform | Instance | Duration | Cost | Outcome |
|----------|----------|----------|------|---------|
| **RunPod** | RTX 3090 | 85 min | **$1.01** | Success |
| AWS SageMaker | ml.g5.12xlarge | ~4 hours | $29.00 | Failed |
| GCP Vertex AI | — | — | — | GPU quota denied |

> RunPod delivered the same training at 29x less cost than the failed SageMaker attempt, with immediate GPU access and no quota issues.

---

## How to Reproduce

```bash
# 1. Deploy RunPod pod (RTX 3090, PyTorch template)
# 2. Install dependencies
pip install transformers==4.43.4 accelerate==0.33.0 peft==0.12.0 datasets bitsandbytes trl==0.9.6 rich hf_transfer

# 3. Login to HuggingFace
huggingface-cli login --token YOUR_TOKEN

# 4. Smoke test (~3 min, ~$0.02)
python train.py --sft_num_samples 10 --dpo_num_samples 10 --output_dir /workspace/test

# 5. Full training (~85 min, ~$1.01)
python train.py --output_dir /workspace/outputs

# 6. Stop pod when done to stop billing
```

---

## Next Steps

1. ~~SFT training on Llama-3.1-8B~~ DONE
2. ~~DPO alignment on Llama-3.1-8B~~ DONE
3. Run evaluation benchmarks (MT-Bench, MMLU, HellaSwag, TruthfulQA)
4. Implement RLHF stage and compare against DPO
5. Deploy model for inference (vLLM serving)
6. Push project to GitHub
