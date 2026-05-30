# Changelog

## Unreleased
- Add cross-model evaluation (Mistral-7B, Llama-3.2-3B)
- Add SimPO and BSFT ablations
- Add 500-prompt LLM-judge benchmark
- Add token probability and attention shift analysis
- Add temperature sweep experiments

## v5
- Merged-SFT strategy: SFT adapter merged into base weights before DPO
- QLoRA r=16 α=32, 4-bit NF4, β=0.05, LR=1e-5
- 875×3ep SFT config, loss 1.41
- DPO reward accuracy 82% (peak 88%)
- Live Streamlit dashboard

## v4
- Switched DPO from stacked adapters to merged-SFT strategy
- Peak reward accuracy jumped from 68% → 83%

## v3
- A100 SXM training run
- Technical instruction dataset (3.9K examples)

## v2
- RTX A5000 run, reward accuracy 75%

## v1
- Initial pipeline on RTX 3090

