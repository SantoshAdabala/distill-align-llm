# Results — distill-align-llm

This is the detailed results companion to the [README](../README.md). The project began as "does DPO hurt factual knowledge?" and became a study of how a self-authored, self-judged factuality benchmark inflates its own score and hides confident fabrication.

The pipeline: Llama-3.1-8B-Instruct, SFT (875 technical examples × 3 epochs, merged) → DPO (β=0.05), QLoRA throughout, trained on RunPod (~$27). The benchmark, TechFact-500 (`data/eval_factuality_v2.jsonl`), had its questions, gold reference answers, and judge all produced by GPT-4o-mini.

---

## Headline: the score does not survive a stronger judge

Same 500 stored responses, re-scored (`scripts/rejudge.py`). Source: `outputs/rejudge/gpt-4o/summary.json`.

| Metric | GPT-4o-mini (self-judge) | GPT-4o (independent) | GPT-4o (clean-key subset) |
|---|:---:|:---:|:---:|
| Factuality pass rate (score ≥ 2) | **84.0%** | 47.8% | ~56% |
| Factuality (normalized mean) | 75.7% | 51.5% | — |
| Confident Fabrication Rate | 7.4% | 30.2% | ~30% |

A 28-point drop in pass rate, and the Confident Fabrication Rate (asserted **and** wrong) quadruples. Three mechanisms explain it.

---

## 1. The answer key is wrong 20–30% of the time

Audit of all 500 gold reference answers with a stronger model (`scripts/audit_references.py`):

- Mean gold-answer quality: **1.98 / 3**
- **108 / 500** gold answers contain a clear factual error

The errors **correlate with the model's own errors**. Several gold answers claim Llama-3.1-8B has 16 attention heads (it has 32) — the same fabrication the model makes. When the model repeats a mistake baked into the key, the judge marks it correct. A self-authored, self-judged benchmark manufactures agreement with the model it grades.

*Caveat:* the LLM auditor occasionally over-flags genuinely-public facts as unverifiable, so 20–30% is approximate; the direction is corroborated by the head-count error appearing in both the key and the model.

---

## 2. The model confidently fabricates ~30% of the time and almost never abstains

Each response classified as asserted / hedged / refused (`scripts/honesty_eval.py`). GPT-4o honesty matrix on the 500 (`outputs/rejudge/gpt-4o/summary.json`):

| | Correct | Wrong |
|---|:---:|:---:|
| **Asserted** | 151 | 151 |
| **Hedged** | 88 | 109 |
| **Refused** | 0 | 1 |

- The model expresses uncertainty on exactly **1 of 500** questions.
- Half of its flat assertions are wrong → **Confident Fabrication Rate ≈ 30%**.
- The CFR holds whether the reference key is clean or corrupt, so it is not an artifact of the bad key.
- The weak self-judge reports CFR at 7.4% — undercounting by ~4×, because confident answers matching a fabricated key score as correct.

---

## 3. A reference-free probe confirms it, and points at SFT

TechFact-Trap (`data/techfact_trap.jsonl`): 35 questions with **no correct answer** (made-up acronyms, fabricated methods/papers, false premises). The only honest responses are to refuse or correct the premise. Run via `scripts/trap_eval.py`.

| Stage | Asserted (fabrication) | Hedged | Refused |
|---|:---:|:---:|:---:|
| Base (instruct) | 54% | 43% | 3% |
| After SFT | **86%** | 14% | **0%** |
| After DPO | 60% | 40% | **0%** |

Fine-tuning eliminates the ability to abstain (refusal → 0). Domain **SFT — not DPO** — is what spikes confident fabrication; DPO walks it back slightly. The honest read: confident fabrication is mostly established during supervised fine-tuning.

*Caveat:* n=35. Refusal → 0 is solid; the between-stage fabrication differences are suggestive, not conclusive.

---

## 4. Judge agreement: independent judges score the answers harder

| Judge pair | Quadratic κ |
|---|:---:|
| GPT-4o vs GPT-4o-mini | 0.48 |
| Claude Opus vs GPT-4o-mini | 0.26 |

On a 100-response sample, **18** answers the weak judge passed were failed outright by Opus — confident fabrications it accepted ("DPO = Domain-Specific Pre-Training", "SimPO uses a reward model", "bf16 was designed by NVIDIA"). Cross-vendor agreement is the weakest, and Opus scores hardest of all. Source: `outputs/human_annotation/opus_vs_gpt4omini.json`.

---

## 5. Human anchor (the tiebreaker)

A 30-response subset hand-rated blind (0–3 scale, no AI scores shown, stratified across the judge's score range; `scripts/human_annotation.py`, seed 42). Both judges compared against the human labels on the same items. Source: `outputs/human_annotation/HUMAN_ANCHOR_30.txt` + `_anchor_order.json`.

| 30-item anchor | Human | GPT-4o-mini | GPT-4o |
|---|:---:|:---:|:---:|
| Mean score (0–3) | **1.33** | 2.43 | 1.77 |
| Factual (score ≥ 2) | **40.0%** | 86.7% | 63.3% |
| Fully correct (score = 3) | 10.0% | 56.7% | 16.7% |
| Correlation with human | — | 0.41 | 0.62 |
| MAE vs human | — | 1.10 | 0.63 |
| Over-rates the human on | — | 21/30 | 15/30 |
| Exact agreement | — | 9/30 | 12/30 |

The self-judge inflates by **~47 points** against a human; the independent GPT-4o still inflates by **~23**. Using a stronger judge helps (higher correlation, lower error, more exact agreements) but does **not** close the gap — every automated judge tested errs generous, in proportion to how confident the answer sounds. "Use a stronger LLM judge" is not a clean fix.

*Caveat:* n=30 gives the human figure a wide interval (~±18pp), and the sample is stratified on the weak judge's score distribution. Read it as "~40%, decisively below both judges," not a precise point estimate.

---

## Limitations

- Correctness labels (except the human anchor) come from LLM judges; the n=30 human anchor confirms the inflation direction but is small (~±18pp).
- Reference-audit corruption rate (20–30%) is approximate; LLM auditor over-flags some public facts.
- Trap set is small (n=35).
- Single model pipeline, single training run, no confidence intervals.

---

## Cost

| Item | Detail | Cost |
|---|---|---|
| Training (SFT + DPO, QLoRA) | RunPod A100 | ~$27 |
| API judging / auditing / re-judge | OpenAI (GPT-4o) + Anthropic (Opus) | API usage |

---

## Reproduce

Most analyses re-use the stored model responses and need only an API key, not a GPU:

```bash
make install

make rejudge             # re-judge stored responses with gpt-4o (needs OPENAI_API_KEY)
make audit               # audit the gold reference answers
python scripts/make_figures.py

# trap evaluation needs the merged models (GPU)
make trap ARGS="--tag llama8b --sft_model outputs/sft_merged --dpo_model outputs/dpo_8b_merged"
```
