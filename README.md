# distill-align-llm

I set out to measure whether DPO hurts a model's factual knowledge. I built the usual setup: an SFT -> DPO pipeline on Llama-3.1-8B, a 500-question technical benchmark, and an LLM judge to score the answers. The model scored 84% factuality, which looked great.

Then I checked the measurement instead of trusting it, and the project turned into something else: a study of how a factuality benchmark that is written and graded by the same weak model inflates the score and hides confident fabrication.

**[Live dashboard ->](https://distill-align-llm-aembgrswzfay6bjupbnjpp.streamlit.app)**

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-TRL%20%7C%20PEFT-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://distill-align-llm-aembgrswzfay6bjupbnjpp.streamlit.app)
[![Model](https://img.shields.io/badge/Model-Llama--3.1--8B--Instruct-0EA5E9?style=flat-square&logo=meta&logoColor=white)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=flat-square)](./LICENSE)

---

## The finding

The benchmark (TechFact-500) had its questions, its gold reference answers, and its judge all produced by GPT-4o-mini. Under that setup the aligned model scores 84%. It does not survive contact with anything stronger.

| | GPT-4o-mini (self-judge) | GPT-4o (independent) | GPT-4o (clean-key subset) |
|---|:---:|:---:|:---:|
| Factuality (pass rate) | **84%** | 48% | **56%** |
| Confident Fabrication Rate | 7% | 30% | 30% |

Three things are going on underneath that 28-point drop:

**The answer key is wrong about 20-30% of the time.** I audited all 500 gold reference answers with a stronger model: mean quality 1.98/3, and 108 of 500 contain a clear factual error. The errors are not random - they *correlate with the model's own errors*. Several gold answers claim Llama-3.1-8B has 16 attention heads (it has 32), which is exactly the fabrication the model makes. When the model repeats a mistake that is baked into the key, the judge marks it correct. A self-authored, self-judged benchmark manufactures agreement with the model it is grading.

**The model confidently fabricates ~30% of the time and almost never says "I don't know."** Classifying each response as asserted / hedged / refused, half of the model's flat assertions are wrong, and it expresses uncertainty on exactly 1 of 500 questions. The Confident Fabrication Rate (asserted *and* wrong) is ~30% and stays there whether the reference key is clean or corrupt, so it is not an artifact of the bad key. The weak self-judge reports it as 7% - undercounting by 4x, because confident answers that match a fabricated key get scored as correct.

**A reference-free probe confirms it, and points at SFT.** I built TechFact-Trap: 35 questions with no correct answer (made-up acronyms, fabricated methods and papers, false premises). The only honest responses are to refuse or correct the premise.

| Stage | Asserted (fabrication) | Hedged | Refused |
|---|:---:|:---:|:---:|
| Base (instruct) | 54% | 43% | 3% |
| After SFT | **86%** | 14% | **0%** |
| After DPO | 60% | 40% | **0%** |

Fine-tuning eliminates the model's ability to abstain (refusal -> 0), and domain SFT - not DPO - is what spikes confident fabrication. DPO actually walks it back a little. This is the honest read: the confident fabrication is mostly established during supervised fine-tuning, not by preference optimization.

**Two independent judges agree the score is inflated.** GPT-4o agrees only moderately with GPT-4o-mini (quadratic kappa 0.48), and Claude Opus - a different vendor - agrees less (kappa 0.26) and scores harder still. On a 100-response sample, 18 answers the weak judge passed were failed outright by Opus; they are confident fabrications it accepted ("DPO = Domain-Specific Pre-Training", "SimPO uses a reward model", "bf16 was designed by NVIDIA").

**A human anchor confirms the inflation - and that a stronger judge only half-fixes it.** I hand-rated a 30-response subset blind (0-3 scale, no AI scores shown, stratified across the judge's score range), then compared both judges against my labels on the same items.

| 30-item human anchor | Human | GPT-4o-mini | GPT-4o |
|---|:---:|:---:|:---:|
| Factual (score >= 2) | **40%** | 87% | 63% |
| Fully correct (score = 3) | 10% | 57% | 17% |
| Correlation with human | - | 0.41 | 0.62 |
| Over-rates the human on | - | 21/30 | 15/30 |

The self-judge inflates by ~47 points against a human; the independent GPT-4o still inflates by ~23. Using a stronger judge moves the score in the right direction - higher correlation, lower error, agrees exactly on 12/30 vs 9/30 - but does not close the gap. Every automated judge tested errs generous, roughly in proportion to how confident the answer sounds. With n=30 the human figure carries a wide interval (~+/-18pp), so read it as "~40%, decisively below both judges," not a point estimate.

---

## What I built

- **Model pipeline** - Llama-3.1-8B-Instruct, SFT (875 technical examples x 3 epochs, merged) -> DPO (beta=0.05), QLoRA throughout. Trained on RunPod for about $27.
- **TechFact-500** (`data/eval_factuality_v2.jsonl`) - the GPT-4o-mini-authored benchmark whose circularity is the subject of the study.
- **TechFact-Trap** (`data/techfact_trap.jsonl`) - 35 no-correct-answer probes for measuring abstention.
- **Honesty scorer** (`scripts/honesty_eval.py`) - asserted/hedged/refused classification and Confident Fabrication Rate.
- **Independent re-judging** (`scripts/rejudge.py`) - re-score stored responses with any judge model.
- **Reference audit** (`scripts/audit_references.py`) - check the gold answer key for factual errors.
- **Trap evaluation** (`scripts/trap_eval.py`) - run base/SFT/DPO on the trap set and measure refusal vs fabrication.
- **Figures** (`scripts/make_figures.py`) and the Streamlit **dashboard** (`dashboard/app.py`).

---

## Reproduce

```bash
git clone https://github.com/SantoshAdabala/distill-align-llm.git
cd distill-align-llm
make install

# dashboard (no GPU, no API key)
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

The evaluation analyses re-use the stored model responses, so most of them need only an API key, not a GPU:

```bash
# re-judge the stored responses with a stronger judge (needs OPENAI_API_KEY)
make rejudge                    # defaults to gpt-4o; override with MODEL=o4-mini

# audit the gold reference answers for factual errors
make audit

# regenerate the paper figures
python scripts/make_figures.py
```

The trap evaluation needs the models (GPU). It runs base/SFT/DPO from merged checkpoints:

```bash
make trap ARGS="--tag llama8b --sft_model outputs/sft_merged --dpo_model outputs/dpo_8b_merged"
```

---

## Repo structure

```
distill-align-llm/
├── data/
│   ├── eval_factuality_v2.jsonl   # TechFact-500 (GPT-4o-mini authored)
│   └── techfact_trap.jsonl        # TechFact-Trap (no correct answer)
├── scripts/
│   ├── honesty_eval.py            # asserted/hedged/refused + CFR
│   ├── rejudge.py                 # re-score with a stronger judge (make rejudge)
│   ├── audit_references.py        # audit the gold answer key (make audit)
│   ├── trap_eval.py               # refusal vs fabrication, base/SFT/DPO (make trap)
│   ├── make_figures.py            # paper figures
│   └── run_sft.py / run_dpo.py    # the training pipeline
├── dashboard/app.py               # Streamlit dashboard
├── src/distill_align/             # config, data, model loading, training wrappers
└── tests/
```

---

## Limitations

- Correctness labels above come from LLM judges (GPT-4o, Opus), now anchored against a 30-response human-rated subset (see "The finding"). Human factuality was ~40% - below even the independent GPT-4o judge - confirming the inflation direction. The anchor is small (n=30, ~+/-18pp) and stratified on the weak judge's score distribution, so "real factuality" should be read as "~40-56%, decisively below 84%," not a precise figure.
- The reference audit uses an LLM auditor that occasionally over-flags genuinely-public facts as unverifiable, so the 20-30% corruption rate is approximate (the direction is corroborated by the cross-checkable head-count error appearing in both the key and the model).
- The trap set is small (n=35); refusal -> 0 is solid, but the fabrication-rate differences between stages are suggestive, not conclusive.
- Single model pipeline, single training run, no confidence intervals.

---

## Tech stack

PyTorch, HuggingFace Transformers / TRL / PEFT, bitsandbytes, OpenAI and Anthropic APIs (judges), matplotlib, Streamlit, Plotly, pytest, ruff.

## License

MIT
