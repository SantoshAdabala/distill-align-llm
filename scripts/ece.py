"""Expected Calibration Error for one model stage.

Teacher-forces stored responses through the model to get a per-answer confidence
(mean token probability), pairs it with a correctness label, and bins the two into a
reliability table. Run once per stage; compare ECE across base/sft/dpo.

Needs, for the stage being measured: that stage's responses (prompt_id,response) and a
correctness source for the same ids. Only the DPO stage currently has both; base/sft need
their eval responses generated and judged first.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_full(model_path):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tok


def answer_confidence(model, tok, prompt, response):
    """Mean probability the model assigns to the tokens of its own answer."""
    pre = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=False, add_generation_prompt=True)
    pre_ids = tok(pre, return_tensors="pt").input_ids[0]
    resp_ids = tok(response, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(resp_ids) == 0:
        return None
    full = torch.cat([pre_ids, resp_ids]).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits = model(full).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    start = len(pre_ids)
    tok_lp = [logprobs[start + i - 1, resp_ids[i]].item() for i in range(len(resp_ids))]
    return float(torch.tensor(tok_lp).mean().exp())


def ece(confidences, correct, n_bins=10):
    import numpy as np
    conf = np.array(confidences)
    acc = np.array(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    table, total = [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            table.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": 0, "conf": None, "acc": None})
            continue
        c, a, w = conf[m].mean(), acc[m].mean(), m.sum() / len(conf)
        total += w * abs(c - a)
        table.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": int(m.sum()),
                      "conf": round(float(c), 3), "acc": round(float(a), 3)})
    return round(total, 4), table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dpo_llama8b")
    ap.add_argument("--model", default="outputs/dpo_8b_merged")
    ap.add_argument("--responses_csv", default="outputs/eval_v2_judge/eval_v2_full_results.csv")
    ap.add_argument("--prompts", default="data/eval_factuality_v2.jsonl")
    ap.add_argument("--correct_csv", default="outputs/rejudge/gpt-4o/scores.csv")
    ap.add_argument("--correct_col", default="new_score")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    prompt_text = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(args.prompts)}
    correct = {r["prompt_id"]: int(r[args.correct_col]) >= 2
               for r in csv.DictReader(open(args.correct_csv))}
    responses = [r for r in csv.DictReader(open(args.responses_csv))
                 if r["prompt_id"] in correct and r["prompt_id"] in prompt_text]

    model, tok = load_full(args.model)
    confs, corr = [], []
    for i, r in enumerate(responses):
        c = answer_confidence(model, tok, prompt_text[r["prompt_id"]], r["response"])
        if c is None:
            continue
        confs.append(c)
        corr.append(correct[r["prompt_id"]])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(responses)}")

    val, table = ece(confs, corr)
    out_dir = Path(args.out_dir or f"outputs/ece/{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"tag": args.tag, "n": len(confs), "ece": val,
              "mean_confidence": round(sum(confs) / len(confs), 3),
              "accuracy": round(sum(corr) / len(corr), 3), "reliability_table": table}
    (out_dir / "ece.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
