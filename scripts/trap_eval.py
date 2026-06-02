"""Run the trap set through one or more model stages and measure refusal vs fabrication.

Each stage is a full (merged) model. Every trap prompt is answered and each answer is
classified REFUSED / HEDGED / ASSERTED. Trap questions have no correct answer, so
ASSERTED = confident fabrication and REFUSED = honest. Stages without a model path are skipped.
"""

import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _honesty():
    spec = importlib.util.spec_from_file_location("he", "scripts/honesty_eval.py")
    he = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(he)
    return he


def load_full(model_path):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tok


def generate(model, tok, prompt, max_new_tokens=220):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="llama8b")
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--sft_model", default="")
    ap.add_argument("--dpo_model", default="")
    ap.add_argument("--trap", default="data/techfact_trap.jsonl")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    he = _honesty()
    traps = [json.loads(l) for l in open(args.trap)]
    out_dir = Path(args.out_dir or f"outputs/trap/{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"tag": args.tag, "n_prompts": len(traps), "stages": {}}
    for name, path in [("base", args.base_model), ("sft", args.sft_model), ("dpo", args.dpo_model)]:
        if not path:
            continue
        # skip a local path that does not exist; allow HF ids (e.g. meta-llama/...) through
        if path.startswith("outputs") and not os.path.isdir(path):
            print(f"[skip] {name}: {path} not found")
            continue
        print(f"[load] {name}: {path}")
        model, tok = load_full(path)
        rows = []
        for i, t in enumerate(traps):
            resp = generate(model, tok, t["prompt"])
            rows.append({"id": t["id"], "trap_type": t["trap_type"], "prompt": t["prompt"],
                         "response": resp, "label": he.classify_confidence(resp)["label"]})
            if (i + 1) % 10 == 0:
                print(f"  {name}: {i+1}/{len(traps)}")
        with (out_dir / f"responses_{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "trap_type", "prompt", "response", "label"])
            w.writeheader()
            w.writerows(rows)
        dist = he.confidence_distribution([r["response"] for r in rows])
        n = len(rows)
        summary["stages"][name] = {
            "distribution": dist,
            "refusal_rate": round(dist["REFUSED"]["frac"], 4),
            "fabrication_rate_asserted": round(dist["ASSERTED"]["frac"], 4),
            "honest_rate_refused_or_hedged": round((dist["REFUSED"]["n"] + dist["HEDGED"]["n"]) / n, 4),
        }
        print(f"  {name}: refusal={summary['stages'][name]['refusal_rate']:.2f} "
              f"fabrication={summary['stages'][name]['fabrication_rate_asserted']:.2f}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
