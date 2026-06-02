"""Run the trap set through one or more model stages and measure refusal vs fabrication.

For each available stage (base / sft / dpo) the model answers every trap prompt, and each
answer is classified REFUSED / HEDGED / ASSERTED. Since trap questions have no correct
answer, ASSERTED = confident fabrication and REFUSED = honest. Stages whose adapter path
is missing are skipped.
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _honesty():
    spec = importlib.util.spec_from_file_location("he", "scripts/honesty_eval.py")
    he = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(he)
    return he


def _bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_stage(stage, base_model, sft_adapter, dpo_base, dpo_adapter):
    from peft import PeftModel
    if stage == "base":
        tok = _tokenizer(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=_bnb(), device_map="auto", torch_dtype=torch.bfloat16)
        return model, tok
    if stage == "sft":
        tok = _tokenizer(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=_bnb(), device_map="auto", torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, sft_adapter)
        return model, tok
    if stage == "dpo":
        tok = _tokenizer(dpo_base)
        base = AutoModelForCausalLM.from_pretrained(
            dpo_base, quantization_config=_bnb(), device_map="auto", torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, dpo_adapter)
        return model, tok
    raise ValueError(stage)


def generate(model, tok, prompt, max_new_tokens=220):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=1.0, pad_token_id=tok.eos_token_id)
    new = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


def stage_available(stage, sft_adapter, dpo_base, dpo_adapter):
    if stage == "base":
        return True
    if stage == "sft":
        return sft_adapter and os.path.isdir(sft_adapter)
    if stage == "dpo":
        return dpo_base and os.path.isdir(dpo_base) and dpo_adapter and os.path.isdir(dpo_adapter)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="llama8b", help="output subfolder name")
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--sft_adapter", default="outputs/sft/final_adapter")
    ap.add_argument("--dpo_base", default="outputs/sft_merged")
    ap.add_argument("--dpo_adapter", default="outputs/dpo/dpo_adapter")
    ap.add_argument("--stages", default="base,sft,dpo")
    ap.add_argument("--trap", default="data/techfact_trap.jsonl")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    he = _honesty()
    traps = [json.loads(l) for l in open(args.trap)]
    out_dir = Path(args.out_dir or f"outputs/trap/{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"tag": args.tag, "n_prompts": len(traps), "stages": {}}
    for stage in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if not stage_available(stage, args.sft_adapter, args.dpo_base, args.dpo_adapter):
            print(f"[skip] {stage}: adapter path missing")
            continue
        print(f"[load] {stage} ...")
        model, tok = load_stage(stage, args.base_model, args.sft_adapter, args.dpo_base, args.dpo_adapter)
        rows = []
        for i, t in enumerate(traps):
            resp = generate(model, tok, t["prompt"])
            label = he.classify_confidence(resp)["label"]
            rows.append({"id": t["id"], "trap_type": t["trap_type"], "prompt": t["prompt"],
                         "response": resp, "label": label})
            if (i + 1) % 10 == 0:
                print(f"  {stage}: {i+1}/{len(traps)}")
        with (out_dir / f"responses_{stage}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "trap_type", "prompt", "response", "label"])
            w.writeheader()
            w.writerows(rows)
        dist = he.confidence_distribution([r["response"] for r in rows])
        n = len(rows)
        summary["stages"][stage] = {
            "distribution": dist,
            "refusal_rate": round(dist["REFUSED"]["frac"], 4),
            "fabrication_rate_asserted": round(dist["ASSERTED"]["frac"], 4),
            "honest_rate_refused_or_hedged": round((dist["REFUSED"]["n"] + dist["HEDGED"]["n"]) / n, 4),
        }
        print(f"  {stage}: refusal={summary['stages'][stage]['refusal_rate']:.2f}  "
              f"fabrication(asserted)={summary['stages'][stage]['fabrication_rate_asserted']:.2f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
