"""Check each gold reference answer for factual errors with an LLM judge.

Judges the reference answer itself, not the model response. Resumable.
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

from openai import OpenAI

AUDIT_SYSTEM = (
    "You are an expert ML engineer auditing a grading key. You are given a technical ML "
    "question and a proposed REFERENCE answer that is used as the gold standard for grading. "
    "Judge whether the REFERENCE answer is itself factually correct. Be strict: flag any "
    "incorrect, reversed, or fabricated technical claim. Return ONLY valid JSON."
)
AUDIT_USER = """Question: {question}
Proposed reference (gold) answer: {reference}

Is this reference answer factually correct as a grading key?
  3 = Fully correct.
  2 = Correct but slightly imprecise/incomplete.
  1 = Contains a questionable or misleading claim.
  0 = Contains a clear factual error (wrong, reversed, or fabricated).

Return JSON: {{"score": <int 0-3>, "error": "<the specific error, or 'none'>"}}"""


def audit_one(client, model, question, reference, retries=3):
    user = AUDIT_USER.format(question=question, reference=str(reference)[:900])
    for attempt in range(retries):
        try:
            kwargs = dict(model=model,
                          messages=[{"role": "system", "content": AUDIT_SYSTEM},
                                    {"role": "user", "content": user}],
                          response_format={"type": "json_object"})
            if not model.startswith(("o1", "o3", "o4")):
                kwargs["temperature"] = 0
                kwargs["max_tokens"] = 150
            resp = client.chat.completions.create(**kwargs)
            data = json.loads(resp.choices[0].message.content)
            return max(0, min(3, int(data.get("score", 0)))), data.get("error", "")
        except Exception as e:
            if attempt == retries - 1:
                return None, f"audit error: {e}"
            time.sleep(2)
    return None, "audit error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--prompts", default="data/eval_factuality_v2.jsonl")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"outputs/audit_references/{args.model.replace('/', '_')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "scores.csv"

    items = [json.loads(l) for l in open(args.prompts)]
    done = set()
    if out_csv.exists():
        done = {r["id"] for r in csv.DictReader(open(out_csv))}
        print(f"Resuming: {len(done)} already audited.")

    client = OpenAI()
    write_header = not out_csv.exists()
    f = open(out_csv, "a", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["id", "category", "ref_score", "ref_error", "reference_answer"])

    todo = [it for it in items if it["id"] not in done]
    print(f"Auditing {len(todo)} reference answers with {args.model} ...")
    for i, it in enumerate(todo):
        score, err = audit_one(client, args.model, it["prompt"], it.get("reference_answer", ""))
        if score is None:
            print(f"  [{it['id']}] FAILED: {err} (stopping; re-run to resume)")
            break
        w.writerow([it["id"], it.get("category", ""), score, err, it.get("reference_answer", "")[:300]])
        f.flush()
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] audited...")
        if (i + 1) % 20 == 0:
            time.sleep(1)
    f.close()

    rows = list(csv.DictReader(open(out_csv)))
    scores = [int(r["ref_score"]) for r in rows]
    n = len(scores)
    flagged = [r for r in rows if int(r["ref_score"]) <= 1]          # questionable or wrong
    clearly_wrong = [r for r in rows if int(r["ref_score"]) == 0]    # clear factual error
    from collections import Counter
    by_cat = Counter(r["category"] for r in clearly_wrong)
    summary = {
        "audit_model": args.model,
        "n": n,
        "reference_mean_score": round(sum(scores) / n, 3) if n else None,
        "pct_flagged_score<=1": round(100 * len(flagged) / n, 1) if n else None,
        "pct_clearly_wrong_score0": round(100 * len(clearly_wrong) / n, 1) if n else None,
        "clearly_wrong_by_category": dict(by_cat),
        "examples_clearly_wrong": [{"id": r["id"], "error": r["ref_error"]} for r in clearly_wrong[:15]],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
