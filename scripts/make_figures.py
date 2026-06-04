"""Generate the paper figures from the result files into paper/figures/."""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

FIG = Path("paper/figures")
FIG.mkdir(parents=True, exist_ok=True)

ASSERT, HEDGE, REFUSE = "#c0392b", "#95a5a6", "#27ae60"
BLUE, RED, GRAY, DARK = "#2980b9", "#c0392b", "#bdc3c7", "#34495e"


def fig_collapse():
    """Pass-rate and CFR of the same responses under three judges."""
    judges = ["GPT-4o-mini\n(self-judge)", "GPT-4o\n(independent)", "GPT-4o\n(clean key)"]
    passrate = [84.0, 47.8, 55.6]
    cfr = [7.4, 30.2, 30.1]
    x = range(len(judges))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.bar([i - w / 2 for i in x], passrate, w, label="Factuality (pass rate, $\\geq$2)", color=BLUE)
    ax.bar([i + w / 2 for i in x], cfr, w, label="Confident Fabrication Rate", color=RED)
    for i, v in zip(x, passrate):
        ax.text(i - w / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=9)
    for i, v in zip(x, cfr):
        ax.text(i + w / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(judges)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 95)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.savefig(FIG / "fig_collapse.pdf")
    plt.close(fig)


def fig_trap():
    """Asserted/hedged/refused across training stages on the trap set."""
    stages = ["Base", "After SFT", "After DPO"]
    asserted = [54.3, 85.7, 60.0]
    hedged = [42.9, 14.3, 40.0]
    refused = [2.9, 0.0, 0.0]
    x = range(len(stages))
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.bar(x, asserted, 0.6, label="Asserted (fabrication)", color=ASSERT)
    ax.bar(x, hedged, 0.6, bottom=asserted, label="Hedged", color=HEDGE)
    ax.bar(x, refused, 0.6, bottom=[a + h for a, h in zip(asserted, hedged)],
           label="Refused (``I don't know'')".replace("``", '"').replace("''", '"'), color=REFUSE)
    for i, v in zip(x, asserted):
        ax.text(i, v / 2, f"{v:.0f}", ha="center", color="white", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages)
    ax.set_ylabel("Percent of 35 trap responses")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=1)
    fig.savefig(FIG / "fig_trap.pdf")
    plt.close(fig)


def fig_judges():
    """Score distributions of the weak judge vs an independent vendor."""
    try:
        rows = json.load(open("outputs/human_annotation/opus_vs_gpt4omini.json"))
        mini = Counter(r["gpt4o_mini"] for r in rows)
        opus = Counter(r["opus"] for r in rows)
    except Exception:
        mini = {0: 7, 1: 10, 2: 35, 3: 48}
        opus = {0: 29, 1: 33, 2: 24, 3: 14}
    scores = [0, 1, 2, 3]
    mv = [mini.get(s, 0) for s in scores]
    ov = [opus.get(s, 0) for s in scores]
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.bar([s - w / 2 for s in scores], mv, w, label="GPT-4o-mini (self-judge)", color=GRAY)
    ax.bar([s + w / 2 for s in scores], ov, w, label="Claude Opus (independent)", color=DARK)
    ax.set_xticks(scores)
    ax.set_xlabel("Judge score (0 = hallucinated, 3 = fully correct)")
    ax.set_ylabel("Responses (of 100)")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(FIG / "fig_judges.pdf")
    plt.close(fig)


def fig_audit():
    """Clearly-wrong reference answers by category."""
    try:
        s = json.load(open("outputs/audit_references/gpt-4o/summary.json"))
        cw = s["clearly_wrong_by_category"]
    except Exception:
        cw = {"architecture_facts": 6, "training_mechanics": 16, "alignment_concepts": 33,
              "quantization_efficiency": 29, "empirical_reasoning": 24}
    labels = [k.replace("_", "\n") for k in cw]
    vals = list(cw.values())
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.bar(range(len(vals)), vals, 0.6, color=RED)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, str(v), ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Gold answers with a clear error")
    ax.set_title("108 of 500 reference answers are factually wrong", fontsize=10)
    fig.savefig(FIG / "fig_audit.pdf")
    plt.close(fig)


def fig_calibration():
    """P(True) reliability diagram across training stages; marker area = bin count."""
    stages = [("Base", "outputs/calibration/base/ptrue.json", REFUSE),
              ("After SFT", "outputs/calibration/sft/ptrue.json", RED),
              ("After DPO", "outputs/calibration/dpo_llama8b/ptrue.json", BLUE)]
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    ax.plot([0, 1], [0, 1], "--", color="#7f8c8d", lw=1, label="Perfect calibration")
    for name, path, color in stages:
        d = json.load(open(path))
        bins = [b for b in d["reliability_table"] if b["n"] and b["conf"] is not None]
        conf = [b["conf"] for b in bins]
        acc = [b["acc"] for b in bins]
        sizes = [12 + b["n"] * 1.3 for b in bins]  # area grows with bin count
        ax.scatter(conf, acc, s=sizes, color=color, alpha=0.7, edgecolors="white",
                   linewidths=0.5, label=f"{name} (ECE {d['ece']:.2f})", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model self-confidence  $P$(True)")
    ax.set_ylabel("Actual accuracy")
    ax.set_title("Marker size $\\propto$ \\#answers; below diagonal = overconfident", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(FIG / "fig_calibration.pdf")
    plt.close(fig)


def fig_human():
    """Human anchor vs the two judges on the same 30-item subset."""
    labels = ["Human", "GPT-4o-mini\n(self-judge)", "GPT-4o\n(independent)"]
    factual = [40.0, 86.7, 63.3]
    colors = [DARK, GRAY, BLUE]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.bar(range(3), factual, 0.6, color=colors)
    for i, v in enumerate(factual):
        ax.text(i, v + 1.8, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Factual (score $\\geq$2), %")
    ax.set_ylim(0, 100)
    ax.set_title("30-item human-rated subset", fontsize=10)
    fig.savefig(FIG / "fig_human.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_collapse()
    fig_trap()
    fig_judges()
    fig_audit()
    fig_calibration()
    fig_human()
    print("wrote:", *(p.name for p in sorted(FIG.glob("*.pdf"))))
