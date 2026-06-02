"""Dashboard for the circular-evaluation study.

Shows how a factuality benchmark authored and judged by one weak model (GPT-4o-mini)
inflates an aligned model's score, and what an independent judge, a reference audit,
a confidence analysis, and a reference-free trap probe reveal instead.

Run: streamlit run dashboard/app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BLUE, RED, GRAY, GREEN, DARK = "#2980b9", "#c0392b", "#95a5a6", "#27ae60", "#34495e"

st.set_page_config(page_title="The Benchmark Hallucinates Too", layout="wide")

st.title("The Benchmark Hallucinates Too")
st.markdown(
    "**Circular LLM evaluation inflates factuality and hides confident fabrication.** "
    "An aligned Llama-3.1-8B model scores 84% on a technical-factuality benchmark whose "
    "questions, answer key, and judge are all GPT-4o-mini. None of that survives scrutiny."
)

# ── headline numbers ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reported factuality", "84%", help="GPT-4o-mini self-judge, pass rate")
c2.metric("Real factuality", "~56%", "-28 pts", delta_color="inverse",
          help="GPT-4o, trustworthy-key subset")
c3.metric("Confident Fabrication Rate", "30%", "+23 pts vs self-judge",
          delta_color="inverse", help="asserted yet wrong")
c4.metric("Says 'I don't know'", "1 / 500", help="abstention rate")

st.divider()

# ── 1. the collapse ───────────────────────────────────────────────────────────
st.header("1. The same answers, three judges")
st.markdown(
    "The benchmark's own author (GPT-4o-mini) scores the model far higher than an "
    "independent judge does. Restricting to items whose reference answer is itself "
    "trustworthy barely helps — the score still sits ~28 points below the self-judged figure."
)
judges = ["GPT-4o-mini<br>(self-judge)", "GPT-4o<br>(independent)", "GPT-4o<br>(clean key)"]
passrate = [84.0, 47.8, 55.6]
cfr = [7.4, 30.2, 30.1]
fig = go.Figure()
fig.add_bar(x=judges, y=passrate, name="Factuality (pass rate)", marker_color=BLUE,
            text=[f"{v:.0f}%" for v in passrate], textposition="outside")
fig.add_bar(x=judges, y=cfr, name="Confident Fabrication Rate", marker_color=RED,
            text=[f"{v:.0f}%" for v in cfr], textposition="outside")
fig.update_layout(barmode="group", height=420, yaxis_title="Percent",
                  yaxis=dict(range=[0, 95]), legend=dict(x=0.62, y=0.98),
                  margin=dict(t=20, b=40))
st.plotly_chart(fig, width="stretch")

st.divider()

# ── 2. the answer key is wrong ────────────────────────────────────────────────
st.header("2. The answer key is itself ~20–30% wrong")
left, right = st.columns([2, 1])
with left:
    cats = {"architecture": 6, "training": 16, "alignment": 33,
            "quantization": 29, "empirical reasoning": 24}
    figa = go.Figure(go.Bar(x=list(cats), y=list(cats.values()), marker_color=RED,
                            text=list(cats.values()), textposition="outside"))
    figa.update_layout(height=360, yaxis_title="Gold answers with a clear error",
                       title="108 of 500 reference answers are factually wrong (GPT-4o audit)",
                       margin=dict(t=50, b=40))
    st.plotly_chart(figa, width="stretch")
with right:
    st.metric("Mean reference quality", "1.98 / 3")
    st.metric("Reference answers flagged", "30.2%")
    st.metric("Clear factual errors", "21.6%")
    st.markdown(
        "Crucially, the key's errors **correlate with the model's**: several gold answers "
        "claim Llama-3.1-8B has *16 attention heads* (it has 32) — the same fabrication the "
        "model makes. When the model matches a fabricated key, the judge scores it correct."
    )

st.divider()

# ── 3. confident fabrication and no abstention ────────────────────────────────
st.header("3. Confident fabrication, and the failure to abstain")
left3, right3 = st.columns([1, 1])
with left3:
    st.markdown("**Confidence vs. correctness (500 responses, GPT-4o judge)**")
    matrix = pd.DataFrame(
        {"Correct": [151, 88, 0], "Wrong": [151, 109, 1]},
        index=["Asserted (302)", "Hedged (197)", "Refused (1)"],
    )
    st.table(matrix)
    st.markdown(
        "Half of the model's flat assertions are wrong (red quadrant = 30% of all answers), "
        "and it expresses uncertainty exactly **once in 500**. CFR is stable at ~30% whether "
        "the key is trustworthy or corrupt, so it is not an artifact of the bad references."
    )
with right3:
    st.markdown("**Trap probe: 35 questions with no correct answer**")
    stages = ["Base", "After SFT", "After DPO"]
    asserted = [54.3, 85.7, 60.0]
    hedged = [42.9, 14.3, 40.0]
    refused = [2.9, 0.0, 0.0]
    figt = go.Figure()
    figt.add_bar(x=stages, y=asserted, name="Asserted (fabrication)", marker_color=RED)
    figt.add_bar(x=stages, y=hedged, name="Hedged", marker_color=GRAY)
    figt.add_bar(x=stages, y=refused, name="Refused", marker_color=GREEN)
    figt.update_layout(barmode="stack", height=340, yaxis_title="Percent of 35",
                       legend=dict(orientation="h", y=-0.15), margin=dict(t=10, b=10))
    st.plotly_chart(figt, width="stretch")
    st.caption("Fine-tuning drives refusal to zero; confident fabrication peaks after SFT, not DPO.")

st.divider()

# ── 4. judges disagree ────────────────────────────────────────────────────────
st.header("4. Stronger judges disagree with the self-judge")
left4, right4 = st.columns([2, 1])
with left4:
    scores = [0, 1, 2, 3]
    mini = [7, 10, 35, 48]
    opus = [29, 33, 24, 14]
    figj = go.Figure()
    figj.add_bar(x=scores, y=mini, name="GPT-4o-mini (self-judge)", marker_color=GRAY)
    figj.add_bar(x=scores, y=opus, name="Claude Opus (independent)", marker_color=DARK)
    figj.update_layout(barmode="group", height=360,
                       xaxis=dict(title="Judge score (0 = hallucinated, 3 = fully correct)",
                                  tickmode="array", tickvals=scores),
                       yaxis_title="Responses (of 100)", margin=dict(t=20, b=40))
    st.plotly_chart(figj, width="stretch")
with right4:
    st.metric("GPT-4o vs GPT-4o-mini", "κ = 0.48")
    st.metric("Claude Opus vs GPT-4o-mini", "κ = 0.26")
    st.markdown(
        "Two independent judges from two vendors agree the self-judged score is inflated. "
        "On 18 of 100 sampled responses the self-judge passed answers Opus failed outright — "
        "inspection shows confident fabrications:"
    )
    st.markdown(
        "- *DPO = \"Domain-Specific Pre-Training\"* (it is Direct Preference Optimization)\n"
        "- *SimPO uses a learned reward model* (it is reference-free)\n"
        "- *bf16 was designed by NVIDIA* (Google Brain)"
    )

st.divider()

st.subheader("Takeaways")
st.markdown(
    "- A factuality benchmark should not be authored **and** judged by the same model.\n"
    "- Reference answers should be independently audited before use.\n"
    "- Report **confident-fabrication** and **abstention** rates alongside accuracy: a model can "
    "look accurate while being confidently wrong a third of the time, and never admitting it."
)

st.markdown("---")
st.markdown(
    "Built by [Santosh Adabala](https://github.com/santoshadabala) "
    "· [GitHub repo](https://github.com/santoshadabala/distill-align-llm)"
)
