"""
AlignLLM Training Dashboard — Streamlit App.

Displays training results from SFT and DPO runs across v1–v4,
including factuality evaluation (Base vs SFT vs DPO).

Run: streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="AlignLLM Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AlignLLM — Training Dashboard")
st.markdown("**End-to-end LLM alignment pipeline**: SFT → DPO on Llama-3.1-8B-Instruct with QLoRA")
st.markdown("*When Alignment Metrics Look Good but Factuality Does Not*")

# ═══════════════════════════════════════════════
# V1 RESULTS (from first RunPod run)
# ═══════════════════════════════════════════════

V1_SFT_METRICS = [
    {"step": 10, "loss": 2.170, "lr": 2e-5, "epoch": 0.03},
    {"step": 20, "loss": 2.044, "lr": 4e-5, "epoch": 0.07},
    {"step": 30, "loss": 1.701, "lr": 6e-5, "epoch": 0.10},
    {"step": 40, "loss": 1.564, "lr": 8e-5, "epoch": 0.13},
    {"step": 50, "loss": 1.493, "lr": 1.0e-4, "epoch": 0.17},
    {"step": 60, "loss": 1.466, "lr": 1.2e-4, "epoch": 0.20},
    {"step": 70, "loss": 1.410, "lr": 1.4e-4, "epoch": 0.24},
    {"step": 80, "loss": 1.372, "lr": 1.6e-4, "epoch": 0.27},
    {"step": 90, "loss": 1.412, "lr": 1.8e-4, "epoch": 0.30},
    {"step": 100, "loss": 1.359, "lr": 2.0e-4, "epoch": 0.34},
    {"step": 110, "loss": 1.343, "lr": 1.9e-4, "epoch": 0.37},
    {"step": 120, "loss": 1.334, "lr": 1.8e-4, "epoch": 0.40},
    {"step": 130, "loss": 1.273, "lr": 1.7e-4, "epoch": 0.44},
    {"step": 140, "loss": 1.263, "lr": 1.6e-4, "epoch": 0.47},
    {"step": 150, "loss": 1.257, "lr": 1.5e-4, "epoch": 0.51},
    {"step": 160, "loss": 1.289, "lr": 1.4e-4, "epoch": 0.54},
    {"step": 170, "loss": 1.230, "lr": 1.3e-4, "epoch": 0.57},
    {"step": 180, "loss": 1.310, "lr": 1.2e-4, "epoch": 0.61},
    {"step": 190, "loss": 1.257, "lr": 1.1e-4, "epoch": 0.64},
    {"step": 200, "loss": 1.259, "lr": 9.8e-5, "epoch": 0.67},
    {"step": 210, "loss": 1.238, "lr": 8.8e-5, "epoch": 0.71},
    {"step": 220, "loss": 1.308, "lr": 7.8e-5, "epoch": 0.74},
    {"step": 230, "loss": 1.303, "lr": 6.8e-5, "epoch": 0.77},
    {"step": 240, "loss": 1.331, "lr": 5.8e-5, "epoch": 0.81},
    {"step": 250, "loss": 1.253, "lr": 4.8e-5, "epoch": 0.84},
    {"step": 260, "loss": 1.261, "lr": 3.8e-5, "epoch": 0.88},
    {"step": 270, "loss": 1.230, "lr": 2.7e-5, "epoch": 0.91},
    {"step": 280, "loss": 1.314, "lr": 1.7e-5, "epoch": 0.94},
    {"step": 297, "loss": 1.312, "lr": 7.1e-6, "epoch": 0.98},
]

V1_DPO_METRICS = [
    {"step": 10, "loss": 0.893, "reward_acc": 0.275, "margin": -0.184},
    {"step": 50, "loss": 0.691, "reward_acc": 0.400, "margin": 0.136},
    {"step": 100, "loss": 0.647, "reward_acc": 0.450, "margin": 0.264},
    {"step": 150, "loss": 0.681, "reward_acc": 0.275, "margin": 0.081},
    {"step": 200, "loss": 0.566, "reward_acc": 0.500, "margin": 0.449},
    {"step": 300, "loss": 0.596, "reward_acc": 0.450, "margin": 0.633},
    {"step": 400, "loss": 0.582, "reward_acc": 0.400, "margin": 0.378},
    {"step": 500, "loss": 0.650, "reward_acc": 0.425, "margin": 0.528},
    {"step": 600, "loss": 0.636, "reward_acc": 0.430, "margin": 0.534},
    {"step": 700, "loss": 0.694, "reward_acc": 0.350, "margin": 0.304},
    {"step": 800, "loss": 0.731, "reward_acc": 0.380, "margin": 0.324},
    {"step": 900, "loss": 0.656, "reward_acc": 0.425, "margin": 0.549},
    {"step": 1000, "loss": 0.680, "reward_acc": 0.350, "margin": 0.436},
    {"step": 1100, "loss": 0.620, "reward_acc": 0.475, "margin": 0.564},
    {"step": 1187, "loss": 0.564, "reward_acc": 0.450, "margin": 0.659},
]

V1_SUMMARY = {
    "model": "meta-llama/Llama-3.1-8B",
    "quantization": "QLoRA (4-bit NF4)",
    "lora_rank": 16,
    "trainable_params": "13.6M (0.30%)",
    "gpu": "NVIDIA RTX 3090 (24 GB)",
    "platform": "RunPod.io",
    "total_cost": "$1.01",
    "sft_time_min": 13.6,
    "sft_loss": 1.3879,
    "sft_steps": 297,
    "dpo_time_min": 71.5,
    "dpo_loss": 0.6976,
    "dpo_steps": 1187,
    "dpo_lr": "5e-5",
    "dpo_dataset": "UltraFeedback (5K samples)",
}


# ═══════════════════════════════════════════════
# V2 RESULTS (load from file if available)
# ═══════════════════════════════════════════════

V2_RESULTS_PATH = Path("dashboard/v2_results.json")


def load_v2_results():
    """Load v2 results from JSON file if available."""
    if V2_RESULTS_PATH.exists():
        with open(V2_RESULTS_PATH) as f:
            return json.load(f)
    return None


v2_data = load_v2_results()

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════

st.sidebar.header("Run Configuration")
st.sidebar.markdown("### v1 (Initial Run)")
st.sidebar.markdown(f"- **Model**: {V1_SUMMARY['model']}")
st.sidebar.markdown(f"- **GPU**: {V1_SUMMARY['gpu']}")
st.sidebar.markdown(f"- **Cost**: {V1_SUMMARY['total_cost']}")
st.sidebar.markdown(f"- **DPO LR**: {V1_SUMMARY['dpo_lr']}")
st.sidebar.markdown(f"- **Dataset**: {V1_SUMMARY['dpo_dataset']}")

if v2_data:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### v4 (Latest — Merged-SFT DPO)")
    st.sidebar.markdown(f"- **Model**: {v2_data.get('model', 'Llama-3.1-8B-Instruct')}")
    st.sidebar.markdown(f"- **GPU**: {v2_data.get('gpu', 'RTX A6000')}")
    st.sidebar.markdown(f"- **DPO LR**: {v2_data.get('dpo_lr', '1e-5')}")
    st.sidebar.markdown(f"- **DPO Beta**: {v2_data.get('dpo_beta', 0.05)}")
    st.sidebar.markdown(f"- **Dataset**: {v2_data.get('dpo_dataset', 'UltraFeedback Cleaned')}")
    st.sidebar.markdown(f"- **Reward Acc**: {v2_data.get('eval_reward_accuracy', 'N/A'):.0%}")
    st.sidebar.markdown(f"- **Cost**: {v2_data.get('total_cost', 'N/A')}")

# ═══════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════

# --- Overview metrics ---
st.header("📊 Training Overview")

if v2_data and "dpo_loss" in v2_data:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SFT Loss (v4)", f"{v2_data.get('sft_loss', 0):.3f}",
                delta=f"{v2_data.get('sft_loss', 0) - V1_SUMMARY['sft_loss']:.3f} vs v1")
    col2.metric("DPO Loss (v4)", f"{v2_data['dpo_loss']:.4f}",
                delta=f"{v2_data['dpo_loss'] - V1_SUMMARY['dpo_loss']:.3f} vs v1")
    col3.metric("Reward Accuracy (v4)", f"{v2_data.get('eval_reward_accuracy', 0):.0%}")
    col4.metric("Total Cost (v4)", v2_data.get("total_cost", "N/A"))

    col1b, col2b, col3b, col4b = st.columns(4)
    col1b.metric("DPO Beta", f"{v2_data.get('dpo_beta', 0.05)}")
    col2b.metric("DPO Steps", f"{v2_data.get('dpo_steps', 0)}")
    col3b.metric("Reward Margin", f"{v2_data.get('eval_reward_margin', 0):.4f}")
    col4b.metric("Experiment", v2_data.get("experiment", "v4"))
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SFT Loss", f"{V1_SUMMARY['sft_loss']:.4f}", delta="-36% from start")
    col2.metric("DPO Loss (v1)", f"{V1_SUMMARY['dpo_loss']:.4f}", delta="near baseline ⚠️")
    col3.metric("Total Time", f"{V1_SUMMARY['sft_time_min'] + V1_SUMMARY['dpo_time_min']:.0f} min")
    col4.metric("Total Cost", V1_SUMMARY["total_cost"])

# --- SFT Training Curve ---
st.header("📈 SFT Training Curve")

sft_df = pd.DataFrame(V1_SFT_METRICS)

fig_sft = go.Figure()
fig_sft.add_trace(go.Scatter(
    x=sft_df["step"], y=sft_df["loss"],
    mode="lines+markers", name="Training Loss",
    line=dict(color="#2196F3", width=2),
    marker=dict(size=4),
))
fig_sft.add_trace(go.Scatter(
    x=sft_df["step"], y=sft_df["lr"] * 10000,
    mode="lines", name="Learning Rate (×10⁴)",
    line=dict(color="#FF9800", width=1, dash="dash"),
    yaxis="y2",
))
fig_sft.update_layout(
    xaxis_title="Step",
    yaxis_title="Loss",
    yaxis2=dict(title="LR (×10⁴)", overlaying="y", side="right"),
    height=400,
    legend=dict(x=0.7, y=0.95),
    margin=dict(l=50, r=50, t=30, b=50),
)
st.plotly_chart(fig_sft, use_container_width=True)

st.success(f"**SFT converged well**: loss dropped from 2.17 → 1.23 (best), final avg 1.39. "
           f"Completed in {V1_SUMMARY['sft_steps']} steps / {V1_SUMMARY['sft_time_min']} min.")

# --- DPO Training Curve ---
st.header("📈 DPO Training Curve")

tab1, tab2 = st.tabs(["v1 (Initial — LR=5e-5, stacked adapter)", "v4 (Merged-SFT, β=0.05, 20% factual pairs)"])

with tab1:
    dpo_df = pd.DataFrame(V1_DPO_METRICS)

    col_left, col_right = st.columns(2)

    with col_left:
        fig_dpo_loss = px.line(dpo_df, x="step", y="loss", markers=True,
                               title="DPO Loss", color_discrete_sequence=["#E91E63"])
        fig_dpo_loss.add_hline(y=0.693, line_dash="dash", line_color="gray",
                               annotation_text="Random baseline (ln2)")
        fig_dpo_loss.update_layout(height=350, margin=dict(t=40, b=40))
        st.plotly_chart(fig_dpo_loss, use_container_width=True)

    with col_right:
        fig_dpo_acc = go.Figure()
        fig_dpo_acc.add_trace(go.Scatter(
            x=dpo_df["step"], y=dpo_df["reward_acc"],
            mode="lines+markers", name="Reward Accuracy",
            line=dict(color="#4CAF50", width=2),
        ))
        fig_dpo_acc.add_hline(y=0.50, line_dash="dash", line_color="red",
                              annotation_text="50% (random)")
        fig_dpo_acc.update_layout(
            title="Reward Accuracy",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            height=350, margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_dpo_acc, use_container_width=True)

    # Margin chart
    fig_margin = px.bar(dpo_df, x="step", y="margin", title="Reward Margin (chosen - rejected)",
                        color="margin", color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0)
    fig_margin.update_layout(height=300, margin=dict(t=40, b=40))
    st.plotly_chart(fig_margin, use_container_width=True)

    st.warning("**v1 Assessment**: DPO loss (~0.69) near random baseline. "
               "Reward accuracy fluctuated 27–50%, indicating weak alignment. "
               "Reward margins improved from -0.18 to +0.66, showing some learning.")

with tab2:
    if v2_data and "dpo_metrics" in v2_data:
        v2_dpo_df = pd.DataFrame(v2_data["dpo_metrics"])

        col_left2, col_right2 = st.columns(2)

        with col_left2:
            fig_v2_loss = px.line(v2_dpo_df, x="step", y="loss", markers=True,
                                  title="DPO Loss (v4 — Merged-SFT)", color_discrete_sequence=["#9C27B0"])
            fig_v2_loss.add_hline(y=0.693, line_dash="dash", line_color="gray",
                                  annotation_text="Random baseline (ln2)")
            fig_v2_loss.update_layout(height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig_v2_loss, use_container_width=True)

        with col_right2:
            fig_v2_acc = go.Figure()
            fig_v2_acc.add_trace(go.Scatter(
                x=v2_dpo_df["step"], y=v2_dpo_df["reward_acc"],
                mode="lines+markers", name="Reward Accuracy (v4)",
                line=dict(color="#4CAF50", width=2),
            ))
            fig_v2_acc.add_hline(y=0.50, line_dash="dash", line_color="red",
                                 annotation_text="50% (random)")
            fig_v2_acc.add_hline(y=0.80, line_dash="dot", line_color="green",
                                 annotation_text="80% target")
            fig_v2_acc.update_layout(
                title="Reward Accuracy (v4)",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1]),
                height=350, margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_v2_acc, use_container_width=True)

        # Margin chart
        fig_v4_margin = px.bar(v2_dpo_df, x="step", y="margin",
                               title="Reward Margin (v4 — chosen vs rejected)",
                               color="margin", color_continuous_scale="RdYlGn",
                               color_continuous_midpoint=0)
        fig_v4_margin.update_layout(height=300, margin=dict(t=40, b=40))
        st.plotly_chart(fig_v4_margin, use_container_width=True)

        eval_acc = v2_data.get("eval_reward_accuracy", 0)
        if eval_acc >= 0.75:
            st.success(f"**v4 Strong alignment** — reward accuracy: {eval_acc:.0%}. "
                       f"Merged-SFT + β=0.05 produces best DPO training dynamics.")
        elif eval_acc >= 0.65:
            st.success(f"**v4 Good alignment** — reward accuracy: {eval_acc:.0%}")
        elif eval_acc >= 0.55:
            st.info(f"**v4 Moderate alignment** — reward accuracy: {eval_acc:.0%}")
        else:
            st.warning(f"**v4 Weak alignment** — reward accuracy: {eval_acc:.0%}")
    else:
        st.info("🔄 v4 results not yet available. Run the improved DPO training on RunPod, "
                "then save results to `dashboard/v2_results.json`.")

# --- Factuality Comparison ---
if v2_data and "factuality" in v2_data:
    st.header("🎯 Factuality Evaluation: Base vs SFT vs DPO")

    st.markdown("""
    **Key Finding:** Despite strong DPO alignment (80% reward accuracy), all model stages
    perform poorly on strict factual recall of niche ML terminology. The base model itself
    doesn't reliably know these terms.
    """)

    fact_data = v2_data["factuality"]
    fact_df = pd.DataFrame([
        {"Model Stage": "Base (Llama-3.1-8B-Instruct)", "Passed": fact_data["base"]["passed"],
         "Total": fact_data["base"]["total"], "Accuracy": fact_data["base"]["accuracy"]},
        {"Model Stage": "SFT (OpenHermes + Technical)", "Passed": fact_data["sft"]["passed"],
         "Total": fact_data["sft"]["total"], "Accuracy": fact_data["sft"]["accuracy"]},
        {"Model Stage": "DPO (Merged-SFT, β=0.05)", "Passed": fact_data["dpo"]["passed"],
         "Total": fact_data["dpo"]["total"], "Accuracy": fact_data["dpo"]["accuracy"]},
    ])

    col_fact1, col_fact2 = st.columns([1, 2])

    with col_fact1:
        st.dataframe(fact_df[["Model Stage", "Passed", "Accuracy"]].style.format({"Accuracy": "{:.1%}"}),
                     use_container_width=True, hide_index=True)

        st.markdown("**Interpretation:**")
        st.markdown("""
        - Base model doesn't know niche ML terms (9.8%)
        - 875 SFT examples / 1 epoch insufficient to teach factual recall
        - DPO's contribution to factuality loss is secondary (~4pp)
        - Primary bottleneck: SFT data quantity, not DPO interference
        """)

    with col_fact2:
        fig_fact = go.Figure()
        colors = ["#2196F3", "#FF9800", "#E91E63"]
        stages = fact_df["Model Stage"].tolist()
        accuracies = [a * 100 for a in fact_df["Accuracy"].tolist()]

        fig_fact.add_trace(go.Bar(
            x=stages, y=accuracies,
            marker_color=colors,
            text=[f"{a:.1f}%" for a in accuracies],
            textposition="outside",
        ))
        fig_fact.update_layout(
            title="Factuality Accuracy by Model Stage (51 prompts, strict keyword matching)",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 20]),
            height=400,
            margin=dict(t=50, b=80),
            showlegend=False,
        )
        st.plotly_chart(fig_fact, use_container_width=True)

    # Metric mismatch visualization
    st.subheader("📉 The Metric-Factuality Mismatch")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        fig_mismatch = go.Figure()
        fig_mismatch.add_trace(go.Bar(
            name="DPO Reward Accuracy",
            x=["Alignment Metric"], y=[v2_data.get("eval_reward_accuracy", 0.80) * 100],
            marker_color="#4CAF50",
        ))
        fig_mismatch.add_trace(go.Bar(
            name="Factuality (DPO)",
            x=["Factuality"], y=[fact_data["dpo"]["accuracy"] * 100],
            marker_color="#E91E63",
        ))
        fig_mismatch.update_layout(
            title="Alignment ≠ Factuality",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 100]),
            height=350,
            barmode="group",
        )
        st.plotly_chart(fig_mismatch, use_container_width=True)

    with col_m2:
        st.markdown("### The Disconnect")
        st.markdown("""
        | Metric | Score |
        |--------|-------|
        | DPO Reward Accuracy | **80%** ✅ |
        | SFT Token Accuracy | **78%** ✅ |
        | SFT Eval Loss | **0.825** ✅ |
        | Domain Factuality | **5.9%** ❌ |

        **Training metrics look great. Factuality does not.**

        This is the core finding: standard alignment metrics
        (reward accuracy, loss) do not capture factual degradation.
        """)

    st.info("💡 **Next experiments needed:** SFT with 3-5 epochs, SFT with 2.5K-5K examples, "
            "semantic/LLM-judge factuality eval (not just keyword matching).")

# --- Version Comparison Table ---
st.header("📋 Version Comparison")

version_data = {
    "Version": ["v1", "v2", "v3", "v4 (latest)"],
    "SFT Data": ["Alpaca 1K", "Alpaca 1K", "OpenHermes+Tech 3.9K", "OpenHermes+Tech 3.9K"],
    "DPO Config": ["Stacked, β=0.1", "Stacked, β=0.1", "Stacked, β=0.1", "Merged-SFT, β=0.05"],
    "Peak Reward Acc": ["50%", "75%", "68%", "83%"],
    "DPO Loss": ["0.70", "0.76", "0.77", "0.54"],
    "Factuality": ["—", "—", "9.8% (DPO only)", "Base 9.8% / SFT 7.8% / DPO 5.9%"],
}
version_df = pd.DataFrame(version_data)
st.dataframe(version_df, use_container_width=True, hide_index=True)

# --- Model Comparison ---
COMPARISON_PATH = Path("dashboard/sample_comparisons.json")

if COMPARISON_PATH.exists():
    st.header("💬 Response Comparison: Base vs SFT vs DPO")

    with open(COMPARISON_PATH) as f:
        comparisons = json.load(f)

    for i, entry in enumerate(comparisons):
        with st.expander(f"Prompt {i+1}: {entry['prompt'][:80]}...", expanded=(i == 0)):
            col_base, col_sft, col_dpo = st.columns(3)
            with col_base:
                st.markdown("**🔵 Base Model**")
                st.text_area("", entry["base"][:500], height=200, key=f"base_{i}", disabled=True)
            with col_sft:
                st.markdown("**🟡 After SFT**")
                st.text_area("", entry["sft"][:500], height=200, key=f"sft_{i}", disabled=True)
            with col_dpo:
                st.markdown("**🟢 After DPO**")
                st.text_area("", entry["dpo"][:500], height=200, key=f"dpo_{i}", disabled=True)

# --- Config Details ---
st.header("⚙️ Configuration")

col_config1, col_config2 = st.columns(2)

with col_config1:
    st.markdown("### Model & Training")
    st.json({
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "quantization": "QLoRA (4-bit NF4, double quant)",
        "lora_rank": 16,
        "lora_alpha": 32,
        "trainable_params": "13.6M (0.30%)",
        "sft_data": "OpenHermes 3K + Technical 875 + Uncertainty 15",
        "dpo_data": "UltraFeedback 5K + Factual pairs (20% upsampled)",
    })

with col_config2:
    st.markdown("### Infrastructure")
    st.json({
        "gpu": "NVIDIA RTX A6000 (48 GB)",
        "platform": "RunPod.io",
        "total_cost": "~$5 across all runs",
        "sft_duration": "12.5 min",
        "dpo_duration": "67.6 min (v4)",
        "dpo_beta": 0.05,
        "dpo_lr": "1e-5",
        "adapter_strategy": "Merged-SFT (bake SFT into base before DPO)",
    })

# --- Footer ---
st.markdown("---")
st.markdown(
    "Built by [Santosh Adabala](https://github.com/santoshadabala) "
    "| [GitHub Repo](https://github.com/santoshadabala/distill-align-llm)"
)
