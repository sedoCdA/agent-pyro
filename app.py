"""
=============================================================
AgentPyro — Streamlit Dashboard
=============================================================
Purpose:
    Interactive web UI for the Agentic AI Security Risk Predictor.
    Enter an agent action log entry via form controls and get a
    live access decision with confidence score, probability
    breakdown, SHAP explanation, and risk level.

Run:
    streamlit run app.py

Requirements:
    pip install streamlit plotly
=============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AgentPyro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Add src/ to path so we can import inference.py ───────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from inference import predict_access_decision

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0c10;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f1218;
    border-right: 1px solid #1e2433;
}

/* Header */
.pyro-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.5px;
    margin-bottom: 0;
    line-height: 1.1;
}
.pyro-sub {
    font-size: 0.85rem;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}

/* Decision cards */
.decision-card {
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}
.card-allowed   { background: linear-gradient(135deg, #052e16 0%, #14532d 100%); border: 1px solid #16a34a; }
.card-blocked   { background: linear-gradient(135deg, #2d0a0a 0%, #450a0a 100%); border: 1px solid #dc2626; }
.card-human     { background: linear-gradient(135deg, #1c1508 0%, #32200a 100%); border: 1px solid #d97706; }

.decision-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 6px;
}
.decision-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1;
}
.conf-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    margin-top: 10px;
}
.risk-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-left: 8px;
}
.risk-LOW      { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
.risk-MEDIUM   { background: #1c1508; color: #fbbf24; border: 1px solid #d97706; }
.risk-HIGH     { background: #2d0a0a; color: #f87171; border: 1px solid #dc2626; }
.risk-CRITICAL { background: #1e0a2e; color: #c084fc; border: 1px solid #9333ea; }

/* Shap reason rows */
.reason-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid #1e2433;
    font-size: 0.88rem;
}
.reason-feat  { color: #94a3b8; font-family: 'Space Mono', monospace; font-size: 0.78rem; }
.reason-pos   { color: #f87171; font-family: 'Space Mono', monospace; font-size: 0.82rem; }
.reason-neg   { color: #4ade80; font-family: 'Space Mono', monospace; font-size: 0.82rem; }
.reason-val   { color: #475569; font-size: 0.75rem; font-family: 'Space Mono', monospace; }

/* Metric tiles */
.metric-tile {
    background: #0f1218;
    border: 1px solid #1e2433;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-number {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f8fafc;
}
.metric-label {
    font-size: 0.72rem;
    color: #475569;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 3px;
}

/* Divider */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #334155;
    margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid #1e2433;
    padding-bottom: 6px;
}

/* Input labels */
label { color: #94a3b8 !important; font-size: 0.82rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Slider track */
.stSlider > div > div > div { background: #1e2433 !important; }

/* Selectbox */
.stSelectbox > div > div { background: #0f1218 !important; border: 1px solid #1e2433 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar — Input Form ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="pyro-header">🔥 AgentPyro</div>', unsafe_allow_html=True)
    st.markdown('<div class="pyro-sub">AGENTIC AI SECURITY RISK PREDICTOR</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Agent Identity</div>', unsafe_allow_html=True)

    agent_role = st.selectbox("Agent Role", [
        "customer_support_agent", "finance_agent", "hr_onboarding_agent",
        "it_ops_agent", "sales_agent", "data_analyst_agent",
    ])
    user_role = st.selectbox("User Role", [
        "admin", "analyst", "manager", "vendor", "developer", "auditor",
    ])
    agent_autonomy_level = st.slider("Agent Autonomy Level", 1, 5, 3,
        help="1 = highly supervised, 5 = fully autonomous")

    st.markdown('<div class="section-title">Requested Action</div>', unsafe_allow_html=True)

    requested_action = st.selectbox("Action", [
        "read_record", "export_report", "send_email", "change_permission",
        "delete_record", "create_user", "modify_config", "access_logs",
    ])
    tool_requested = st.selectbox("Tool / API", [
        "crm_api", "file_storage_api", "hris_api", "external_webhook",
        "database_api", "email_api", "audit_log_api", "identity_api",
    ])
    resource_type = st.selectbox("Resource Type", [
        "customer_profile", "api_key_secret", "sales_pipeline",
        "employee_record", "financial_report", "system_config", "audit_log",
    ])
    resource_sensitivity = st.slider("Resource Sensitivity", 1, 5, 3,
        help="1 = public, 5 = top secret")

    st.markdown('<div class="section-title">Risk Signals</div>', unsafe_allow_html=True)

    action_risk_score       = st.slider("Action Risk Score",        0, 100, 30)
    data_exfiltration_risk  = st.slider("Data Exfiltration Risk",   0, 100, 20)

    col1, col2 = st.columns(2)
    with col1:
        permission_match         = st.checkbox("Permission Match",    value=True)
        prompt_injection         = st.checkbox("Prompt Injection",    value=False)
    with col2:
        human_approval_required  = st.checkbox("Human Approval Req", value=False)
        audit_log_available      = st.checkbox("Audit Log Available", value=True)

    previous_failed_attempts = st.slider("Previous Failed Attempts", 0, 10, 0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  ANALYSE ACTION")


# ── Quick stats bar ───────────────────────────────────────────────────────────
st.markdown('<div class="pyro-header">Access Decision Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="pyro-sub">REAL-TIME AGENT ACTION RISK ANALYSIS WITH SHAP EXPLANATIONS</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-tile"><div class="metric-number">99.55%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-tile"><div class="metric-number">99.09%</div><div class="metric-label">Macro F1 Score</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-tile"><div class="metric-number">99.80%</div><div class="metric-label">Mean Confidence</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-tile"><div class="metric-number">LightGBM</div><div class="metric-label">Primary Model</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    log_entry = {
        "agent_role":                agent_role,
        "agent_autonomy_level":      agent_autonomy_level,
        "user_role":                 user_role,
        "requested_action":          requested_action,
        "tool_requested":            tool_requested,
        "resource_type":             resource_type,
        "resource_sensitivity":      resource_sensitivity,
        "permission_match":          int(permission_match),
        "action_risk_score":         action_risk_score,
        "prompt_injection_detected": int(prompt_injection),
        "data_exfiltration_risk":    data_exfiltration_risk,
        "human_approval_required":   int(human_approval_required),
        "previous_failed_attempts":  previous_failed_attempts,
        "audit_log_available":       int(audit_log_available),
    }

    with st.spinner("Analysing agent action ..."):
        result = predict_access_decision(log_entry)

    decision = result["decision"]
    conf     = result["confidence"]
    risk     = result["risk_level"]

    # ── Decision card ──
    card_class = {
        "Allowed":              "card-allowed",
        "Blocked":              "card-blocked",
        "Needs_Human_Approval": "card-human",
    }[decision]

    emoji = {"Allowed": "✅", "Blocked": "🚫", "Needs_Human_Approval": "⚠️"}[decision]
    label = decision.replace("_", " ")

    risk_colors = {"LOW": "#4ade80", "MEDIUM": "#fbbf24",
                   "HIGH": "#f87171", "CRITICAL": "#c084fc"}
    risk_color  = risk_colors.get(risk, "#f8fafc")

    st.markdown(f"""
    <div class="decision-card {card_class}">
        <div class="decision-label">Access Decision</div>
        <div class="decision-value">{emoji} {label}</div>
        <span class="conf-pill">Confidence: {conf:.2%}</span>
        <span class="risk-pill risk-{risk}">{risk}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Two columns: probabilities + SHAP ──
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Class Probabilities</div>', unsafe_allow_html=True)

        probs  = result["probabilities"]
        labels = list(probs.keys())
        values = list(probs.values())
        colors = ["#16a34a", "#dc2626", "#d97706"]

        fig = go.Figure(go.Bar(
            x=values,
            y=[l.replace("_", " ") for l in labels],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            text=[f"{v:.2%}" for v in values],
            textposition="outside",
            textfont=dict(family="Space Mono", size=11, color="#94a3b8"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=60, t=10, b=10),
            height=180,
            xaxis=dict(
                range=[0, 1.15],
                showgrid=False, zeroline=False,
                tickformat=".0%", tickfont=dict(color="#475569", size=10),
            ),
            yaxis=dict(
                showgrid=False, zeroline=False,
                tickfont=dict(family="Space Mono", color="#94a3b8", size=11),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Confidence gauge
        st.markdown('<div class="section-title">Confidence Gauge</div>', unsafe_allow_html=True)
        gauge_color = {"Allowed": "#16a34a", "Blocked": "#dc2626",
                       "Needs_Human_Approval": "#d97706"}[decision]
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf * 100,
            number={"suffix": "%", "font": {"family": "Space Mono",
                                             "size": 28, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#334155",
                          "tickfont": {"color": "#475569", "size": 10}},
                "bar":  {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "#0f1218",
                "bordercolor": "#1e2433",
                "steps": [
                    {"range": [0, 50],  "color": "#0f1218"},
                    {"range": [50, 90], "color": "#151b26"},
                    {"range": [90, 100],"color": "#1a2030"},
                ],
                "threshold": {"line": {"color": gauge_color, "width": 2},
                               "thickness": 0.85, "value": conf * 100},
            },
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=20, r=20, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown('<div class="section-title">Top SHAP Reasons</div>', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.78rem; color:#475569; margin-bottom:10px;'>Why this decision was made — feature contributions toward the predicted class</div>",
                    unsafe_allow_html=True)

        reasons_html = ""
        for i, r in enumerate(result["top_reasons"], 1):
            direction = "▲" if r["shap_value"] > 0 else "▼"
            cls       = "reason-pos" if r["shap_value"] > 0 else "reason-neg"
            reasons_html += f"""
            <div class="reason-row">
                <span style="color:#64748b; font-family:'Space Mono',monospace; font-size:0.72rem; min-width:16px;">{i}.</span>
                <span class="reason-feat" style="flex:1; margin-left:8px;">{r['feature']}</span>
                <span class="{cls}">{direction} {r['shap_value']:+.4f}</span>
                <span class="reason-val" style="margin-left:12px;">val={r['feature_value']:.2f}</span>
            </div>
            """
        st.markdown(f"<div style='background:#0f1218; border:1px solid #1e2433; border-radius:10px; padding:12px 16px;'>{reasons_html}</div>",
                    unsafe_allow_html=True)

        # SHAP bar chart
        st.markdown('<div class="section-title">SHAP Waterfall</div>', unsafe_allow_html=True)
        reasons  = result["top_reasons"]
        feats    = [r["feature"] for r in reasons][::-1]
        shap_vals= [r["shap_value"] for r in reasons][::-1]
        bar_colors = ["#dc2626" if v > 0 else "#16a34a" for v in shap_vals]

        fig3 = go.Figure(go.Bar(
            x=shap_vals,
            y=feats,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:+.4f}" for v in shap_vals],
            textposition="outside",
            textfont=dict(family="Space Mono", size=10, color="#94a3b8"),
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=220,
            margin=dict(l=0, r=70, t=5, b=5),
            xaxis=dict(showgrid=False, zeroline=True,
                       zerolinecolor="#334155", zerolinewidth=1,
                       tickfont=dict(color="#475569", size=10)),
            yaxis=dict(showgrid=False, zeroline=False,
                       tickfont=dict(family="Space Mono", color="#94a3b8", size=10)),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── Raw log entry expander ──
    with st.expander("📋  View raw log entry"):
        st.json(log_entry)

else:
    # ── Empty state ──
    st.markdown("""
    <div style="
        border: 1px dashed #1e2433;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        margin-top: 1rem;
    ">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔍</div>
        <div style="font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #334155; letter-spacing: 0.1em;">
            CONFIGURE AN AGENT ACTION IN THE SIDEBAR<br>
            <span style="color: #1d4ed8;">THEN CLICK ANALYSE ACTION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show example scenarios as reference
    st.markdown('<div class="section-title" style="margin-top:2rem;">Example Scenarios</div>',
                unsafe_allow_html=True)

    scenarios = [
        ("✅ Likely Allowed",              "analyst + read_record + low risk + permission match"),
        ("🚫 Likely Blocked",              "vendor + export_report + no permission + high risk"),
        ("⚠️  Likely Needs Human Approval", "manager + change_permission + medium risk + approval flag"),
    ]
    for title, desc in scenarios:
        st.markdown(f"""
        <div style="background:#0f1218; border:1px solid #1e2433; border-radius:8px;
                    padding:10px 14px; margin-bottom:8px; display:flex; align-items:center; gap:12px;">
            <span style="font-family:'Space Mono',monospace; font-size:0.82rem; color:#f8fafc;">{title}</span>
            <span style="font-size:0.75rem; color:#475569;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)