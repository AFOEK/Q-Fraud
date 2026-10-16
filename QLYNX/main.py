"""QLYNX | AML, Fraud and Player Protection — Streamlit investigator prototype."""
from __future__ import annotations

import json
import html
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from risk_ui import (RISK_COLORS, RISK_NAMES, RISK_PALE, PRIORITY_COLORS,
                     add_triage, task_band, badge, task_badge, colored_table, risk_cards_html)
from graph_view import (context_for_account, aggregate_graph, parse_edges, linked_graph, render_graph)
from workspace_v3 import CSS as WORKSPACE_CSS, queue_view, account360, relationships
import streamlit as st

from copilot import ask_ollama, evidence_packet, fallback_note
from model_registry import (
    ROOT, TASKS, TASK_LABELS, LGBM, OPERATIONAL, NEURAL_CHECKPOINTS,
    manifest, model_status, score_lightgbm, score_v4_demo, infer_known_accounts,
)

st.set_page_config(
    page_title="QLYNX | Anti-Money Laundering, Fraud & Player Protection",
    page_icon="🛡️", layout="wide", initial_sidebar_state="expanded",
)

PAGES = (
    "Command Centre", "Investigation Queue", "Account 360°", "Historical Context",
    "Relationship Explorer", "Data Intake", "Decisions & Audit", "Model Laboratory",
    "LLM Investigator Copilot", "Appearance",
)
NAV_ICONS = {
    "Command Centre": ":material/dashboard:",
    "Investigation Queue": ":material/filter_list:",
    "Account 360°": ":material/person_search:",
    "Historical Context": ":material/show_chart:",
    "Relationship Explorer": ":material/hub:",
    "Data Intake": ":material/cloud_upload:",
    "Decisions & Audit": ":material/fact_check:",
    "Model Laboratory": ":material/science:",
    "LLM Investigator Copilot": ":material/auto_awesome:",
    "Appearance": ":material/palette:",
}
if "page" not in st.session_state:
    st.session_state.page = "Command Centre"
if "source" not in st.session_state:
    st.session_state.source = "Demo snapshot"
if "llm_url" not in st.session_state:
    st.session_state.llm_url = "http://127.0.0.1:11434"
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "qwen2.5:3b"

st.markdown("""<style>
:root{--ql-border:#e3e9f2;--ql-text:#15213c;--ql-secondary:#65748d}
html,body,[class*="css"]{font-family:Inter,"Segoe UI",Arial,sans-serif}
.block-container{max-width:1280px;padding-top:1.4rem;padding-bottom:3rem}
h1{font-size:2rem !important;letter-spacing:-.04em !important;font-weight:760 !important}
h2,h3{letter-spacing:-.025em !important}
section[data-testid="stSidebar"]{border-right:1px solid #e1e7ef;background:#f6f8fc}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:1rem}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.35rem !important}
section[data-testid="stSidebar"] div[data-testid="stButton"]{width:100% !important}
section[data-testid="stSidebar"] div[data-testid="stButton"] button{
 width:100% !important;min-height:40px !important;height:auto !important;max-width:100% !important;
 display:flex !important;align-items:center !important;justify-content:flex-start !important;text-align:left !important;
 border:1px solid transparent !important;box-shadow:none !important;border-radius:9px !important;
 background:transparent !important;color:#40526e !important;padding:.55rem .7rem !important;
 font:500 .89rem/1.3 Inter,"Segoe UI",Arial,sans-serif !important;
 transition:background .15s ease,color .15s ease !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button p{font:inherit !important;letter-spacing:0 !important}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover{background:#e9eef7 !important;color:#15213c !important}
/* current nav key, set from current session state at each rerun */
__ACTIVE_SELECTOR__
[data-testid="stMetric"]{border:1px solid var(--ql-border);border-radius:13px;padding:.88rem 1.1rem;background:#fff}
[data-testid="stMetricValue"]{font-weight:760;color:#192644}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:12px}
[data-testid="stMain"] [data-testid="stButton"] button{border-radius:9px}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
.ql-kicker{text-transform:uppercase;letter-spacing:.12em;color:#7385a4;font-size:.72rem;font-weight:760}
.ql-caption{font-size:.88rem;color:#72829f;margin-top:-.25rem;margin-bottom:1.15rem}
.ql-brand{display:flex;align-items:center;gap:10px;font-size:1.3rem;font-weight:800;letter-spacing:-.045em;color:#15213c;padding:7px 0 3px}
.ql-brand-mark{width:31px;height:31px;border-radius:9px;background:#172640;color:white;display:inline-flex;align-items:center;justify-content:center;font-size:1.2rem}
.ql-tagline{color:#718098;font-size:.79rem;line-height:1.5;margin:4px 0 7px}
.ql-section{color:#8391a6;font-size:.67rem;font-weight:750;letter-spacing:.12em;margin:17px 0 6px;text-transform:uppercase}
.ql-risk-legend{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0 8px}
.ql-pill,.ql-task-pill{display:inline-flex;align-items:center;gap:5px;font-weight:700;font-size:.74rem;padding:4px 9px;border:1px solid transparent;border-radius:30px;white-space:nowrap}
.ql-pill-dot{display:inline-block;width:7px;height:7px;border-radius:50%}
.ql-risk-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:0 0 1.5rem}
.ql-risk-card{min-height:192px;border-radius:14px;border:1px solid #e0e7f2;border-top:4px solid var(--risk);padding:17px 19px;background:var(--risk-bg);box-shadow:0 2px 6px #26344a09}
.ql-risk-card.ql-fraud{--risk:#d92d43;--risk-bg:#fff9f9}
.ql-risk-card.ql-aml{--risk:#ed8a23;--risk-bg:#fffbf5}
.ql-risk-card.ql-protection{--risk:#219c67;--risk-bg:#f6fcf9}
.ql-risk-header{display:flex;align-items:center;gap:8px;font-weight:750;color:#18243b;font-size:1rem}
.ql-risk-indicator{width:10px;height:10px;flex-shrink:0;border-radius:50%;background:var(--risk)}
.ql-risk-subtitle{font-size:.8rem;color:#718097;margin-top:5px}
.ql-risk-number{font-weight:790;font-size:2rem;color:var(--risk);margin-top:15px;line-height:1.1;letter-spacing:-.045em}
.ql-risk-number span{font-size:.8rem;letter-spacing:0;font-weight:500;color:#718097}
.ql-risk-track{height:8px;background:#e8edf5;border-radius:100px;margin:13px 0 9px;overflow:hidden}
.ql-risk-track div{background:var(--risk);height:100%;border-radius:100px;min-width:2px}
.ql-risk-bottom{color:#718097;font-size:.78rem}
.ql-score-card{background:var(--risk-bg);border:1px solid var(--risk-border);border-top:3px solid var(--risk);padding:15px 16px;border-radius:12px;margin:10px 0}
.ql-score-card strong{display:block;font-size:.84rem;color:#53647e;margin-bottom:9px}
.ql-score-number{font-size:1.8rem;color:var(--risk);font-weight:790;letter-spacing:-.03em}
.ql-score-foot{font-size:.77rem;color:#72829d;margin-top:7px}
@media(max-width:850px){.ql-risk-grid{grid-template-columns:1fr}.block-container{padding-left:1rem;padding-right:1rem}}
</style>""".replace("__ACTIVE_SELECTOR__", "section[data-testid='stSidebar'] .st-key-nav_"+str(PAGES.index(st.session_state.page))+" button{background:#e5eaf4 !important;color:#142442 !important;font-weight:730 !important;border-left:3px solid #253b60 !important;}"), unsafe_allow_html=True)
st.markdown("<style>" + WORKSPACE_CSS + "</style>", unsafe_allow_html=True)

@st.cache_data(show_spinner="Reading demo snapshot…")
def demo_predictions():
    return pd.read_csv(ROOT / "demo/demo_inference_results.csv", dtype={"account_id": str})

@st.cache_data(show_spinner=False)
def demo_features():
    return pd.read_csv(ROOT / "demo/demo_accounts_upload.csv", dtype={"account_id": str})

@st.cache_data(show_spinner="Reading investigator queue…")
def full_queue():
    return pd.read_parquet(ROOT / "data/investigator_queue.parquet")

@st.cache_data(show_spinner=False)
def demo_evidence():
    return pd.read_csv(ROOT / "demo/demo_model_evidence.csv", dtype={"account_id": str})


def go(page):
    st.session_state.page = page


def data_view():
    if st.session_state.source == "Latest inference":
        if "fresh_scores" in st.session_state:
            return st.session_state.fresh_scores
        return demo_predictions()
    if st.session_state.source == "Exported 20K queue":
        return full_queue()
    return demo_predictions()


def task_score_column(df, task):
    for col in (f"{task}_score", manifest()["operational_models"][task]["score_column"]):
        if col in df:
            return col
    return None


def alerts(df):
    out = {}
    for task in TASKS:
        flag = f"{task}_alert"
        if flag in df:
            out[task] = df[flag].fillna(False).astype(bool)
        else:
            score_col = task_score_column(df, task)
            threshold = float(manifest()["operational_models"][task]["threshold"])
            out[task] = (df[score_col] >= threshold) if score_col else pd.Series(False, index=df.index)
    return out


def active_row():
    view = data_view()
    if view.empty:
        return None
    ids = view.account_id.astype(str)
    current = st.session_state.get("active_account", str(ids.iloc[0]))
    if current not in ids.values:
        current = str(ids.iloc[0]); st.session_state.active_account = current
    return view.loc[ids == current].iloc[0]


def select_account(view, prefix):
    if view.empty:
        st.warning("No accounts are available.")
        return None
    ids = view.account_id.astype(str).tolist()
    initial = st.session_state.get("active_account", ids[0])
    index = ids.index(initial) if initial in ids else 0
    chosen = st.selectbox("Select account", ids, index=index, key=f"{prefix}_select")
    st.session_state.active_account = chosen
    return view.loc[view.account_id.astype(str) == chosen].iloc[0]


def model_cards(df):
    flags = alerts(df)
    metrics = [("Monitored Accounts", f"{len(df):,}", "Accounts included in this view"),
               ("Flagged Accounts", f"{int(pd.DataFrame(flags).any(axis=1).sum()):,}", "Accounts with ≥1 operational alert"),
               ("Alerted Account Rate", f"{100 * pd.DataFrame(flags).any(axis=1).mean():.2f}%", "Share with alerts")]
    cols = st.columns(3, gap="medium")
    for col, (title, value, caption) in zip(cols, metrics):
        with col:
            st.metric(title, value)
            st.caption(caption)
    st.subheader("Risk Detection Summary")
    st.caption("Operational alerts from the three selected LightGBM models")
    st.markdown(risk_cards_html(flags, len(df)), unsafe_allow_html=True)
    return flags


with st.sidebar:
    st.markdown('<div class="ql-brand"><span class="ql-brand-mark">⬡</span> QLYNX</div>', unsafe_allow_html=True)
    st.markdown('<div class="ql-tagline">Anti-Money Laundering, Fraud &amp; Player Protection</div>', unsafe_allow_html=True)
    st.markdown('<div class="ql-risk-legend">'+''.join(task_badge(t, label) for t, label in
        (("fraud", "Fraud"), ("aml", "AML"), ("protection", "Protection")))+'</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="ql-section">Investigation</div>', unsafe_allow_html=True)
    for page_name in PAGES[:5]:
        st.button(page_name, key=f"nav_{PAGES.index(page_name)}", icon=NAV_ICONS[page_name], on_click=go, args=(page_name,), use_container_width=True)
    st.markdown('<div class="ql-section">Operations</div>', unsafe_allow_html=True)
    for page_name in PAGES[5:9]:
        st.button(page_name, key=f"nav_{PAGES.index(page_name)}", icon=NAV_ICONS[page_name], on_click=go, args=(page_name,), use_container_width=True)
    st.markdown('<div class="ql-section">Preferences</div>', unsafe_allow_html=True)
    st.button("Appearance", key="nav_9", icon=NAV_ICONS["Appearance"], on_click=go, args=("Appearance",), use_container_width=True)
    st.divider()
    st.markdown("**Demo Investigator**")
    st.caption("Local research demonstration · Synthetic data")
    st.caption(f"**{int((model_status().Status == 'Available').sum())}/{len(model_status())}** model checkpoints available")

st.markdown("**◈  QLYNX — Anti-Money Laundering, Fraud & Player Protection**")
st.markdown('<div class="ql-kicker">Investigation workspace / '+st.session_state.page+'</div>', unsafe_allow_html=True)
page = st.session_state.page
if page != "Account 360°":
    st.title(page)

if page == "Command Centre":
    st.markdown('<div class="ql-caption">Operational fraud, AML and player protection monitoring.</div>', unsafe_allow_html=True)
    left, right = st.columns([4, 2])
    with left:
        st.selectbox("Data source", ["Demo snapshot"] + (["Latest inference"] if "fresh_scores" in st.session_state else []) + ["Exported 20K queue"],
                     key="source")
    with right:
        st.write("")
        st.button("Run new scan →", type="primary", use_container_width=True, on_click=go, args=("Data Intake",))
    if st.session_state.source == "Exported 20K queue":
        st.info("Showing the 20,000 exported investigation queue entries, not all 1,199,885 accounts. "
                "The global population counters in the artifact manifest refer to the complete dataset.")
    elif st.session_state.source == "Demo snapshot":
        st.caption("Bundled 130-account demonstration. Select Data Intake to produce fresh checkpoint-backed operational scores.")
    else:
        st.success("Latest inference from uploaded compatible engineered features.")
    df = data_view()
    flags = model_cards(df)
    st.subheader("Alert Distribution")
    fig = px.bar(pd.DataFrame({"Category": [TASK_LABELS[t] for t in TASKS], "Alerts": [int(flags[t].sum()) for t in TASKS]}),
                 x="Category", y="Alerts", text="Alerts", color="Category",
                 color_discrete_map={TASK_LABELS[t]: RISK_COLORS[t] for t in TASKS})
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Quantum / Classical V5 — Research Benchmark")
    st.caption("Held-out synthetic test evaluation, not live 130-account demo performance. Quantum V5 uses a classically simulated six-qubit circuit.")
    q1, q2, q3 = st.columns(3, gap="medium")
    q1.metric("Quantum V5 macro-AUPRC", "0.902237")
    q2.metric("Strong classical V5 macro-AUPRC", "0.897335")
    q3.metric("Extra fraud cases at top 0.5%", "+36", help="Quantum 766 vs strong classical 730, among 896 reviewed accounts on the synthetic held-out test.")
    st.button("Compare all 18 models →", on_click=go, args=("Model Laboratory",))
    with st.expander("Model provenance and limitations"):
        st.json({"selected_models": OPERATIONAL, "alert_thresholds": {
            t: manifest()["operational_models"][t]["threshold"] for t in TASKS},
            "experimental": "V5 quantum & strong classical three-seed ensembles", "source": "Synthetic benchmark"})

elif page == "Investigation Queue":
    sources = ["Demo snapshot"] + (["Latest inference"] if "fresh_scores" in st.session_state else []) + ["Exported 20K queue"]
    def _switch_source(name):
        st.session_state.source = name
    def _open_account(account_id):
        st.session_state.active_account = str(account_id)
        go("Account 360°")
    queue_view(data=data_view(), manifest=manifest(), sources=sources,
               source_name=st.session_state.source, set_source=_switch_source,
               on_account=_open_account, audit_path=ROOT / "audit/decisions.jsonl",
               on_intake=lambda: go("Data Intake"))

elif page == "Account 360°":
    row = active_row()
    if st.session_state.source == "Demo snapshot":
        source_features = demo_features()
        source_evidence = demo_evidence()
    elif st.session_state.source == "Latest inference":
        source_features = st.session_state.get("fresh_input")
        source_evidence = st.session_state.get("fresh_explanations", pd.DataFrame())
    else:
        source_features = None
        source_evidence = pd.DataFrame()
    account360(row=row, data=data_view(), manifest=manifest(),
               source_name=st.session_state.source, features=source_features,
               evidence=source_evidence, on_queue=lambda: go("Investigation Queue"),
               on_copilot=lambda: go("LLM Investigator Copilot"),
               on_audit=lambda: go("Decisions & Audit"), run_neural=infer_known_accounts,
               explain_model=lambda features, explanations: score_lightgbm(features, list(OPERATIONAL.values()), explanations=explanations))

elif page == "Historical Context":
    st.caption("Three 30-day observation windows, where aggregates are available.")
    row = select_account(data_view(), "history")
    if row is not None:
        if st.session_state.source == "Latest inference":
            extra = st.session_state.get("fresh_input")
        elif st.session_state.source == "Demo snapshot":
            extra = demo_features()
        else:
            extra = data_view()
        found = extra.loc[extra.account_id.astype(str) == str(row.account_id)] if extra is not None else pd.DataFrame()
        hist = found.iloc[0] if not found.empty else row
        pairs = []
        for category in ("wager", "deposit", "tx"):
            for label, suffix in (("Days 1–30", "first30"), ("Days 31–60", "middle30"), ("Days 61–90", "last30")):
                col = f"{category}_{suffix}"
                if col in hist and pd.notna(hist[col]):
                    pairs.append({"Window": label, "Measure": category.title(), "Amount / Count": float(hist[col])})
        if pairs:
            fig = px.line(pd.DataFrame(pairs), x="Window", y="Amount / Count", color="Measure", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Aggregated windows, not timestamped individual events. Transaction measures may use different units.")
        else:
            st.info("This exported account has no three-window values. Raw event histories are not included in the release.")

elif page == "Relationship Explorer":
    row = select_account(data_view(), "relationships_v3")
    if st.session_state.source == "Demo snapshot":
        source_features = demo_features()
    elif st.session_state.source == "Latest inference":
        source_features = st.session_state.get("fresh_input")
    else:
        source_features = None
    relationships(row=row, features=source_features)

elif page == "Data Intake":
    st.caption("Fresh operational scoring of compatible account-level engineered features. Not raw transaction/event ingestion.")
    uploaded = st.file_uploader("Upload engineered account feature CSV", type="csv", key="intake")
    if uploaded is None:
        st.info("No upload selected; the bundled 130-row demo feature CSV will be used.")
    if st.button("Run fraud + AML + player protection", type="primary"):
        try:
            features = pd.read_csv(uploaded, dtype={"account_id": str}) if uploaded else demo_features()
            with st.spinner("Loading selected LightGBM weights and scoring features…"):
                scored, shap = score_lightgbm(features, list(OPERATIONAL.values()), explanations=True)
            st.session_state.fresh_scores = scored
            st.session_state.fresh_input = features
            st.session_state.fresh_explanations = shap
            st.session_state.source = "Latest inference"
            st.success(f"Fresh operational inference complete: {len(scored):,} accounts; "
                       f"{int((scored.alert_count > 0).sum()):,} flagged.")
        except Exception as exc:
            st.error(f"Could not score this dataset: {exc}")
    if "fresh_scores" in st.session_state:
        st.dataframe(st.session_state.fresh_scores.head(250), hide_index=True, use_container_width=True)
        st.download_button("Export scored cases", st.session_state.fresh_scores.to_csv(index=False),
                           file_name="qlynx_operational_scores.csv", mime="text/csv")
        st.button("Open Command Centre →", on_click=go, args=("Command Centre",))

elif page == "Model Laboratory":
    st.caption("All model artifacts inventoried. Operational alerts and experimental ranking models remain separate.")
    counts = model_status()
    st.dataframe(counts, hide_index=True, use_container_width=True)
    st.subheader("A. Compare all classical LightGBM boosters")
    chosen = st.multiselect("Selected LightGBM variants", list(LGBM), default=list(LGBM))
    if st.button("Run selected LightGBM models"):
        try:
            frame = st.session_state.get("fresh_input", demo_features())
            with st.spinner("Scoring with original, contextual and hard-negative variants…"):
                result, _ = score_lightgbm(frame, chosen)
            st.session_state.classical_comparison = result
            st.success(f"Used {len(chosen)} trained boosters on {len(frame)} accounts.")
        except Exception as exc:
            st.error(str(exc))
    if "classical_comparison" in st.session_state:
        df = st.session_state.classical_comparison
        selected_id = st.selectbox("Inspect classical comparisons for account", df.account_id.astype(str).tolist())
        chosen_row = df.loc[df.account_id.astype(str) == selected_id].iloc[0]
        display = pd.DataFrame({"Model": [x for x in LGBM if x in chosen_row],
                                "Score": [float(chosen_row[x]) for x in LGBM if x in chosen_row]})
        st.plotly_chart(px.bar(display, x="Model", y="Score", range_y=[0, 1]), use_container_width=True)
        st.dataframe(df.head(200), hide_index=True, use_container_width=True)
    st.divider()
    st.subheader("B. Run V4 and V5 neural checkpoints")
    st.caption("V4 supports bundled 130 demo IDs. V5 supports the bundled 130 demo IDs; add the original frozen 1.2M-account inference_inputs/ and predictions/ folders to support the full snapshot.")
    sample_ids = demo_features().account_id.astype(str).head(30).tolist()
    neural_id = st.text_input("Known account ID", value=sample_ids[0])
    run_v4 = st.checkbox("Run V4 classical + quantum weights", value=True)
    run_v5 = st.checkbox("Run all 6 V5 classical + quantum weights", value=True)
    if st.button("Run neural model inference", type="primary"):
        try:
            output = pd.DataFrame({"account_id": [neural_id.strip()]})
            with st.spinner("Running saved neural weights (CPU may take time for V5)…"):
                if run_v4:
                    output = output.merge(score_v4_demo([neural_id.strip()]), on="account_id")
                if run_v5:
                    output = output.merge(infer_known_accounts([neural_id.strip()]), on="account_id")
            st.session_state.neural_comparison = output
            st.success("Neural checkpoint inference passed reproduction checks.")
        except Exception as exc:
            st.error(f"Neural inference failed: {exc}")
    if "neural_comparison" in st.session_state:
        out = st.session_state.neural_comparison
        st.dataframe(out, hide_index=True, use_container_width=True)
        if len(out):
            records = [{"Model / task": col, "Score": float(out.iloc[0][col])}
                       for col in out if col != "account_id"]
            st.plotly_chart(px.bar(pd.DataFrame(records), x="Model / task", y="Score", range_y=[0, 1]), use_container_width=True)
    st.caption("V5: six-qubit, three-layer state-vector circuit; simulated on classical hardware.")

elif page == "LLM Investigator Copilot":
    st.caption("Optional local Ollama; grounded only in available account evidence and scored model outputs.")
    row = select_account(data_view(), "copilot")
    if row is not None:
        st.info("LLM outputs are drafts for human review. They do not provide a diagnosis, regulatory finding, or enforcement decision.")
        with st.expander("LLM connection", expanded=False):
            st.text_input("Ollama base URL", key="llm_url")
            st.text_input("Ollama model", key="llm_model")
            st.code(f"ollama pull {st.session_state.llm_model}\nollama serve", language="bash")
        if st.session_state.source == "Latest inference":
            ev = st.session_state.get("fresh_explanations", pd.DataFrame())
        elif st.session_state.source == "Demo snapshot":
            ev = demo_evidence().rename(columns={"observed_feature_value": "observed_value"})
        else:
            ev = pd.DataFrame()
        if len(ev):
            ev = ev.loc[ev.account_id.astype(str) == str(row.account_id)]
        context = row.to_dict()
        # Include observed aggregate values where the scored upload retains its original feature rows.
        if st.session_state.source == "Latest inference" and "fresh_input" in st.session_state:
            match = st.session_state.fresh_input
            match = match.loc[match.account_id.astype(str) == str(row.account_id)]
            if len(match):
                context.update(match.iloc[0].to_dict())
        elif st.session_state.source == "Demo snapshot":
            match = demo_features()
            match = match.loc[match.account_id.astype(str) == str(row.account_id)]
            if len(match):
                context.update(match.iloc[0].to_dict())
        neural = {}
        if "neural_comparison" in st.session_state:
            neural_row = st.session_state.neural_comparison.iloc[0]
            if str(neural_row.account_id) == str(row.account_id):
                neural = {k: round(float(v), 7) for k, v in neural_row.items() if k != "account_id"}
        packet = evidence_packet(row.account_id, context, contributions=ev, neural=neural)
        with st.expander("Exactly what the LLM receives"):
            st.json(packet)
        question = st.text_area("Investigator question", value="Draft a concise evidence-grounded investigator report for this account.")
        c1, c2 = st.columns([1, 1])
        with c1:
            generate = st.button("Generate with Ollama", type="primary", use_container_width=True)
        with c2:
            deterministic = st.button("Deterministic fallback", use_container_width=True)
        if generate or deterministic:
            key = str(row.account_id)
            try:
                with st.spinner("Generating evidence-grounded case note…"):
                    note = fallback_note(packet) if deterministic else ask_ollama(packet, question,
                        model=st.session_state.llm_model, url=st.session_state.llm_url)
                st.session_state.case_note = (key, note)
            except Exception as exc:
                st.error(f"Ollama unavailable: {exc}. Select the deterministic fallback or start your local server.")
        if st.session_state.get("case_note", (None,))[0] == str(row.account_id):
            note = st.session_state.case_note[1]
            st.markdown(note)
            st.download_button("Download draft case note", note, file_name=f"qlynx_{row.account_id}_note.md", mime="text/markdown")

elif page == "Decisions & Audit":
    st.caption("Human investigator disposition with a local JSONL audit trail. This prototype cannot suspend accounts.")
    row = select_account(data_view(), "audit")
    if row is not None:
        audit_path = ROOT / "audit/decisions.jsonl"
        with st.form("audit_entry", clear_on_submit=False):
            reviewer = st.text_input("Investigator ID")
            action = st.selectbox("Disposition", ["No action / close", "Monitor", "Verify identity", "Investigate fraud",
                                                  "Refer for AML review", "Refer for player protection review", "Recommend restriction review"])
            rationale = st.text_area("Evidence, alternative explanation and rationale")
            policy_review = st.checkbox("Required internal policy review completed (for restriction referral)")
            submit = st.form_submit_button("Record decision", type="primary")
        if submit:
            if not reviewer.strip() or not rationale.strip():
                st.error("Both investigator ID and written rationale are required.")
            elif action == "Recommend restriction review" and not policy_review:
                st.error("Confirm authorization and policy review before recording this high-impact referral.")
            else:
                decision = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "account_id": str(row.account_id),
                            "investigator": reviewer.strip(), "action": action, "rationale": rationale.strip(),
                            "policy_review_confirmed": bool(policy_review), "data_source": st.session_state.source,
                            "prototype_only": True}
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
                st.success("Logged local analyst disposition. No external action was taken.")
        if audit_path.exists():
            history = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            history = pd.DataFrame(history)
            st.dataframe(history.loc[history.account_id.astype(str) == str(row.account_id)], hide_index=True, use_container_width=True)

elif page == "Appearance":
    st.caption("QLYNX follows the Streamlit application theme. The dashboard uses a light neutral card-based layout by default.")
    st.code("""[theme]
base = "light"
primaryColor = "#111b32"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafd"
textColor = "#111b32"
font = "sans serif""", language="toml")
    st.caption("Save these settings in .streamlit/config.toml and restart Streamlit to match the reference screenshot.")

st.divider()
st.caption("QLYNX · Synthetic research prototype. Scores are uncalibrated; quantum execution is simulated; decisions require human review.")
