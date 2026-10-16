"""Risk display and review-priority policy for QLYNX.

Priority is a transparent, demo-only TRIAGE heuristic; model outputs and the
original threshold-based alert flags are not modified or recalibrated.
"""
from __future__ import annotations

import html
import numpy as np
import pandas as pd

RISK_COLORS = {
    "fraud": "#d92d43",        # red
    "aml": "#ed8a23",          # orange
    "protection": "#219c67",   # green
}
RISK_PALE = {
    "fraud": "#fff0f1",
    "aml": "#fff5e9",
    "protection": "#eaf8f1",
}
RISK_NAMES = {
    "fraud": "Fraud", "aml": "Anti-Money Laundering", "protection": "Player Protection"
}
PRIORITY_COLORS = {
    "Critical": ("#6D28D9", "#F3E8FF"),  # deep violet
    "High": ("#BE185D", "#FCE7F3"),      # magenta
    "Review": ("#1D4ED8", "#DBEAFE"),    # royal blue
    "Watch": ("#0E7490", "#CFFAFE"),     # cyan
    "Routine": ("#475569", "#F1F5F9"),   # slate gray
}

def get_score_series(frame: pd.DataFrame, task: str, manifest: dict):
    for column in (f"{task}_score", manifest["operational_models"][task]["score_column"]):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def add_triage(frame: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Create comparable display columns plus priority from *alert thresholds*.

    Critical = multiple active risk categories OR very high score >=0.95.
    High = a single alert with score >=0.70 or with normalized margin >=0.50.
    Review = any remaining alert.
    Watch = no alert but within 80% of a policy threshold in any category.
    Routine = other accounts. These are UI heuristics, not new model labels.
    """
    view = frame.copy()
    n = len(view)
    alert_flags = []
    margin_scores = []
    max_score = np.full(n, -1.0)
    near = np.zeros(n, dtype=bool)
    for task in RISK_COLORS:
        score = get_score_series(view, task, manifest)
        threshold = float(manifest["operational_models"][task]["threshold"])
        # Use model's operational thresholds with exact score column it was trained for.
        triggered = (score >= threshold).fillna(False).to_numpy(dtype=bool)
        view[f"{task}_score"] = score
        view[f"{task}_alert"] = triggered
        alert_flags.append(triggered)
        max_score = np.maximum(max_score, score.fillna(-1.0).to_numpy(dtype=float))
        margin = ((score - threshold) / max(1 - threshold, 1e-10)).clip(lower=0)
        margin_scores.append(margin.fillna(0).to_numpy(dtype=float))
        near |= (score >= threshold * .80).fillna(False).to_numpy(dtype=bool)
    active = np.stack(alert_flags, axis=1).sum(axis=1) if n else np.zeros(0, dtype=int)
    strongest_margin = np.max(np.stack(margin_scores, axis=1), axis=1) if n else np.zeros(0)
    critical = (active >= 2) | ((active > 0) & (max_score >= .95))
    high = (active > 0) & ((max_score >= .70) | (strongest_margin >= .50))
    view["alert_count"] = active
    view["priority"] = np.select([critical, high, active > 0, near], ["Critical", "High", "Review", "Watch"], default="Routine")
    view["_priority_sort"] = np.select([critical, high, active > 0, near], [4, 3, 2, 1], default=0)
    view["_risk_peak"] = np.maximum(0, max_score)
    return view


def task_band(score: float, threshold: float):
    if not np.isfinite(score):
        return "Unavailable"
    if score >= max(.95, threshold):
        return "Critical"
    if score >= threshold and (score >= .70 or (score - threshold) / max(1 - threshold, .001) >= .5):
        return "High"
    if score >= threshold:
        return "Review"
    if score >= threshold * .8:
        return "Watch"
    return "Routine"


def badge(value: str):
    color, pale = PRIORITY_COLORS.get(str(value), ("#5f6c80", "#edf1f6"))
    return f'<span class="ql-pill" style="background:{pale};color:{color};border-color:{color}33">{html.escape(str(value))}</span>'


def task_badge(task: str, text: str | None = None):
    return (f'<span class="ql-task-pill" style="color:{RISK_COLORS[task]};background:{RISK_PALE[task]}">'
            f'<span class="ql-pill-dot" style="background:{RISK_COLORS[task]}"></span>'
            f'{html.escape(str(text or RISK_NAMES[task]))}</span>')


def colored_table(frame: pd.DataFrame, score_task: str | None = None):
    """Style score cells and priorities; compatible with st.dataframe(PandasStyler)."""
    view = frame.copy()
    format_map = {c: "{:.4f}" for c in ["fraud_score", "aml_score", "protection_score", "Risk score"] if c in view}
    styler = view.style.format(format_map, na_rep="—")
    def priority_css(v):
        fg, bg = PRIORITY_COLORS.get(str(v), ("#5f6c80", "#edf1f6"))
        return f"background-color:{bg}; color:{fg}; font-weight:750;"
    for field in ("priority", "Priority", "Overall priority", "Band"):
        if field in view:
            styler = styler.map(priority_css, subset=[field])
    for task, color in RISK_COLORS.items():
        for field in (f"{task}_score", RISK_NAMES[task]):
            if field not in view:
                continue
            def color_cell(v, task=task, color=color):
                try:
                    number = float(v)
                except (ValueError, TypeError):
                    return ""
                # Colored text, not a probability confidence claim.
                return f"color:{color};font-weight:700;" if np.isfinite(number) else ""
            styler = styler.map(color_cell, subset=[field])
    if "Risk score" in view and score_task in RISK_COLORS:
        color = RISK_COLORS[score_task]
        styler = styler.map(lambda _: f"color:{color};font-weight:750;", subset=["Risk score"])
    return styler


def risk_cards_html(flags: dict, n: int):
    subtitles = {"fraud": "Fraud V3 · hard-negative 4×", "aml": "AML Context V2", "protection": "Protection Context V2"}
    cards = []
    for task, flag in flags.items():
        count = int(flag.sum())
        rate = 100 * count / max(n, 1)
        cards.append(f'''
          <div class="ql-risk-card ql-{task}">
             <div class="ql-risk-header"><span class="ql-risk-indicator"></span>{RISK_NAMES[task]}</div>
             <div class="ql-risk-subtitle">{subtitles[task]}</div>
             <div class="ql-risk-number">{count:,}<span> alerts</span></div>
             <div class="ql-risk-track"><div style="width:{rate:.5f}%"></div></div>
             <div class="ql-risk-bottom">{rate:.2f}% of monitored accounts</div>
          </div>''')
    return '<div class="ql-risk-grid">' + ''.join(cards) + '</div>'
