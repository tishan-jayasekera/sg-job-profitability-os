"""
UI Components for the Active Delivery Command Center.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import config
from src.data.job_lifecycle import get_job_task_attribution
from src.data.semantic import get_category_col, safe_quote_job_task
from src.exports import export_action_brief
from src.metrics.client_group_subsidy import compute_client_group_subsidy_context
from src.metrics.delivery_control import compute_root_cause_drivers


RISK_COLORS = {
    "Red": ("#dc3545", "#fff5f5"),
    "Amber": ("#ffc107", "#fffbf0"),
    "Green": ("#28a745", "#f0fff0"),
}
RISK_ICONS = {"Red": "🔴", "Amber": "🟡", "Green": "🟢"}


def inject_delivery_control_theme() -> None:
    _inject_delivery_control_css()


def summarize_alerts(jobs_df: pd.DataFrame) -> Dict[str, float]:
    red_jobs = jobs_df[jobs_df["risk_band"] == "Red"]
    amber_jobs = jobs_df[jobs_df["risk_band"] == "Amber"]
    return {
        "red_count": float(len(red_jobs)),
        "amber_count": float(len(amber_jobs)),
        "margin_at_risk": float(_compute_margin_at_risk(red_jobs)),
    }


def _risk_meta(risk_score: float | None, fallback_band: str = "Green") -> Dict[str, str]:
    if pd.notna(risk_score):
        score = float(risk_score)
        if score >= 80:
            return {
                "label": "Critical",
                "band": "Red",
                "color": "#EF4444",
                "chip_class": "dc-chip-risk-red",
                "dot": "🔴",
            }
        if score >= 50:
            return {
                "label": "Watch",
                "band": "Amber",
                "color": "#F59E0B",
                "chip_class": "dc-chip-risk-amber",
                "dot": "🟡",
            }
        return {
            "label": "Healthy",
            "band": "Green",
            "color": "#22C55E",
            "chip_class": "dc-chip-risk-green",
            "dot": "🟢",
        }

    normalized = str(fallback_band).strip().lower()
    if normalized == "red":
        return {"label": "Critical", "band": "Red", "color": "#EF4444", "chip_class": "dc-chip-risk-red", "dot": "🔴"}
    if normalized == "amber":
        return {"label": "Watch", "band": "Amber", "color": "#F59E0B", "chip_class": "dc-chip-risk-amber", "dot": "🟡"}
    return {"label": "Healthy", "band": "Green", "color": "#22C55E", "chip_class": "dc-chip-risk-green", "dot": "🟢"}


def _inject_delivery_control_css() -> None:
    if st.session_state.get("_dc_css_injected", False):
        return

    st.markdown(
        """
        <style>
        :root {
            --dc-bg: #F8F9FB;
            --dc-card: #FFFFFF;
            --dc-border: #E5E7EB;
            --dc-text: #111827;
            --dc-text-2: #6B7280;
            --dc-text-muted: #9CA3AF;
            --dc-primary: #2563EB;
            --dc-red: #DC2626;
            --dc-amber: #D97706;
            --dc-green: #16A34A;
            --dc-mono: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            --dc-sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        .stApp {
            background: var(--dc-bg);
            font-family: var(--dc-sans);
            color: var(--dc-text);
        }
        .block-container {
            padding-top: 0.55rem;
            padding-bottom: 1.2rem;
            max-width: 1560px;
        }
        h2, h3 {
            letter-spacing: -0.02em;
            color: var(--dc-text);
        }
        div[data-testid="stMarkdownContainer"] p,
        label,
        span {
            font-family: var(--dc-sans);
        }
        .dc-page-title {
            font-size: 28px;
            font-weight: 760;
            color: var(--dc-text);
            letter-spacing: -0.5px;
            margin: 0.15rem 0 0.75rem 0;
        }
        .dc-command-bar {
            background: #FFFFFF;
            border: 1px solid var(--dc-border);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            padding: 0.7rem 0.8rem 0.5rem 0.8rem;
            margin-bottom: 0.65rem;
        }
        .dc-command-divider {
            border-bottom: 1px solid var(--dc-border);
            margin-top: 0.55rem;
        }
        .dc-filter-label {
            font-size: 11px;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: var(--dc-text-muted);
            font-weight: 700;
            margin-bottom: 0.22rem;
        }
        .dc-pill {
            border-radius: 999px;
            border: 1px solid transparent;
            padding: 0.45rem 0.75rem;
            min-height: 60px;
        }
        .dc-pill-title {
            font-size: 10px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--dc-text-muted);
        }
        .dc-pill-value {
            font-size: 18px;
            line-height: 1.15;
            font-weight: 760;
            margin-top: 0.12rem;
            font-variant-numeric: tabular-nums;
            font-family: var(--dc-mono);
        }
        .dc-pill-sub {
            font-size: 11px;
            color: var(--dc-text-2);
            margin-top: 0.12rem;
            font-variant-numeric: tabular-nums;
            font-family: var(--dc-mono);
        }
        .dc-pill-critical {
            background: #FEF2F2;
            border-color: #FECACA;
        }
        .dc-pill-critical .dc-pill-value {
            color: var(--dc-red);
        }
        .dc-pill-watch {
            background: #FFFBEB;
            border-color: #FDE68A;
        }
        .dc-pill-watch .dc-pill-value {
            color: var(--dc-amber);
        }
        .dc-alert-card {
            border-radius: 8px;
            border: 1px solid var(--dc-border);
            padding: 0.6rem 0.75rem;
            background: #FFFFFF;
        }
        .dc-alert-critical {
            background: #FEF2F2;
            border-color: #FECACA;
        }
        .dc-alert-watch {
            background: #FFFBEB;
            border-color: #FDE68A;
        }
        .dc-alert-good {
            background: #F0FDF4;
            border-color: #BBF7D0;
        }
        .dc-alert-value {
            font-size: 18px;
            line-height: 1.15;
            font-weight: 760;
            margin-top: 0.12rem;
            font-family: var(--dc-mono);
        }
        .dc-alert-sub {
            font-size: 11px;
            color: var(--dc-text-2);
            margin-top: 0.15rem;
            font-family: var(--dc-mono);
        }
        .dc-card-shell {
            background: var(--dc-card);
            border: 1px solid var(--dc-border);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            padding: 0.45rem 0.6rem 0.6rem 0.6rem;
        }
        .dc-kicker {
            font-size: 11px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--dc-text-muted);
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .dc-job-header {
            background: #FFFFFF;
            border: 1px solid var(--dc-border);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.85rem;
        }
        .dc-job-id {
            font-size: 11px;
            letter-spacing: 0.09em;
            color: var(--dc-text-muted);
            text-transform: uppercase;
            font-weight: 700;
            font-family: var(--dc-mono);
            margin-bottom: 0.15rem;
        }
        .dc-job-title {
            font-size: 16px;
            font-weight: 760;
            color: var(--dc-text);
            line-height: 1.35;
        }
        .dc-tag-row {
            margin-top: 0.45rem;
        }
        .dc-chip {
            display: inline-block;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            padding: 0.18rem 0.5rem;
            margin-right: 0.32rem;
            margin-bottom: 0.2rem;
            border: 1px solid #D1D5DB;
            background: #F3F4F6;
            color: #374151;
            line-height: 1.15;
        }
        .dc-chip-risk-red {
            background: var(--dc-red);
            border-color: var(--dc-red);
            color: #FFFFFF;
        }
        .dc-chip-risk-amber {
            background: var(--dc-amber);
            border-color: var(--dc-amber);
            color: #FFFFFF;
        }
        .dc-chip-risk-green {
            background: var(--dc-green);
            border-color: var(--dc-green);
            color: #FFFFFF;
        }
        .dc-queue-shell {
            background: #FFFFFF;
            border: 1px solid var(--dc-border);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            padding: 0.5rem 0.55rem 0.45rem 0.55rem;
        }
        .dc-queue-scroll {
            border-top: 1px solid #F1F5F9;
            margin-top: 0.45rem;
            padding-top: 0.45rem;
        }
        .dc-queue-foot {
            margin-top: 0.35rem;
        }
        .dc-queue-subtle {
            color: var(--dc-text-muted);
            font-size: 11px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .dc-risk-dot-red { color: #EF4444; }
        .dc-risk-dot-amber { color: #F59E0B; }
        .dc-risk-dot-green { color: #22C55E; }
        div[data-baseweb="select"] > div {
            border-color: #D1D5DB !important;
            min-height: 34px !important;
            background: #FFFFFF !important;
            box-shadow: none !important;
        }
        div[data-baseweb="select"] svg {
            color: #6B7280 !important;
        }
        div[data-baseweb="radio"] label {
            font-size: 12px !important;
            color: #374151 !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--dc-border);
            background: #FFFFFF;
            border-radius: 8px;
            padding: 0.4rem 0.55rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 10px !important;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--dc-text-muted);
            font-weight: 700;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            color: var(--dc-text);
            font-weight: 760;
            font-variant-numeric: tabular-nums;
            font-family: var(--dc-mono);
            line-height: 1.05;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.8rem !important;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            font-family: var(--dc-mono);
        }
        .dc-lens-caption {
            margin: 0.25rem 0 0.55rem 0;
            padding: 0.35rem 0.5rem;
            border: 1px solid var(--dc-border);
            border-radius: 6px;
            background: #F9FAFB;
            color: #4B5563;
            font-size: 12px;
        }
        .dc-section-divider {
            margin: 0.6rem 0 0.6rem 0;
            border-top: 1px solid var(--dc-border);
        }
        .dc-block {
            border: 1px solid var(--dc-border);
            border-radius: 8px;
            background: #FFFFFF;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            padding: 0.62rem 0.72rem;
            margin-bottom: 0.6rem;
        }
        .dc-footnote {
            font-size: 10px;
            color: #9CA3AF;
            font-style: italic;
            margin-top: 0.28rem;
        }
        [data-testid="stButton"] > button {
            white-space: pre-line;
            text-align: left;
            justify-content: flex-start;
            line-height: 1.25;
            font-family: var(--dc-sans);
            font-size: 12px;
            border-radius: 8px;
        }
        [data-testid="stButton"] > button[kind="secondary"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            color: #111827;
            padding: 0.55rem 0.62rem;
        }
        [data-testid="stButton"] > button[kind="secondary"]:hover {
            background: #F9FAFB;
            border-color: #CBD5E1;
        }
        [data-testid="stButton"] > button[kind="primary"] {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            color: #111827;
            box-shadow: inset 3px 0 0 #2563EB;
            padding: 0.55rem 0.62rem;
        }
        [data-testid="stButton"] > button[kind="tertiary"] {
            border: 1px solid #D1D5DB;
            background: transparent;
            color: #374151;
            justify-content: center;
            text-align: center;
            padding: 0.34rem 0.65rem;
            font-weight: 600;
        }
        [data-testid="stButton"] > button[kind="tertiary"]:hover {
            border-color: #9CA3AF;
            background: #F3F4F6;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--dc-border);
            border-radius: 8px;
            background: #FFFFFF;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        [data-testid="stExpander"] summary {
            font-size: 12px !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6B7280;
            font-weight: 700;
        }
        [data-testid="stCheckbox"] label, [data-testid="stToggle"] label {
            font-size: 12px;
            color: #374151;
        }
        .dc-table-note {
            font-size: 11px;
            color: #9CA3AF;
            margin-top: -0.2rem;
            margin-bottom: 0.25rem;
        }
        [data-testid="stDataFrame"] [role="gridcell"] {
            font-size: 12px !important;
        }
        [data-testid="stDataFrame"] [role="columnheader"] {
            font-size: 11px !important;
            letter-spacing: 0.04em;
            color: #6B7280 !important;
            text-transform: uppercase;
        }
        [data-testid="stVerticalBlock"] > div[style*="overflow: auto"]::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        [data-testid="stVerticalBlock"] > div[style*="overflow: auto"]::-webkit-scrollbar-track {
            background: #F3F4F6;
            border-radius: 8px;
        }
        [data-testid="stVerticalBlock"] > div[style*="overflow: auto"]::-webkit-scrollbar-thumb {
            background: #D1D5DB;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_dc_css_injected"] = True


def render_alert_strip(jobs_df: pd.DataFrame) -> None:
    """
    Render compact alert banner showing portfolio status.

    Design: Two-cell layout
    - Left cell: 🔴 CRITICAL count + $ at risk
    - Right cell: 🟡 WATCH count
    """
    _inject_delivery_control_css()

    red_jobs = jobs_df[jobs_df["risk_band"] == "Red"]
    amber_jobs = jobs_df[jobs_df["risk_band"] == "Amber"]

    margin_at_risk = _compute_margin_at_risk(red_jobs)

    col1, col2 = st.columns(2)

    with col1:
        if len(red_jobs) > 0:
            st.markdown(
                f"""
                <div class="dc-alert-card dc-alert-critical">
                    <div class="dc-kicker">Critical Exposure</div>
                    <div class="dc-alert-value">{len(red_jobs)} Jobs</div>
                    <div class="dc-alert-sub">${margin_at_risk:,.0f} margin at risk</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="dc-alert-card dc-alert-good">
                    <div class="dc-kicker">Critical Exposure</div>
                    <div class="dc-alert-value">0 Jobs</div>
                    <div class="dc-alert-sub">No critical jobs</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        if len(amber_jobs) > 0:
            st.markdown(
                f"""
                <div class="dc-alert-card dc-alert-watch">
                    <div class="dc-kicker">Watchlist</div>
                    <div class="dc-alert-value">{len(amber_jobs)} Jobs</div>
                    <div class="dc-alert-sub">Need attention this cycle</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="dc-alert-card dc-alert-good">
                    <div class="dc-kicker">Watchlist</div>
                    <div class="dc-alert-value">0 Jobs</div>
                    <div class="dc-alert-sub">No jobs on watch</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_job_queue(
    jobs_df: pd.DataFrame,
    job_name_lookup: Dict[str, str],
    include_green: bool = False,
) -> Optional[str]:
    """
    Render compact job queue as selectable cards.

    Returns selected job_no.
    """
    _inject_delivery_control_css()

    with st.container(border=True):
        st.markdown('<div class="dc-kicker">Priority Queue</div>', unsafe_allow_html=True)

        if include_green:
            display_df = jobs_df
        else:
            display_df = jobs_df[jobs_df["risk_band"].isin(["Red", "Amber"])]

        sort_col, _ = st.columns([1.15, 1.0])
        with sort_col:
            sort_option = st.selectbox(
                "Sort by",
                ["Risk Score", "Margin at Risk", "Hours Overrun", "Recent Activity"],
                key="job_queue_sort",
                label_visibility="collapsed",
            )
        display_df = _apply_sort(display_df, sort_option)

        if "job_queue_limit" not in st.session_state:
            st.session_state.job_queue_limit = 10

        display_df = display_df.head(st.session_state.job_queue_limit)
        selected_job = None
        selected_current = st.session_state.get("selected_job")

        with st.container(height=760, border=False):
            for _, row in display_df.iterrows():
                job_no = str(row["job_no"])
                job_name = str(job_name_lookup.get(job_no, job_no))
                issue = str(_format_primary_issue(row))
                action = str(row.get("recommended_action", "Review job status"))
                risk_score = row.get("risk_score", np.nan)
                risk_meta = _risk_meta(risk_score, str(row.get("risk_band", "Green")))
                score_text = f"{float(risk_score):.0f}" if pd.notna(risk_score) else "—"
                runtime_days = row.get("runtime_days", np.nan)
                runtime_text = f"{int(runtime_days)} days" if pd.notna(runtime_days) else "N/A"
                margin_pct = row.get("forecast_margin_pct", np.nan)
                if pd.notna(margin_pct):
                    fill = int(np.clip(round((float(margin_pct) + 20) / 12), 0, 10))
                    bar = "█" * fill + "·" * (10 - fill)
                    bar_text = f"{bar}  {float(margin_pct):+.0f}%"
                else:
                    bar_text = "··········  n/a"
                line1 = f"{risk_meta['dot']}{score_text:>3}  {job_no} — {job_name[:52]}"
                line2 = f"Runtime: {runtime_text}"
                line3 = bar_text
                line4 = issue
                line5 = f"→ {action[:50]}"
                card_text = "\n".join([line1, line2, line3, line4, line5])
                is_selected = selected_current == job_no

                if st.button(
                    card_text,
                    key=f"job_card_{job_no}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    selected_job = job_no
                    st.session_state.selected_job = job_no

        remaining = len(jobs_df) - st.session_state.job_queue_limit
        if remaining > 0:
            spacer_l, more_col, spacer_r = st.columns([1, 1.4, 1])
            with more_col:
                if st.button(
                    f"Show {min(10, remaining)} more",
                    key="job_queue_more",
                    type="tertiary",
                    use_container_width=True,
                ):
                    st.session_state.job_queue_limit += 10

        st.toggle("Include Green jobs", key="include_green_jobs")
        return selected_job or st.session_state.get("selected_job")


def render_selected_job_panel(
    df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    job_no: str,
    job_name_lookup: Dict[str, str],
    df_all: Optional[pd.DataFrame] = None,
) -> None:
    """
    Render the selected job detail panel.
    """
    _inject_delivery_control_css()

    if df_all is None:
        df_all = df

    job_row = jobs_df[jobs_df["job_no"] == job_no].iloc[0]
    job_name = job_row.get("job_name") or job_name_lookup.get(job_no, job_no)

    dept = job_row.get("department_final", "")
    cat = job_row.get("job_category", "")
    risk_meta = _risk_meta(job_row.get("risk_score", np.nan), str(job_row.get("risk_band", "Green")))
    risk_label = risk_meta["label"]
    risk_chip_class = risk_meta["chip_class"]
    st.markdown(
        f"""
        <div class="dc-job-header">
            <div class="dc-job-id">{escape(str(job_no))}</div>
            <div class="dc-job-title">{escape(str(job_name))}</div>
            <div class="dc-tag-row">
                <span class="dc-chip">Delivery : {escape(str(dept))}</span>
                <span class="dc-chip">{escape(str(cat))}</span>
                <span class="dc-chip {risk_chip_class}">{escape(str(risk_label))}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_job_snapshot(job_row)
    _render_client_group_subsidy_context(df_all, jobs_df, job_no)

    drivers = compute_root_cause_drivers(df, job_row)
    _render_why_at_risk(drivers)
    _render_recommended_actions(job_row, drivers)

    with st.expander("▼ Expand Full Diagnosis", expanded=True):
        _render_full_diagnosis(df, job_no, job_row)

    with st.expander("Definitions", expanded=False):
        _render_definitions()

    brief_bytes, brief_name = export_action_brief(job_no, job_row, drivers)
    st.download_button(
        "📤 Export Action Brief",
        data=brief_bytes,
        file_name=brief_name,
        mime="text/markdown",
    )


def _render_job_snapshot(row: pd.Series) -> None:
    """Render compact snapshot box."""
    quoted = row.get("quoted_hours", 0) or 0
    actual = row.get("actual_hours", 0) or 0
    variance_pct = (actual - quoted) / quoted * 100 if quoted > 0 else 0

    margin = row.get("forecast_margin_pct", 0)
    bench = row.get("median_margin_pct", 0)
    delta = margin - bench if pd.notna(margin) and pd.notna(bench) else None

    burn_current = row.get("burn_rate_per_day", np.nan)
    burn_prev = row.get("burn_rate_prev_per_day", np.nan)
    burn_weekly = burn_current * 7 if pd.notna(burn_current) else np.nan
    burn_delta = None
    if pd.notna(burn_current) and pd.notna(burn_prev):
        burn_delta = (burn_current - burn_prev) * 7

    consumed_pct = (actual / quoted * 100) if quoted > 0 else np.nan
    ring_deg = int(np.clip((consumed_pct if pd.notna(consumed_pct) else 0) * 3.6, 0, 360))
    hours_delta_color = "#DC2626" if variance_pct > 0 else "#16A34A"
    hours_delta_arrow = "↑" if variance_pct > 0 else "↓"
    margin_delta_label = f"{delta:+.0f}pp vs bench" if delta is not None else "No benchmark"
    margin_delta_color = "#16A34A" if (delta is not None and delta >= 0) else "#DC2626"
    burn_delta_label = f"{burn_delta:+.0f} hrs/wk" if burn_delta is not None else "No prior window"
    burn_delta_color = "#DC2626" if (burn_delta is not None and burn_delta < 0) else "#16A34A"

    with st.container(border=True):
        st.markdown('<div class="dc-kicker">Snapshot</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="small")

        with c1:
            st.markdown(
                f"""
                <div class="dc-block">
                    <div class="dc-filter-label">Hours</div>
                    <div style="display:flex; align-items:center; gap:0.62rem;">
                        <div style="width:54px;height:54px;border-radius:50%;background:conic-gradient(#2563EB {ring_deg}deg, #E5E7EB {ring_deg}deg);display:flex;align-items:center;justify-content:center;">
                            <div style="width:43px;height:43px;border-radius:50%;background:#fff;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#6B7280;font-family:var(--dc-mono);">
                                {consumed_pct:.0f}%"""
                + """
                            </div>
                        </div>
                        <div style="font-size:30px;line-height:1.0;font-weight:760;color:#111827;font-family:var(--dc-mono);">
                            """
                + f"{actual:.0f} / {quoted:.0f}"
                + """
                        </div>
                    </div>
                    <div style="margin-top:0.3rem;font-size:12px;font-weight:700;color:"""
                + hours_delta_color
                + """;font-family:var(--dc-mono);">"""
                + f"{hours_delta_arrow} {variance_pct:+.0f}%"
                + """</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="dc-block">
                    <div class="dc-filter-label">Margin</div>
                    <div style="font-size:32px;line-height:1.0;font-weight:760;color:#111827;font-family:var(--dc-mono);">
                        {margin:.0f}%
                    </div>
                    <div style="margin-top:0.33rem;font-size:12px;font-weight:700;color:{margin_delta_color};font-family:var(--dc-mono);">
                        {margin_delta_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            burn_label = f"{burn_weekly:.0f} hrs/wk" if pd.notna(burn_weekly) else "—"
            st.markdown(
                f"""
                <div class="dc-block">
                    <div class="dc-filter-label">Burn Rate</div>
                    <div style="font-size:32px;line-height:1.0;font-weight:760;color:#111827;font-family:var(--dc-mono);">
                        {burn_label}
                    </div>
                    <div style="margin-top:0.33rem;font-size:12px;font-weight:700;color:{burn_delta_color};font-family:var(--dc-mono);">
                        {burn_delta_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="dc-section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="dc-footnote">Burn rate = average hours/day over the last 28 days, scaled to weekly hours. Delta compares the prior 28-day window.</div>',
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def _compute_subsidy_context_cached(
    df_all: pd.DataFrame,
    jobs_df: pd.DataFrame,
    selected_job_no: str,
    lookback_months: Optional[int],
    scope: str,
) -> Dict:
    return compute_client_group_subsidy_context(
        df_all=df_all,
        jobs_df=jobs_df,
        selected_job_no=selected_job_no,
        lookback_months=lookback_months,
        scope=scope,
    )


def _fmt_signed_currency(value: float) -> str:
    if pd.isna(value):
        return "—"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _fmt_currency(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"${value:,.0f}"


def _fmt_percent(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.1f}%"


def _render_client_group_subsidy_context(
    df_all: pd.DataFrame,
    jobs_df: pd.DataFrame,
    selected_job_no: str,
) -> None:
    with st.container(border=True):
        st.markdown('<div class="dc-kicker">Client Group Subsidy Lens</div>', unsafe_allow_html=True)

        controls_left, controls_right = st.columns([1, 2])
        with controls_left:
            st.markdown('<div class="dc-filter-label">Window</div>', unsafe_allow_html=True)
            window_label = st.selectbox(
                "Window",
                options=["3m", "6m", "12m", "All"],
                index=2,
                key="dc_subsidy_window",
                label_visibility="collapsed",
            )
        with controls_right:
            st.markdown('<div class="dc-filter-label">Scope</div>', unsafe_allow_html=True)
            scope_label = st.radio(
                "Scope",
                options=["All jobs", "Active jobs only"],
                horizontal=True,
                key="dc_subsidy_scope",
                label_visibility="collapsed",
            )

        window_map = {"3m": 3, "6m": 6, "12m": 12, "All": None}
        scope_map = {"All jobs": "all", "Active jobs only": "active_only"}

        context = _compute_subsidy_context_cached(
            df_all=df_all,
            jobs_df=jobs_df,
            selected_job_no=selected_job_no,
            lookback_months=window_map[window_label],
            scope=scope_map[scope_label],
        )

        status = context.get("status")
        if status == "missing_group_column":
            st.info(
                "Client group field not found (checked: client_group_rev_job_month, client_group_rev_job, client_group, client)."
            )
            return
        if status == "missing_group_value":
            st.warning("Selected job has no usable client-group value in the selected window.")
            return
        if status == "empty_group":
            st.info("No peer jobs available for subsidy analysis in this scope/window.")
            return
        if status != "ok":
            st.info("No peer jobs available for subsidy analysis in this scope/window.")
            return

        summary = context["summary"]
        group_value = context["group_value"]
        group_col = context["group_col"]

        st.markdown(
            (
                "<div class='dc-lens-caption'>"
                f"Group: {group_value}  • Field: {group_col}  • Scope: {scope_label}  • Window: {window_label}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        kpi_1, kpi_2, kpi_3 = st.columns(3)
        with kpi_1:
            st.metric("Selected Job Margin ($)", _fmt_signed_currency(summary["selected_margin"]))
        with kpi_2:
            st.metric("Group Margin ($)", _fmt_signed_currency(summary["group_margin"]))
        with kpi_3:
            group_margin_pct = summary["group_margin_pct"]
            st.metric("Group Margin % (pp)", f"{group_margin_pct:.1f}pp" if pd.notna(group_margin_pct) else "—")
        kpi_4, kpi_5, kpi_6 = st.columns(3)
        with kpi_4:
            coverage_ratio = summary["coverage_ratio"]
            st.metric("Subsidy Coverage (x)", f"{coverage_ratio:.2f}x" if pd.notna(coverage_ratio) else "—")
        with kpi_5:
            st.metric("Subsidizer Jobs (#)", f"{int(summary['subsidizer_job_count'])}")
        with kpi_6:
            st.metric("Active Red/Amber (#)", f"{int(summary['red_job_count'])}/{int(summary['amber_job_count'])}")

        verdict = summary["verdict"]
        selected_loss_abs = summary["selected_loss_abs"]
        positive_pool = summary["positive_peer_margin_pool"]
        coverage_ratio = summary["coverage_ratio"]
        buffer_after_subsidy = summary["buffer_after_subsidy"]

        coverage_text = f"{coverage_ratio:.2f}x" if pd.notna(coverage_ratio) else "—"
        banner_text = (
            f"Selected job loss: {_fmt_currency(selected_loss_abs)}  •  "
            f"Peer margin buffer: {_fmt_currency(positive_pool)}  •  "
            f"Coverage: {coverage_text}  •  "
            f"Net buffer after coverage: {_fmt_signed_currency(buffer_after_subsidy)}"
        )

        if verdict == "No Subsidy Needed":
            st.success(f"✅ **{verdict}**  \n{banner_text}")
        elif verdict == "Fully Subsidized":
            st.success(f"✅ **{verdict}**  \n{banner_text}")
        elif verdict in ["Partially Subsidized", "Weak Subsidy"]:
            st.warning(f"⚠️ **{verdict}**  \n{banner_text}")
        else:
            st.error(f"⚠️ **{verdict}**  \n{banner_text}")

        jobs_table = context["jobs"].copy()
        if len(jobs_table) == 0:
            st.info("No peer jobs available for subsidy analysis in this scope/window.")
            return

        plot_df = jobs_table.copy()
        plot_df["job_name"] = plot_df["job_name"].fillna(plot_df["job_no"])
        plot_df["job_name_short"] = plot_df["job_name"].astype(str).str.slice(0, 36)
        plot_df["label"] = np.where(
            plot_df["is_selected"],
            plot_df["job_no"].astype(str) + " — " + plot_df["job_name_short"] + " (Selected)",
            plot_df["job_no"].astype(str) + " — " + plot_df["job_name_short"],
        )
        plot_df = plot_df.sort_values("margin", ascending=True)

        bar_colors = np.where(plot_df["margin"] >= 0, "#16A34A", "#DC2626")
        line_widths = np.where(plot_df["is_selected"], 3, 0.5)
        line_colors = np.where(plot_df["is_selected"], "#111827", "rgba(17,24,39,0.25)")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=plot_df["label"],
            x=plot_df["margin"],
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color=line_colors, width=line_widths),
            ),
            customdata=np.stack([plot_df["revenue"].values, plot_df["cost"].values], axis=1),
            hovertemplate=(
                "%{y}<br>"
                "Margin: %{x:$,.0f}<br>"
                "Revenue: %{customdata[0]:$,.0f}<br>"
                "Cost: %{customdata[1]:$,.0f}"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            title="Margin Contribution by Job (Client Group)",
            xaxis_title="Margin",
            yaxis_title="",
            height=max(320, 28 * len(plot_df)),
            showlegend=False,
            template="plotly_white",
            margin=dict(l=0, r=6, t=55, b=12),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)", zeroline=True, zerolinewidth=1)
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Interpretation", expanded=False):
            if pd.notna(coverage_ratio) and coverage_ratio >= 1:
                st.markdown("- Coverage ratio is at least 1.0x: low immediate economic risk but watch concentration.")
            elif pd.notna(coverage_ratio) and 0 < coverage_ratio < 1:
                st.markdown("- Coverage ratio is between 0 and 1.0x: partially offset, still value-destructive.")
            elif pd.notna(coverage_ratio) and np.isclose(coverage_ratio, 0.0):
                st.markdown("- Coverage ratio is 0.0x: no portfolio offset.")
            else:
                st.markdown("- Selected job is not currently loss-making in this window, so subsidy is not required.")

            if positive_pool > 0 and pd.notna(summary["subsidy_concentration_pct"]):
                st.markdown(
                    f"- Subsidy concentration: top subsidizer contributes {summary['subsidy_concentration_pct']:.1f}% of the positive margin pool."
                )

        st.markdown("<div class='dc-section-divider'></div>", unsafe_allow_html=True)
        st.markdown("**Peer Jobs in Selected Client Group**")
        table_cols = [
            "is_selected",
            "job_no",
            "job_name",
            "risk_band",
            "revenue",
            "cost",
            "margin",
            "margin_pct",
            "contribution_pct_to_group_margin",
        ]
        table_df = jobs_table[table_cols].sort_values("margin", ascending=False).copy()
        table_df["is_selected"] = np.where(table_df["is_selected"], "★", "")
        table_df["job_name"] = table_df["job_name"].fillna("").astype(str)
        table_df["job_name_full"] = table_df["job_name"]
        table_df["job_name"] = table_df["job_name"].str.slice(0, 44)
        table_df.loc[table_df["job_name_full"].str.len() > 44, "job_name"] = (
            table_df.loc[table_df["job_name_full"].str.len() > 44, "job_name"] + "…"
        )
        table_show = table_df.drop(columns=["job_name_full"])

        def _row_bg(row: pd.Series) -> list[str]:
            if row["is_selected"] == "★":
                bg = "#EFF6FF"
            elif row.name % 2 == 0:
                bg = "#FFFFFF"
            else:
                bg = "#F9FAFB"
            return [f"background-color: {bg}"] * len(row)

        styled = (
            table_show.style
            .format({
                "revenue": "${:,.0f}",
                "cost": "${:,.0f}",
                "margin": "${:,.0f}",
                "margin_pct": "{:.1f}%",
                "contribution_pct_to_group_margin": "{:.1f}%",
            })
            .apply(_row_bg, axis=1)
            .set_properties(
                subset=["revenue", "cost", "margin", "margin_pct", "contribution_pct_to_group_margin"],
                **{"font-family": "var(--dc-mono)", "text-align": "right"},
            )
            .set_properties(subset=["job_no", "job_name", "risk_band"], **{"text-align": "left"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown('<div class="dc-table-note">Selected row is highlighted in blue.</div>', unsafe_allow_html=True)


def _render_why_at_risk(drivers: List[Dict]) -> None:
    """Render ranked risk drivers."""
    with st.container(border=True):
        st.markdown('<div class="dc-kicker">Why At Risk</div>', unsafe_allow_html=True)

        if not drivers:
            st.success("No significant risk drivers")
            return

        for i, driver in enumerate(drivers[:3], 1):
            name = str(driver["driver_name"])
            evidence = str(driver.get("evidence_detail", driver.get("evidence_value", "")))
            evidence_value = str(driver.get("evidence_value", ""))
            severity_dot = "🔴" if i == 1 else ("🟡" if i == 2 else "🟢")
            st.markdown(
                f"""
                <div class="dc-block">
                    <div style="display:flex;justify-content:space-between;align-items:baseline;gap:0.6rem;">
                        <div style="font-weight:760; color:#111827; margin-bottom:0.15rem;">
                            {severity_dot} {i}. {escape(name)}
                        </div>
                        <div style="font-weight:760; color:#DC2626; font-family:var(--dc-mono);">
                            {escape(evidence_value)}
                        </div>
                    </div>
                    <div style="font-size:12px; color:#4B5563;">{escape(evidence)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_recommended_actions(row: pd.Series, drivers: List[Dict]) -> None:
    """Render checkable action items."""
    with st.container(border=True):
        st.markdown('<div class="dc-kicker">Recommended Actions</div>', unsafe_allow_html=True)

        actions = []
        for driver in drivers[:3]:
            action = driver.get("recommendation", "")
            if action:
                actions.append(action)

        if not actions:
            actions.append("Review job status with PM")

        for i, action in enumerate(actions):
            st.markdown('<div class="dc-block" style="margin-bottom:0.45rem;">', unsafe_allow_html=True)
            st.checkbox(f"🛠 {action}", key=f"action_{row['job_no']}_{i}")
            st.markdown("</div>", unsafe_allow_html=True)


def _render_full_diagnosis(df: pd.DataFrame, job_no: str, job_row: pd.Series) -> None:
    """
    Render full diagnosis panel (expandable).

    Includes:
    - Task consumption vs quote with benchmark overlay
    - Staff contribution by task with margin-erosion context
    """
    st.markdown('<div class="dc-kicker">Task Consumption vs Quote</div>', unsafe_allow_html=True)
    task_df = get_job_task_attribution(df, job_no)
    if len(task_df) > 0:
        task_df = task_df.sort_values("variance", ascending=False)
        task_plot = task_df.head(10).copy()

        bench = _compute_task_benchmarks(df, job_row)
        task_plot = task_plot.merge(bench, on="task_name", how="left")
        fte = _compute_task_fte_equiv(df, job_no)
        task_plot = task_plot.merge(fte, on="task_name", how="left")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Quoted",
            y=task_plot["task_name"],
            x=task_plot["quoted_hours"],
            orientation="h",
            marker_color="#6c757d",
        ))
        fte_labels = [
            f"{value:.1f} FTE" if pd.notna(value) and value > 0 else ""
            for value in task_plot.get("fte_equiv", pd.Series(dtype=float))
        ]
        fig.add_trace(go.Bar(
            name="Actual",
            y=task_plot["task_name"],
            x=task_plot["actual_hours"],
            orientation="h",
            marker_color=[
                "#dc3545" if a > q else "#28a745"
                for a, q in zip(task_plot["actual_hours"], task_plot["quoted_hours"])
            ],
            text=fte_labels,
            textposition="inside",
            textfont=dict(color="white"),
        ))

        bench_points = pd.DataFrame()
        if "bench_actual_hours" in task_plot.columns:
            bench_points = task_plot[task_plot["bench_actual_hours"].notna()]
        if len(bench_points) > 0:
            fig.add_trace(go.Scatter(
                name="Benchmark (median completed)",
                y=bench_points["task_name"],
                x=bench_points["bench_actual_hours"],
                mode="markers",
                marker=dict(color="#111827", size=8),
            ))

        fig.update_layout(
            barmode="group",
            height=max(260, len(task_plot) * 30),
            xaxis_title="Hours",
            yaxis_title="",
            template="plotly_white",
            legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
            margin=dict(l=0, r=8, t=10, b=10),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#374151"),
        )
        fig.update_xaxes(showgrid=False, tickfont=dict(size=12, color="#6B7280"), title_font=dict(size=12, color="#6B7280"))
        fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color="#6B7280"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Red bars indicate tasks running over quoted hours. Benchmarks reflect median actual hours on completed "
            "jobs in the same department/category. Labels on actual bars show FTE-equivalent effort."
        )
    else:
        st.caption("No task data available.")

    st.markdown('<div class="dc-kicker">Task Contribution by Staff (Overrun Focus)</div>', unsafe_allow_html=True)
    staff_task_df = _compute_task_staff_contribution(df, job_no)
    if len(staff_task_df) > 0:
        total_erosion = staff_task_df["erosion_value"].sum()
        staff_task_df["erosion_pct_total"] = (
            staff_task_df["erosion_value"] / total_erosion * 100 if total_erosion > 0 else 0
        )

        def _top_contributors(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("actual_hours", ascending=False).copy()
            group["cum_share"] = group["task_share"].cumsum()
            keep = group["cum_share"] <= 0.8
            if len(group) > 0 and not keep.any():
                keep.iloc[0] = True
            if len(group) > 0 and keep.any():
                first_over = group.index[group["cum_share"] > 0.8]
                if len(first_over) > 0:
                    keep.loc[first_over[0]] = True
            return group[keep]

        focused = staff_task_df.groupby("task_name", group_keys=False).apply(_top_contributors)

        task_meta = staff_task_df.groupby("task_name").agg(
            task_cost=("task_cost", "sum"),
            quoted_hours=("task_quoted_hours", "first"),
        ).reset_index()
        task_meta = task_meta.set_index("task_name")

        task_labels = {
            task: (
                f"{task} • ${task_meta.loc[task, 'task_cost']:,.0f}"
                if task in task_meta.index and pd.notna(task_meta.loc[task, "task_cost"])
                else task
            )
            for task in focused["task_name"].unique()
        }
        focused["task_label"] = focused["task_name"].map(task_labels)

        fig_task = go.Figure()
        for staff_name in focused["staff_name"].unique():
            subset = focused[focused["staff_name"] == staff_name]
            text_labels = [
                f"{staff_name} ({share:.0f}%)" if share >= 10 else ""
                for share in subset["task_share"] * 100
            ]
            fig_task.add_trace(go.Bar(
                name=staff_name,
                y=subset["task_label"],
                x=subset["actual_hours"],
                orientation="h",
                text=text_labels,
                textposition="inside",
                textfont=dict(color="white"),
                customdata=subset["erosion_pct_total"],
                hovertemplate=(
                    "%{y}<br>%{fullData.name}<br>"
                    "Hours: %{x:.0f}<br>"
                    "Task Share: %{customdata:.1f}% of total erosion"
                    "<extra></extra>"
                ),
            ))

        quoted_points = task_meta.reset_index()
        if "quoted_hours" in quoted_points.columns:
            quoted_points["task_label"] = quoted_points["task_name"].map(task_labels)
            fig_task.add_trace(go.Scatter(
                name="Quoted Hours (target)",
                y=quoted_points["task_label"],
                x=quoted_points["quoted_hours"],
                mode="markers+text",
                marker=dict(color="#111827", size=11, symbol="diamond", line=dict(color="white", width=1)),
                text=[f"{val:.0f}h" if pd.notna(val) else "" for val in quoted_points["quoted_hours"]],
                textposition="middle right",
                textfont=dict(color="#111827", size=10),
            ))

        fig_task.update_layout(
            barmode="stack",
            height=max(260, len(task_meta) * 30),
            xaxis_title="Hours",
            yaxis_title="",
            template="plotly_white",
            legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
            margin=dict(l=0, r=8, t=10, b=10),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#374151"),
        )
        fig_task.update_xaxes(showgrid=False, tickfont=dict(size=12, color="#6B7280"), title_font=dict(size=12, color="#6B7280"))
        fig_task.update_yaxes(showgrid=False, tickfont=dict(size=12, color="#6B7280"))
        st.plotly_chart(fig_task, use_container_width=True)
        st.caption(
            "Each task shows the staff who contribute to at least ~80% of its hours. "
            "Quoted hours are marked with an X, and task labels include total cost."
        )
    else:
        st.caption("No staff-task contribution data available.")



def _compute_margin_at_risk(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return df.get("margin_at_risk", pd.Series(dtype=float)).fillna(0).sum()


def _apply_sort(df: pd.DataFrame, sort_option: str) -> pd.DataFrame:
    if len(df) == 0:
        return df

    if sort_option == "Margin at Risk":
        return df.sort_values("margin_at_risk", ascending=False)
    if sort_option == "Hours Overrun":
        return df.sort_values("hours_overrun", ascending=False)
    if sort_option == "Recent Activity":
        if "last_activity" in df.columns:
            return df.sort_values("last_activity", ascending=False)
        return df

    return df.sort_values("risk_score", ascending=False)


def _format_primary_issue(row: pd.Series) -> str:
    """Format primary issue with specific numbers."""
    driver = row.get("primary_driver", "")

    if "scope" in driver.lower() or "hours overrun" in driver.lower():
        pct = row.get("scope_creep_pct")
        if pd.isna(pct):
            pct = row.get("hours_variance_pct", 0)
        return f"Scope/hours +{pct:.0f}% over quote"

    if "rate" in driver.lower():
        rate_var = row.get("rate_variance", 0)
        return f"Rate leakage ${rate_var:.0f}/hr"

    if "margin" in driver.lower():
        margin = row.get("forecast_margin_pct", 0)
        bench = row.get("median_margin_pct", 0)
        return f"Margin {margin:.0f}% (bench: {bench:.0f}%)"

    if "running" in driver.lower() or "runtime" in driver.lower():
        delta = row.get("runtime_delta_days", 0)
        return f"Running {delta:.0f} days over benchmark"

    return driver or "Monitor"


def _compute_task_benchmarks(df: pd.DataFrame, job_row: pd.Series) -> pd.DataFrame:
    category_col = get_category_col(df)
    dept = job_row.get("department_final")
    cat = job_row.get("job_category")

    df_completed = df.copy()
    if "job_completed_date" in df_completed.columns:
        df_completed = df_completed[df_completed["job_completed_date"].notna()]
    elif "job_status" in df_completed.columns:
        df_completed = df_completed[df_completed["job_status"].str.lower().str.contains("completed", na=False)]
    else:
        df_completed = df_completed.iloc[0:0]

    if dept:
        df_completed = df_completed[df_completed["department_final"] == dept]
    if cat and category_col in df_completed.columns:
        df_completed = df_completed[df_completed[category_col] == cat]

    if len(df_completed) == 0:
        return pd.DataFrame(columns=["task_name", "bench_actual_hours"])

    actuals = df_completed.groupby(["job_no", "task_name"])["hours_raw"].sum().reset_index()
    bench_actual = actuals.groupby("task_name")["hours_raw"].median().reset_index()
    bench_actual = bench_actual.rename(columns={"hours_raw": "bench_actual_hours"})

    return bench_actual


def _compute_task_fte_equiv(df: pd.DataFrame, job_no: str) -> pd.DataFrame:
    job_df = df[df["job_no"] == job_no].copy()
    if len(job_df) == 0:
        return pd.DataFrame(columns=["task_name", "fte_equiv"])

    if "fte_hours_scaling" in job_df.columns:
        scaling = job_df["fte_hours_scaling"].fillna(config.DEFAULT_FTE_SCALING)
    else:
        scaling = config.DEFAULT_FTE_SCALING

    denom = config.CAPACITY_HOURS_PER_WEEK * scaling
    denom = denom.replace(0, np.nan) if isinstance(denom, pd.Series) else denom
    job_df["fte_equiv"] = job_df["hours_raw"] / denom
    job_df["fte_equiv"] = job_df["fte_equiv"].replace([np.inf, -np.inf], np.nan)

    fte_by_task = job_df.groupby("task_name")["fte_equiv"].sum().reset_index()
    return fte_by_task


def _compute_task_staff_contribution(df: pd.DataFrame, job_no: str) -> pd.DataFrame:
    job_df = df[df["job_no"] == job_no].copy()
    if len(job_df) == 0:
        return pd.DataFrame()

    staff_task = job_df.groupby(["task_name", "staff_name"]).agg(
        actual_hours=("hours_raw", "sum"),
        task_cost=("base_cost", "sum") if "base_cost" in job_df.columns else ("hours_raw", "sum"),
    ).reset_index()
    staff_task["task_total_hours"] = staff_task.groupby("task_name")["actual_hours"].transform("sum")
    staff_task["task_share"] = np.where(
        staff_task["task_total_hours"] > 0,
        staff_task["actual_hours"] / staff_task["task_total_hours"],
        0,
    )

    job_task_quote = safe_quote_job_task(job_df)
    if len(job_task_quote) == 0 or "quoted_time_total" not in job_task_quote.columns:
        staff_task["task_quoted_hours"] = 0.0
        staff_task["quoted_alloc"] = 0.0
        staff_task["overrun_hours"] = staff_task["actual_hours"]
        staff_task["quote_rate"] = 0.0
    else:
        task_totals = job_df.groupby("task_name")["hours_raw"].sum().reset_index()
        task_totals = task_totals.rename(columns={"hours_raw": "task_actual_hours"})
        staff_task = staff_task.merge(task_totals, on="task_name", how="left")
        quote_cols = ["task_name", "quoted_time_total"]
        if "quoted_amount_total" in job_task_quote.columns:
            quote_cols.append("quoted_amount_total")
        staff_task = staff_task.merge(
            job_task_quote[quote_cols],
            on="task_name",
            how="left",
        )
        staff_task["quoted_time_total"] = staff_task["quoted_time_total"].fillna(0)
        if "quoted_amount_total" not in staff_task.columns:
            staff_task["quoted_amount_total"] = 0
        staff_task["quoted_amount_total"] = staff_task["quoted_amount_total"].fillna(0)
        staff_task["task_actual_hours"] = staff_task["task_actual_hours"].replace(0, np.nan)
        staff_task["quoted_alloc"] = (
            staff_task["actual_hours"] / staff_task["task_actual_hours"]
        ) * staff_task["quoted_time_total"]
        staff_task["quoted_alloc"] = staff_task["quoted_alloc"].fillna(0)
        staff_task["overrun_hours"] = staff_task["actual_hours"] - staff_task["quoted_alloc"]
        staff_task["task_quoted_hours"] = staff_task["quoted_time_total"]
        staff_task["quote_rate"] = np.where(
            staff_task["quoted_time_total"] > 0,
            staff_task["quoted_amount_total"] / staff_task["quoted_time_total"],
            0,
        )

    staff_task["erosion_value"] = (
        staff_task["overrun_hours"].clip(lower=0) * staff_task["quote_rate"]
    )

    return staff_task


def _render_definitions() -> None:
    st.markdown("**Burn Rate:** Average hours per day over the last 28 days, shown as hours per week.")
    st.markdown("**Margin at Risk:** Benchmark margin minus forecast margin, applied to forecast revenue.")
    st.markdown("**Scope Creep:** Share of hours on tasks not matched to a quote.")
    st.markdown("**Hours Overrun:** EAC hours minus quoted hours (positive indicates overrun).")
    st.markdown("**Risk Score:** Composite score (0–100) from margin, revenue lag, hours overrun, rate leakage, and runtime.")
