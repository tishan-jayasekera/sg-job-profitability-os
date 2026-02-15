"""
Delivery Control Tower UI Components.

Provides the four functions imported by pages/4_Active_Delivery.py:
- inject_delivery_control_theme()
- summarize_alerts()
- render_job_queue()
- render_selected_job_panel()
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.job_lifecycle import (
    compute_task_level_benchmarks,
    get_job_staff_attribution,
    get_job_task_attribution,
)
from src.metrics.client_group_subsidy import (
    compute_client_group_subsidy_context,
    resolve_group_column,
)
from src.metrics.delivery_control import compute_root_cause_drivers
from src.ui.completion_forecast_components import render_completion_forecast_section
from src.ui.formatting import fmt_count, fmt_currency, fmt_hours, fmt_percent, fmt_rate


def inject_delivery_control_theme() -> None:
    """Inject CSS for the delivery control tower page."""
    st.markdown(
        """
        <style>
        :root {
            --dc-border: #e4e7ec;
            --dc-muted: #667085;
            --dc-ink: #1f2937;
            --dc-selected: #2563eb;
        }
        .dc-page-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 0.5rem;
        }
        .dc-filter-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: #555;
            margin-bottom: 2px;
        }
        .dc-pill {
            padding: 10px 14px;
            border-radius: 8px;
            text-align: center;
        }
        .dc-pill-critical {
            background: #fce4e4;
            border: 1px solid #f5a5a5;
        }
        .dc-pill-watch {
            background: #fff8e1;
            border: 1px solid #ffe082;
        }
        .dc-pill-title {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #666;
        }
        .dc-pill-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1a1a2e;
        }
        .dc-pill-sub {
            font-size: 0.72rem;
            color: #888;
        }
        .dc-command-divider {
            height: 1px;
            background: #e0e0e0;
            margin: 8px 0 0 0;
        }
        .dc-job-header {
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 16px;
        }
        .dc-job-header-title {
            font-size: 1.1em;
            font-weight: 600;
            color: #1a1a2e;
        }
        .dc-job-header-sub {
            font-size: 0.85em;
            color: #555;
            margin-top: 4px;
        }
        .dc-section-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--dc-muted);
            margin: 0.2rem 0 0.45rem 0;
        }
        .dc-section-title {
            font-size: 0.95rem;
            font-weight: 650;
            color: #111827;
            margin: 0 0 0.5rem 0;
        }
        .dc-soft-divider {
            height: 1px;
            background: linear-gradient(90deg, #e2e8f0 0%, rgba(226, 232, 240, 0.25) 100%);
            margin: 0.7rem 0;
        }
        .dc-queue-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 0.32rem;
            margin: 0 0 0.45rem 0;
        }
        .dc-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 0.15rem 0.5rem;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        .dc-chip-red {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        .dc-chip-amber {
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #fde68a;
        }
        .dc-chip-green {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }

        /* KPI sizing: reduce oversized metric numerals and improve hierarchy */
        div[data-testid="metric-container"] {
            border: 1px solid var(--dc-border);
            border-radius: 10px;
            padding: 0.55rem 0.7rem;
            background: #fff;
        }
        div[data-testid="metric-container"] > label {
            font-size: 0.74rem !important;
            letter-spacing: 0.01em;
            color: var(--dc-muted) !important;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.95rem !important;
            line-height: 1.08 !important;
            color: var(--dc-ink);
            font-weight: 650;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.88rem !important;
        }

        /* Queue readability and selected-item emphasis */
        .dc-queue-title {
            font-size: 0.8rem;
            color: var(--dc-muted);
            font-weight: 600;
            letter-spacing: 0.02em;
            margin: 0.1rem 0 0.35rem 0;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 0.38rem;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] {
            margin: 0 !important;
            padding: 0.48rem 0.58rem;
            border: 1px solid var(--dc-border);
            border-radius: 10px;
            background: #fff;
            align-items: flex-start;
            transition: background 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child {
            font-size: 0.91rem;
            line-height: 1.35;
            color: var(--dc-ink);
            font-weight: 500;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
            border-color: #c9d3e2;
            background: #f9fbff;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            border-color: var(--dc-selected);
            background: #eef5ff;
            box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.18);
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:last-child {
            color: #0f172a;
            font-weight: 650;
        }
        details[data-testid="stExpander"] {
            border: 1px solid var(--dc-border);
            border-radius: 10px;
            background: #fff;
            overflow: hidden;
        }
        details[data-testid="stExpander"] > summary {
            background: #f8fafc;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.3rem;
            border-bottom: 1px solid var(--dc-border);
            margin-bottom: 0.4rem;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            height: 34px;
            padding: 0 0.9rem;
            border-radius: 8px 8px 0 0;
            color: var(--dc-muted);
            font-weight: 600;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            color: #1d4ed8;
            background: #eff6ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def summarize_alerts(jobs_df: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize alert counts from the jobs dataframe.

    Returns dict with keys: red_count, amber_count, margin_at_risk
    """
    if len(jobs_df) == 0:
        return {"red_count": 0, "amber_count": 0, "margin_at_risk": 0}

    red_count = int((jobs_df["risk_band"] == "Red").sum()) if "risk_band" in jobs_df.columns else 0
    amber_count = int((jobs_df["risk_band"] == "Amber").sum()) if "risk_band" in jobs_df.columns else 0
    margin_at_risk = float(jobs_df["margin_at_risk"].sum()) if "margin_at_risk" in jobs_df.columns else 0

    return {
        "red_count": red_count,
        "amber_count": amber_count,
        "margin_at_risk": margin_at_risk if pd.notna(margin_at_risk) else 0,
    }


def _build_queue_action_export(
    display_df: pd.DataFrame,
    job_name_lookup: Dict[str, str],
) -> pd.DataFrame:
    """Build an export-ready diagnosis/action list for the visible queue jobs."""
    if len(display_df) == 0:
        return pd.DataFrame(
            columns=[
                "job_no",
                "job_name",
                "risk_band",
                "risk_score",
                "diagnosis",
                "action_list",
                "recommended_action",
                "margin_to_date_pct",
                "forecast_margin_pct",
                "quote_consumed_pct",
                "hours_overrun",
                "burn_rate_per_day",
                "runtime_days",
            ]
        )

    export = display_df.copy()
    export["job_no"] = export["job_no"].astype(str)
    export["job_name"] = export["job_no"].map(lambda j: job_name_lookup.get(str(j), "")).fillna("")
    if "risk_band" not in export.columns:
        export["risk_band"] = "Unknown"
    export["risk_band"] = export["risk_band"].astype(str)
    export["risk_score"] = pd.to_numeric(export.get("risk_score"), errors="coerce")
    export["primary_driver"] = export.get("primary_driver", "").fillna("").astype(str)
    export["recommended_action"] = export.get("recommended_action", "").fillna("").astype(str)

    export["diagnosis"] = np.where(
        export["primary_driver"].str.strip().ne(""),
        export["primary_driver"],
        "Monitor",
    )

    def _action_steps(row: pd.Series) -> str:
        base = str(row.get("recommended_action", "") or "").strip()
        band = str(row.get("risk_band", "Unknown"))
        if band == "Red":
            suffix = "Triage today; assign owner + due date; track daily."
        elif band == "Amber":
            suffix = "Review this week; lock remediation plan; track twice weekly."
        else:
            suffix = "Continue standard review cadence."
        return f"{base} {suffix}".strip()

    export["action_list"] = export.apply(_action_steps, axis=1)

    export["margin_to_date_pct"] = pd.to_numeric(export.get("margin_pct_to_date"), errors="coerce").round(1)
    export["forecast_margin_pct"] = pd.to_numeric(export.get("forecast_margin_pct"), errors="coerce").round(1)
    export["quote_consumed_pct"] = pd.to_numeric(export.get("pct_consumed"), errors="coerce").round(1)
    export["hours_overrun"] = pd.to_numeric(export.get("hours_overrun"), errors="coerce").round(1)
    export["burn_rate_per_day"] = pd.to_numeric(export.get("burn_rate_per_day"), errors="coerce").round(2)
    export["runtime_days"] = pd.to_numeric(export.get("runtime_days"), errors="coerce").round(0)

    risk_order = {"Red": 0, "Amber": 1, "Green": 2}
    export["_risk_rank"] = export["risk_band"].map(risk_order).fillna(99)

    columns = [
        "job_no",
        "job_name",
        "risk_band",
        "risk_score",
        "diagnosis",
        "action_list",
        "recommended_action",
        "margin_to_date_pct",
        "forecast_margin_pct",
        "quote_consumed_pct",
        "hours_overrun",
        "burn_rate_per_day",
        "runtime_days",
    ]
    keep = [c for c in columns if c in export.columns]
    export = export.sort_values(["_risk_rank", "risk_score"], ascending=[True, False], na_position="last")
    export = export[keep]
    return export


def render_job_queue(
    jobs_df: pd.DataFrame,
    job_name_lookup: Dict[str, str],
    include_green: bool = False,
) -> Optional[str]:
    """Render the left-panel priority queue and return selected job_no."""
    if len(jobs_df) == 0:
        st.info("No jobs matching current filters.")
        return None

    if "include_green_jobs" not in st.session_state:
        st.session_state["include_green_jobs"] = include_green
    include_green_jobs = st.checkbox("Include on-track jobs", key="include_green_jobs")

    display_df = jobs_df if include_green_jobs else jobs_df[jobs_df["risk_band"] != "Green"]
    if len(display_df) == 0:
        st.info("No jobs matching current filters.")
        return None

    st.markdown(
        f'<div class="dc-queue-title">Priority Queue ({len(display_df)} jobs)</div>',
        unsafe_allow_html=True,
    )
    risk_counts = display_df["risk_band"].astype(str).value_counts()
    st.markdown(
        "<div class=\"dc-queue-stats\">"
        f"<span class=\"dc-chip dc-chip-red\">{int(risk_counts.get('Red', 0))} Red</span>"
        f"<span class=\"dc-chip dc-chip-amber\">{int(risk_counts.get('Amber', 0))} Amber</span>"
        f"<span class=\"dc-chip dc-chip-green\">{int(risk_counts.get('Green', 0))} Green</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    action_export_df = _build_queue_action_export(display_df, job_name_lookup)
    if len(action_export_df) > 0:
        csv_bytes = action_export_df.to_csv(index=False).encode("utf-8")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "Export Action List (CSV)",
            data=csv_bytes,
            file_name=f"active_delivery_action_list_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Exports diagnosis and action steps for all jobs currently shown in this queue view.",
        )

    def _truncate_text(value: str, max_len: int = 72) -> str:
        text = " ".join(str(value).split())
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "…"

    current_idx = 0
    current_selected = st.session_state.get("selected_job")
    current_selected_str = str(current_selected) if current_selected is not None else None

    risk_emoji = {"Red": "🔴", "Amber": "🟡", "Green": "🟢"}
    options = []
    job_nos = []
    for _, row in display_df.iterrows():
        job_no = str(row.get("job_no", ""))
        risk_band = str(row.get("risk_band", "") or "")
        name = _truncate_text(str(job_name_lookup.get(job_no, "") or ""), max_len=64)
        badge = risk_emoji.get(risk_band, "⚪")
        label = f"{badge} {job_no} [{risk_band}]"
        if name:
            label += f" — {name}"

        meta_parts = []
        driver = str(row.get("primary_driver", "") or "")
        if driver and driver != "Monitor":
            meta_parts.append(_truncate_text(driver, max_len=40))
        margin_to_date = pd.to_numeric(row.get("margin_pct_to_date"), errors="coerce")
        if pd.notna(margin_to_date):
            meta_parts.append(f"MTD {margin_to_date:.1f}%")
        if meta_parts:
            label += f"  ·  {' | '.join(meta_parts[:2])}"

        if current_selected_str == job_no:
            label = f"→ {label}"

        options.append(label)
        job_nos.append(job_no)

    if current_selected_str in job_nos:
        current_idx = job_nos.index(current_selected_str)

    if "job_queue_radio" in st.session_state:
        stored_idx = st.session_state.get("job_queue_radio")
        if not isinstance(stored_idx, int) or stored_idx < 0 or stored_idx >= len(options):
            st.session_state["job_queue_radio"] = current_idx

    selected_idx = st.radio(
        "Priority Queue",
        options=list(range(len(options))),
        format_func=lambda i: options[i],
        index=current_idx,
        key="job_queue_radio",
        label_visibility="collapsed",
    )

    selected_job = str(job_nos[selected_idx])
    st.session_state.selected_job = selected_job
    return selected_job


def render_selected_job_panel(
    df_scope: pd.DataFrame,
    jobs_df: pd.DataFrame,
    selected_job: str,
    job_name_lookup: Dict[str, str],
    df_all: Optional[pd.DataFrame] = None,
) -> None:
    """Render the right-panel details for the selected job."""
    base_df = df_all if df_all is not None else df_scope

    job_row = jobs_df[jobs_df["job_no"].astype(str) == str(selected_job)]
    if len(job_row) == 0:
        st.warning(f"Job {selected_job} not found in current dataset.")
        return
    job_row = job_row.iloc[0]

    job_name = job_name_lookup.get(str(selected_job), "")
    risk_band = str(job_row.get("risk_band", "Unknown"))
    risk_colors = {"Red": "#dc3545", "Amber": "#ffc107", "Green": "#28a745"}
    band_color = risk_colors.get(risk_band, "#6c757d")

    st.markdown(
        f'<div class="dc-job-header" style="background:{band_color}15; border-left:4px solid {band_color};">'
        f'<div class="dc-job-header-title">{selected_job} — {job_name if job_name else "Unnamed"}</div>'
        f'<div class="dc-job-header-sub">Risk: <strong style="color:{band_color}">{risk_band}</strong>'
        f' &nbsp;|&nbsp; {job_row.get("primary_driver", "Monitor")}'
        f' &nbsp;|&nbsp; {job_row.get("recommended_action", "")}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    margin_to_date = pd.to_numeric(job_row.get("margin_pct_to_date"), errors="coerce")
    pct_consumed = pd.to_numeric(job_row.get("pct_consumed"), errors="coerce")
    hours_overrun = pd.to_numeric(job_row.get("hours_overrun"), errors="coerce")
    margin_at_risk = pd.to_numeric(job_row.get("margin_at_risk"), errors="coerce")
    mar_confidence = str(job_row.get("margin_at_risk_confidence", "")).lower()

    st.markdown('<div class="dc-section-label">Job Health Snapshot</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Margin-to-Date %",
            f"{margin_to_date:.1f}%" if pd.notna(margin_to_date) else "—",
            help="Actual (revenue - cost) / revenue to date",
        )
    with k2:
        st.metric(
            "Quote Consumed",
            f"{pct_consumed:.0f}%" if pd.notna(pct_consumed) else "—",
            help="Actual hours / quoted hours",
        )
    with k3:
        st.metric(
            "Hours Overrun",
            f"{hours_overrun:.0f} hrs" if pd.notna(hours_overrun) else "—",
            help="Hours over quote (0 if under)",
        )
    with k4:
        if mar_confidence == "low":
            mar_display = "N/A (early stage)"
        elif pd.notna(margin_at_risk):
            mar_display = f"${margin_at_risk:,.0f}"
            if mar_confidence == "medium":
                mar_display += " (est.)"
        else:
            mar_display = "—"
        st.metric(
            "Margin at Risk",
            mar_display,
            help="Estimated margin gap vs peer benchmark",
        )

    burn_rate = pd.to_numeric(job_row.get("burn_rate_per_day"), errors="coerce")
    burn_prev = pd.to_numeric(job_row.get("burn_rate_prev_per_day"), errors="coerce")
    burn_delta = burn_rate - burn_prev if pd.notna(burn_rate) and pd.notna(burn_prev) else None

    runtime_days = pd.to_numeric(job_row.get("runtime_days"), errors="coerce")
    runtime_delta = pd.to_numeric(job_row.get("runtime_delta_days"), errors="coerce")

    last_activity = pd.to_datetime(job_row.get("last_activity"), errors="coerce")
    last_activity_str = last_activity.strftime("%Y-%m-%d") if pd.notna(last_activity) else "—"

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric(
            "Burn Rate",
            f"{burn_rate:.1f} hrs/day" if pd.notna(burn_rate) else "—",
            delta=f"{burn_delta:+.1f} hrs/day" if burn_delta is not None else None,
            delta_color="inverse",
        )
    with d2:
        st.metric(
            "Runtime",
            f"{runtime_days:.0f} days" if pd.notna(runtime_days) else "—",
            delta=f"{runtime_delta:+.0f} days vs peer median" if pd.notna(runtime_delta) else None,
        )
    with d3:
        st.metric("Last Activity", last_activity_str)

    dept = job_row.get("department_final")
    cat = job_row.get("job_category")
    peers = jobs_df[
        (jobs_df["department_final"] == dept)
        & (jobs_df["job_category"] == cat)
    ].copy()
    peer_count = len(peers)

    st.markdown('<div class="dc-soft-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dc-section-label">Peer Benchmarking</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="dc-section-title">Active Peers ({peer_count} jobs in {dept} / {cat})</div>',
        unsafe_allow_html=True,
    )
    if peer_count < 2:
        st.caption("Insufficient active peers for comparison in this category.")
    else:
        peers_other = peers[peers["job_no"].astype(str) != str(selected_job)]
        active_peer_median = pd.to_numeric(peers_other["margin_pct_to_date"], errors="coerce").median()
        historical_benchmark = pd.to_numeric(job_row.get("median_margin_pct"), errors="coerce")

        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric(
                "This Job Margin %",
                f"{margin_to_date:.1f}%" if pd.notna(margin_to_date) else "—",
                help="Actual margin to date for this job",
            )
        with p2:
            st.metric(
                "Active Peer Median %",
                f"{active_peer_median:.1f}%" if pd.notna(active_peer_median) else "—",
                help="Median margin across active peers in same category",
            )
        with p3:
            st.metric(
                "Historical Benchmark %",
                f"{historical_benchmark:.1f}%" if pd.notna(historical_benchmark) else "—",
                help="Median final margin of completed jobs in same category",
            )

    st.markdown('<div class="dc-soft-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dc-section-label">Diagnosis</div>', unsafe_allow_html=True)
    with st.expander("Root Cause Analysis", expanded=False):
        drivers = compute_root_cause_drivers(df_scope, job_row)
        if drivers:
            for driver in drivers:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{driver['driver_name']}** — {driver.get('evidence_detail', '')}")
                    st.caption(
                        f"{driver['evidence_metric']}: {driver['evidence_value']} "
                        f"(benchmark: {driver['benchmark_value']})"
                    )
                with c2:
                    st.markdown(f"**Action:** {driver['recommendation']}")
        else:
            st.info("No significant risk drivers detected.")

    st.markdown('<div class="dc-soft-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dc-section-label">Execution Breakdown</div>', unsafe_allow_html=True)
    tab_task, tab_staff = st.tabs(["Task Breakdown", "Staff Attribution"])

    with tab_task:
        task_df = get_job_task_attribution(df_scope, selected_job)
        # fallback for type mismatch
        if len(task_df) == 0 and "job_no" in df_scope.columns:
            matches = df_scope.loc[
                df_scope["job_no"].astype(str) == str(selected_job), "job_no"
            ]
            if len(matches) > 0:
                task_df = get_job_task_attribution(df_scope, matches.iloc[0])

        if len(task_df) == 0:
            st.info("No task attribution data available for this job.")
        else:
            # Get peer benchmarks for the same dept+category
            dept = job_row.get("department_final", "")
            cat = job_row.get("job_category", "")
            bench_df = compute_task_level_benchmarks(
                base_df,
                str(selected_job),
                dept,
                cat,
            )

            # Merge benchmark into task data
            if len(bench_df) > 0:
                task_df = task_df.merge(bench_df, on="task_name", how="left")
            else:
                task_df["peer_median_hours"] = np.nan
                task_df["peer_job_count"] = 0

            task_df = task_df.sort_values("actual_hours", ascending=False)

            # ── Chart: Task (y) with cost-to-date, stacked actual by staff + quoted target marker ──
            job_scope_df = df_scope[df_scope["job_no"].astype(str) == str(selected_job)].copy()
            if len(job_scope_df) == 0 and "job_no" in df_scope.columns:
                matches = df_scope.loc[
                    df_scope["job_no"].astype(str) == str(selected_job), "job_no"
                ]
                if len(matches) > 0:
                    job_scope_df = df_scope[df_scope["job_no"] == matches.iloc[0]].copy()

            can_build_staff_task = (
                len(job_scope_df) > 0
                and "task_name" in job_scope_df.columns
                and "hours_raw" in job_scope_df.columns
            )

            if can_build_staff_task:
                if "staff_name" not in job_scope_df.columns:
                    job_scope_df["staff_name"] = "Unassigned"
                if "base_cost" not in job_scope_df.columns:
                    job_scope_df["base_cost"] = 0.0

                job_scope_df["task_name"] = job_scope_df["task_name"].astype(str)
                job_scope_df["staff_name"] = (
                    job_scope_df["staff_name"]
                    .fillna("Unassigned")
                    .astype(str)
                    .str.strip()
                    .replace("", "Unassigned")
                )
                job_scope_df["hours_raw"] = pd.to_numeric(
                    job_scope_df["hours_raw"], errors="coerce"
                ).fillna(0.0)
                job_scope_df["base_cost"] = pd.to_numeric(
                    job_scope_df["base_cost"], errors="coerce"
                ).fillna(0.0)

                task_cost_map = job_scope_df.groupby("task_name")["base_cost"].sum()
                task_staff = (
                    job_scope_df.groupby(["task_name", "staff_name"], dropna=False)["hours_raw"]
                    .sum()
                    .reset_index(name="actual_hours")
                )

                top_task_df = task_df.copy()
                top_task_df["task_name"] = top_task_df["task_name"].astype(str)
                top_task_df["_display_hours"] = (
                    top_task_df[["actual_hours", "quoted_hours"]]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .max(axis=1)
                )
                top_task_df = top_task_df.sort_values("_display_hours", ascending=False).head(14)
                top_task_names = top_task_df["task_name"].tolist()

                if len(top_task_names) > 0 and len(task_staff) > 0:
                    task_staff["task_name"] = task_staff["task_name"].astype(str)
                    task_staff["staff_name"] = task_staff["staff_name"].astype(str)

                    staff_pivot = (
                        task_staff.pivot_table(
                            index="task_name",
                            columns="staff_name",
                            values="actual_hours",
                            aggfunc="sum",
                            fill_value=0.0,
                        )
                        .reindex(top_task_names, fill_value=0.0)
                    )
                    staff_order = staff_pivot.sum(axis=0).sort_values(ascending=False).index.tolist()

                    quote_map = (
                        top_task_df.set_index("task_name")["quoted_hours"]
                        .apply(pd.to_numeric, errors="coerce")
                        .fillna(0.0)
                    )
                    quote_values = quote_map.reindex(top_task_names).fillna(0.0).values

                    y_labels = [
                        f"{task} - {fmt_currency(float(task_cost_map.get(task, 0.0)))}"
                        for task in top_task_names
                    ]
                    palette = [
                        "#2563eb", "#7db7e8", "#ef4444", "#8b5cf6", "#14b8a6",
                        "#f59e0b", "#22c55e", "#fb7185", "#94a3b8", "#0ea5e9",
                        "#eab308", "#f97316", "#10b981", "#a855f7", "#3b82f6",
                    ]

                    fig = go.Figure()
                    for i, staff in enumerate(staff_order):
                        hours_vals = (
                            pd.to_numeric(staff_pivot[staff], errors="coerce")
                            .fillna(0.0)
                            .values
                        )
                        if np.nansum(hours_vals) <= 0:
                            continue
                        fig.add_trace(
                            go.Bar(
                                y=y_labels,
                                x=hours_vals,
                                customdata=np.array(top_task_names, dtype=object),
                                name=staff,
                                orientation="h",
                                marker_color=palette[i % len(palette)],
                                hovertemplate=(
                                    "Task: %{customdata}<br>"
                                    f"Staff: {staff}<br>"
                                    "Actual Hours: %{x:.1f}h<extra></extra>"
                                ),
                            )
                        )

                    fig.add_trace(
                        go.Scatter(
                            y=y_labels,
                            x=quote_values,
                            customdata=np.array(top_task_names, dtype=object),
                            mode="markers+text",
                            name="Quoted Hours (target)",
                            marker=dict(
                                symbol="diamond",
                                size=9,
                                color="#111827",
                                line=dict(color="#ffffff", width=1),
                            ),
                            text=[f"{v:.0f}h" for v in quote_values],
                            textposition="middle right",
                            textfont=dict(size=10, color="#111827"),
                            hovertemplate=(
                                "Task: %{customdata}<br>"
                                "Quoted Target: %{x:.1f}h<extra></extra>"
                            ),
                        )
                    )

                    fig.update_layout(
                        barmode="stack",
                        title="Task Execution by Staff (Actual Hours) vs Quoted Target",
                        xaxis_title="Hours",
                        yaxis_title="Task - Cost to Date",
                        template="plotly_white",
                        height=max(360, len(top_task_names) * 44 + 120),
                        margin=dict(l=260, r=30, t=50, b=55),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                        hovermode="y unified",
                    )
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Stacked bars show actual task hours split by staff. "
                        "Black markers show quoted hour targets. "
                        "Y-axis labels include task cost-to-date."
                    )
                else:
                    can_build_staff_task = False

            if not can_build_staff_task:
                top_tasks = task_df.head(10)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=top_tasks["task_name"],
                    x=top_tasks["quoted_hours"].fillna(0),
                    name="Quoted",
                    orientation="h",
                    marker_color="#9ecae1",
                ))
                fig.add_trace(go.Bar(
                    y=top_tasks["task_name"],
                    x=top_tasks["actual_hours"].fillna(0),
                    name="Actual",
                    orientation="h",
                    marker_color="#3182bd",
                ))
                if top_tasks["peer_median_hours"].notna().any():
                    fig.add_trace(go.Bar(
                        y=top_tasks["task_name"],
                        x=top_tasks["peer_median_hours"].fillna(0),
                        name="Peer Benchmark",
                        orientation="h",
                        marker_color="#ff7f0e",
                    ))
                fig.update_layout(
                    barmode="group",
                    title="Task Hours: Quoted vs Actual vs Peer Benchmark",
                    xaxis_title="Hours",
                    yaxis_title="",
                    template="plotly_white",
                    height=max(300, len(top_tasks) * 40 + 100),
                    margin=dict(l=180, r=30, t=40, b=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

            # ── Data table ──
            display = task_df.copy()
            col_map = {
                "task_name": "Task",
                "quoted_hours": "Quoted Hrs",
                "actual_hours": "Actual Hrs",
                "peer_median_hours": "Peer Benchmark Hrs",
                "variance": "Variance",
                "variance_pct": "Variance %",
            }
            display = display.rename(columns=col_map)
            for col in ["Quoted Hrs", "Actual Hrs", "Peer Benchmark Hrs", "Variance"]:
                if col in display.columns:
                    display[col] = display[col].apply(fmt_hours)
            if "Variance %" in display.columns:
                display["Variance %"] = display["Variance %"].apply(fmt_percent)

            show_cols = [
                c for c in [
                    "Task", "Quoted Hrs", "Actual Hrs",
                    "Peer Benchmark Hrs", "Variance", "Variance %",
                ] if c in display.columns
            ]
            st.dataframe(display[show_cols], use_container_width=True, hide_index=True)

    with tab_staff:
        staff_df = get_job_staff_attribution(df_scope, selected_job)
        if len(staff_df) == 0 and "job_no" in df_scope.columns:
            matches = df_scope.loc[
                df_scope["job_no"].astype(str) == str(selected_job), "job_no"
            ]
            if len(matches) > 0:
                staff_df = get_job_staff_attribution(df_scope, matches.iloc[0])

        if len(staff_df) == 0:
            st.info("No staff attribution data available for this job.")
        else:
            staff_df = staff_df.sort_values("hours", ascending=False)

            # Quoted rate for reference
            quoted_rate = pd.to_numeric(job_row.get("quote_rate", np.nan), errors="coerce")

            # ── Chart: Staff cost rate vs quoted rate ──
            top_staff = staff_df.head(10)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top_staff["staff_name"],
                x=top_staff["effective_cost_rate"].fillna(0),
                name="Effective Cost Rate",
                orientation="h",
                marker_color="#3182bd",
            ))
            if "effective_rev_rate" in top_staff.columns:
                fig.add_trace(go.Bar(
                    y=top_staff["staff_name"],
                    x=top_staff["effective_rev_rate"].fillna(0),
                    name="Revenue Rate",
                    orientation="h",
                    marker_color="#2ca02c",
                ))
            if pd.notna(quoted_rate) and quoted_rate > 0:
                fig.add_vline(
                    x=quoted_rate,
                    line_dash="dash",
                    line_color="#dc3545",
                    annotation_text=f"Quoted: {fmt_rate(quoted_rate)}",
                    annotation_position="top right",
                )
            fig.update_layout(
                barmode="group",
                title="Staff Rate Analysis (Cost Rate vs Revenue Rate)",
                xaxis_title="$/hr",
                yaxis_title="",
                template="plotly_white",
                height=max(300, len(top_staff) * 40 + 100),
                margin=dict(l=180, r=30, t=40, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

            # ── Data table with enriched columns ──
            display = staff_df.copy()
            display["margin_contribution"] = display["revenue"].fillna(0) - display["cost"].fillna(0)
            col_map = {
                "staff_name": "Staff",
                "hours": "Hours",
                "cost": "Cost",
                "revenue": "Revenue",
                "effective_cost_rate": "Cost Rate",
                "effective_rev_rate": "Rev Rate",
                "margin_contribution": "Margin $",
                "tasks_worked": "Tasks",
            }
            display = display.rename(columns=col_map)
            if "Hours" in display.columns:
                display["Hours"] = display["Hours"].apply(fmt_hours)
            for col in ["Cost", "Revenue", "Margin $"]:
                if col in display.columns:
                    display[col] = display[col].apply(fmt_currency)
            for col in ["Cost Rate", "Rev Rate"]:
                if col in display.columns:
                    display[col] = display[col].apply(fmt_rate)

            show_cols = [
                c for c in [
                    "Staff", "Hours", "Cost Rate", "Rev Rate",
                    "Cost", "Revenue", "Margin $", "Tasks",
                ] if c in display.columns
            ]
            st.dataframe(display[show_cols], use_container_width=True, hide_index=True)

    # ── Section G: Client Group Subsidization ──────────────────────────
    _render_client_subsidy_section(
        df_all if df_all is not None else df_scope,
        jobs_df,
        str(selected_job),
        job_name_lookup,
    )

    # ── Section H: Completion Forecast ──────────────────────────
    render_completion_forecast_section(
        df_all=df_all if df_all is not None else df_scope,
        df_scope=df_scope,
        jobs_df=jobs_df,
        selected_job=str(selected_job),
        job_name_lookup=job_name_lookup,
    )


def _render_client_subsidy_section(
    df_all: pd.DataFrame,
    jobs_df: pd.DataFrame,
    selected_job: str,
    job_name_lookup: Dict[str, str],
) -> None:
    """Render client group subsidization analysis for the selected job."""
    st.markdown("---")
    st.markdown('<div class="dc-section-label">Client Group Context</div>', unsafe_allow_html=True)

    show_client_ltv = st.toggle(
        "Show Client LTV (all jobs)",
        value=False,
        key=f"dc_client_ltv_{selected_job}",
        help="Switch to client-level grouping and include all jobs for lifetime-value context.",
    )
    preferred_group_col = "client" if show_client_ltv else None
    selected_scope = "all" if show_client_ltv else "active_only"
    selected_lookback = None

    if show_client_ltv:
        st.caption("LTV view: client-level grouping across all jobs and full history.")
    else:
        st.caption("Default view shows active jobs under this client group with lifetime margin metrics.")

    if preferred_group_col is not None:
        group_col = resolve_group_column(df_all, preferred=preferred_group_col)
    else:
        group_col = resolve_group_column(df_all)
    if group_col is None:
        st.caption("No client group column available in dataset.")
        return

    ctx = compute_client_group_subsidy_context(
        df_all,
        jobs_df,
        selected_job,
        lookback_months=selected_lookback,
        scope=selected_scope,
        preferred_group_col=preferred_group_col,
    )

    if ctx["status"] != "ok":
        st.caption(f"Unable to compute client group context (status: {ctx['status']}).")
        return

    summary = ctx["summary"]
    group_value = ctx["group_value"]
    peer_jobs = ctx["jobs"]
    group_label = "Client" if ctx["group_col"] == "client" else "Client Group"

    # ── Verdict banner ──
    verdict = summary["verdict"]
    verdict_colors = {
        "Fully Subsidized": ("#28a745", "This job's losses are fully covered by profitable sibling jobs."),
        "Partially Subsidized": ("#ffc107", "Profitable siblings cover some, but not all, of this job's losses."),
        "Weak Subsidy": ("#dc3545", "Very limited coverage from sibling jobs — group margin under pressure."),
        "Not Subsidized": ("#dc3545", "No profitable sibling jobs to offset this job's losses."),
        "No Subsidy Needed": ("#28a745", "This job is profitable — no subsidy required."),
    }
    vcolor, vdesc = verdict_colors.get(verdict, ("#6c757d", ""))

    st.markdown(
        f'<div style="padding:10px 16px;border-radius:6px;background:{vcolor}15;'
        f'border-left:4px solid {vcolor};margin-bottom:12px;">'
        f'<strong style="color:{vcolor}">{verdict}</strong>'
        f' &nbsp;—&nbsp; {group_label}: <strong>{group_value}</strong>'
        f'<br><span style="font-size:0.85em;color:#555">{vdesc}</span></div>',
        unsafe_allow_html=True,
    )

    # ── KPI strip ──
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Group Margin",
            fmt_currency(summary["group_margin"]),
            help="Total margin across all jobs in this client group",
        )
    with k2:
        gm_pct = summary["group_margin_pct"]
        st.metric(
            "Group Margin %",
            f"{gm_pct:.1f}%" if pd.notna(gm_pct) else "—",
            help="Group-level margin percentage",
        )
    with k3:
        cr = summary["coverage_ratio"]
        st.metric(
            "Coverage Ratio",
            f"{cr:.1f}x" if pd.notna(cr) else "N/A",
            help="How many times peer profit covers this job's loss (>1 = fully covered)",
        )
    with k4:
        st.metric(
            "Buffer After Subsidy",
            fmt_currency(summary["buffer_after_subsidy"]),
            help="Net peer margin after absorbing this job's loss (includes profitable and loss-making peers)",
        )

    # ── Additional context metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Jobs in Group", fmt_count(summary["job_count"]))
    with c2:
        st.metric("Active Jobs", fmt_count(summary["active_job_count"]))
    with c3:
        st.metric("Loss-Making Jobs", fmt_count(summary["loss_job_count"]))
    with c4:
        sc = summary["subsidy_concentration_pct"]
        st.metric(
            "Top Subsidizer Share",
            f"{sc:.0f}%" if pd.notna(sc) else "—",
            help="% of positive peer margin from single largest contributor (high = concentrated risk)",
        )

    # ── Peer jobs table ──
    if len(peer_jobs) > 0:
        if group_label == "Client":
            st.markdown("#### Jobs Under This Client")
        else:
            st.markdown("#### Jobs Under This Client Group")

        display = peer_jobs.copy()
        display["job_label"] = display["job_no"].apply(
            lambda j: f"→ {j}" if str(j) == str(selected_job) else str(j)
        )
        lookup_names = display["job_no"].astype(str).map(job_name_lookup)
        existing_names = display.get("job_name", pd.Series(index=display.index, dtype=object))
        lookup_mask = lookup_names.notna() & lookup_names.astype(str).str.strip().ne("")
        display["job_name"] = lookup_names.where(
            lookup_mask,
            existing_names,
        )
        existing_mask = display["job_name"].notna() & display["job_name"].astype(str).str.strip().ne("")
        display["job_name"] = display["job_name"].where(
            existing_mask,
            display["job_no"].astype(str),
        )

        # Format columns
        col_map = {
            "job_label": "Job",
            "job_name": "Name",
            "risk_band": "Risk",
            "revenue": "Revenue",
            "cost": "Cost",
            "margin": "Margin",
            "margin_pct": "Margin %",
            "contribution_pct_to_group_margin": "Contribution %",
        }
        display = display.rename(columns=col_map)
        for col in ["Revenue", "Cost", "Margin"]:
            if col in display.columns:
                display[col] = display[col].apply(fmt_currency)
        if "Margin %" in display.columns:
            display["Margin %"] = display["Margin %"].apply(fmt_percent)
        if "Contribution %" in display.columns:
            display["Contribution %"] = display["Contribution %"].apply(
                lambda v: f"{v:.1f}%" if pd.notna(v) else "—"
            )

        show_cols = [
            c for c in [
                "Job", "Name", "Risk", "Revenue",
                "Margin", "Margin %", "Contribution %",
            ] if c in display.columns
        ]

        sort_col = "is_selected" if "is_selected" in display.columns else "Job"
        st.dataframe(
            display.sort_values(sort_col, ascending=False)[show_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No peer jobs found under this client group.")
