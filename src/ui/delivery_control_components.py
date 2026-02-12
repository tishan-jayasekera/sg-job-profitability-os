"""
Delivery Control Tower UI Components.

Provides the four functions imported by pages/4_Active_Delivery.py:
- inject_delivery_control_theme()
- summarize_alerts()
- render_job_queue()
- render_selected_job_panel()
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from src.data.job_lifecycle import get_job_staff_attribution, get_job_task_attribution
from src.metrics.delivery_control import compute_root_cause_drivers
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent


def inject_delivery_control_theme() -> None:
    """Inject CSS for the delivery control tower page."""
    st.markdown(
        """
        <style>
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

    risk_emoji = {"Red": "🔴", "Amber": "🟡", "Green": "🟢"}
    options = []
    job_nos = []
    for _, row in display_df.iterrows():
        job_no = str(row.get("job_no", ""))
        name = str(job_name_lookup.get(job_no, "") or "")
        badge = risk_emoji.get(str(row.get("risk_band", "")), "⚪")
        label = f"{badge} {job_no}"
        if name:
            label += f" - {name}"
        driver = str(row.get("primary_driver", "") or "")
        if driver and driver != "Monitor":
            label += f"  |  {driver}"
        options.append(label)
        job_nos.append(job_no)

    current_idx = 0
    current_selected = st.session_state.get("selected_job")
    current_selected_str = str(current_selected) if current_selected is not None else None
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
    _ = df_all

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

    st.markdown(f"### Active Peers ({peer_count} jobs in {dept} / {cat})")
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

    tab_task, tab_staff = st.tabs(["Task Breakdown", "Staff Attribution"])

    with tab_task:
        task_df = get_job_task_attribution(df_scope, selected_job)
        if len(task_df) == 0 and "job_no" in df_scope.columns:
            matches = df_scope.loc[df_scope["job_no"].astype(str) == str(selected_job), "job_no"]
            if len(matches) > 0:
                task_df = get_job_task_attribution(df_scope, matches.iloc[0])

        if len(task_df) == 0:
            st.info("No task attribution data available for this job.")
        else:
            display = task_df.sort_values("variance", ascending=False).copy()
            display = display.rename(
                columns={
                    "task_name": "Task",
                    "quoted_hours": "Quoted Hrs",
                    "actual_hours": "Actual Hrs",
                    "variance": "Variance",
                    "variance_pct": "Variance %",
                }
            )
            for col in ["Quoted Hrs", "Actual Hrs", "Variance"]:
                if col in display.columns:
                    display[col] = display[col].apply(fmt_hours)
            if "Variance %" in display.columns:
                display["Variance %"] = display["Variance %"].apply(fmt_percent)

            show_cols = [c for c in ["Task", "Quoted Hrs", "Actual Hrs", "Variance", "Variance %"] if c in display.columns]
            st.dataframe(display[show_cols], use_container_width=True, hide_index=True)

    with tab_staff:
        staff_df = get_job_staff_attribution(df_scope, selected_job)
        if len(staff_df) == 0 and "job_no" in df_scope.columns:
            matches = df_scope.loc[df_scope["job_no"].astype(str) == str(selected_job), "job_no"]
            if len(matches) > 0:
                staff_df = get_job_staff_attribution(df_scope, matches.iloc[0])

        if len(staff_df) == 0:
            st.info("No staff attribution data available for this job.")
        else:
            display = staff_df.sort_values("hours", ascending=False).copy()
            display = display.rename(
                columns={
                    "staff_name": "Staff",
                    "hours": "Hours",
                    "cost": "Cost",
                    "tasks_worked": "Tasks Worked",
                }
            )
            if "Hours" in display.columns:
                display["Hours"] = display["Hours"].apply(fmt_hours)
            if "Cost" in display.columns:
                display["Cost"] = display["Cost"].apply(fmt_currency)

            show_cols = [c for c in ["Staff", "Hours", "Cost", "Tasks Worked"] if c in display.columns]
            st.dataframe(display[show_cols], use_container_width=True, hide_index=True)
