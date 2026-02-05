"""
UI Components for the Active Delivery Command Center.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import config
from src.data.job_lifecycle import get_job_task_attribution
from src.data.semantic import get_category_col, safe_quote_job_task
from src.exports import export_action_brief
from src.metrics.delivery_control import compute_root_cause_drivers


RISK_COLORS = {
    "Red": ("#dc3545", "#fff5f5"),
    "Amber": ("#ffc107", "#fffbf0"),
    "Green": ("#28a745", "#f0fff0"),
}
RISK_ICONS = {"Red": "🔴", "Amber": "🟡", "Green": "🟢"}


def render_alert_strip(jobs_df: pd.DataFrame) -> None:
    """
    Render compact alert banner showing portfolio status.

    Design: Two-cell layout
    - Left cell: 🔴 CRITICAL count + $ at risk
    - Right cell: 🟡 WATCH count
    """
    red_jobs = jobs_df[jobs_df["risk_band"] == "Red"]
    amber_jobs = jobs_df[jobs_df["risk_band"] == "Amber"]

    margin_at_risk = _compute_margin_at_risk(red_jobs)

    col1, col2 = st.columns(2)

    with col1:
        if len(red_jobs) > 0:
            st.error(
                f"🔴 **CRITICAL:** {len(red_jobs)} jobs, ${margin_at_risk:,.0f} margin at risk"
            )
        else:
            st.success("🟢 No critical jobs")

    with col2:
        if len(amber_jobs) > 0:
            st.warning(f"🟡 **WATCH:** {len(amber_jobs)} jobs need attention")
        else:
            st.info("No jobs on watch list")


def render_job_queue(
    jobs_df: pd.DataFrame,
    job_name_lookup: Dict[str, str],
    include_green: bool = False,
) -> Optional[str]:
    """
    Render compact job queue as selectable cards.

    Returns selected job_no.
    """
    if include_green:
        display_df = jobs_df
    else:
        display_df = jobs_df[jobs_df["risk_band"].isin(["Red", "Amber"])]

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

    for _, row in display_df.iterrows():
        job_no = row["job_no"]
        job_name = job_name_lookup.get(job_no, job_no)
        risk_band = row.get("risk_band", "Green")
        icon = RISK_ICONS.get(risk_band, "⚪")

        issue = _format_primary_issue(row)
        action = row.get("recommended_action", "Review job status")

        label = (
            f"{icon} {job_no}\n"
            f"{job_name[:35]}\n\n"
            f"{issue}\n"
            f"→ {action[:45]}"
        )

        if st.button(label, key=f"job_card_{job_no}", use_container_width=True):
            selected_job = job_no
            st.session_state.selected_job = job_no

    remaining = len(jobs_df) - st.session_state.job_queue_limit
    if remaining > 0:
        if st.button(f"Show {min(10, remaining)} more", key="job_queue_more"):
            st.session_state.job_queue_limit += 10

    st.checkbox("Include Green jobs", key="include_green_jobs")

    return selected_job or st.session_state.get("selected_job")


def render_selected_job_panel(
    df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    job_no: str,
    job_name_lookup: Dict[str, str],
) -> None:
    """
    Render the selected job detail panel.
    """
    job_row = jobs_df[jobs_df["job_no"] == job_no].iloc[0]
    job_name = job_row.get("job_name") or job_name_lookup.get(job_no, job_no)

    st.markdown(f"### {job_no} — {job_name}")

    dept = job_row.get("department_final", "")
    cat = job_row.get("job_category", "")
    band = job_row.get("risk_band", "Green")
    icon = {"Red": "🔴 Critical", "Amber": "🟡 Watch", "Green": "🟢 On Track"}.get(band, "⚪")
    st.caption(f"{dept} • {cat} • {icon}")

    _render_job_snapshot(job_row)

    drivers = compute_root_cause_drivers(df, job_row)
    _render_why_at_risk(drivers)
    _render_recommended_actions(job_row, drivers)

    with st.expander("▼ Expand Full Diagnosis", expanded=False):
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

    st.markdown("**SNAPSHOT**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Hours",
            f"{actual:.0f} / {quoted:.0f}",
            f"{variance_pct:+.0f}%",
            delta_color="inverse",
        )

    with col2:
        delta_label = f"{delta:+.0f}pp vs bench" if delta is not None else None
        st.metric(
            "Margin",
            f"{margin:.0f}%",
            delta_label,
            delta_color="normal" if delta is not None and delta >= 0 else "inverse",
        )

    with col3:
        burn_label = f"{burn_weekly:.0f} hrs/wk" if pd.notna(burn_weekly) else "—"
        burn_delta_label = f"{burn_delta:+.0f} hrs/wk" if burn_delta is not None else None
        st.metric("Burn Rate", burn_label, burn_delta_label)

    st.caption(
        "Burn rate = average hours/day over the last 28 days, scaled to a weekly rate. "
        "Delta compares the prior 28-day window."
    )


def _render_why_at_risk(drivers: List[Dict]) -> None:
    """Render ranked risk drivers."""
    st.markdown("**WHY AT RISK**")

    if not drivers:
        st.success("No significant risk drivers")
        return

    for i, driver in enumerate(drivers[:3], 1):
        name = driver["driver_name"]
        evidence = driver.get("evidence_detail", driver.get("evidence_value", ""))
        st.markdown(
            f"**{i}. {name}** — {driver['evidence_value']}  \n"
            f"<span style='color: #666; font-size: 0.9em;'>{evidence}</span>",
            unsafe_allow_html=True,
        )


def _render_recommended_actions(row: pd.Series, drivers: List[Dict]) -> None:
    """Render checkable action items."""
    st.markdown("**RECOMMENDED ACTIONS**")

    actions = []
    for driver in drivers[:3]:
        action = driver.get("recommendation", "")
        if action:
            actions.append(action)

    if not actions:
        actions.append("Review job status with PM")

    for i, action in enumerate(actions):
        st.checkbox(action, key=f"action_{row['job_no']}_{i}")


def _render_full_diagnosis(df: pd.DataFrame, job_no: str, job_row: pd.Series) -> None:
    """
    Render full diagnosis panel (expandable).

    Includes:
    - Task consumption vs quote with benchmark overlay
    - Staff contribution by task with margin-erosion context
    """
    st.markdown("**Task Consumption vs Quote**")
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
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Red bars indicate tasks running over quoted hours. Benchmarks reflect median actual hours on completed "
            "jobs in the same department/category. Labels on actual bars show FTE-equivalent effort."
        )
    else:
        st.caption("No task data available.")

    st.markdown("**Task Contribution by Staff (Overrun Focus)**")
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
            legend=dict(orientation="h", y=1.05),
        )
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
