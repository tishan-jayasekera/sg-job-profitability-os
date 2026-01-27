"""
Forecast & Bottlenecks Page

Predict remaining work vs empirical capacity using benchmarks + team velocity.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.data.semantic import get_category_col
from src.data.cohorts import get_active_jobs
from src.data.profiles import (
    compute_task_expertise,
    compute_category_expertise,
    compute_staff_capacity,
    compute_expected_load,
    compute_headroom,
)
from src.staffing.engine import recommend_staff_for_plan
from src.modeling.benchmarks import build_category_benchmarks
from src.modeling.forecast import forecast_remaining_work, solve_bottlenecks
from src.modeling.supply import build_velocity_for_active_jobs
from src.ui.formatting import fmt_hours, fmt_percent
from src.config import config


st.set_page_config(page_title="Forecast & Bottlenecks", page_icon="🧭", layout="wide")


def _get_job_meta(df: pd.DataFrame) -> pd.DataFrame:
    category_col = get_category_col(df)
    cols = ["job_no", "department_final", category_col]
    meta = df[cols].drop_duplicates()
    meta = meta.rename(columns={category_col: "category_rev_job"})
    if "job_due_date" in df.columns:
        due = df.groupby("job_no")["job_due_date"].first().reset_index()
        meta = meta.merge(due, on="job_no", how="left")
    return meta


def _completed_jobs(df: pd.DataFrame) -> pd.DataFrame:
    if "job_completed_date" in df.columns:
        return df[df["job_completed_date"].notna()].copy()
    if "job_status" in df.columns:
        return df[df["job_status"].str.lower().str.contains("completed", na=False)].copy()
    return df.iloc[0:0].copy()


def _capacity_weekly(df: pd.DataFrame, weeks: int) -> float:
    if len(df) == 0:
        return np.nan
    date_col = "work_date" if "work_date" in df.columns else "month_key"
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    latest = df[date_col].max()
    cutoff = latest - pd.Timedelta(weeks=weeks)
    recent = df[df[date_col] >= cutoff]
    if len(recent) == 0:
        recent = df
    if "fte_hours_scaling" not in recent.columns:
        recent["fte_hours_scaling"] = 1.0
    staff_fte = recent.groupby("staff_name")["fte_hours_scaling"].median().sum()
    return staff_fte * config.CAPACITY_HOURS_PER_WEEK


def _weekly_trend(df_active: pd.DataFrame, df_completed: pd.DataFrame, dept: str | None, cat: str | None) -> tuple[pd.Series, pd.Series, float]:
    date_col = "work_date" if "work_date" in df_active.columns else "month_key"
    active = df_active.copy()
    active[date_col] = pd.to_datetime(active[date_col], errors="coerce")
    if dept:
        active = active[active["department_final"] == dept]
    if cat:
        active = active[active["category_rev_job"] == cat]
    weekly_active = active.set_index(date_col)["hours_raw"].resample("W").sum()

    completed = df_completed.copy()
    if len(completed) > 0:
        completed[date_col] = pd.to_datetime(completed[date_col], errors="coerce")
        if dept:
            completed = completed[completed["department_final"] == dept]
        if cat:
            completed = completed[completed["category_rev_job"] == cat]
        weekly_completed = completed.set_index(date_col)["hours_raw"].resample("W").sum()
    else:
        weekly_completed = pd.Series(dtype=float)

    cap_week = _capacity_weekly(active, config.LOAD_TRAILING_WEEKS)
    return weekly_active, weekly_completed, cap_week


def _recommend_staff_for_gaps(task_gaps: pd.DataFrame, task_expertise: pd.DataFrame, category_expertise: pd.DataFrame, headroom_df: pd.DataFrame, dept: str, cat: str) -> dict:
    if len(task_gaps) == 0:
        return {}
    plan = {
        "department": dept,
        "category": cat,
        "tasks": [
            {"task_name": row["task_name"], "hours": row["remaining_task_hours"]}
            for _, row in task_gaps.iterrows()
        ],
    }
    eligibility_config = {
        "recency_months": config.ELIGIBILITY_RECENCY_MONTHS,
        "min_hours": config.ELIGIBILITY_MIN_HOURS,
        "min_jobs": config.ELIGIBILITY_MIN_JOBS,
    }
    recs, _ = recommend_staff_for_plan(
        plan,
        task_expertise,
        category_expertise,
        headroom_df,
        eligibility_config,
        top_n=1,
    )
    if len(recs) == 0:
        return {}
    return recs.groupby("task_name")["staff_name"].first().to_dict()


def main():
    st.title("Forecast & Bottlenecks")
    st.caption("Remaining work vs empirical capacity using completed-job benchmarks + team velocity.")

    df = load_fact_timesheet()
    category_col = get_category_col(df)
    df = df.rename(columns={category_col: "category_rev_job"})

    active_jobs = get_active_jobs(df)
    df_active = df[df["job_no"].isin(active_jobs)].copy()
    if len(df_active) == 0:
        st.warning("No active jobs found.")
        return

    bench_summary, task_mix = build_category_benchmarks(df)
    remaining = forecast_remaining_work(df_active, bench_summary, task_mix)
    velocity = build_velocity_for_active_jobs(df, active_jobs, weeks=config.LOAD_TRAILING_WEEKS)
    task_level, job_level = solve_bottlenecks(remaining, velocity, df_active)

    job_meta = _get_job_meta(df_active)
    job_level = job_level.merge(job_meta, on="job_no", how="left")

    # =========================
    # CHAIN CONTROLS (TOP-DOWN)
    # =========================
    st.subheader("Chain Controls")
    dept_options = ["All"] + sorted(job_level["department_final"].dropna().unique().tolist())
    dept = st.selectbox("Department", dept_options)
    scope = job_level.copy()
    if dept != "All":
        scope = scope[scope["department_final"] == dept]

    cat_options = ["All"] + sorted(scope["category_rev_job"].dropna().unique().tolist())
    category = st.selectbox("Job Category", cat_options)
    if category != "All":
        scope = scope[scope["category_rev_job"] == category]

    job_options = sorted(scope["job_no"].dropna().unique().tolist())
    selected_job = st.selectbox("Job", job_options)

    selected_meta = job_level[job_level["job_no"] == selected_job].iloc[0]
    selected_task = task_level[task_level["job_no"] == selected_job].copy()
    selected_bench = bench_summary[
        (bench_summary["department_final"] == selected_meta["department_final"]) &
        (bench_summary["category_rev_job"] == selected_meta["category_rev_job"])
    ]

    # =========================
    # SECTION 1 — COMPANY FORECAST
    # =========================
    st.subheader("Section 1 — Company Forecast (Top‑Down)")
    company_eta = job_level["job_eta_weeks"].replace([np.inf, -np.inf], np.nan).max()
    company_risk = (job_level["status"] == "At Risk").sum()
    company_blocked = (job_level["status"] == "Blocked").sum()
    company_total_remaining = task_level["remaining_task_hours"].sum()
    zero_velocity_tasks = task_level[
        (task_level["remaining_task_hours"] > 0) & (task_level["team_velocity_hours_week"] == 0)
    ]
    # Company Demand Trend
    st.markdown("**Company Demand Trend**")
    date_col = "work_date" if "work_date" in df_active.columns else "month_key"
    df_active[date_col] = pd.to_datetime(df_active[date_col], errors="coerce")
    df_completed = _completed_jobs(df)
    if len(df_completed) > 0:
        df_completed[date_col] = pd.to_datetime(df_completed[date_col], errors="coerce")
    weekly_hours = df_active.set_index(date_col)["hours_raw"].resample("W").sum()
    weekly_jobs = df_active.groupby(pd.Grouper(key=date_col, freq="W"))["job_no"].nunique()
    weekly_completed = (
        df_completed.set_index(date_col)["hours_raw"].resample("W").sum()
        if len(df_completed) > 0 else pd.Series(dtype=float)
    )
    capacity_week = _capacity_weekly(df_active, config.LOAD_TRAILING_WEEKS)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_hours.index,
        y=weekly_hours.values,
        mode="lines+markers",
        name="Active Job Hours (weekly)",
    ))
    if len(weekly_completed) > 0:
        fig.add_trace(go.Scatter(
            x=weekly_completed.index,
            y=weekly_completed.values,
            mode="lines+markers",
            name="Completed Work (weekly)",
        ))
    fig.add_trace(go.Scatter(
        x=weekly_jobs.index,
        y=weekly_jobs.values,
        mode="lines+markers",
        name="Active Jobs (count)",
        yaxis="y2",
    ))
    if pd.notna(capacity_week) and capacity_week > 0:
        fig.add_hline(y=capacity_week, line_dash="dash", annotation_text="Imputed Capacity / week")
    fig.update_layout(
        height=320,
        xaxis_title="Week",
        yaxis=dict(title="Hours"),
        yaxis2=dict(title="Active Jobs", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Forecast Period (Next N Weeks)**")
    n_weeks = st.slider("Forecast weeks", 4, 16, 8)
    projected_weekly_demand = weekly_hours.mean() if len(weekly_hours) > 0 else 0
    forecast_total_demand = projected_weekly_demand * n_weeks
    forecast_total_capacity = capacity_week * n_weeks if pd.notna(capacity_week) else np.nan
    forecast_gap = forecast_total_capacity - forecast_total_demand if pd.notna(forecast_total_capacity) else np.nan
    forecast_cols = st.columns(4)
    with forecast_cols[0]:
        st.metric("Projected Demand (hrs)", fmt_hours(forecast_total_demand))
    with forecast_cols[1]:
        st.metric("Projected Capacity (hrs)", fmt_hours(forecast_total_capacity))
    with forecast_cols[2]:
        st.metric("Forecast Gap", fmt_hours(forecast_gap))
    with forecast_cols[3]:
        st.metric("Avg Weekly Demand", fmt_hours(projected_weekly_demand))

    if len(weekly_hours) > 0:
        last_actual = weekly_hours.index.max()
        forecast_end = last_actual + pd.Timedelta(weeks=n_weeks)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=weekly_hours.index,
            y=weekly_hours.values,
            mode="lines+markers",
            name="Actual (weekly)",
        ))
        if pd.notna(capacity_week) and capacity_week > 0:
            fig_forecast.add_hline(y=capacity_week, line_dash="dash", annotation_text="Capacity / week")
        fig_forecast.add_shape(
            type="rect",
            x0=last_actual,
            x1=forecast_end,
            y0=0,
            y1=max(weekly_hours.max(), capacity_week if pd.notna(capacity_week) else 0),
            fillcolor="rgba(200,200,200,0.25)",
            line_width=0,
            layer="below",
        )
        fig_forecast.update_layout(
            height=260,
            xaxis_title="Week",
            yaxis_title="Hours",
            margin=dict(t=30, l=10, r=10, b=10),
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    surplus_week = capacity_week - projected_weekly_demand if pd.notna(capacity_week) else np.nan
    surplus_fte = surplus_week / config.CAPACITY_HOURS_PER_WEEK if pd.notna(surplus_week) else np.nan
    st.caption(
        f"Insight: Forecast period assumes steady demand at {fmt_hours(projected_weekly_demand)} hrs/wk. "
        f"Projected capacity surplus = {fmt_hours(surplus_week)} hrs/wk "
        f"({surplus_fte:.2f} FTE/wk) if positive."
    )

    if len(weekly_hours) > 0 and pd.notna(capacity_week):
        forecast_index = pd.date_range(start=last_actual + pd.Timedelta(weeks=1), periods=n_weeks, freq="W")
        gap_series = pd.Series([capacity_week - projected_weekly_demand] * n_weeks, index=forecast_index)
        gap_fig = go.Figure()
        gap_fig.add_trace(go.Bar(
            x=gap_series.index,
            y=gap_series.values,
            name="Capacity − Demand (hrs)",
            marker_color=["#d95f02" if v < 0 else "#1b9e77" for v in gap_series.values],
        ))
        gap_fig.update_layout(
            height=240,
            xaxis_title="Forecast Week",
            yaxis_title="Capacity Gap (hrs)",
            margin=dict(t=30, l=10, r=10, b=10),
        )
        st.plotly_chart(gap_fig, use_container_width=True)

        deficit_weeks = (gap_series < 0).sum()
        if deficit_weeks > 0:
            st.warning(
                f"Key insight: {deficit_weeks} of the next {n_weeks} weeks show a capacity deficit. "
                "Consider rebalancing workload or adding capacity."
            )
        else:
            st.success("Key insight: No forecasted capacity deficit in the selected period.")

    # Company Remaining Work
    st.markdown("**Company Remaining Work**")
    rem_cols = st.columns(4)
    with rem_cols[0]:
        st.metric("Active Jobs", f"{job_level['job_no'].nunique():,}")
    with rem_cols[1]:
        st.metric("Remaining Hours", fmt_hours(company_total_remaining))
    with rem_cols[2]:
        st.metric("ETA (max)", f"{company_eta:.1f} wks" if pd.notna(company_eta) else "—")
    with rem_cols[3]:
        st.metric("At Risk / Blocked", f"{company_risk} / {company_blocked}")

    # Company Bottleneck Risk
    st.markdown("**Company Bottleneck Risk**")
    risk_cols = st.columns(3)
    with risk_cols[0]:
        st.metric("Zero‑Velocity Tasks", f"{len(zero_velocity_tasks):,}")
    with risk_cols[1]:
        st.metric("Jobs w/ Bottlenecks", f"{zero_velocity_tasks['job_no'].nunique():,}")
    with risk_cols[2]:
        st.metric("Blocked Jobs", f"{company_blocked:,}")

    st.divider()

    # =========================
    # SECTION 2 — DEPARTMENT VIEW
    # =========================
    st.subheader("Section 2 — Department Forecast")
    st.markdown("**Department Demand Trend**")
    dept_list = sorted(job_level["department_final"].dropna().unique().tolist())
    if dept == "All":
        selected_depts = st.multiselect("Departments to compare", dept_list, default=dept_list[:2])
    else:
        selected_depts = [dept]
    dept_trend_cols = st.columns(2)
    for idx, d in enumerate(selected_depts):
        with dept_trend_cols[idx % 2]:
            weekly_active, weekly_completed, cap_week = _weekly_trend(df_active, df_completed, d, None)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weekly_active.index, y=weekly_active.values, mode="lines+markers", name="Active"))
            if len(weekly_completed) > 0:
                fig.add_trace(go.Scatter(x=weekly_completed.index, y=weekly_completed.values, mode="lines+markers", name="Completed"))
            if pd.notna(cap_week) and cap_week > 0:
                fig.add_hline(y=cap_week, line_dash="dash", annotation_text="Capacity / week")
            fig.update_layout(height=220, title=d, xaxis_title="Week", yaxis_title="Hours", margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Department Remaining Work**")
    dept_summary = job_level.groupby("department_final").agg(
        jobs=("job_no", "nunique"),
        at_risk=("status", lambda x: (x == "At Risk").sum()),
        blocked=("status", lambda x: (x == "Blocked").sum()),
    ).reset_index()
    dept_remaining = task_level.groupby("department_final")["remaining_task_hours"].sum().reset_index()
    dept_summary = dept_summary.merge(dept_remaining, on="department_final", how="left")
    st.dataframe(
        dept_summary.rename(columns={
            "department_final": "Department",
            "remaining_task_hours": "Remaining Hours",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={"Remaining Hours": st.column_config.NumberColumn(format="%.1f")},
    )
    st.markdown("**Department Bottleneck Risk**")
    dept_bottlenecks = task_level[
        (task_level["remaining_task_hours"] > 0) & (task_level["team_velocity_hours_week"] == 0)
    ].groupby("department_final")["task_name"].count().rename("zero_velocity_tasks").reset_index()
    st.dataframe(
        dept_bottlenecks.rename(columns={"department_final": "Department", "zero_velocity_tasks": "Zero‑Velocity Tasks"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # =========================
    # SECTION 3 — CATEGORY VIEW
    # =========================
    st.subheader("Section 3 — Category Forecast")
    st.markdown("**Category Demand Trend**")
    cat_list = sorted(scope["category_rev_job"].dropna().unique().tolist())
    if category == "All":
        selected_cats = st.multiselect("Categories to compare", cat_list, default=cat_list[:2])
    else:
        selected_cats = [category]
    cat_trend_cols = st.columns(2)
    for idx, c in enumerate(selected_cats):
        with cat_trend_cols[idx % 2]:
            weekly_active, weekly_completed, cap_week = _weekly_trend(df_active, df_completed, dept if dept != "All" else None, c)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weekly_active.index, y=weekly_active.values, mode="lines+markers", name="Active"))
            if len(weekly_completed) > 0:
                fig.add_trace(go.Scatter(x=weekly_completed.index, y=weekly_completed.values, mode="lines+markers", name="Completed"))
            if pd.notna(cap_week) and cap_week > 0:
                fig.add_hline(y=cap_week, line_dash="dash", annotation_text="Capacity / week")
            fig.update_layout(height=220, title=c, xaxis_title="Week", yaxis_title="Hours", margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Category Remaining Work**")
    cat_summary = scope.groupby("category_rev_job").agg(
        jobs=("job_no", "nunique"),
        at_risk=("status", lambda x: (x == "At Risk").sum()),
        blocked=("status", lambda x: (x == "Blocked").sum()),
    ).reset_index()
    cat_remaining = task_level.groupby("category_rev_job")["remaining_task_hours"].sum().reset_index()
    cat_summary = cat_summary.merge(cat_remaining, on="category_rev_job", how="left")
    st.dataframe(
        cat_summary.rename(columns={
            "category_rev_job": "Category",
            "remaining_task_hours": "Remaining Hours",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={"Remaining Hours": st.column_config.NumberColumn(format="%.1f")},
    )
    st.markdown("**Category Bottleneck Risk**")
    cat_bottlenecks = task_level[
        (task_level["remaining_task_hours"] > 0) & (task_level["team_velocity_hours_week"] == 0)
    ].groupby("category_rev_job")["task_name"].count().rename("zero_velocity_tasks").reset_index()
    st.dataframe(
        cat_bottlenecks.rename(columns={"category_rev_job": "Category", "zero_velocity_tasks": "Zero‑Velocity Tasks"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # KPI strip
    st.subheader("Section 4 — Job Forecast Summary")
    actual_hours = df_active[df_active["job_no"] == selected_job]["hours_raw"].sum()
    projected = selected_task["projected_eac_hours"].max() if len(selected_task) > 0 else np.nan
    eta = selected_meta["job_eta_weeks"]
    due_weeks = selected_meta.get("due_weeks", np.nan)
    status = selected_meta.get("status", "Unknown")

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.metric("Projected EAC (p50)", fmt_hours(projected))
    with kpi_cols[1]:
        st.metric("Actual to Date", fmt_hours(actual_hours))
    with kpi_cols[2]:
        st.metric("ETA (weeks)", f"{eta:.1f}" if pd.notna(eta) and not np.isinf(eta) else "—")
    with kpi_cols[3]:
        st.metric("Due (weeks)", f"{due_weeks:.1f}" if pd.notna(due_weeks) else "—")
    with kpi_cols[4]:
        st.metric("Status", status)

    st.divider()

    # Chart 1: Shape vs Reality
    st.subheader("Section 5 — Task Shape vs Reality")
    if len(selected_task) > 0:
        chart_df = selected_task[["task_name", "expected_task_hours", "actual_task_hours", "remaining_task_hours"]].copy()
        chart_df = chart_df.fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_df["task_name"],
            y=chart_df["expected_task_hours"],
            name="Benchmark Shape (Expected)",
        ))
        fig.add_trace(go.Bar(
            x=chart_df["task_name"],
            y=chart_df["actual_task_hours"],
            name="Actuals",
        ))
        fig.add_trace(go.Bar(
            x=chart_df["task_name"],
            y=chart_df["remaining_task_hours"],
            name="Forecast Remaining",
        ))
        fig.update_layout(
            barmode="stack",
            height=360,
            xaxis_title="Task",
            yaxis_title="Hours",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No task-level forecast available for this job.")

    st.divider()

    # Chart 2: Bottleneck matrix
    st.subheader("Section 6 — Bottleneck Matrix (Task → FTE)")
    if len(selected_task) > 0:
        bottleneck_rows = selected_task.copy()
        bottleneck_rows["Est. Weeks"] = np.where(
            bottleneck_rows["team_velocity_hours_week"] > 0,
            bottleneck_rows["remaining_task_hours"] / bottleneck_rows["team_velocity_hours_week"],
            np.inf,
        )
        bottleneck_rows["Action"] = np.where(
            (bottleneck_rows["remaining_task_hours"] > 0) & (bottleneck_rows["team_velocity_hours_week"] == 0),
            "⚠️ Add resource",
            "Monitor",
        )

        # Recommendations
        task_expertise = compute_task_expertise(df, config.PROFILE_TRAINING_MONTHS, config.RECENCY_HALF_LIFE_MONTHS)
        category_expertise = compute_category_expertise(df, config.PROFILE_TRAINING_MONTHS, config.RECENCY_HALF_LIFE_MONTHS)
        capacity_df = compute_staff_capacity(df_active, config.LOAD_TRAILING_WEEKS)
        expected_df = compute_expected_load(df_active, config.LOAD_TRAILING_WEEKS, config.LOAD_TRAILING_WEEKS)
        headroom_df = compute_headroom(capacity_df, expected_df)

        gap_tasks = bottleneck_rows[
            (bottleneck_rows["remaining_task_hours"] > 0) &
            (bottleneck_rows["team_velocity_hours_week"] == 0)
        ][["task_name", "remaining_task_hours"]]

        rec_map = _recommend_staff_for_gaps(
            gap_tasks,
            task_expertise,
            category_expertise,
            headroom_df,
            selected_meta["department_final"],
            selected_meta["category_rev_job"],
        )
        bottleneck_rows["Recommendation"] = bottleneck_rows["task_name"].map(rec_map).fillna("—")

        st.dataframe(
            bottleneck_rows[[
                "task_name",
                "remaining_task_hours",
                "team_velocity_hours_week",
                "Est. Weeks",
                "Action",
                "Recommendation",
            ]].rename(columns={
                "task_name": "Task",
                "remaining_task_hours": "Remaining Hrs",
                "team_velocity_hours_week": "Velocity (hrs/wk)",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Remaining Hrs": st.column_config.NumberColumn(format="%.1f"),
                "Velocity (hrs/wk)": st.column_config.NumberColumn(format="%.1f"),
                "Est. Weeks": st.column_config.NumberColumn(format="%.1f"),
            },
        )
    else:
        st.info("No bottlenecks detected (or no task data).")


if __name__ == "__main__":
    main()
