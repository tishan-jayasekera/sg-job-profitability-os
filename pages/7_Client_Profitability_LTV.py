"""
Client Profitability & LTV Dashboard
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.data.cohorts import filter_by_time_window
from src.data.semantic import profitability_rollup, get_category_col, safe_quote_rollup, filter_jobs_by_state
from src.data.client_analytics import (
    compute_client_portfolio_summary,
    compute_client_quadrants,
    compute_client_queue,
    compute_client_job_ledger,
    compute_client_department_profit,
    compute_client_task_mix,
    compute_global_task_median_mix,
    compute_company_cost_rate,
    compute_client_ltv,
    compute_client_tenure_months,
)
from src.ui.components import (
    render_client_portfolio_health,
    render_client_quadrant_scatter,
    render_client_intervention_queue,
    render_client_deep_dive,
    render_client_driver_forensics,
    render_client_ltv_section,
    render_client_ltv_methodology_expander,
)
from src.ui import charts as charts
from src.ui.state import init_state, get_state, set_state
from src.ui.formatting import build_job_name_lookup, format_job_label


st.set_page_config(page_title="Client Profitability & LTV", page_icon="📈", layout="wide")

init_state()


def _health_grade(margin_pct: float, profit: float) -> str:
    if pd.isna(margin_pct) and pd.isna(profit):
        return "Unknown"
    if profit < 0:
        return "Red"
    if margin_pct >= 25:
        return "Green"
    if margin_pct >= 10:
        return "Amber"
    return "Red"


def _build_margin_trend(monthly: pd.DataFrame) -> px.bar:
    if len(monthly) == 0:
        return px.bar(title="Margin % (Last 12 Months)")
    trend = monthly.copy()
    trend = trend.tail(12)
    fig = px.bar(
        trend,
        x=trend.columns[0],
        y="margin_pct",
        title="Margin % (Last 12 Months)",
    )
    fig.update_layout(yaxis_title="Margin %", xaxis_title="")
    return fig


def main():
    st.title("Client Profitability & LTV Control Tower")
    st.caption("Segment clients by economic value, diagnose margin leakage, and drill to task/FTE drivers.")

    df_all = load_fact_timesheet()
    if len(df_all) == 0:
        st.warning("No fact_timesheet_day_enriched data available.")
        return

    if "month_key" in df_all.columns:
        df_all["month_key"] = pd.to_datetime(df_all["month_key"])

    st.sidebar.header("Controls")

    def _reset_client_selection():
        if "selected_client_name" in st.session_state:
            st.session_state["selected_client_name"] = []

    time_window = st.sidebar.selectbox(
        "Time Window",
        options=["6m", "12m", "24m", "all"],
        format_func=lambda x: f"Last {x}" if x != "all" else "All Time",
        key="time_window",
        on_change=_reset_client_selection,
    )
    job_state_options = ["All", "Active", "Completed"]
    selected_state = st.sidebar.selectbox(
        "Job State",
        options=job_state_options,
        key="job_state_filter",
        on_change=_reset_client_selection,
    )

    df_window = filter_by_time_window(df_all, time_window, date_col="month_key")
    if selected_state in ["Active", "Completed"]:
        state_jobs = filter_jobs_by_state(df_all, selected_state)
        if "job_no" in state_jobs.columns:
            df_window = df_window[df_window["job_no"].isin(state_jobs["job_no"].unique())]
    else:
        df_window = filter_jobs_by_state(df_window, selected_state)
    if "client" not in df_window.columns:
        st.warning("Client field missing from dataset.")
        return

    # SECTION 1 — Executive Portfolio Health
    summary = compute_client_portfolio_summary(df_window)
    render_client_portfolio_health(summary)

    # SECTION 2 — Portfolio Quadrant Scatter
    y_axis_label = st.radio(
        "Y-axis",
        ["Absolute Profit", "Margin %"],
        horizontal=True,
        key="client_quadrant_y_axis",
    )
    y_mode = "margin_pct" if y_axis_label == "Margin %" else "profit"
    quadrant_df, median_x, median_y = compute_client_quadrants(df_window, y_mode=y_mode)
    quote_rollup = safe_quote_rollup(df_window, ["client"])
    if len(quote_rollup) > 0 and "quoted_amount" in quote_rollup.columns:
        valid_clients = quote_rollup[quote_rollup["quoted_amount"] > 0]["client"].unique().tolist()
        quadrant_df = quadrant_df[quadrant_df["client"].isin(valid_clients)]

    y_title = "Margin %" if y_mode == "margin_pct" else "Profit"
    scatter_fig = charts.client_quadrant_scatter(
        quadrant_df,
        x_col="revenue",
        y_col="y_value",
        size_col="hours",
        quadrant_col="quadrant",
        median_x=median_x,
        median_y=median_y,
        title="Client Portfolio: Revenue vs Profitability",
        x_title="Revenue",
        y_title=y_title,
    )
    render_client_quadrant_scatter(scatter_fig)

    quadrant_options = ["Partners", "Underperformers", "Niche", "Drain"]
    selected_quadrant = st.selectbox(
        "Focus Quadrant",
        options=quadrant_options,
        key="selected_client_quadrant",
    )

    # SECTION 3 — Intervention Queue
    mode_options = ["Intervention", "Growth"]
    selected_mode = st.radio(
        "Queue Mode",
        options=mode_options,
        key="client_queue_mode",
        horizontal=True,
    )
    shortlist_size = st.slider(
        "Shortlist Size",
        min_value=5,
        max_value=30,
        value=int(get_state("client_shortlist_size") or 10),
        step=1,
        key="client_shortlist_size",
    )

    queue_df = compute_client_queue(df_window, selected_quadrant, selected_mode, y_mode=y_mode)
    render_client_intervention_queue(queue_df, shortlist_size)

    # SECTION 4 — Selected Client Deep-Dive
    if len(quadrant_df) > 0:
        client_pool = quadrant_df[quadrant_df["quadrant"] == selected_quadrant]["client"].unique().tolist()
    else:
        client_pool = []

    if not client_pool:
        client_pool = sorted(df_window["client"].dropna().unique().tolist())

    prior_clients = st.session_state.get("selected_client_name")
    if isinstance(prior_clients, str):
        prior_clients = [prior_clients]
    elif isinstance(prior_clients, list):
        prior_clients = prior_clients
    else:
        prior_clients = []
    default_clients = [c for c in prior_clients if c in client_pool]
    if "selected_client_name" in st.session_state and not isinstance(st.session_state.get("selected_client_name"), list):
        del st.session_state["selected_client_name"]
    selected_clients = st.multiselect(
        "Select Client(s)",
        options=client_pool,
        default=default_clients,
        key="selected_client_name",
    )
    if len(selected_clients) == 0:
        st.info("Select at least one client to continue.")
        return

    df_client_window = df_window[df_window["client"].isin(selected_clients)].copy()

    # Canonical drill chain filters
    chain_key_map = {
        "job": "client_job_filter",
        "dept": "client_department_filter",
        "category": "client_category_filter",
        "task": "client_task_filter",
        "staff": "client_staff_filter",
    }
    for key in chain_key_map.values():
        if key not in st.session_state:
            st.session_state[key] = "All"

    def _reset_chain_from(level: str):
        order = ["job", "dept", "category", "task", "staff"]
        if level not in order:
            return
        start = order.index(level) + 1
        for downstream in order[start:]:
            st.session_state[chain_key_map[downstream]] = "All"

    chain_cols = st.columns(5)
    with chain_cols[0]:
        job_name_lookup = build_job_name_lookup(df_client_window)
        job_options = ["All"] + sorted(df_client_window["job_no"].dropna().unique().tolist())
        if st.session_state[chain_key_map["job"]] not in job_options:
            st.session_state[chain_key_map["job"]] = "All"
        selected_job = st.selectbox(
            "Job",
            job_options,
            key=chain_key_map["job"],
            on_change=_reset_chain_from,
            args=("job",),
            format_func=lambda j: format_job_label(j, job_name_lookup),
        )
    if selected_job != "All":
        df_client_window = df_client_window[df_client_window["job_no"] == selected_job]

    with chain_cols[1]:
        dept_options = ["All"] + sorted(df_client_window["department_final"].dropna().unique().tolist())
        if st.session_state[chain_key_map["dept"]] not in dept_options:
            st.session_state[chain_key_map["dept"]] = "All"
        selected_dept = st.selectbox(
            "Department",
            dept_options,
            key=chain_key_map["dept"],
            on_change=_reset_chain_from,
            args=("dept",),
        )
    if selected_dept != "All":
        df_client_window = df_client_window[df_client_window["department_final"] == selected_dept]

    category_col = get_category_col(df_client_window)
    with chain_cols[2]:
        category_options = ["All"] + sorted(df_client_window[category_col].dropna().unique().tolist())
        if st.session_state[chain_key_map["category"]] not in category_options:
            st.session_state[chain_key_map["category"]] = "All"
        selected_category = st.selectbox(
            "Job Category",
            category_options,
            key=chain_key_map["category"],
            on_change=_reset_chain_from,
            args=("category",),
        )
    if selected_category != "All":
        df_client_window = df_client_window[df_client_window[category_col] == selected_category]

    with chain_cols[3]:
        task_options = ["All"] + sorted(df_client_window["task_name"].dropna().unique().tolist())
        if st.session_state[chain_key_map["task"]] not in task_options:
            st.session_state[chain_key_map["task"]] = "All"
        selected_task = st.selectbox(
            "Task",
            task_options,
            key=chain_key_map["task"],
            on_change=_reset_chain_from,
            args=("task",),
        )
    if selected_task != "All":
        df_client_window = df_client_window[df_client_window["task_name"] == selected_task]

    with chain_cols[4]:
        staff_options = ["All"] + sorted(df_client_window["staff_name"].dropna().unique().tolist())
        if st.session_state[chain_key_map["staff"]] not in staff_options:
            st.session_state[chain_key_map["staff"]] = "All"
        selected_staff = st.selectbox(
            "FTE",
            staff_options,
            key=chain_key_map["staff"],
        )
    if selected_staff != "All":
        df_client_window = df_client_window[df_client_window["staff_name"] == selected_staff]

    client_rollup = profitability_rollup(df_client_window)
    quote_rollup = safe_quote_rollup(df_client_window, [])
    quoted_revenue = np.nan
    quoted_hours = np.nan
    quoted_cost = np.nan
    quoted_margin_pct = np.nan
    if len(quote_rollup) > 0:
        quoted_revenue = quote_rollup["quoted_amount"].iloc[0]
        quoted_hours = quote_rollup["quoted_hours"].iloc[0]
        total_hours = df_client_window["hours_raw"].sum()
        total_cost = df_client_window["base_cost"].sum()
        cost_rate = total_cost / total_hours if total_hours > 0 else np.nan
        if pd.notna(quoted_hours) and pd.notna(cost_rate):
            quoted_cost = quoted_hours * cost_rate
        if pd.notna(quoted_revenue) and quoted_revenue > 0 and pd.notna(quoted_cost):
            quoted_margin_pct = (quoted_revenue - quoted_cost) / quoted_revenue * 100

    if len(client_rollup) > 0:
        client_rollup = client_rollup.iloc[0]
        jobs_count = df_client_window["job_no"].nunique()
        metrics = {
            "Revenue": client_rollup["revenue"],
            "Profit": client_rollup["margin"],
            "Margin %": client_rollup["margin_pct"],
            "Realised Rate": client_rollup["realised_rate"],
            "Jobs": jobs_count,
            "Quoted Revenue": quoted_revenue,
            "Quoted Cost": quoted_cost,
            "Quoted Margin %": quoted_margin_pct,
            "Quoted Hours": quoted_hours,
        }
        grade = _health_grade(client_rollup["margin_pct"], client_rollup["margin"])
    else:
        metrics = {
            "Revenue": 0,
            "Profit": 0,
            "Margin %": np.nan,
            "Realised Rate": np.nan,
            "Jobs": 0,
            "Quoted Revenue": quoted_revenue,
            "Quoted Cost": quoted_cost,
            "Quoted Margin %": quoted_margin_pct,
            "Quoted Hours": quoted_hours,
        }
        grade = "Unknown"

    ledger = compute_client_job_ledger(df_client_window)
    dept_profit = compute_client_department_profit(df_client_window)
    if len(dept_profit) > 0:
        dept_fig = charts.horizontal_bar(
            dept_profit,
            x="margin",
            y="department_final",
            title="Profit by Department",
        )
    else:
        dept_fig = charts.horizontal_bar(pd.DataFrame({"margin": [], "department_final": []}), x="margin", y="department_final")

    render_client_deep_dive(", ".join(selected_clients), metrics, grade, ledger, dept_fig)

    # SECTION 5 — Driver Forensics
    client_task = compute_client_task_mix(df_client_window)
    global_task = compute_global_task_median_mix(df_all)
    task_compare = pd.merge(client_task, global_task, on="task_name", how="left")
    task_compare["global_share_pct"] = task_compare["global_share_pct"].fillna(0)
    task_compare["client_share_pct"] = task_compare["share_pct"]
    task_compare["delta_pp"] = task_compare["client_share_pct"] - task_compare["global_share_pct"]
    task_compare = task_compare[task_compare["delta_pp"] > 10].sort_values("delta_pp", ascending=False)

    task_time_fig = None
    staff_cost_time_fig = None
    task_benchmark_fig = None
    delivery_burn_fig = None
    erosion_table = None
    if "task_name" in df_client_window.columns and "staff_name" in df_client_window.columns:
        task_hours = df_client_window.groupby("task_name")["hours_raw"].sum().reset_index()
        top_tasks = task_hours.sort_values("hours_raw", ascending=False).head(8)["task_name"].tolist()
        if "month_key" in df_client_window.columns:
            task_time = df_client_window[df_client_window["task_name"].isin(top_tasks)].copy()
            task_time = task_time.groupby(["month_key", "task_name"])["hours_raw"].sum().reset_index()
            task_time_fig = px.bar(
                task_time,
                x="month_key",
                y="hours_raw",
                color="task_name",
                title="Task Hours Over Time (Top Tasks)",
            )
            task_time_fig.update_layout(xaxis_title="Month", yaxis_title="Hours", legend_title="Task", barmode="stack")
            task_time_fig.update_xaxes(tickformat="%b %Y")

    staffing = pd.DataFrame()
    senior_flag = False
    if "staff_name" in df_client_window.columns:
        if "role" not in df_client_window.columns:
            df_client_window["role"] = "—"
        else:
            df_client_window["role"] = df_client_window["role"].fillna("—")
        staffing = df_client_window.groupby(["staff_name", "role"]).agg(
            hours=("hours_raw", "sum"),
            cost=("base_cost", "sum"),
        ).reset_index()
        staffing["cost_rate"] = np.where(
            staffing["hours"] > 0,
            staffing["cost"] / staffing["hours"],
            np.nan,
        )
        staffing = staffing.sort_values("hours", ascending=False).head(20)

        company_rate = compute_company_cost_rate(df_all)
        client_rate = (
            df_client_window["base_cost"].sum() / df_client_window["hours_raw"].sum()
            if df_client_window["hours_raw"].sum() > 0
            else np.nan
        )
        if pd.notna(company_rate) and pd.notna(client_rate):
            senior_flag = client_rate > company_rate * 1.2
        if "month_key" in df_client_window.columns:
            staff_time = df_client_window[df_client_window["staff_name"].isin(staffing["staff_name"].head(6))].copy()
            staff_time = staff_time.groupby(["month_key", "staff_name"])["base_cost"].sum().reset_index()
            staff_cost_time_fig = px.area(
                staff_time,
                x="month_key",
                y="base_cost",
                color="staff_name",
                title="Staff Cost Over Time (Top Staff)",
            )
            staff_cost_time_fig.update_layout(xaxis_title="Month", yaxis_title="Cost", legend_title="Staff")
            staff_cost_time_fig.update_xaxes(tickformat="%b %Y")

    # Benchmark comparison chart (task mix vs global median)
    if len(client_task) > 0 and len(global_task) > 0:
        task_compare_all = pd.merge(client_task, global_task, on="task_name", how="left")
        task_compare_all["global_share_pct"] = task_compare_all["global_share_pct"].fillna(0)
        task_compare_all = task_compare_all.sort_values("hours_raw", ascending=False).head(8)
        task_compare_long = task_compare_all.melt(
            id_vars=["task_name"],
            value_vars=["share_pct", "global_share_pct"],
            var_name="series",
            value_name="share_pct_value",
        )
        task_compare_long["series"] = task_compare_long["series"].replace(
            {"share_pct": "Client", "global_share_pct": "Benchmark"}
        )
        task_benchmark_fig = px.bar(
            task_compare_long,
            x="task_name",
            y="share_pct_value",
            color="series",
            barmode="group",
            title="Task Mix vs Global Median (Top Tasks)",
        )
        task_benchmark_fig.update_layout(xaxis_title="Task", yaxis_title="Share %", legend_title="")

    # Delivery burn vs quote + expected end date
    if "month_key" in df_client_window.columns:
        monthly_hours = df_client_window.groupby("month_key")["hours_raw"].sum().reset_index()
        monthly_hours = monthly_hours.sort_values("month_key")
        monthly_hours["cumulative_hours"] = monthly_hours["hours_raw"].cumsum()
        delivery_burn_fig = go.Figure()
        delivery_burn_fig.add_trace(
            go.Scatter(
                x=monthly_hours["month_key"],
                y=monthly_hours["cumulative_hours"],
                mode="lines+markers",
                name="Cumulative Hours",
            )
        )
        quote_rollup_total = safe_quote_rollup(df_client_window, [])
        if len(quote_rollup_total) > 0 and "quoted_hours" in quote_rollup_total.columns:
            quoted_hours = quote_rollup_total["quoted_hours"].iloc[0]
            if pd.notna(quoted_hours) and quoted_hours > 0:
                delivery_burn_fig.add_hline(
                    y=quoted_hours,
                    line_dash="dash",
                    line_color="#444",
                    annotation_text="Quoted Hours",
                    annotation_position="top left",
                )
        if "job_due_date" in df_client_window.columns:
            job_due = df_client_window[["job_no", "job_due_date"]].drop_duplicates()
            job_due["job_due_date"] = pd.to_datetime(job_due["job_due_date"], errors="coerce")
            expected_end = job_due["job_due_date"].dropna().median()
            if pd.notna(expected_end):
                expected_end = pd.to_datetime(expected_end)
                expected_end_dt = expected_end.to_pydatetime()
                delivery_burn_fig.add_shape(
                    type="line",
                    x0=expected_end_dt,
                    x1=expected_end_dt,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(dash="dot", color="#888"),
                )
                delivery_burn_fig.add_annotation(
                    x=expected_end_dt,
                    y=1.02,
                    xref="x",
                    yref="paper",
                    text="Expected end",
                    showarrow=False,
                )
        delivery_burn_fig.update_layout(
            title="Delivery Burn vs Quote",
            xaxis_title="Month",
            yaxis_title="Cumulative Hours",
        )
        job_fin = df_client_window.groupby("job_no").agg(
            revenue=("rev_alloc", "sum"),
            cost=("base_cost", "sum"),
            hours=("hours_raw", "sum"),
        ).reset_index()
        job_fin["margin"] = job_fin["revenue"] - job_fin["cost"]
        job_fin["margin_pct"] = np.where(
            job_fin["revenue"] > 0,
            job_fin["margin"] / job_fin["revenue"] * 100,
            np.nan,
        )
        quote_by_job = safe_quote_rollup(df_client_window, ["job_no"])
        if len(quote_by_job) > 0:
            job_fin = job_fin.merge(
                quote_by_job[["job_no", "quoted_hours"]],
                on="job_no",
                how="left",
            )
        else:
            job_fin["quoted_hours"] = np.nan
        job_fin["hours_overrun_pct"] = np.where(
            job_fin["quoted_hours"] > 0,
            (job_fin["hours"] - job_fin["quoted_hours"]) / job_fin["quoted_hours"] * 100,
            np.nan,
        )
        erosion = job_fin[
            (job_fin["margin_pct"] < 10) | (job_fin["hours_overrun_pct"] > 10)
        ].copy()
        if len(erosion) > 0:
            erosion = erosion.sort_values(
                ["margin_pct", "hours_overrun_pct"],
                ascending=[True, False],
            ).head(12)
            erosion_table = erosion.rename(columns={
                "job_no": "Job",
                "margin_pct": "Margin %",
                "hours_overrun_pct": "Hours Overrun %",
                "revenue": "Revenue",
                "cost": "Cost",
                "margin": "Margin",
            })

    render_client_driver_forensics(
        task_compare,
        staffing,
        senior_flag,
        task_time_fig=task_time_fig,
        staff_cost_time_fig=staff_cost_time_fig,
        task_benchmark_fig=task_benchmark_fig,
        delivery_burn_fig=delivery_burn_fig,
        erosion_table=erosion_table,
    )

    # SECTION 6 — LTV & Trends (unfiltered data)
    ltv_client = selected_clients[0]
    if len(selected_clients) > 1:
        st.info(f"LTV view reflects the first selected client: {ltv_client}.")
    ltv = compute_client_ltv(df_all, ltv_client)
    tenure_months = compute_client_tenure_months(df_all, ltv_client)

    cumulative = ltv.get("cumulative", pd.DataFrame())
    if len(cumulative) > 0:
        cumulative_fig = charts.cumulative_profit_line(
            cumulative,
            x="months_since_start",
            y="cumulative_profit",
            title="Cumulative Profit Since First Job",
        )
    else:
        cumulative_fig = charts.cumulative_profit_line(pd.DataFrame({"months_since_start": [], "cumulative_profit": []}), x="months_since_start", y="cumulative_profit")

    monthly = ltv.get("monthly", pd.DataFrame())
    margin_fig = _build_margin_trend(monthly)

    render_client_ltv_section(cumulative_fig, margin_fig, tenure_months)
    render_client_ltv_methodology_expander()


if __name__ == "__main__":
    main()
