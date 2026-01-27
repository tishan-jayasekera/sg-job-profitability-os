"""
Job Mix & Implied FTE Demand Page

Analyze job intake patterns and implied capacity demand.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state, get_state, set_state
from src.ui.layout import section_header
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_count
from src.ui.charts import time_series, quadrant_scatter, horizontal_bar
from src.data.loader import load_fact_timesheet
from src.data.semantic import safe_quote_job_task, get_category_col
from src.data.cohorts import filter_by_time_window
from src.config import config


st.set_page_config(page_title="Job Mix & Demand", page_icon="📊", layout="wide")

init_state()

def _compute_staff_month_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute staff-month totals with effective FTE scaling from timesheets."""
    if "staff_name" not in df.columns or "month_key" not in df.columns:
        return pd.DataFrame()

    df_base = df.copy()
    if "hours_raw" not in df_base.columns:
        df_base["hours_raw"] = 0.0

    if "fte_hours_scaling" in df_base.columns:
        df_base["fte_hours_scaling"] = df_base["fte_hours_scaling"].fillna(
            config.DEFAULT_FTE_SCALING
        )

        def _weighted_fte(group: pd.DataFrame) -> float:
            weights = group["hours_raw"].fillna(0)
            total = weights.sum()
            if total > 0:
                return float(np.average(group["fte_hours_scaling"], weights=weights))
            return float(group["fte_hours_scaling"].mean())

        grouped = df_base.groupby(["staff_name", "month_key"])
        total_hours = grouped["hours_raw"].sum().rename("total_hours")
        fte_scaling = grouped.apply(_weighted_fte).rename("fte_scaling")
        staff_month_totals = pd.concat([total_hours, fte_scaling], axis=1).reset_index()
    else:
        staff_month_totals = df_base.groupby(["staff_name", "month_key"]).agg(
            total_hours=("hours_raw", "sum"),
        ).reset_index()
        staff_month_totals["fte_scaling"] = config.DEFAULT_FTE_SCALING

    return staff_month_totals


def calculate_job_cohorts(df: pd.DataFrame, cohort_type: str = "first_activity") -> pd.DataFrame:
    """
    Calculate job cohorts by intake period.
    
    cohort_type: 'first_activity', 'first_revenue', or 'quote_date'
    """
    if "job_no" not in df.columns:
        return pd.DataFrame()
    
    if cohort_type == "first_activity":
        # First month with hours > 0
        cohort = df[df["hours_raw"] > 0].groupby("job_no")["month_key"].min().reset_index()
        cohort.columns = ["job_no", "cohort_month"]
    
    elif cohort_type == "first_revenue":
        # First month with revenue > 0
        cohort = df[df["rev_alloc"] > 0].groupby("job_no")["month_key"].min().reset_index()
        cohort.columns = ["job_no", "cohort_month"]
    
    else:
        # Default to first activity
        cohort = df[df["hours_raw"] > 0].groupby("job_no")["month_key"].min().reset_index()
        cohort.columns = ["job_no", "cohort_month"]
    
    return cohort


def calculate_job_mix_metrics(df: pd.DataFrame, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate job mix metrics by cohort month."""
    
    # Get job-level attributes
    category_col = get_category_col(df)
    job_attrs = df.groupby("job_no").agg(
        department_final=("department_final", "first"),
        job_category=(category_col, "first"),
    ).reset_index()
    
    # Safe quote totals per job
    job_task = safe_quote_job_task(df)
    if len(job_task) > 0:
        job_quotes = job_task.groupby("job_no").agg(
            quoted_hours=("quoted_time_total", "sum"),
            quoted_amount=("quoted_amount_total", "sum"),
        ).reset_index()
    else:
        job_quotes = pd.DataFrame({"job_no": [], "quoted_hours": [], "quoted_amount": []})

    job_actuals = df.groupby("job_no", dropna=False).agg(
        actual_hours=("hours_raw", "sum"),
    ).reset_index()
    
    # Merge
    jobs = job_attrs.merge(cohort_df, on="job_no", how="inner")
    jobs = jobs.merge(job_quotes, on="job_no", how="left")
    jobs = jobs.merge(job_actuals, on="job_no", how="left")
    
    # Fill NaN
    jobs["quoted_hours"] = jobs["quoted_hours"].fillna(0)
    jobs["quoted_amount"] = jobs["quoted_amount"].fillna(0)
    jobs["actual_hours"] = jobs["actual_hours"].fillna(0)
    
    # Aggregate by cohort month
    monthly = jobs.groupby("cohort_month").agg(
        job_count=("job_no", "nunique"),
        total_quoted_hours=("quoted_hours", "sum"),
        total_quoted_amount=("quoted_amount", "sum"),
        avg_quoted_hours=("quoted_hours", "mean"),
        avg_quoted_amount=("quoted_amount", "mean"),
        total_actual_hours=("actual_hours", "sum"),
        avg_actual_hours=("actual_hours", "mean"),
    ).reset_index()

    # Derived metrics
    monthly["value_per_quoted_hour"] = np.where(
        monthly["total_quoted_hours"] > 0,
        monthly["total_quoted_amount"] / monthly["total_quoted_hours"],
        np.nan
    )

    monthly["quoted_vs_actual_ratio"] = np.where(
        monthly["total_actual_hours"] > 0,
        monthly["total_quoted_hours"] / monthly["total_actual_hours"],
        np.nan
    )
    
    # Implied FTE demand
    # FTE = quoted_hours / (weeks_in_month * 38)
    weeks_per_month = 4.33
    monthly["implied_fte_quoted"] = (
        monthly["total_quoted_hours"] / (weeks_per_month * config.CAPACITY_HOURS_PER_WEEK)
    )
    monthly["implied_fte_actual"] = (
        monthly["total_actual_hours"] / (weeks_per_month * config.CAPACITY_HOURS_PER_WEEK)
    )

    monthly["cohort_month"] = pd.to_datetime(monthly["cohort_month"])
    
    return monthly.sort_values("cohort_month")


def calculate_demand_vs_supply(df: pd.DataFrame, monthly_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate demand vs supply comparison."""

    # Get supply capacity by month using effective FTE from timesheets
    staff_month_totals = _compute_staff_month_totals(df)
    if len(staff_month_totals) == 0:
        return pd.DataFrame()

    supply = staff_month_totals.groupby("month_key").agg(
        staff_count=("staff_name", "nunique"),
        avg_fte=("fte_scaling", "mean"),
        supply_fte=("fte_scaling", "sum"),
    ).reset_index()

    # Supply capacity
    weeks_per_month = 4.33
    supply["supply_capacity_hours"] = (
        supply["supply_fte"] * weeks_per_month * config.CAPACITY_HOURS_PER_WEEK
    )
    
    # Merge with demand
    comparison = monthly_metrics.merge(
        supply.rename(columns={"month_key": "cohort_month"}),
        on="cohort_month",
        how="left"
    )
    
    # Calculate slack
    comparison["implied_utilisation_quoted"] = np.where(
        comparison["supply_capacity_hours"] > 0,
        comparison["total_quoted_hours"] / comparison["supply_capacity_hours"] * 100,
        np.nan
    )

    comparison["implied_utilisation_actual"] = np.where(
        comparison["supply_capacity_hours"] > 0,
        comparison["total_actual_hours"] / comparison["supply_capacity_hours"] * 100,
        np.nan
    )

    comparison["slack_pct_quoted"] = 100 - comparison["implied_utilisation_quoted"]
    comparison["slack_pct_actual"] = 100 - comparison["implied_utilisation_actual"]
    
    return comparison


def calculate_capacity_vs_delivery(df: pd.DataFrame, df_base: pd.DataFrame = None) -> pd.DataFrame:
    """Calculate operational capacity vs delivered hours by calendar month.

    Capacity is allocated by staff-month share to avoid double counting across groups.
    """
    if df_base is None:
        df_base = df

    weeks_per_month = 4.33
    staff_month_totals = _compute_staff_month_totals(df_base)
    if len(staff_month_totals) == 0:
        return pd.DataFrame()

    slice_hours = df.groupby(["staff_name", "month_key"]).agg(
        slice_hours=("hours_raw", "sum"),
    ).reset_index()

    allocation = slice_hours.merge(staff_month_totals, on=["staff_name", "month_key"], how="left")
    allocation["hour_share"] = np.where(
        allocation["total_hours"] > 0,
        allocation["slice_hours"] / allocation["total_hours"],
        0,
    )
    allocation["supply_fte"] = allocation["fte_scaling"] * allocation["hour_share"]
    allocation["capacity_hours"] = (
        allocation["supply_fte"] * weeks_per_month * config.CAPACITY_HOURS_PER_WEEK
    )

    supply = allocation.groupby("month_key").agg(
        staff_count=("staff_name", "nunique"),
        supply_fte=("supply_fte", "sum"),
        capacity_hours=("capacity_hours", "sum"),
    ).reset_index()
    supply["avg_fte"] = np.where(
        supply["staff_count"] > 0,
        supply["supply_fte"] / supply["staff_count"],
        np.nan,
    )

    delivery = df.groupby("month_key").agg(
        actual_hours=("hours_raw", "sum"),
    ).reset_index()

    if "is_billable" in df.columns:
        billable = df[df["is_billable"] == True].groupby("month_key")["hours_raw"].sum()
        nonbillable = df[df["is_billable"] == False].groupby("month_key")["hours_raw"].sum()
        delivery = delivery.merge(billable.rename("billable_hours"), on="month_key", how="left")
        delivery = delivery.merge(nonbillable.rename("nonbillable_hours"), on="month_key", how="left")
        delivery["billable_hours"] = delivery["billable_hours"].fillna(0)
        delivery["nonbillable_hours"] = delivery["nonbillable_hours"].fillna(0)
    else:
        delivery["billable_hours"] = 0
        delivery["nonbillable_hours"] = 0

    capacity = supply.merge(delivery, on="month_key", how="left")
    capacity["actual_hours"] = capacity["actual_hours"].fillna(0)
    capacity["utilisation_pct"] = np.where(
        capacity["capacity_hours"] > 0,
        capacity["actual_hours"] / capacity["capacity_hours"] * 100,
        np.nan,
    )
    capacity["billable_utilisation_pct"] = np.where(
        capacity["capacity_hours"] > 0,
        capacity["billable_hours"] / capacity["capacity_hours"] * 100,
        np.nan,
    )
    capacity["slack_hours"] = capacity["capacity_hours"] - capacity["actual_hours"]
    capacity["slack_pct"] = 100 - capacity["utilisation_pct"]
    capacity["month_key"] = pd.to_datetime(capacity["month_key"])

    return capacity.sort_values("month_key")


def compute_capacity_summary(
    df: pd.DataFrame, group_col: str, df_base: pd.DataFrame = None
) -> pd.DataFrame:
    """Summarize allocated capacity and delivery by group across the selected time window."""
    if group_col not in df.columns:
        return pd.DataFrame()
    if df_base is None:
        df_base = df

    weeks_per_month = 4.33
    staff_month_totals = _compute_staff_month_totals(df_base)
    if len(staff_month_totals) == 0:
        return pd.DataFrame()

    group_cols = []
    for col in [group_col, "staff_name", "month_key"]:
        if col not in group_cols:
            group_cols.append(col)

    slice_hours = df.groupby(group_cols).agg(
        slice_hours=("hours_raw", "sum"),
    ).reset_index()
    allocation = slice_hours.merge(
        staff_month_totals, on=["staff_name", "month_key"], how="left"
    )
    allocation["hour_share"] = np.where(
        allocation["total_hours"] > 0,
        allocation["slice_hours"] / allocation["total_hours"],
        0,
    )
    allocation["supply_fte"] = allocation["fte_scaling"] * allocation["hour_share"]
    allocation["capacity_hours"] = (
        allocation["supply_fte"] * weeks_per_month * config.CAPACITY_HOURS_PER_WEEK
    )

    monthly = allocation.groupby([group_col, "month_key"]).agg(
        staff_count=("staff_name", "nunique"),
        supply_fte=("supply_fte", "sum"),
        capacity_hours=("capacity_hours", "sum"),
        actual_hours=("slice_hours", "sum"),
    ).reset_index()

    if "is_billable" in df.columns:
        billable = df[df["is_billable"] == True].groupby([group_col, "month_key"])[
            "hours_raw"
        ].sum().rename("billable_hours")
        nonbillable = df[df["is_billable"] == False].groupby([group_col, "month_key"])[
            "hours_raw"
        ].sum().rename("nonbillable_hours")
        monthly = monthly.merge(billable, on=[group_col, "month_key"], how="left")
        monthly = monthly.merge(nonbillable, on=[group_col, "month_key"], how="left")
        monthly["billable_hours"] = monthly["billable_hours"].fillna(0)
        monthly["nonbillable_hours"] = monthly["nonbillable_hours"].fillna(0)
    else:
        monthly["billable_hours"] = 0
        monthly["nonbillable_hours"] = 0

    summary = monthly.groupby(group_col).agg(
        months=("month_key", "nunique"),
        avg_fte_equiv=("supply_fte", "mean"),
        capacity_hours=("capacity_hours", "sum"),
        actual_hours=("actual_hours", "sum"),
        billable_hours=("billable_hours", "sum"),
        nonbillable_hours=("nonbillable_hours", "sum"),
    ).reset_index()

    summary["utilisation_pct"] = np.where(
        summary["capacity_hours"] > 0,
        summary["actual_hours"] / summary["capacity_hours"] * 100,
        np.nan,
    )
    summary["billable_utilisation_pct"] = np.where(
        summary["capacity_hours"] > 0,
        summary["billable_hours"] / summary["capacity_hours"] * 100,
        np.nan,
    )
    summary["slack_hours"] = summary["capacity_hours"] - summary["actual_hours"]
    summary["slack_pct"] = 100 - summary["utilisation_pct"]
    summary["billable_ratio"] = np.where(
        summary["actual_hours"] > 0,
        summary["billable_hours"] / summary["actual_hours"],
        np.nan,
    )

    return summary.sort_values("slack_hours", ascending=False)


def compute_job_volume_deltas(
    df: pd.DataFrame, group_col: str, time_window: str
) -> pd.DataFrame:
    """Compute job volume change vs prior period for a group."""
    if group_col not in df.columns or "job_no" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["month_key"] = pd.to_datetime(df["month_key"])
    available = sorted(df["month_key"].dropna().unique())
    if len(available) < 2:
        return pd.DataFrame()

    if time_window == "6m":
        months = 6
    elif time_window == "12m":
        months = 12
    elif time_window == "24m":
        months = 24
    else:
        months = min(12, len(available))

    if len(available) < months:
        months = len(available)
    if months < 2:
        return pd.DataFrame()

    mid = months // 2
    recent_months = available[-mid:]
    prior_months = available[-(2 * mid):-mid] if len(available) >= 2 * mid else available[:mid]

    recent = df[df["month_key"].isin(recent_months)]
    prior = df[df["month_key"].isin(prior_months)]

    recent_jobs = recent.groupby(group_col)["job_no"].nunique().rename("recent_jobs")
    prior_jobs = prior.groupby(group_col)["job_no"].nunique().rename("prior_jobs")
    deltas = pd.concat([recent_jobs, prior_jobs], axis=1).fillna(0).reset_index()
    deltas["job_volume_delta"] = np.where(
        deltas["prior_jobs"] > 0,
        (deltas["recent_jobs"] - deltas["prior_jobs"]) / deltas["prior_jobs"],
        np.nan,
    )

    return deltas


def main():
    st.title("Job Mix & Implied FTE Demand")
    st.caption("Analyze job intake patterns and capacity implications")
    
    # Load data
    df_all_raw = load_fact_timesheet()
    df_all_raw["month_key"] = pd.to_datetime(df_all_raw["month_key"])
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    cohort_options = {
        "first_activity": "First Activity Month",
        "first_revenue": "First Revenue Month",
    }
    
    cohort_type = st.sidebar.selectbox(
        "Cohort Definition",
        options=list(cohort_options.keys()),
        format_func=lambda x: cohort_options[x]
    )
    set_state("cohort_definition", cohort_type)
    
    time_window = st.sidebar.selectbox(
        "Time Window",
        options=["6m", "12m", "24m", "all"],
        format_func=lambda x: f"Last {x}" if x != "all" else "All Time",
        index=1
    )
    
    departments = ["All"] + sorted(df_all_raw["department_final"].dropna().unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)
    
    if selected_dept != "All":
        df_all = df_all_raw[df_all_raw["department_final"] == selected_dept]
    else:
        df_all = df_all_raw
    
    # Calculate cohorts
    cohort_df = calculate_job_cohorts(df_all, cohort_type)
    
    if len(cohort_df) == 0:
        st.warning("No job data available for analysis.")
        return
    
    # Base window for capacity allocation (always full company, time-windowed)
    df_base_window = filter_by_time_window(df_all_raw, time_window, date_col="month_key")

    # Calculate metrics
    monthly_metrics = calculate_job_mix_metrics(df_all, cohort_df)
    monthly_metrics = filter_by_time_window(monthly_metrics, time_window, date_col="cohort_month")
    monthly_metrics["underquoted_hours"] = (
        monthly_metrics["total_actual_hours"] - monthly_metrics["total_quoted_hours"]
    ).clip(lower=0)
    df_window = filter_by_time_window(df_all, time_window, date_col="month_key")
    capacity_delivery = calculate_capacity_vs_delivery(df_window, df_base=df_base_window)

    # Job-level data for loss diagnosis and quadrant drill
    category_col = get_category_col(df_all)
    job_task = safe_quote_job_task(df_all)
    if len(job_task) > 0:
        job_quotes = job_task.groupby("job_no").agg(
            quoted_hours=("quoted_time_total", "sum"),
            quoted_amount=("quoted_amount_total", "sum"),
        ).reset_index()
    else:
        job_quotes = pd.DataFrame({"job_no": [], "quoted_hours": [], "quoted_amount": []})

    job_actuals = df_all.groupby("job_no", dropna=False).agg(
        actual_hours=("hours_raw", "sum"),
    ).reset_index()

    job_attrs = df_all.groupby("job_no").agg(
        department_final=("department_final", "first"),
        job_category=(category_col, "first"),
    ).reset_index()

    job_level = job_attrs.merge(job_quotes, on="job_no", how="left")
    job_level = job_level.merge(job_actuals, on="job_no", how="left")
    job_level = job_level.merge(cohort_df, on="job_no", how="left")
    job_level["quoted_hours"] = job_level["quoted_hours"].fillna(0)
    job_level["quoted_amount"] = job_level["quoted_amount"].fillna(0)
    job_level["actual_hours"] = job_level["actual_hours"].fillna(0)
    job_level["underquoted_hours"] = (job_level["actual_hours"] - job_level["quoted_hours"]).clip(lower=0)
    job_level["underquoted_ratio"] = np.where(
        job_level["actual_hours"] > 0,
        job_level["quoted_hours"] / job_level["actual_hours"],
        np.nan,
    )
    job_level["cohort_month"] = pd.to_datetime(job_level["cohort_month"])
    job_level = filter_by_time_window(job_level, time_window, date_col="cohort_month")
    
    # =========================================================================
    # SECTION A: KPI STRIP
    # =========================================================================
    section_header("Job Mix Summary")
    
    kpi_cols = st.columns(8)
    
    total_jobs = monthly_metrics["job_count"].sum()
    total_quoted_hours = monthly_metrics["total_quoted_hours"].sum()
    total_quoted_amount = monthly_metrics["total_quoted_amount"].sum()
    total_actual_hours = monthly_metrics["total_actual_hours"].sum()
    avg_hours_per_job = total_quoted_hours / total_jobs if total_jobs > 0 else 0
    avg_amount_per_job = total_quoted_amount / total_jobs if total_jobs > 0 else 0
    value_per_quoted_hour = total_quoted_amount / total_quoted_hours if total_quoted_hours > 0 else 0
    quoted_vs_actual_ratio = total_quoted_hours / total_actual_hours if total_actual_hours > 0 else np.nan
    
    with kpi_cols[0]:
        st.metric("Jobs", fmt_count(total_jobs))
    
    with kpi_cols[1]:
        st.metric("Total Quoted", fmt_currency(total_quoted_amount))
    
    with kpi_cols[2]:
        st.metric("Avg $/Job", fmt_currency(avg_amount_per_job))
    
    with kpi_cols[3]:
        st.metric("Total Hours", fmt_hours(total_quoted_hours))
    
    with kpi_cols[4]:
        st.metric("Avg Hrs/Job", fmt_hours(avg_hours_per_job))
    
    with kpi_cols[5]:
        st.metric("Total Actual Hours", fmt_hours(total_actual_hours))

    with kpi_cols[6]:
        st.metric("Quoted/Actual Hrs", f"{quoted_vs_actual_ratio:.2f}x" if pd.notna(quoted_vs_actual_ratio) else "—")

    with kpi_cols[7]:
        st.metric("$/Quoted Hr", fmt_currency(value_per_quoted_hour))

    with st.expander("How these are calculated", expanded=False):
        st.markdown(
            """
**Executive question**  
Are we quoting the right amount of work, and does actual delivery track what we planned?

**Approach**  
We count distinct jobs, use quotes to measure planned scope, and compare that plan to real timesheet
hours. This yields a clean view of pricing and workload sizing.

**Key calculations (plain English)**  
- **Jobs**: unique job count in the period.  
- **Total Quoted Hours**: promised hours, summed safely at job‑task level.  
- **Total Actual Hours**: hours actually worked.  
- **Avg $/Job**: quoted dollars per job.  
- **$/Quoted Hr**: value per quoted hour (pricing signal).  
- **Quoted vs Actual**: planned hours ÷ actual hours (below 1.0 = under‑quoted).

**Why “safe” totals matter**  
Quotes repeat on each timesheet row. We dedupe at (job_no, task_name) before summing so the
planned hours aren’t inflated.
            """
        )
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION B: TREND CHARTS
    # =========================================================================
    section_header("Monthly Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_bar(
            x=monthly_metrics["cohort_month"],
            y=monthly_metrics["job_count"],
            name="Jobs",
            marker_color="#4c78a8",
        )
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["avg_quoted_amount"],
                name="Avg Quote Value",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#f58518", width=2),
            )
        )
        fig.update_layout(
            title="Job Volume vs Quote Value",
            yaxis=dict(title="Jobs"),
            yaxis2=dict(title="Avg Quote Value ($)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["avg_quoted_hours"],
                name="Avg Quoted Hours",
                mode="lines+markers",
                line=dict(color="#54a24b", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["avg_actual_hours"],
                name="Avg Actual Hours",
                mode="lines+markers",
                line=dict(color="#e45756", width=2),
            )
        )
        fig.update_layout(
            title="Avg Hours per Job: Quoted vs Actual",
            yaxis=dict(title="Hours"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["total_quoted_hours"],
                name="Quoted Hours",
                mode="lines+markers",
                line=dict(color="#72b7b2", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["total_actual_hours"],
                name="Actual Hours",
                mode="lines+markers",
                line=dict(color="#b279a2", width=2),
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["underquoted_hours"],
                name="Under-Quoted Hours",
                marker_color="rgba(228,87,86,0.35)",
            )
        )
        fig.update_layout(
            title="Total Hours: Quoted vs Actual",
            yaxis=dict(title="Hours"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = time_series(
            monthly_metrics,
            x="cohort_month",
            y="value_per_quoted_hour",
            title="Value per Quoted Hour",
            y_title="$/hr",
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(job_level) > 0:
        top_depts = (
            job_level.groupby("department_final")["underquoted_hours"]
            .sum()
            .sort_values(ascending=False)
            .head(4)
            .index
            .tolist()
        )
        dept_month = job_level[job_level["department_final"].isin(top_depts)].groupby(
            ["cohort_month", "department_final"]
        )["underquoted_hours"].sum().reset_index()
        if len(dept_month) > 0:
            fig = go.Figure()
            for dept in top_depts:
                dept_slice = dept_month[dept_month["department_final"] == dept]
                fig.add_bar(
                    x=dept_slice["cohort_month"],
                    y=dept_slice["underquoted_hours"],
                    name=dept,
                )
            fig.update_layout(
                title="Under-Quoted Hours by Department (Top 4)",
                yaxis=dict(title="Under-Quoted Hours"),
                barmode="stack",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION C: JOB QUADRANT
    # =========================================================================
    section_header("Job Value vs Effort", "Identify 'high value / low hours' segment")
    
    if len(job_level) > 0:
        job_quotes = job_level[(job_level["quoted_hours"] > 0) & (job_level["quoted_amount"] > 0)].copy()

        if len(job_quotes) > 5:
            # Chain filter for this section
            chain_cols = [
                ("department_final", "Department"),
                ("job_category", "Category"),
            ]
            chain_filters = {}
            chain_box = st.columns(len(chain_cols))
            for idx, (col, label) in enumerate(chain_cols):
                if col not in job_quotes.columns:
                    continue
                options = ["All"] + sorted(job_quotes[col].dropna().unique().tolist())
                with chain_box[idx]:
                    choice = st.selectbox(f"{label} (this section)", options, key=f"job_value_chain_{col}")
                if choice != "All":
                    chain_filters[col] = choice

            if chain_filters:
                for col, value in chain_filters.items():
                    job_quotes = job_quotes[job_quotes[col] == value]

            fig = quadrant_scatter(
                job_quotes,
                x="quoted_hours",
                y="quoted_amount",
                hover_name="job_no",
                title="Job Portfolio: Value vs Effort",
                x_title="Quoted Hours",
                y_title="Quoted Amount ($)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats by quadrant
            med_hours = job_quotes["quoted_hours"].median()
            med_amount = job_quotes["quoted_amount"].median()

            job_quotes["quadrant"] = np.where(
                (job_quotes["quoted_hours"] < med_hours) & (job_quotes["quoted_amount"] > med_amount),
                "High Value / Low Effort",
                np.where(
                    (job_quotes["quoted_hours"] >= med_hours) & (job_quotes["quoted_amount"] > med_amount),
                    "High Value / High Effort",
                    np.where(
                        (job_quotes["quoted_hours"] < med_hours) & (job_quotes["quoted_amount"] <= med_amount),
                        "Low Value / Low Effort",
                        "Low Value / High Effort",
                    ),
                ),
            )

            quadrant_summary = job_quotes.groupby("quadrant").agg(
                jobs=("job_no", "count"),
                avg_hours=("quoted_hours", "mean"),
                avg_value=("quoted_amount", "mean"),
            ).reset_index()

            st.dataframe(
                quadrant_summary.rename(columns={
                    "quadrant": "Quadrant",
                    "jobs": "Jobs",
                    "avg_hours": "Avg Hours",
                    "avg_value": "Avg Value",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Avg Hours": st.column_config.NumberColumn(format="%.0f"),
                    "Avg Value": st.column_config.NumberColumn(format="$%.0f"),
                },
            )

            quadrant_options = [
                "High Value / Low Effort",
                "High Value / High Effort",
                "Low Value / Low Effort",
                "Low Value / High Effort",
            ]
            selected_quadrant = st.selectbox(
                "Inspect Jobs in Quadrant",
                options=quadrant_options,
                index=0,
            )

            quadrant_jobs = job_quotes[job_quotes["quadrant"] == selected_quadrant].copy()
            quadrant_jobs["quoted_vs_actual"] = np.where(
                quadrant_jobs["actual_hours"] > 0,
                quadrant_jobs["quoted_hours"] / quadrant_jobs["actual_hours"],
                np.nan,
            )

            st.subheader("Quadrant Portfolio: Jobs + Profitability Context")
            st.caption(
                "This table converts the quadrant into a decision list: who the client is, how much was quoted, "
                "what has been earned so far, and whether margin is holding up."
            )

            job_financials = df_window.groupby("job_no").agg(
                revenue_to_date=("rev_alloc", "sum") if "rev_alloc" in df_window.columns else ("job_no", "count"),
                cost_to_date=("base_cost", "sum") if "base_cost" in df_window.columns else ("job_no", "count"),
                hours_to_date=("hours_raw", "sum"),
            ).reset_index()
            job_financials["margin_to_date"] = job_financials["revenue_to_date"] - job_financials["cost_to_date"]
            job_financials["margin_pct_to_date"] = np.where(
                job_financials["revenue_to_date"] > 0,
                job_financials["margin_to_date"] / job_financials["revenue_to_date"] * 100,
                np.nan,
            )

            job_meta = df_all.groupby("job_no").agg(
                client=("client", "first") if "client" in df_all.columns else ("job_no", "first"),
                job_status=("job_status", "first") if "job_status" in df_all.columns else ("job_no", "first"),
                job_completed_date=("job_completed_date", "first") if "job_completed_date" in df_all.columns else ("job_no", "first"),
            ).reset_index()
            if "job_completed_date" in job_meta.columns:
                job_meta["is_active"] = job_meta["job_completed_date"].isna()
            elif "job_status" in job_meta.columns:
                job_meta["is_active"] = ~job_meta["job_status"].str.lower().str.contains("completed", na=False)
            else:
                job_meta["is_active"] = True

            quadrant_detail = quadrant_jobs.merge(job_financials, on="job_no", how="left")
            quadrant_detail = quadrant_detail.merge(job_meta, on="job_no", how="left")
            quadrant_detail["quote_to_revenue"] = np.where(
                quadrant_detail["quoted_amount"] > 0,
                quadrant_detail["revenue_to_date"] / quadrant_detail["quoted_amount"],
                np.nan,
            )
            quadrant_detail["hours_overrun_pct"] = np.where(
                quadrant_detail["quoted_hours"] > 0,
                (quadrant_detail["actual_hours"] - quadrant_detail["quoted_hours"]) / quadrant_detail["quoted_hours"] * 100,
                np.nan,
            )
            quadrant_detail["quote_rate"] = np.where(
                quadrant_detail["quoted_hours"] > 0,
                quadrant_detail["quoted_amount"] / quadrant_detail["quoted_hours"],
                np.nan,
            )
            quadrant_detail["realised_rate"] = np.where(
                quadrant_detail["actual_hours"] > 0,
                quadrant_detail["revenue_to_date"] / quadrant_detail["actual_hours"],
                np.nan,
            )

            st.subheader("Active Jobs at Risk (Margin Erosion)")
            st.caption(
                "Focus list for delivery leaders. These are active jobs where margins are under pressure "
                "or trending below quote economics."
            )

            def _risk_reasons(row: pd.Series) -> str:
                reasons = []
                if pd.notna(row.get("margin_pct_to_date")) and row["margin_pct_to_date"] < 15:
                    reasons.append("Low margin % to date")
                if pd.notna(row.get("quote_to_revenue")) and row["quote_to_revenue"] < 0.7:
                    reasons.append("Revenue lagging quote")
                if pd.notna(row.get("hours_overrun_pct")) and row["hours_overrun_pct"] > 10:
                    reasons.append("Hours overrun vs quote")
                if pd.notna(row.get("realised_rate")) and pd.notna(row.get("quote_rate")):
                    if row["realised_rate"] < row["quote_rate"] * 0.85:
                        reasons.append("Realised rate below quote rate")
                return "; ".join(reasons)

            active_risk = quadrant_detail[quadrant_detail["is_active"] == True].copy()
            active_risk["risk_reason"] = active_risk.apply(_risk_reasons, axis=1)
            active_risk = active_risk[active_risk["risk_reason"] != ""]
            active_risk = active_risk.sort_values(["margin_pct_to_date", "quote_to_revenue"], ascending=[True, True])

            if len(active_risk) > 0:
                st.dataframe(
                    active_risk.head(12)[
                        [
                            "job_no",
                            "client",
                            "department_final",
                            "job_category",
                            "margin_pct_to_date",
                            "quote_to_revenue",
                            "hours_overrun_pct",
                            "realised_rate",
                            "quote_rate",
                            "risk_reason",
                        ]
                    ].rename(columns={
                        "job_no": "Job",
                        "client": "Client",
                        "department_final": "Department",
                        "job_category": "Category",
                        "margin_pct_to_date": "Margin %",
                        "quote_to_revenue": "Revenue / Quote",
                        "hours_overrun_pct": "Hours Overrun %",
                        "realised_rate": "Realised Rate",
                        "quote_rate": "Quote Rate",
                        "risk_reason": "Why At Risk",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Revenue / Quote": st.column_config.NumberColumn(format="%.2fx"),
                        "Hours Overrun %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Realised Rate": st.column_config.NumberColumn(format="$%.0f"),
                        "Quote Rate": st.column_config.NumberColumn(format="$%.0f"),
                    },
                )
            else:
                st.success("No active jobs flagged for margin erosion in this quadrant.")

            st.dataframe(
                quadrant_detail.sort_values("quoted_amount", ascending=False)[
                    [
                        "job_no",
                        "client",
                        "department_final",
                        "job_category",
                        "quoted_amount",
                        "revenue_to_date",
                        "margin_to_date",
                        "margin_pct_to_date",
                        "quoted_hours",
                        "actual_hours",
                        "hours_overrun_pct",
                        "quote_to_revenue",
                        "is_active",
                    ]
                ].rename(columns={
                    "job_no": "Job",
                    "client": "Client",
                    "department_final": "Department",
                    "job_category": "Category",
                    "quoted_amount": "Quoted $",
                    "revenue_to_date": "Revenue to Date",
                    "margin_to_date": "Margin to Date",
                    "margin_pct_to_date": "Margin %",
                    "quoted_hours": "Quoted Hours",
                    "actual_hours": "Actual Hours",
                    "hours_overrun_pct": "Hours Overrun %",
                    "quote_to_revenue": "Revenue / Quote",
                    "is_active": "Active",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Quoted $": st.column_config.NumberColumn(format="$%.0f"),
                    "Revenue to Date": st.column_config.NumberColumn(format="$%.0f"),
                    "Margin to Date": st.column_config.NumberColumn(format="$%.0f"),
                    "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Quoted Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Actual Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Hours Overrun %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Revenue / Quote": st.column_config.NumberColumn(format="%.2fx"),
                    "Active": st.column_config.CheckboxColumn(),
                },
            )

            st.subheader("Job Deep‑Dive: Margin Drivers and Overruns")
            job_options = quadrant_detail["job_no"].dropna().unique().tolist()
            if job_options:
                selected_job = st.selectbox("Select a job to trace margin drivers", job_options, key="job_value_job_select")
                df_job = df_window[df_window["job_no"] == selected_job].copy()

                st.markdown("**1) Profitability to Date (Selected Window)**")
                job_fin = quadrant_detail[quadrant_detail["job_no"] == selected_job].iloc[0]
                kpi_cols = st.columns(5)
                with kpi_cols[0]:
                    st.metric("Revenue to Date", fmt_currency(job_fin["revenue_to_date"]))
                with kpi_cols[1]:
                    st.metric("Cost to Date", fmt_currency(job_fin["cost_to_date"]))
                with kpi_cols[2]:
                    st.metric("Margin to Date", fmt_currency(job_fin["margin_to_date"]))
                with kpi_cols[3]:
                    st.metric("Margin %", fmt_percent(job_fin["margin_pct_to_date"]))
                with kpi_cols[4]:
                    st.metric("Revenue / Quote", f"{job_fin['quote_to_revenue']:.2f}x" if pd.notna(job_fin["quote_to_revenue"]) else "—")

                st.markdown("**2) Task Cost Leaders & Overruns**")
                if len(df_job) > 0:
                    task_cost = df_job.groupby("task_name").agg(
                        hours=("hours_raw", "sum"),
                        cost=("base_cost", "sum") if "base_cost" in df_job.columns else ("hours_raw", "sum"),
                        revenue=("rev_alloc", "sum") if "rev_alloc" in df_job.columns else ("hours_raw", "sum"),
                    ).reset_index()

                    jt = safe_quote_job_task(df_job)
                    if len(jt) > 0 and "quoted_time_total" in jt.columns:
                        task_cost = task_cost.merge(
                            jt[["task_name", "quoted_time_total"]].rename(columns={"quoted_time_total": "quoted_hours"}),
                            on="task_name",
                            how="left",
                        )
                    else:
                        task_cost["quoted_hours"] = np.nan

                    task_cost["hours_overrun"] = task_cost["hours"] - task_cost["quoted_hours"]
                    task_cost["hours_overrun_pct"] = np.where(
                        task_cost["quoted_hours"] > 0,
                        task_cost["hours_overrun"] / task_cost["quoted_hours"] * 100,
                        np.nan,
                    )
                    task_cost["margin"] = task_cost["revenue"] - task_cost["cost"]
                    task_cost["margin_pct"] = np.where(
                        task_cost["revenue"] > 0,
                        task_cost["margin"] / task_cost["revenue"] * 100,
                        np.nan,
                    )

                    st.dataframe(
                        task_cost.sort_values("cost", ascending=False).head(12).rename(columns={
                            "task_name": "Task",
                            "hours": "Hours",
                            "cost": "Cost",
                            "revenue": "Revenue",
                            "margin": "Margin",
                            "margin_pct": "Margin %",
                            "quoted_hours": "Quoted Hours",
                            "hours_overrun_pct": "Hours Overrun %",
                        }),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Hours": st.column_config.NumberColumn(format="%.1f"),
                            "Cost": st.column_config.NumberColumn(format="$%.0f"),
                            "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                            "Margin": st.column_config.NumberColumn(format="$%.0f"),
                            "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                            "Quoted Hours": st.column_config.NumberColumn(format="%.1f"),
                            "Hours Overrun %": st.column_config.NumberColumn(format="%.1f%%"),
                        },
                    )

                st.markdown("**3) Staff Drivers (FTE‑aware)**")
                if len(df_job) > 0 and "staff_name" in df_job.columns:
                    staff_cost = df_job.groupby("staff_name").agg(
                        hours=("hours_raw", "sum"),
                        cost=("base_cost", "sum") if "base_cost" in df_job.columns else ("hours_raw", "sum"),
                        fte_scaling=("fte_hours_scaling", "first") if "fte_hours_scaling" in df_job.columns else ("staff_name", "count"),
                    ).reset_index()
                    staff_cost = staff_cost.sort_values("cost", ascending=False).head(10)
                    st.dataframe(
                        staff_cost.rename(columns={
                            "staff_name": "Staff",
                            "hours": "Hours",
                            "cost": "Cost",
                            "fte_scaling": "FTE Scaling",
                        }),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Hours": st.column_config.NumberColumn(format="%.1f"),
                            "Cost": st.column_config.NumberColumn(format="$%.0f"),
                            "FTE Scaling": st.column_config.NumberColumn(format="%.2f"),
                        },
                    )

            # Profitability evolution (selected quadrant + chain filters)
            if "rev_alloc" in df_window.columns and "base_cost" in df_window.columns:
                profit_month = df_window.groupby(["month_key", "job_no"]).agg(
                    revenue=("rev_alloc", "sum"),
                    cost=("base_cost", "sum"),
                    hours=("hours_raw", "sum"),
                ).reset_index()
                profit_month["margin"] = profit_month["revenue"] - profit_month["cost"]
                profit_month["margin_pct"] = np.where(
                    profit_month["revenue"] > 0,
                    profit_month["margin"] / profit_month["revenue"] * 100,
                    np.nan,
                )
                profit_month["realised_rate"] = np.where(
                    profit_month["hours"] > 0,
                    profit_month["revenue"] / profit_month["hours"],
                    np.nan,
                )

                prof_jobs = set(quadrant_jobs["job_no"].unique().tolist())
                prof_slice = profit_month[profit_month["job_no"].isin(prof_jobs)]
                if len(prof_slice) > 0:
                    trend = prof_slice.groupby("month_key").agg(
                        revenue=("revenue", "sum"),
                        cost=("cost", "sum"),
                        hours=("hours", "sum"),
                    ).reset_index()
                    trend["margin"] = trend["revenue"] - trend["cost"]
                    trend["margin_pct"] = np.where(
                        trend["revenue"] > 0,
                        trend["margin"] / trend["revenue"] * 100,
                        np.nan,
                    )
                    trend["realised_rate"] = np.where(
                        trend["hours"] > 0,
                        trend["revenue"] / trend["hours"],
                        np.nan,
                    )

                    st.subheader("Profitability Evolution (Selected Quadrant)")
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=trend["month_key"],
                            y=trend["margin_pct"],
                            name="Margin %",
                            mode="lines+markers",
                            line=dict(color="#54a24b", width=2),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=trend["month_key"],
                            y=trend["realised_rate"],
                            name="Realised Rate",
                            mode="lines+markers",
                            yaxis="y2",
                            line=dict(color="#4c78a8", width=2),
                        )
                    )
                    fig.update_layout(
                        yaxis=dict(title="Margin %"),
                        yaxis2=dict(title="Realised Rate", overlaying="y", side="right"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        title="Margin % and Realised Rate Over Time",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(
                        trend.rename(columns={
                            "month_key": "Month",
                            "revenue": "Revenue",
                            "cost": "Cost",
                            "margin": "Margin",
                            "margin_pct": "Margin %",
                            "realised_rate": "Realised Rate",
                        }),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Month": st.column_config.DateColumn(format="YYYY-MM"),
                            "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                            "Cost": st.column_config.NumberColumn(format="$%.0f"),
                            "Margin": st.column_config.NumberColumn(format="$%.0f"),
                            "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                            "Realised Rate": st.column_config.NumberColumn(format="$%.0f"),
                        },
                    )

            st.info(
                """
Operational signals:
- High Value / Low Effort: productize and standardize; protect scope and pricing.
- High Value / High Effort: review estimation, resourcing, and margin guardrails.
- Low Value / Low Effort: automate or bundle; consider off-peak capacity.
- Low Value / High Effort: strong candidates for re-pricing or exit.
                """
            )
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION D: OPERATIONAL CAPACITY (DELIVERY VS SUPPLY)
    # =========================================================================
    section_header("Operational Capacity", "Actual work delivered vs available capacity")

    if len(capacity_delivery) > 0:
        growth_pct = st.slider("Sales growth scenario (demand increase)", 0, 50, 0, step=5)
        capacity_delivery["scenario_hours"] = capacity_delivery["actual_hours"] * (1 + growth_pct / 100)
        capacity_delivery["scenario_utilisation"] = np.where(
            capacity_delivery["capacity_hours"] > 0,
            capacity_delivery["scenario_hours"] / capacity_delivery["capacity_hours"] * 100,
            np.nan,
        )

        col1, col2 = st.columns(2)

        with col1:
            dept_summary = compute_capacity_summary(df_window, "department_final", df_base=df_base_window)
            if len(dept_summary) > 0:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=dept_summary["department_final"],
                        y=dept_summary["actual_hours"],
                        name="Actual Hours",
                        marker_color="#4c78a8",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        x=dept_summary["department_final"],
                        y=dept_summary["slack_hours"],
                        name="Slack Hours",
                        marker_color="#e45756",
                    )
                )
                fig.update_layout(
                    title="Capacity Composition by Department",
                    yaxis=dict(title="Hours"),
                    barmode="stack",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=capacity_delivery["month_key"],
                    y=capacity_delivery["capacity_hours"],
                    name="Capacity Hours",
                    mode="lines+markers",
                    line=dict(color="#4c78a8", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=capacity_delivery["month_key"],
                    y=capacity_delivery["actual_hours"],
                    name="Actual Delivered Hours",
                    mode="lines+markers",
                    line=dict(color="#f58518", width=2),
                )
            )
            fig.update_layout(
                title="Capacity vs Delivered Hours",
                yaxis=dict(title="Hours"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=capacity_delivery["month_key"],
                    y=capacity_delivery["utilisation_pct"],
                    name="Utilisation (Total Hours)",
                    mode="lines+markers",
                    line=dict(color="#54a24b", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=capacity_delivery["month_key"],
                    y=capacity_delivery["scenario_utilisation"],
                    name=f"Utilisation (Total, +{growth_pct}%)",
                    mode="lines+markers",
                    line=dict(color="#e45756", width=2, dash="dash"),
                )
            )
            fig.update_layout(
                title="Total Utilisation Under Growth Scenario",
                yaxis=dict(title="%"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig.add_hline(y=100, line_dash="dash", line_color="red",
                          annotation_text="Full Capacity")
            st.plotly_chart(fig, use_container_width=True)

        latest = capacity_delivery.sort_values("month_key").tail(1)
        if len(latest) > 0:
            latest_row = latest.iloc[0]
            kpi_cols = st.columns(7)
            with kpi_cols[0]:
                st.metric("FTE Equiv", f"{latest_row['supply_fte']:.2f}")
            with kpi_cols[1]:
                st.metric("Avg FTE", f"{latest_row['avg_fte']:.2f}")
            with kpi_cols[2]:
                st.metric("Capacity Hours", fmt_hours(latest_row["capacity_hours"]))
            with kpi_cols[3]:
                st.metric("Actual Hours", fmt_hours(latest_row["actual_hours"]))
            with kpi_cols[4]:
                st.metric("Slack Hours", fmt_hours(latest_row["slack_hours"]))
            with kpi_cols[5]:
                st.metric("Utilisation (Total)", fmt_percent(latest_row["utilisation_pct"]))
            with kpi_cols[6]:
                st.metric("Billable Utilisation", fmt_percent(latest_row["billable_utilisation_pct"]))

        st.info(
            """
Decision guide:
- If total utilisation is consistently low and slack is high, you have capacity — sales/BD focus.
- If total utilisation is near 100% and slack is low/negative, you need hiring or reprioritisation.
- If total utilisation is moderate but billable utilisation is low, this is a resourcing/ops mix issue.
            """
        )

        st.caption(
            "Executive summary: we translate real timesheet activity into a fair monthly capacity baseline, "
            "then compare it to what was delivered. This shows true slack or overload without double counting "
            "people who work across teams."
        )

        with st.expander("Methodology and reconciliation (operational view)", expanded=False):
            st.markdown(
                """
**Executive question**  
Do we have true spare capacity, or are we fully loaded once we account for how people actually work?

**Design principle**  
Capacity must reconcile across teams. If a person splits time across multiple areas, we split their capacity
the same way — so totals add up cleanly and no one is double‑counted.

**How we build supply (capacity)**  
1) **Start with real activity**: only staff who logged time in the month are counted.  
2) **Compute effective FTE**:  
   - If `fte_hours_scaling` exists, use a hours‑weighted average for that staff‑month.  
   - If it doesn’t, assume 1.0.  
3) **Convert to monthly capacity**: 38 hours/week × 4.33 weeks × effective FTE.  
4) **Allocate capacity across slices**: split a person’s capacity in proportion to their timesheet hours.

**How we measure demand**  
- **Actual Hours**: total hours delivered in the month.  
- **Billable vs Non‑Billable**: split using the `is_billable` flag.

**How to read the output**  
- **Utilisation (Total)** = (Billable + Non‑Billable Hours) ÷ Capacity Hours.  
- **Billable Utilisation** = Billable Hours ÷ Capacity Hours.  
- **Slack Hours** = Capacity Hours − Actual Hours.  
- **Slack %** = 100% − Utilisation (Total).  

This produces an apples‑to‑apples view of capacity vs delivery that reconciles at the company total.
                """
            )

            st.dataframe(
                capacity_delivery[[
                    "month_key",
                    "supply_fte",
                    "avg_fte",
                    "capacity_hours",
                    "actual_hours",
                    "billable_hours",
                    "nonbillable_hours",
                    "utilisation_pct",
                    "billable_utilisation_pct",
                    "slack_hours",
                ]].rename(columns={
                    "month_key": "Month",
                    "supply_fte": "FTE Equiv",
                    "avg_fte": "Avg FTE",
                    "capacity_hours": "Capacity Hours",
                    "actual_hours": "Actual Hours",
                    "billable_hours": "Billable Hours",
                    "nonbillable_hours": "Non‑Billable Hours",
                    "utilisation_pct": "Utilisation (Total)",
                    "billable_utilisation_pct": "Billable Utilisation",
                    "slack_hours": "Slack Hours",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Month": st.column_config.DateColumn(format="YYYY-MM"),
                    "FTE Equiv": st.column_config.NumberColumn(format="%.2f"),
                    "Avg FTE": st.column_config.NumberColumn(format="%.2f"),
                    "Capacity Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Actual Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Billable Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Non‑Billable Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Utilisation (Total)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Billable Utilisation": st.column_config.NumberColumn(format="%.1f%%"),
                    "Slack Hours": st.column_config.NumberColumn(format="%.1f"),
                },
            )

    st.markdown("---")

    # =========================================================================
    # SECTION E: OPERATIONAL DRILLDOWN (DEPARTMENT → CATEGORY → STAFF → BREAKDOWN → TASK)
    # =========================================================================
    section_header("Operational Drilldown", "Follow the delivery chain to locate slack or overload")
    st.caption(
        "How to read this drilldown: capacity is allocated using real timesheet shares, so when you move "
        "from department → category → staff → task, the totals always reconcile. "
        "“Avg FTE (equiv)” reflects capacity‑weighted staffing, not simple headcount."
    )

    category_col = "category_rev_job" if "category_rev_job" in df_window.columns else get_category_col(df_window)
    chain = [
        ("department_final", "Department"),
        (category_col, "Category"),
        ("staff_name", "Staff"),
        ("breakdown", "Breakdown"),
        ("task_name", "Job Task"),
    ]

    drill_df = df_window.copy()
    selections = {}
    cols = st.columns(len(chain))
    for idx, (col, label) in enumerate(chain):
        if col not in drill_df.columns:
            continue
        options = ["All"] + sorted(drill_df[col].dropna().unique().tolist())
        with cols[idx]:
            choice = st.selectbox(label, options, key=f"cap_chain_{col}")
        if choice != "All":
            drill_df = drill_df[drill_df[col] == choice]
            selections[col] = choice

    if len(drill_df) == 0:
        st.warning("No data available for the selected drilldown filters.")
    else:
        drill_capacity = calculate_capacity_vs_delivery(drill_df, df_base=df_base_window)
        drill_capacity = filter_by_time_window(drill_capacity, time_window, date_col="month_key")

        total_capacity = drill_capacity["capacity_hours"].sum()
        total_actual = drill_capacity["actual_hours"].sum()
        total_billable = drill_capacity["billable_hours"].sum() if "billable_hours" in drill_capacity.columns else 0
        total_slack = total_capacity - total_actual
        utilisation_total = (total_actual / total_capacity * 100) if total_capacity > 0 else np.nan
        utilisation_billable = (total_billable / total_capacity * 100) if total_capacity > 0 else np.nan

        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            st.metric("Capacity Hours", fmt_hours(total_capacity))
        with kpi_cols[1]:
            st.metric("Actual Hours", fmt_hours(total_actual))
        with kpi_cols[2]:
            st.metric("Slack Hours", fmt_hours(total_slack))
        with kpi_cols[3]:
            st.metric("Utilisation (Total)", fmt_percent(utilisation_total))
        with kpi_cols[4]:
            st.metric("Billable Utilisation", fmt_percent(utilisation_billable))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=drill_capacity["month_key"],
                y=drill_capacity["capacity_hours"],
                name="Capacity Hours",
                mode="lines+markers",
                line=dict(color="#4c78a8", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=drill_capacity["month_key"],
                y=drill_capacity["actual_hours"],
                name="Actual Hours",
                mode="lines+markers",
                line=dict(color="#f58518", width=2),
            )
        )
        fig.update_layout(
            title="Capacity vs Delivery (Filtered)",
            yaxis=dict(title="Hours"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        next_level = None
        for col, label in chain:
            if col not in selections and col in drill_df.columns:
                next_level = (col, label)
                break

        if next_level:
            summary = compute_capacity_summary(drill_df, next_level[0], df_base=df_base_window)
            if len(summary) > 0:
                st.subheader(f"Next Level Breakdown: {next_level[1]}")
                st.dataframe(
                    summary.rename(columns={
                        next_level[0]: next_level[1],
                        "avg_fte_equiv": "Avg FTE (equiv)",
                        "capacity_hours": "Capacity Hours",
                        "actual_hours": "Actual Hours",
                        "billable_hours": "Billable Hours",
                        "nonbillable_hours": "Non‑Billable Hours",
                        "utilisation_pct": "Utilisation (Total)",
                        "billable_utilisation_pct": "Billable Utilisation",
                        "slack_hours": "Slack Hours",
                        "slack_pct": "Slack %",
                        "billable_ratio": "Billable Ratio",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Capacity Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Actual Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Billable Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Non‑Billable Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Utilisation (Total)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Billable Utilisation": st.column_config.NumberColumn(format="%.1f%%"),
                        "Slack Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Slack %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Billable Ratio": st.column_config.NumberColumn(format="%.1f%%"),
                        "Avg FTE (equiv)": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=summary[next_level[0]],
                        y=summary["actual_hours"],
                        name="Actual Hours",
                        marker_color="#4c78a8",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        x=summary[next_level[0]],
                        y=summary["slack_hours"],
                        name="Slack Hours",
                        marker_color="#e45756",
                    )
                )
                fig.update_layout(
                    title=f"Capacity Composition by {next_level[1]}",
                    yaxis=dict(title="Hours"),
                    barmode="stack",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"drill_stack_{next_level[0]}")

        st.subheader("Slack Hotspots (Current Selection)")
        st.caption(
            "Slack hotspots highlight where capacity exists after accounting for actual work. "
            "We allocate each person’s capacity based on where they spent their time, using "
            "`fte_hours_scaling` when available (otherwise assuming 1.0). "
            "This avoids double counting and makes hotspots comparable across levels."
        )
        hotspot_levels = [
            ("department_final", "Department"),
            (category_col, "Category"),
            ("staff_name", "Staff"),
            ("task_name", "Job Task"),
        ]
        for col, label in hotspot_levels:
            if col not in drill_df.columns:
                continue
            hotspot = compute_capacity_summary(drill_df, col, df_base=df_base_window)
            if len(hotspot) == 0:
                continue
            with st.expander(f"Methodology: {label} slack calculation", expanded=False):
                st.markdown(
                    """
**Executive question**  
Where is real spare capacity once we account for how people actually spent their time?

**Why this method is fair**  
We anchor capacity to real timesheets and then split it across the same slices of work.  
That way, people who work across areas aren’t counted twice.

**How we calculate slack (in four steps)**  
1) **Estimate effective FTE per staff‑month**  
   - Use hours‑weighted `fte_hours_scaling` when available; otherwise assume 1.0.  
2) **Convert FTE to monthly capacity**  
   - 38 hours/week × 4.33 weeks × effective FTE.  
3) **Allocate capacity to the slice**  
   - Based on each slice’s share of that person’s hours.  
4) **Compute slack**  
   - Slack = allocated capacity − actual hours.  
   - Slack % = 100% − utilisation.  

**Interpretation**  
High slack means unutilised capacity; negative slack indicates overload.
                    """
                )
            top_hotspot = hotspot.sort_values("slack_hours", ascending=False).head(8)
            st.markdown(f"**Top Slack by {label}**")
            st.dataframe(
                top_hotspot.rename(columns={
                    col: label,
                    "capacity_hours": "Capacity Hours",
                    "actual_hours": "Actual Hours",
                    "slack_hours": "Slack Hours",
                    "slack_pct": "Slack %",
                    "utilisation_pct": "Utilisation (Total)",
                    "billable_utilisation_pct": "Billable Utilisation",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Capacity Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Actual Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Slack Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Slack %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Utilisation (Total)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Billable Utilisation": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

            deltas = compute_job_volume_deltas(drill_df, col, time_window)
            if len(deltas) > 0:
                actionable = top_hotspot.merge(deltas, on=col, how="left")
                actionable["why_flag"] = np.where(
                    (actionable["job_volume_delta"] < -0.1) & (actionable["billable_ratio"] > 0.6),
                    "Low job volume",
                    np.where(
                        (actionable["billable_ratio"] < 0.5) & (actionable["job_volume_delta"] > -0.05),
                        "High non‑billable load",
                        np.where(
                            (actionable["billable_ratio"] < 0.5) & (actionable["job_volume_delta"] < -0.1),
                            "Both: volume + non‑billable",
                            "Mixed / investigate",
                        ),
                    ),
                )
                st.markdown(f"**Actionable Why: {label}**")
                actionable_view = actionable.rename(columns={
                    col: label,
                    "job_volume_delta": "Job Volume Δ",
                    "billable_ratio": "Billable Ratio",
                    "slack_hours": "Slack Hours",
                    "slack_pct": "Slack %",
                    "why_flag": "Likely Driver",
                })[
                    [label, "Slack Hours", "Slack %", "Job Volume Δ", "Billable Ratio", "Likely Driver"]
                ]
                st.dataframe(
                    actionable_view,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Slack Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Slack %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Job Volume Δ": st.column_config.NumberColumn(format="%.1f%%"),
                        "Billable Ratio": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )

    # =========================================================================
    # SECTION F: SO WHAT (CONSULTANT VIEW)
    # =========================================================================
    section_header("So What", "What the trend implies about pricing vs delivery")

    monthly_sorted = monthly_metrics.sort_values("cohort_month")
    if len(monthly_sorted) >= 6:
        recent = monthly_sorted.tail(3)
        prior = monthly_sorted.tail(6).head(3)

        recent_jobs = recent["job_count"].mean()
        prior_jobs = prior["job_count"].mean()
        recent_quote = recent["avg_quoted_amount"].mean()
        prior_quote = prior["avg_quoted_amount"].mean()
        recent_ratio = (
            recent["total_quoted_hours"].sum() / recent["total_actual_hours"].sum()
            if recent["total_actual_hours"].sum() > 0 else np.nan
        )
        prior_ratio = (
            prior["total_quoted_hours"].sum() / prior["total_actual_hours"].sum()
            if prior["total_actual_hours"].sum() > 0 else np.nan
        )

        jobs_delta = (recent_jobs - prior_jobs) / prior_jobs if prior_jobs > 0 else np.nan
        quote_delta = (recent_quote - prior_quote) / prior_quote if prior_quote > 0 else np.nan
        ratio_delta = recent_ratio - prior_ratio if pd.notna(recent_ratio) and pd.notna(prior_ratio) else np.nan

        jobs_delta_text = fmt_percent(jobs_delta * 100) if pd.notna(jobs_delta) else "n/a"
        quote_delta_text = fmt_percent(quote_delta * 100) if pd.notna(quote_delta) else "n/a"
        ratio_delta_text = f"{ratio_delta:+.2f}x" if pd.notna(ratio_delta) else "n/a"

        st.info(
            f"""
**Recent 3 months vs prior 3 months**
- Job volume change: {jobs_delta_text}
- Avg quote value change: {quote_delta_text}
- Quoted vs actual hours ratio change: {ratio_delta_text}

**Interpretation**
- If job volume is falling while quote value rises, you are winning fewer, larger jobs.
- If quoted vs actual ratio is < 1.0, delivery is heavier than quoted (margin compression).
- If quoted vs actual ratio is > 1.0, quotes exceed delivery (potential slack/underutilisation).
            """
        )

        loss_dept = job_level.groupby("department_final").agg(
            jobs=("job_no", "nunique"),
            actual_hours=("actual_hours", "sum"),
            underquoted_hours=("underquoted_hours", "sum"),
        ).reset_index()
        loss_dept["underquoted_share"] = np.where(
            loss_dept["actual_hours"] > 0,
            loss_dept["underquoted_hours"] / loss_dept["actual_hours"],
            np.nan,
        )
        loss_dept = loss_dept.sort_values("underquoted_hours", ascending=False).head(5)

        loss_cat = job_level.groupby("job_category").agg(
            jobs=("job_no", "nunique"),
            actual_hours=("actual_hours", "sum"),
            underquoted_hours=("underquoted_hours", "sum"),
        ).reset_index()
        loss_cat["underquoted_share"] = np.where(
            loss_cat["actual_hours"] > 0,
            loss_cat["underquoted_hours"] / loss_cat["actual_hours"],
            np.nan,
        )
        loss_cat = loss_cat.sort_values("underquoted_hours", ascending=False).head(5)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Loss Hotspots by Department")
            st.dataframe(
                loss_dept.rename(columns={
                    "department_final": "Department",
                    "jobs": "Jobs",
                    "actual_hours": "Actual Hours",
                    "underquoted_hours": "Under-Quoted Hours",
                    "underquoted_share": "Under-Quoted Share",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Actual Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Under-Quoted Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Under-Quoted Share": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

        with col2:
            st.subheader("Loss Hotspots by Job Category")
            st.dataframe(
                loss_cat.rename(columns={
                    "job_category": "Category",
                    "jobs": "Jobs",
                    "actual_hours": "Actual Hours",
                    "underquoted_hours": "Under-Quoted Hours",
                    "underquoted_share": "Under-Quoted Share",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Actual Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Under-Quoted Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Under-Quoted Share": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
    else:
        st.caption("Need at least 6 months of data to compute the trend comparison.")


if __name__ == "__main__":
    main()
