"""
Job Mix & Implied FTE Demand Page

Analyze job intake patterns and implied capacity demand.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state, get_state, set_state
from src.ui.layout import section_header
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_count
from src.ui.charts import time_series, quadrant_scatter, horizontal_bar
from src.ui.intervention_components import (
    render_quadrant_health_summary,
    render_intervention_queue,
    render_selected_job_brief,
    render_driver_analysis,
    render_peer_context,
    render_quadrant_trend,
    render_methodology_expander,
)
from src.modeling.intervention import (
    compute_intervention_risk_score,
    build_intervention_queue,
    compute_quadrant_health_summary,
    get_peer_context,
)
from src.data.loader import load_fact_timesheet
from src.data.semantic import safe_quote_job_task, get_category_col, safe_quote_rollup, filter_jobs_by_state
from src.data.client_analytics import compute_global_task_median_mix, compute_company_cost_rate
from src.data.cohorts import filter_by_time_window
from src.config import config


st.set_page_config(page_title="Job Mix & Demand", page_icon="📊", layout="wide")

init_state()

def _section_intro(title: str, subtitle: str, why: str):
    st.markdown(f"### {title}")
    st.caption(subtitle)
    st.markdown(f"**Why it matters:** {why}")

def _section_start(variant: str):
    st.markdown(f"<div class='section {variant}'>", unsafe_allow_html=True)

def _section_end():
    st.markdown("</div>", unsafe_allow_html=True)

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

    # Actual hours in cohort month (align demand to intake month)
    df_with_cohort = df.merge(cohort_df, on="job_no", how="inner")
    df_with_cohort["month_key"] = pd.to_datetime(df_with_cohort["month_key"])
    df_with_cohort["cohort_month"] = pd.to_datetime(df_with_cohort["cohort_month"])
    df_cohort_month = df_with_cohort[df_with_cohort["month_key"] == df_with_cohort["cohort_month"]]
    job_actuals = df_cohort_month.groupby("job_no", dropna=False).agg(
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

    st.markdown(
        """
        <style>
        .page-hero {
            background: linear-gradient(135deg, #f4efe4 0%, #fff7e8 100%);
            border: 1px solid #eadfcb;
            border-radius: 12px;
            padding: 14px 16px;
            margin: 6px 0 14px 0;
        }
        .guide-row { display: flex; gap: 12px; }
        .guide-card {
            flex: 1;
            background: #ffffff;
            border: 1px solid #e6e1d5;
            border-radius: 10px;
            padding: 12px 12px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .guide-card h4 { margin: 0 0 6px 0; }
        .section {
            border-radius: 12px;
            padding: 12px 14px;
            margin: 14px 0;
            border: 1px solid #e6e1d5;
            background: #fcfbf8;
        }
        .profit-frame {
            background: linear-gradient(135deg, #f7f2e7 0%, #fff9ee 100%);
            border: 1px solid #eadfcb;
            border-radius: 12px;
            padding: 14px 16px;
            margin: 8px 0 10px 0;
        }
        .profit-frame h4 { margin: 0 0 4px 0; }
        .profit-frame p { margin: 0; color: #5a5346; }
        .signal-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        .signal-card {
            background: #ffffff;
            border: 1px solid #e6e1d5;
            border-radius: 10px;
            padding: 10px 12px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .signal-card small { color: #6c6455; }
        .signal-card strong { display: block; font-size: 18px; margin-top: 2px; }
        @media (max-width: 900px) {
            .signal-row { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 600px) {
            .signal-row { grid-template-columns: 1fr; }
        }
        .band-portfolio { border-left: 6px solid #4c78a8; }
        .band-value { border-left: 6px solid #f58518; }
        .band-capacity { border-left: 6px solid #54a24b; }
        .band-drill { border-left: 6px solid #e45756; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
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
    job_state_options = ["All", "Active", "Completed"]
    current_state = get_state("job_state_filter") if get_state("job_state_filter") in job_state_options else "All"
    selected_state = st.sidebar.selectbox(
        "Job State",
        options=job_state_options,
        index=job_state_options.index(current_state),
        key="filter_job_state",
    )
    set_state("job_state_filter", selected_state)

    exclude_sg_alloc = st.sidebar.checkbox(
        "Exclude: Social Garden Invoice Allocation",
        value=True,
        help="Removes the [Job Task] Name 'Social Garden Invoice Allocation' from all analyses."
    )
    
    departments = ["All"] + sorted(df_all_raw["department_final"].dropna().unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)
    
    if exclude_sg_alloc and "task_name" in df_all_raw.columns:
        df_all_raw = df_all_raw[df_all_raw["task_name"] != "Social Garden Invoice Allocation"]

    if selected_dept != "All":
        df_all = df_all_raw[df_all_raw["department_final"] == selected_dept]
    else:
        df_all = df_all_raw

    if selected_state in ["Active", "Completed"]:
        state_jobs = filter_jobs_by_state(df_all_raw, selected_state)
        df_all = df_all[df_all["job_no"].isin(state_jobs["job_no"].unique())]

    st.markdown(
        """
        <div class="page-hero">
            <div class="guide-row">
                <div class="guide-card">
                    <h4>1) Portfolio Mix</h4>
                    <div>What work is coming in, how big it is, and pricing signals.</div>
                </div>
                <div class="guide-card">
                    <h4>2) Value vs Effort</h4>
                    <div>Which jobs create value vs consume capacity, and why.</div>
                </div>
                <div class="guide-card">
                    <h4>3) Capacity vs Delivery</h4>
                    <div>Are we over- or under-loaded, and where?</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    # Job-level summary for quoted vs actual (to-date)
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

    category_col = get_category_col(df_all)
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
    job_level = job_level[job_level["cohort_month"].notna()]
    job_level_window = filter_by_time_window(job_level, time_window, date_col="cohort_month")

    monthly_actuals = job_level_window.groupby("cohort_month").agg(
        total_actual_hours_to_date=("actual_hours", "sum"),
        avg_actual_hours_to_date=("actual_hours", "mean"),
        total_quoted_hours_to_date=("quoted_hours", "sum"),
        underquoted_hours_to_date=("underquoted_hours", "sum"),
    ).reset_index().sort_values("cohort_month")

    # SECTION A: KPI STRIP
    # =========================================================================
    _section_start("band-portfolio")
    _section_intro(
        "1) Portfolio Mix — Intake, Scale, and Pricing",
        "How many jobs are coming in, how large they are, and what they imply for demand.",
        "This sets the baseline for forecasting demand and pricing discipline."
    )
    
    kpi_cols = st.columns(8)
    
    total_jobs = monthly_metrics["job_count"].sum()
    total_quoted_hours = job_level_window["quoted_hours"].sum()
    total_quoted_amount = job_level_window["quoted_amount"].sum()
    total_actual_hours = job_level_window["actual_hours"].sum()
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
    _section_end()
    
    # =========================================================================
    # SECTION B: TREND CHARTS
    # =========================================================================
    _section_start("band-portfolio")
    _section_intro(
        "1.1) Monthly Trends — Volume vs Value",
        "Track whether job volume and quote values are moving together or diverging.",
        "Highlights pricing consistency and shifts in demand mix."
    )
    
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
            hovermode="x unified",
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
                y=monthly_actuals["avg_actual_hours_to_date"],
                name="Avg Actual Hours",
                mode="lines+markers",
                line=dict(color="#e45756", width=2),
            )
        )
        fig.update_layout(
            title="Avg Hours per Job: Quoted vs Actual",
            yaxis=dict(title="Hours"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
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
                y=monthly_actuals["total_actual_hours_to_date"],
                name="Actual Hours",
                mode="lines+markers",
                line=dict(color="#b279a2", width=2),
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthly_metrics["cohort_month"],
                y=monthly_actuals["underquoted_hours_to_date"],
                name="Under-Quoted Hours",
                marker_color="rgba(228,87,86,0.35)",
            )
        )
        fig.update_layout(
            title="Total Hours: Quoted vs Actual",
            yaxis=dict(title="Hours"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # Job Mix & Scope Creep Diagnostics (Additive)
    # -----------------------------------------------------------------
    st.markdown("#### Job Mix & Scope Creep Diagnostics")
    st.caption("Track category mix shifts, hours overrun, and margin erosion signals from department to company level.")

    mix_cols = st.columns(2)
    with mix_cols[0]:
        if len(job_level_window) > 0 and "job_category" in job_level_window.columns:
            top_cats = (
                job_level_window.groupby("job_category")["job_no"]
                .nunique()
                .sort_values(ascending=False)
                .head(6)
                .index
                .tolist()
            )
            cat_month = job_level_window[job_level_window["job_category"].isin(top_cats)].groupby(
                ["cohort_month", "job_category"]
            ).agg(
                jobs=("job_no", "nunique"),
                hours=("actual_hours", "sum"),
            ).reset_index()
            fig = go.Figure()
            for cat in top_cats:
                slice_df = cat_month[cat_month["job_category"] == cat]
                fig.add_bar(
                    x=slice_df["cohort_month"],
                    y=slice_df["jobs"],
                    name=cat,
                )
            fig.update_layout(
                title="Job Mix by Category (Top 6)",
                barmode="stack",
                yaxis=dict(title="Jobs"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                template="plotly_white",
                height=360,
                margin=dict(l=50, r=30, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    with mix_cols[1]:
        if len(job_level_window) > 0:
            overrun_month = job_level_window.groupby("cohort_month").agg(
                actual_hours=("actual_hours", "sum"),
                quoted_hours=("quoted_hours", "sum"),
                underquoted_hours=("underquoted_hours", "sum"),
            ).reset_index()
            overrun_month["underquoted_share"] = np.where(
                overrun_month["actual_hours"] > 0,
                overrun_month["underquoted_hours"] / overrun_month["actual_hours"] * 100,
                np.nan,
            )
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=overrun_month["cohort_month"],
                    y=overrun_month["underquoted_hours"],
                    name="Under-Quoted Hours",
                    marker_color="rgba(228,87,86,0.5)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=overrun_month["cohort_month"],
                    y=overrun_month["underquoted_share"],
                    name="Under-Quoted Share %",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color="#4c78a8", width=2),
                )
            )
            fig.update_layout(
                title="Hours Overrun / Scope Creep",
                yaxis=dict(title="Under-Quoted Hours"),
                yaxis2=dict(title="Under-Quoted Share %", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                template="plotly_white",
                height=360,
                margin=dict(l=50, r=40, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Overrun drivers: job categories + tasks
            st.markdown("**Overrun Drivers**")
            driver_cols = st.columns([1, 0.04, 1])
            with driver_cols[0]:
                if "job_category" in job_level_window.columns:
                    cat_overrun = job_level_window.groupby("job_category").agg(
                        jobs=("job_no", "nunique"),
                        underquoted_hours=("underquoted_hours", "sum"),
                        actual_hours=("actual_hours", "sum"),
                    ).reset_index()
                    cat_overrun["underquoted_share"] = np.where(
                        cat_overrun["actual_hours"] > 0,
                        cat_overrun["underquoted_hours"] / cat_overrun["actual_hours"] * 100,
                        np.nan,
                    )
                    cat_overrun = cat_overrun.sort_values("underquoted_hours", ascending=False).head(6)
                    fig = go.Figure()
                    fig.add_bar(
                        x=cat_overrun["underquoted_hours"],
                        y=cat_overrun["job_category"],
                        orientation="h",
                        marker_color="#e45756",
                        name="Under-Quoted Hours",
                    )
                    fig.update_layout(
                        title="Top Job Categories Driving Overruns",
                        xaxis=dict(title="Under-Quoted Hours"),
                        yaxis=dict(title="Category", categoryorder="total ascending"),
                        template="plotly_white",
                        height=320,
                        margin=dict(l=80, r=20, t=50, b=40),
                        legend=dict(orientation="h"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with driver_cols[2]:
                if "task_name" in df_window.columns and "job_no" in job_level_window.columns:
                    overrun_jobs = job_level_window[job_level_window["underquoted_hours"] > 0]["job_no"].unique()
                    task_overrun = df_window[df_window["job_no"].isin(overrun_jobs)].groupby("task_name").agg(
                        hours=("hours_raw", "sum"),
                    ).reset_index()
                    task_overrun = task_overrun.sort_values("hours", ascending=False).head(8)
                    fig = go.Figure()
                    fig.add_bar(
                        x=task_overrun["hours"],
                        y=task_overrun["task_name"],
                        orientation="h",
                        marker_color="#f58518",
                        name="Hours (Overrun Jobs)",
                    )
                    fig.update_layout(
                        title="Top Tasks Inside Overrun Jobs",
                        xaxis=dict(title="Hours"),
                        yaxis=dict(title="Task", categoryorder="total ascending"),
                        template="plotly_white",
                        height=320,
                        margin=dict(l=120, r=20, t=50, b=40),
                        legend=dict(orientation="h"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    erosion_cols = st.columns(2)
    with erosion_cols[0]:
        if "department_final" in job_level_window.columns:
            dept_margin = job_level_window.groupby(["cohort_month", "department_final"]).agg(
                revenue=("quoted_amount", "sum"),
                actual_hours=("actual_hours", "sum"),
                cost=("actual_hours", "sum"),
            ).reset_index()
            if len(dept_margin) > 0 and "base_cost" in df_all.columns:
                cost_map = df_all.groupby("job_no").agg(cost=("base_cost", "sum")).reset_index()
                job_cost = job_level_window.merge(cost_map, on="job_no", how="left")
                dept_margin = job_cost.groupby(["cohort_month", "department_final"]).agg(
                    revenue=("quoted_amount", "sum"),
                    cost=("cost", "sum"),
                ).reset_index()
                dept_margin["margin_pct"] = np.where(
                    dept_margin["revenue"] > 0,
                    (dept_margin["revenue"] - dept_margin["cost"]) / dept_margin["revenue"] * 100,
                    np.nan,
                )
                top_depts = (
                    dept_margin.groupby("department_final")["revenue"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(4)
                    .index
                    .tolist()
                )
                fig = go.Figure()
                for dept in top_depts:
                    slice_df = dept_margin[dept_margin["department_final"] == dept]
                    fig.add_trace(
                        go.Scatter(
                            x=slice_df["cohort_month"],
                            y=slice_df["margin_pct"],
                            name=dept,
                            mode="lines+markers",
                        )
                    )
                fig.update_layout(
                    title="Margin % by Department (Top 4)",
                    yaxis=dict(title="Margin %"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

    with erosion_cols[1]:
        if "rev_alloc" in df_all.columns and "base_cost" in df_all.columns:
            company_month = df_all.groupby("month_key").agg(
                revenue=("rev_alloc", "sum"),
                cost=("base_cost", "sum"),
            ).reset_index()
            company_month["margin_pct"] = np.where(
                company_month["revenue"] > 0,
                (company_month["revenue"] - company_month["cost"]) / company_month["revenue"] * 100,
                np.nan,
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=company_month["month_key"],
                    y=company_month["margin_pct"],
                    name="Company Margin %",
                    mode="lines+markers",
                    line=dict(color="#54a24b", width=2),
                )
            )
            fig.update_layout(
                title="Company Margin % Trend",
                yaxis=dict(title="Margin %"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                template="plotly_white",
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

    if len(job_level_window) > 0:
        top_depts = (
            job_level_window.groupby("department_final")["underquoted_hours"]
            .sum()
            .sort_values(ascending=False)
            .head(4)
            .index
            .tolist()
        )
        dept_month = job_level_window[job_level_window["department_final"].isin(top_depts)].groupby(
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
    _section_end()
    
    # =========================================================================
    # SECTION C: JOB QUADRANT
    # =========================================================================
    _section_start("band-value")
    _section_intro(
        "2) Value vs Effort — Portfolio Segmentation",
        "Map jobs by quoted value and quoted hours, then drill into profitability and delivery risk.",
        "Turns the portfolio into an action list: protect high‑value work and fix margin erosion."
    )
    
    if len(job_level_window) > 0:
        job_quotes = job_level_window[(job_level_window["quoted_hours"] > 0) & (job_level_window["quoted_amount"] > 0)].copy()

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

            # Build margin to date for visual view (effort, value, margin)
            job_financials_3d = df_window.groupby("job_no").agg(
                revenue_to_date=("rev_alloc", "sum") if "rev_alloc" in df_window.columns else ("job_no", "count"),
                cost_to_date=("base_cost", "sum") if "base_cost" in df_window.columns else ("job_no", "count"),
            ).reset_index()
            job_financials_3d["margin_to_date"] = job_financials_3d["revenue_to_date"] - job_financials_3d["cost_to_date"]
            job_financials_3d["margin_pct_to_date"] = np.where(
                job_financials_3d["revenue_to_date"] > 0,
                job_financials_3d["margin_to_date"] / job_financials_3d["revenue_to_date"] * 100,
                np.nan,
            )

            summary_candidates = [
                "[Job] Job Summary",
                "job_summary",
                "job_summary_quote",
                "job_summary_raw",
                "job_name",
                "job_name_raw",
                "deliverable",
                "task_label",
            ]
            summary_col = next((c for c in summary_candidates if c in df_all.columns), None)

            job_meta_3d = df_all.groupby("job_no").agg(
                client=("client", "first") if "client" in df_all.columns else ("job_no", "first"),
                job_summary=(summary_col, "first") if summary_col else ("job_no", "first"),
            ).reset_index()

            job_3d = job_quotes.merge(job_financials_3d, on="job_no", how="left")
            job_3d = job_3d.merge(job_meta_3d, on="job_no", how="left")
            job_3d = job_3d.dropna(subset=["quoted_hours", "quoted_amount"])

            x_med = job_3d["quoted_hours"].median()
            y_med = job_3d["quoted_amount"].median()
            margin_guard = 20.0

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=job_3d["quoted_hours"],
                    y=job_3d["quoted_amount"],
                    mode="markers",
                    marker=dict(
                        size=7,
                        color=job_3d["margin_pct_to_date"],
                        colorscale="RdYlGn",
                        cmin=-20,
                        cmax=50,
                        colorbar=dict(title="Margin %"),
                        line=dict(width=0.4, color="#444"),
                        opacity=0.85,
                    ),
                    text=job_3d["job_no"],
                    customdata=job_3d[["margin_pct_to_date", "client", "job_summary"]].values,
                    hovertemplate=(
                        "Job: %{text}<br>"
                        "Client: %{customdata[1]}<br>"
                        "Job Summary: %{customdata[2]}<br>"
                        "Quoted Hours: %{x:.1f}<br>"
                        "Quoted $: %{y:$,.0f}<br>"
                        "Margin %: %{customdata[0]:.1f}%<extra></extra>"
                    ),
                )
            )
            fig.add_vline(x=x_med, line_dash="dash", line_color="#4c78a8")
            fig.add_hline(y=y_med, line_dash="dash", line_color="#f58518")
            fig.add_annotation(x=x_med, y=job_3d["quoted_amount"].max(), text="Median Effort", showarrow=False, yshift=10)
            fig.add_annotation(x=job_3d["quoted_hours"].max(), y=y_med, text="Median Value", showarrow=False, xshift=10)
            fig.add_annotation(
                x=job_3d["quoted_hours"].min(),
                y=job_3d["quoted_amount"].min(),
                text=f"Green ≥ {margin_guard:.0f}% margin",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.7)",
            )
            fig.update_layout(
                title="Job Portfolio: Effort × Value (Color = Margin %)",
                xaxis_title="Quoted Hours (Effort)",
                yaxis_title="Quoted Amount ($) (Value)",
                margin=dict(l=10, r=10, b=40, t=50),
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

            def _risk_action(row: pd.Series) -> str:
                actions = []
                if pd.notna(row.get("hours_overrun_pct")) and row["hours_overrun_pct"] > 10:
                    actions.append("Scope reset + reforecast hours")
                if pd.notna(row.get("quote_to_revenue")) and row["quote_to_revenue"] < 0.7:
                    actions.append("Accelerate billing / milestone invoice")
                if pd.notna(row.get("realised_rate")) and pd.notna(row.get("quote_rate")):
                    if row["realised_rate"] < row["quote_rate"] * 0.85:
                        actions.append("Staffing mix review (rate uplift)")
                if pd.notna(row.get("margin_pct_to_date")) and row["margin_pct_to_date"] < 15:
                    actions.append("Margin guardrails / pricing review")
                if not actions:
                    return "Monitor"
                return " | ".join(dict.fromkeys(actions))

            active_risk = quadrant_detail[quadrant_detail["is_active"] == True].copy()
            active_risk["risk_reason"] = active_risk.apply(_risk_reasons, axis=1)
            active_risk["recommended_action"] = active_risk.apply(_risk_action, axis=1)
            active_risk = active_risk[active_risk["risk_reason"] != ""]
            active_risk = active_risk.sort_values(["margin_pct_to_date", "quote_to_revenue"], ascending=[True, True])

            st.subheader("Quadrant Portfolio: Jobs + Profitability Context")
            st.caption(
                "A decision-focused view of who is in the quadrant, how value is being captured, and where margins are leaking."
            )

            def _fmt_pct(value: float, decimals: int = 1) -> str:
                return f"{value:.{decimals}f}%" if pd.notna(value) else "—"

            total_jobs_q = quadrant_detail["job_no"].nunique()
            active_jobs_q = int(quadrant_detail["is_active"].sum()) if "is_active" in quadrant_detail.columns else 0
            risk_share = (len(active_risk) / active_jobs_q * 100) if active_jobs_q > 0 else np.nan

            story_cols = st.columns(4)
            story_cols[0].metric("Jobs in Quadrant", f"{total_jobs_q:,}")
            story_cols[1].metric("Active Jobs", f"{active_jobs_q:,}")
            story_cols[2].metric("Median Margin %", _fmt_pct(quadrant_detail["margin_pct_to_date"].median()))
            story_cols[3].metric("Active Jobs at Risk", _fmt_pct(risk_share))

            st.markdown(
                "**How to use this view:** start with the risk watchlist, then scan the full ledger to spot"
                " systemic patterns (pricing, scope control, staffing mix)."
            )

            st.markdown("#### 1) Risk Watchlist (Active Jobs)")
            st.caption(
                "Jobs where delivery economics are breaking: low margins, weak revenue capture, or hours overruns."
            )

            if len(active_risk) > 0:
                st.caption(
                    f"{len(active_risk)} active jobs flagged in this quadrant. "
                    "Prioritize items with low margin %, weak revenue capture, or overruns."
                )
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
                            "recommended_action",
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
                        "recommended_action": "Recommended Action",
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

            st.markdown("#### 2) Full Portfolio Ledger (All Jobs)")
            with st.expander("Show full job list", expanded=False):
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

            st.subheader("Job Deep‑Dive: Operational Health & Right‑Sizing")
            st.caption("Start with a chain filter, shortlist jobs, then drill into one job with clear benchmarks.")

            # Chain-first filters for deep-dive
            chain_cols = st.columns(3)
            with chain_cols[0]:
                dept_options = ["All"] + sorted(df_window["department_final"].dropna().unique().tolist())
                deep_dept = st.selectbox("Department (Deep‑Dive)", dept_options, key="deep_dept")
            with chain_cols[1]:
                df_cat = df_window if deep_dept == "All" else df_window[df_window["department_final"] == deep_dept]
                cat_col = "category_rev_job" if "category_rev_job" in df_cat.columns else "job_category"
                cat_options = ["All"] + sorted(df_cat[cat_col].dropna().unique().tolist())
                deep_cat = st.selectbox("Category (Deep‑Dive)", cat_options, key="deep_cat")
            with chain_cols[2]:
                status_choice = st.selectbox("Status", ["Active", "Completed", "All"], key="deep_status")

            deep_df = df_window.copy()
            if deep_dept != "All":
                deep_df = deep_df[deep_df["department_final"] == deep_dept]
            if deep_cat != "All":
                if cat_col in deep_df.columns:
                    deep_df = deep_df[deep_df[cat_col] == deep_cat]
                else:
                    deep_df = deep_df[deep_df["job_category"] == deep_cat]

            # Job completion map
            completion = deep_df.groupby("job_no").agg(
                completed_date=("job_completed_date", "first") if "job_completed_date" in deep_df.columns else ("job_no", "first"),
                job_status=("job_status", "first") if "job_status" in deep_df.columns else ("job_no", "first"),
            ).reset_index()
            if "job_completed_date" in completion.columns:
                completion["is_completed"] = completion["completed_date"].notna()
            elif "job_status" in completion.columns:
                completion["is_completed"] = completion["job_status"].str.lower().str.contains("completed", na=False)
            else:
                completion["is_completed"] = False

            if status_choice != "All":
                want_completed = status_choice == "Completed"
                completion = completion[completion["is_completed"] == want_completed]
            deep_df = deep_df[deep_df["job_no"].isin(completion["job_no"].unique())]

            # Quick scope summary
            scope_cols = st.columns(4)
            with scope_cols[0]:
                st.metric("Jobs in Scope", f"{deep_df['job_no'].nunique():,}")
            with scope_cols[1]:
                st.metric("Active Jobs", f"{completion[completion['is_completed'] == False]['job_no'].nunique():,}")
            with scope_cols[2]:
                st.metric("Completed Jobs", f"{completion[completion['is_completed'] == True]['job_no'].nunique():,}")
            with scope_cols[3]:
                if "rev_alloc" in deep_df.columns and "base_cost" in deep_df.columns:
                    prof = deep_df.groupby("job_no").agg(
                        revenue=("rev_alloc", "sum"),
                        cost=("base_cost", "sum"),
                    ).reset_index()
                    prof["margin_pct"] = np.where(
                        prof["revenue"] > 0,
                        (prof["revenue"] - prof["cost"]) / prof["revenue"] * 100,
                        np.nan,
                    )
                    st.metric("Median Margin %", f"{prof['margin_pct'].median():.1f}%" if len(prof) > 0 else "—")
                else:
                    st.metric("Median Margin %", "—")

            # Shortlist jobs
            job_meta_deep = df_all.groupby("job_no").agg(
                client=("client", "first") if "client" in df_all.columns else ("job_no", "first"),
                job_summary=(summary_col, "first") if "summary_col" in locals() and summary_col else ("job_no", "first"),
            ).reset_index()

            job_metrics = deep_df.groupby("job_no").agg(
                actual_hours=("hours_raw", "sum"),
                revenue=("rev_alloc", "sum") if "rev_alloc" in deep_df.columns else ("job_no", "count"),
                cost=("base_cost", "sum") if "base_cost" in deep_df.columns else ("job_no", "count"),
            ).reset_index()
            job_metrics["margin"] = job_metrics["revenue"] - job_metrics["cost"]
            job_metrics["margin_pct"] = np.where(
                job_metrics["revenue"] > 0,
                job_metrics["margin"] / job_metrics["revenue"] * 100,
                np.nan,
            )
            job_metrics = job_metrics.merge(completion[["job_no", "is_completed"]], on="job_no", how="left")
            job_metrics = job_metrics.merge(job_meta_deep, on="job_no", how="left")

            shortlist_n = st.slider("Shortlist size", 5, 30, 10, key="deep_shortlist")
            sort_option = st.selectbox(
                "Sort shortlist by",
                ["Lowest Margin %", "Highest Cost", "Highest Hours"],
                key="deep_sort",
            )
            if sort_option == "Highest Cost":
                shortlist = job_metrics.sort_values("cost", ascending=False).head(shortlist_n)
            elif sort_option == "Highest Hours":
                shortlist = job_metrics.sort_values("actual_hours", ascending=False).head(shortlist_n)
            else:
                shortlist = job_metrics.sort_values("margin_pct", ascending=True).head(shortlist_n)

            st.markdown("**Shortlist (Lowest Margin %)**")
            st.dataframe(
                shortlist.rename(columns={
                    "job_no": "Job",
                    "client": "Client",
                    "job_summary": "Job Summary",
                    "actual_hours": "Hours",
                    "revenue": "Revenue",
                    "cost": "Cost",
                    "margin": "Margin",
                    "margin_pct": "Margin %",
                    "is_completed": "Completed",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                    "Cost": st.column_config.NumberColumn(format="$%.0f"),
                    "Margin": st.column_config.NumberColumn(format="$%.0f"),
                    "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Completed": st.column_config.CheckboxColumn(),
                },
            )

            selected_job = st.selectbox(
                "Select job for deep‑dive",
                shortlist["job_no"].tolist() if len(shortlist) > 0 else [],
                key="deep_job_select",
            )

            if selected_job:
                df_job = deep_df[deep_df["job_no"] == selected_job].copy()
                if selected_job in quadrant_detail["job_no"].values:
                    job_row = quadrant_detail[quadrant_detail["job_no"] == selected_job].iloc[0]
                else:
                    job_row = deep_df[deep_df["job_no"] == selected_job].iloc[0]
                job_dept = job_row.get("department_final")
                job_cat = job_row.get("job_category")
                job_active = bool(job_row.get("is_active")) if "is_active" in job_row else False

                # -------------------------------------------------------------
                # Driver Forensics (Selected Job)
                # -------------------------------------------------------------
                st.subheader("Driver Forensics — Selected Job")
                st.caption("Root-cause margin erosion using task mix, staffing cost, and delivery burn.")

                tabs = st.tabs(["Delivery Drivers", "Staffing Mix"])

                with tabs[0]:
                    task_time_fig = None
                    task_benchmark_fig = None
                    delivery_burn_fig = None

                    if "task_name" in df_job.columns:
                        task_hours = df_job.groupby("task_name")["hours_raw"].sum().reset_index()
                        top_tasks = task_hours.sort_values("hours_raw", ascending=False).head(8)["task_name"].tolist()

                        if "month_key" in df_job.columns:
                            task_time = df_job[df_job["task_name"].isin(top_tasks)].copy()
                            task_time = task_time.groupby(["month_key", "task_name"])["hours_raw"].sum().reset_index()
                            task_time_fig = px.bar(
                                task_time,
                                x="month_key",
                                y="hours_raw",
                                color="task_name",
                                title="Task Hours Over Time (Top Tasks)",
                            )
                            task_time_fig.update_layout(
                                barmode="stack",
                                xaxis_title="Month",
                                yaxis_title="Hours",
                                legend_title="Task",
                            )
                            task_time_fig.update_xaxes(tickformat="%b %Y")

                        global_task = compute_global_task_median_mix(df_all)
                        if len(global_task) > 0:
                            client_task = df_job.groupby("task_name")["hours_raw"].sum().reset_index()
                            total_hours = client_task["hours_raw"].sum()
                            client_task["share_pct"] = np.where(
                                total_hours > 0,
                                client_task["hours_raw"] / total_hours * 100,
                                0,
                            )
                            task_compare = pd.merge(client_task, global_task, on="task_name", how="left")
                            task_compare["global_share_pct"] = task_compare["global_share_pct"].fillna(0)
                            task_compare = task_compare.sort_values("hours_raw", ascending=False).head(8)
                            task_compare_long = task_compare.melt(
                                id_vars=["task_name"],
                                value_vars=["share_pct", "global_share_pct"],
                                var_name="series",
                                value_name="share_pct_value",
                            )
                            task_compare_long["series"] = task_compare_long["series"].replace(
                                {"share_pct": "Job", "global_share_pct": "Benchmark"}
                            )
                            task_benchmark_fig = px.bar(
                                task_compare_long,
                                x="task_name",
                                y="share_pct_value",
                                color="series",
                                barmode="group",
                                title="Task Mix vs Global Median (Top Tasks)",
                            )
                            task_benchmark_fig.update_layout(
                                xaxis_title="Task",
                                yaxis_title="Share %",
                                legend_title="",
                            )

                    if "month_key" in df_job.columns:
                        monthly_hours = df_job.groupby("month_key")["hours_raw"].sum().reset_index()
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
                        quote_rollup_total = safe_quote_rollup(df_job, [])
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
                        if "job_due_date" in df_job.columns:
                            job_due = df_job[["job_no", "job_due_date"]].drop_duplicates()
                            job_due["job_due_date"] = pd.to_datetime(job_due["job_due_date"], errors="coerce")
                            expected_end = job_due["job_due_date"].dropna().median()
                            if pd.notna(expected_end):
                                expected_end_dt = pd.to_datetime(expected_end).to_pydatetime()
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

                    if task_benchmark_fig is not None:
                        st.plotly_chart(task_benchmark_fig, use_container_width=True)
                    if task_time_fig is not None:
                        st.plotly_chart(task_time_fig, use_container_width=True)
                    if delivery_burn_fig is not None:
                        st.plotly_chart(delivery_burn_fig, use_container_width=True)

                with tabs[1]:
                    staff_cost_time_fig = None
                    staffing = pd.DataFrame()
                    senior_flag = False
                    if "staff_name" in df_job.columns:
                        if "role" not in df_job.columns:
                            df_job["role"] = "—"
                        else:
                            df_job["role"] = df_job["role"].fillna("—")
                        staffing = df_job.groupby(["staff_name", "role"]).agg(
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
                        job_rate = (
                            df_job["base_cost"].sum() / df_job["hours_raw"].sum()
                            if df_job["hours_raw"].sum() > 0
                            else np.nan
                        )
                        if pd.notna(company_rate) and pd.notna(job_rate):
                            senior_flag = job_rate > company_rate * 1.2
                        if "month_key" in df_job.columns:
                            staff_time = df_job[df_job["staff_name"].isin(staffing["staff_name"].head(6))].copy()
                            staff_time = staff_time.groupby(["month_key", "staff_name"])["base_cost"].sum().reset_index()
                            staff_cost_time_fig = px.area(
                                staff_time,
                                x="month_key",
                                y="base_cost",
                                color="staff_name",
                                title="Staff Cost Over Time (Top Staff)",
                            )
                            staff_cost_time_fig.update_layout(
                                xaxis_title="Month",
                                yaxis_title="Cost",
                                legend_title="Staff",
                            )
                            staff_cost_time_fig.update_xaxes(tickformat="%b %Y")
                            if pd.notna(company_rate):
                                month_hours = df_job.groupby("month_key")["hours_raw"].sum().reset_index()
                                month_hours["benchmark_cost"] = month_hours["hours_raw"] * company_rate
                                staff_cost_time_fig.add_trace(
                                    go.Scatter(
                                        x=month_hours["month_key"],
                                        y=month_hours["benchmark_cost"],
                                        mode="lines",
                                        line=dict(dash="dash", color="#444"),
                                        name="Benchmark cost (company avg rate)",
                                    )
                                )
                    if senior_flag:
                        st.warning("Senior-heavy delivery detected (blended cost rate >20% above company average).")
                    if staff_cost_time_fig is not None:
                        st.plotly_chart(staff_cost_time_fig, use_container_width=True)
                    if len(staffing) == 0:
                        st.info("No staffing mix data available.")
                    else:
                        st.dataframe(
                            staffing.rename(columns={
                                "staff_name": "Staff",
                                "role": "Role",
                                "hours": "Hours",
                                "cost": "Cost",
                                "cost_rate": "Cost Rate",
                            }),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Hours": st.column_config.NumberColumn(format="%.1f"),
                                "Cost": st.column_config.NumberColumn(format="$%.0f"),
                                "Cost Rate": st.column_config.NumberColumn(format="$%.0f"),
                            },
                        )

                df_job = deep_df[deep_df["job_no"] == selected_job].copy()
                if selected_job in quadrant_detail["job_no"].values:
                    job_row = quadrant_detail[quadrant_detail["job_no"] == selected_job].iloc[0]
                else:
                    job_row = deep_df[deep_df["job_no"] == selected_job].iloc[0]
                job_dept = job_row.get("department_final")
                job_cat = job_row.get("job_category")
                job_active = bool(job_row.get("is_active")) if "is_active" in job_row else False

                # Benchmark slice: completed jobs in same dept + category
                df_bench = df_window.copy()
                if "department_final" in df_bench.columns:
                    df_bench = df_bench[df_bench["department_final"] == job_dept]
                bench_cat_col = "category_rev_job" if "category_rev_job" in df_bench.columns else "job_category"
                if bench_cat_col in df_bench.columns:
                    df_bench = df_bench[df_bench[bench_cat_col] == job_cat]

                bench_completion = df_bench.groupby("job_no").agg(
                    completed_date=("job_completed_date", "first") if "job_completed_date" in df_bench.columns else ("job_no", "first"),
                    job_status=("job_status", "first") if "job_status" in df_bench.columns else ("job_no", "first"),
                ).reset_index()
                if "job_completed_date" in bench_completion.columns:
                    bench_completion["is_completed"] = bench_completion["completed_date"].notna()
                elif "job_status" in bench_completion.columns:
                    bench_completion["is_completed"] = bench_completion["job_status"].str.lower().str.contains("completed", na=False)
                else:
                    bench_completion["is_completed"] = False

                completed_jobs = set(bench_completion[bench_completion["is_completed"] == True]["job_no"].tolist())

                # Runtime benchmarks (days and months)
                date_col = "work_date" if "work_date" in df_bench.columns else "month_key"
                df_bench_dates = df_bench.copy()
                df_bench_dates[date_col] = pd.to_datetime(df_bench_dates[date_col], errors="coerce")
                runtime = df_bench_dates.groupby("job_no").agg(
                    start_date=(date_col, "min"),
                    end_date=(date_col, "max"),
                ).reset_index()
                runtime = runtime[runtime["job_no"].isin(completed_jobs)]
                runtime["runtime_days"] = (runtime["end_date"] - runtime["start_date"]).dt.days + 1
                runtime["runtime_months"] = runtime["runtime_days"] / 30.44

                job_runtime = df_job.copy()
                job_runtime[date_col] = pd.to_datetime(job_runtime[date_col], errors="coerce")
                if len(job_runtime) > 0:
                    job_start = job_runtime[date_col].min()
                    job_end = job_runtime[date_col].max()
                    job_runtime_days = (job_end - job_start).days + 1
                    job_runtime_months = job_runtime_days / 30.44
                else:
                    job_runtime_days = np.nan
                    job_runtime_months = np.nan

                runtime_med_days = runtime["runtime_days"].median() if len(runtime) > 0 else np.nan
                runtime_med_months = runtime["runtime_months"].median() if len(runtime) > 0 else np.nan
                delta_days = (job_runtime_days - runtime_med_days) if pd.notna(job_runtime_days) and pd.notna(runtime_med_days) else np.nan

                # Billable mix benchmark
                if "is_billable" in df_bench.columns:
                    job_billable = df_job.copy()
                    job_billable["billable_hours"] = np.where(job_billable["is_billable"], job_billable["hours_raw"], 0)
                    job_billable_share = (
                        job_billable["billable_hours"].sum() / job_billable["hours_raw"].sum()
                        if job_billable["hours_raw"].sum() > 0 else np.nan
                    )

                    bench_billable = df_bench.copy()
                    bench_billable["billable_hours"] = np.where(bench_billable["is_billable"], bench_billable["hours_raw"], 0)
                    billable_by_job = bench_billable.groupby("job_no").agg(
                        billable=("billable_hours", "sum"),
                        total=("hours_raw", "sum"),
                    ).reset_index()
                    billable_by_job = billable_by_job[billable_by_job["job_no"].isin(completed_jobs)]
                    billable_by_job["billable_share"] = np.where(
                        billable_by_job["total"] > 0,
                        billable_by_job["billable"] / billable_by_job["total"],
                        np.nan,
                    )
                    bench_billable_share = billable_by_job["billable_share"].median() if len(billable_by_job) > 0 else np.nan
                else:
                    job_billable_share = np.nan
                    bench_billable_share = np.nan

                # Margin benchmark
                bench_margin = np.nan
                if len(df_bench) > 0 and "rev_alloc" in df_bench.columns and "base_cost" in df_bench.columns:
                    bench_profit = df_bench[df_bench["job_no"].isin(completed_jobs)].groupby("job_no").agg(
                        revenue=("rev_alloc", "sum"),
                        cost=("base_cost", "sum"),
                    ).reset_index()
                    bench_profit["margin_pct"] = np.where(
                        bench_profit["revenue"] > 0,
                        (bench_profit["revenue"] - bench_profit["cost"]) / bench_profit["revenue"] * 100,
                        np.nan,
                    )
                    bench_margin = bench_profit["margin_pct"].median() if len(bench_profit) > 0 else np.nan

                if selected_job in quadrant_detail["job_no"].values:
                    job_fin = quadrant_detail[quadrant_detail["job_no"] == selected_job].iloc[0]
                else:
                    job_revenue = df_job["rev_alloc"].sum() if "rev_alloc" in df_job.columns else np.nan
                    job_cost = df_job["base_cost"].sum() if "base_cost" in df_job.columns else np.nan
                    job_margin_pct = (
                        (job_revenue - job_cost) / job_revenue * 100
                        if pd.notna(job_revenue) and job_revenue > 0
                        else np.nan
                    )
                    jt = safe_quote_job_task(df_job)
                    if len(jt) > 0 and "quoted_amount_total" in jt.columns:
                        job_quote_amount = jt["quoted_amount_total"].sum()
                    else:
                        job_quote_amount = np.nan
                    job_quote_to_revenue = (
                        job_revenue / job_quote_amount
                        if pd.notna(job_revenue) and pd.notna(job_quote_amount) and job_quote_amount > 0
                        else np.nan
                    )
                    job_fin = pd.Series({
                        "margin_pct_to_date": job_margin_pct,
                        "quote_to_revenue": job_quote_to_revenue,
                    })
                margin_delta = (
                    job_fin["margin_pct_to_date"] - bench_margin
                    if pd.notna(job_fin["margin_pct_to_date"]) and pd.notna(bench_margin)
                    else np.nan
                )
                billable_delta = (
                    (job_billable_share - bench_billable_share) * 100
                    if pd.notna(job_billable_share) and pd.notna(bench_billable_share)
                    else np.nan
                )

                # Health score
                health_points = 0
                if pd.notna(delta_days):
                    health_points += 2 if delta_days > 15 else (1 if delta_days > 7 else 0)
                if pd.notna(margin_delta):
                    health_points += 2 if margin_delta < -10 else (1 if margin_delta < -5 else 0)
                if pd.notna(billable_delta):
                    health_points += 2 if billable_delta < -10 else (1 if billable_delta < -5 else 0)
                health_label = "Red" if health_points >= 4 else ("Amber" if health_points >= 2 else "Green")

                actions = []
                if pd.notna(delta_days) and delta_days > 7:
                    actions.append("Reforecast timeline + reset scope")
                if pd.notna(margin_delta) and margin_delta < -5:
                    actions.append("Margin guardrails + pricing review")
                if pd.notna(billable_delta) and billable_delta < -5:
                    actions.append("Shift billable mix / reduce non‑billable")
                if not actions:
                    actions.append("Monitor and hold course")

                st.markdown("**Executive Snapshot**")
                snapshot_cols = st.columns(5)
                with snapshot_cols[0]:
                    st.metric("Status", "Active" if job_active else "Completed")
                with snapshot_cols[1]:
                    st.metric("Health", health_label)
                with snapshot_cols[2]:
                    st.metric("Margin Δ", f"{margin_delta:+.1f}%" if pd.notna(margin_delta) else "—")
                with snapshot_cols[3]:
                    st.metric("Billable Δ", f"{billable_delta:+.1f}pp" if pd.notna(billable_delta) else "—")
                with snapshot_cols[4]:
                    st.metric("Runtime Δ", f"{delta_days:+.0f}d" if pd.notna(delta_days) else "—")
                st.info("Recommended action: " + " | ".join(dict.fromkeys(actions)))

                st.markdown("**Benchmark vs Job (Medians)**")
                compare_df = pd.DataFrame({
                    "Metric": ["Runtime (Days)", "Runtime (Months)", "Margin %", "Billable Share %"],
                    "Job": [
                        job_runtime_days,
                        job_runtime_months,
                        job_fin["margin_pct_to_date"],
                        job_billable_share * 100 if pd.notna(job_billable_share) else np.nan,
                    ],
                    "Benchmark": [
                        runtime_med_days,
                        runtime_med_months,
                        bench_margin,
                        bench_billable_share * 100 if pd.notna(bench_billable_share) else np.nan,
                    ],
                })
                fig = go.Figure()
                fig.add_trace(go.Bar(x=compare_df["Metric"], y=compare_df["Job"], name="Job"))
                fig.add_trace(go.Bar(x=compare_df["Metric"], y=compare_df["Benchmark"], name="Benchmark"))
                fig.update_layout(barmode="group", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Task Cost Leaders & Overruns**")
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
                    st.dataframe(
                        task_cost.sort_values("cost", ascending=False).head(12).rename(columns={
                            "task_name": "Task",
                            "hours": "Hours",
                            "cost": "Cost",
                            "revenue": "Revenue",
                            "quoted_hours": "Quoted Hours",
                            "hours_overrun_pct": "Hours Overrun %",
                        }),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Hours": st.column_config.NumberColumn(format="%.1f"),
                            "Cost": st.column_config.NumberColumn(format="$%.0f"),
                            "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                            "Quoted Hours": st.column_config.NumberColumn(format="%.1f"),
                            "Hours Overrun %": st.column_config.NumberColumn(format="%.1f%%"),
                        },
                    )

                st.markdown("**Task Distribution vs Benchmark (Median Share)**")
                bench_tasks = df_bench[df_bench["job_no"].isin(completed_jobs)].copy()
                if len(bench_tasks) > 0:
                    bench_job_task = bench_tasks.groupby(["job_no", "task_name"])["hours_raw"].sum().reset_index()
                    bench_job_totals = bench_tasks.groupby("job_no")["hours_raw"].sum().rename("job_hours").reset_index()
                    bench_job_task = bench_job_task.merge(bench_job_totals, on="job_no", how="left")
                    bench_job_task["task_share"] = np.where(
                        bench_job_task["job_hours"] > 0,
                        bench_job_task["hours_raw"] / bench_job_task["job_hours"],
                        np.nan,
                    )
                    bench_task_median = bench_job_task.groupby("task_name")["task_share"].median().reset_index()
                    job_task_share = df_job.groupby("task_name")["hours_raw"].sum().reset_index()
                    job_task_share["job_hours"] = df_job["hours_raw"].sum()
                    job_task_share["task_share"] = np.where(
                        job_task_share["job_hours"] > 0,
                        job_task_share["hours_raw"] / job_task_share["job_hours"],
                        np.nan,
                    )
                    task_compare = job_task_share.merge(bench_task_median, on="task_name", how="left", suffixes=("", "_benchmark"))
                    task_compare["share_delta"] = (task_compare["task_share"] - task_compare["task_share_benchmark"]) * 100
                    st.dataframe(
                        task_compare.sort_values("share_delta", ascending=False).head(12).rename(columns={
                            "task_name": "Task",
                            "task_share": "Job Share",
                            "task_share_benchmark": "Benchmark Share",
                            "share_delta": "Delta (pp)",
                        }),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Job Share": st.column_config.NumberColumn(format="%.1f%%"),
                            "Benchmark Share": st.column_config.NumberColumn(format="%.1f%%"),
                            "Delta (pp)": st.column_config.NumberColumn(format="%.1f"),
                        },
                    )

                st.markdown("**Staff Drivers (FTE‑aware)**")
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

            st.info(
                """
Operational signals:
- High Value / Low Effort: productize and standardize; protect scope and pricing.
- High Value / High Effort: review estimation, resourcing, and margin guardrails.
- Low Value / Low Effort: automate or bundle; consider off-peak capacity.
- Low Value / High Effort: strong candidates for re-pricing or exit.
                """
            )
    _section_end()
    
    # =========================================================================
    # SECTION D: OPERATIONAL CAPACITY (DELIVERY VS SUPPLY)
    # =========================================================================
    _section_start("band-capacity")
    _section_intro(
        "3) Capacity vs Delivery — Are We Over or Under‑Loaded?",
        "Compare actual delivery to available capacity, and stress‑test growth scenarios.",
        "This is the operational reality check before committing to new work."
    )

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

**Turnover and staffing changes**  
This method naturally reflects turnover because it only counts people who actually logged time in the month.  
If someone leaves or joins mid‑period, their capacity is included only for the time they were active.

**How we build supply (capacity)**  
1) We start with people who actually logged time in that month.  
2) We treat each person’s available time as a “standard working month,” adjusted if their role is part‑time.  
3) We then split that available time across teams based on where they actually spent their hours.

**How we measure demand**  
- We total the hours delivered in that month.  
- We separate those hours into billable and non‑billable so you can see the mix.

**How to read the output**  
- **Utilisation (Total)** shows how much of the available time was used.  
- **Billable Utilisation** shows how much of the available time went to billable work.  
- **Slack Hours** show unused time.  

**Plain‑English example**  
Two people worked this month. Ava is full‑time and Ben is half‑time.  
Together they have about 1.5 “people‑months” of capacity.  
If Ava spent 70% of her time in Department A and 30% in Department B, we split her available time the same way.  
Ben spent all his time in Department B, so all of his capacity goes there.  
Now when we compare capacity to actual hours in each department, the totals still add up and no one is counted twice.

This gives a fair, plain‑English view of whether we have spare capacity or are overloaded, without double‑counting people who work across teams.
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
    _section_end()

    # =========================================================================
    # SECTION E: OPERATIONAL DRILLDOWN (DEPARTMENT → CATEGORY → STAFF → BREAKDOWN → TASK)
    # =========================================================================
    _section_start("band-drill")
    _section_intro(
        "3.1) Operational Drilldown — Where Capacity Lives",
        "Walk the chain from department → category → staff → task to locate slack or overload.",
        "Ensures decisions are tied to who can actually deliver the work."
    )
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
We anchor capacity to real timesheets and split it across the same slices of work.  
This means a person’s available time is only counted once, then shared across teams in the same proportions as their actual work.

**How we calculate slack (in four steps)**  
1) **Estimate effective FTE per staff‑month**  
   - If we have a part‑time scaling factor, we use a weighted average for that month.  
   - If not, we assume full‑time.  
2) **Convert FTE to a monthly capacity baseline**  
   - We use a standard working month (38 hours/week × 4.33 weeks) and scale it by effective FTE.  
3) **Allocate capacity to each slice**  
   - We split each person’s capacity across departments/categories in the same share as their hours.  
4) **Compute slack**  
   - Slack = allocated capacity − actual hours.  
   - Slack % = 100% − utilisation.  

**Defensibility check**  
- Capacity allocation always reconciles back to the company total.  
- People who work across areas are not double‑counted.  
- The method reflects how work actually happened, not a static org chart.
 - Turnover is handled by using actual monthly activity, not a fixed headcount list.

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

        loss_dept = job_level_window.groupby("department_final").agg(
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

        loss_cat = job_level_window.groupby("job_category").agg(
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
    _section_end()

    # =========================================================================
    # SECTION G: DEPARTMENT CAPACITY COMPOSITION (RECONCILED)
    # =========================================================================
    _section_start("band-capacity-composition")
    _section_intro(
        "Department Capacity Composition (Reconciled)",
        "Shows who really makes up each department and what work they actually do.",
        "Use this to separate dedicated capacity from shared capacity before making resourcing decisions."
    )
    st.caption(
        "This section uses the current drill selection. It allocates each person’s capacity across departments "
        "based on where they actually spent time, so totals reconcile and shared staff aren’t double‑counted."
    )

    base_staff_df = df_base_window if df_base_window is not None else df_window
    if "department_final" in drill_df.columns and "staff_name" in drill_df.columns:
        staff_total_hours = base_staff_df.groupby("staff_name")["hours_raw"].sum().rename("staff_total_hours")
        dept_staff_hours = drill_df.groupby(["department_final", "staff_name"])["hours_raw"].sum().reset_index()
        dept_staff_hours = dept_staff_hours.rename(columns={"hours_raw": "dept_hours"})

        staff_capacity = None
        if "fte_hours_scaling" in base_staff_df.columns and "month_key" in base_staff_df.columns:
            base_for_capacity = base_staff_df.copy()
            base_for_capacity["month_key"] = pd.to_datetime(base_for_capacity["month_key"], errors="coerce")
            base_for_capacity["fte_hours_scaling"] = base_for_capacity["fte_hours_scaling"].fillna(1.0)
            base_for_capacity["weighted_fte"] = base_for_capacity["hours_raw"] * base_for_capacity["fte_hours_scaling"]
            staff_month = base_for_capacity.groupby(["staff_name", "month_key"]).agg(
                hours=("hours_raw", "sum"),
                weighted_fte=("weighted_fte", "sum"),
            ).reset_index()
            staff_month["effective_fte"] = np.where(
                staff_month["hours"] > 0,
                staff_month["weighted_fte"] / staff_month["hours"],
                1.0,
            )
            staff_month["capacity_hours"] = staff_month["effective_fte"] * 4.33 * config.CAPACITY_HOURS_PER_WEEK
            staff_capacity = staff_month.groupby("staff_name")["capacity_hours"].sum().rename("staff_capacity_hours")

        composition = dept_staff_hours.merge(staff_total_hours.reset_index(), on="staff_name", how="left")
        if staff_capacity is not None:
            composition = composition.merge(staff_capacity.reset_index(), on="staff_name", how="left")
        else:
            composition["staff_capacity_hours"] = np.nan

        composition["dept_share_of_staff_time"] = np.where(
            composition["staff_total_hours"] > 0,
            composition["dept_hours"] / composition["staff_total_hours"] * 100,
            np.nan,
        )
        composition["staff_capacity_hours"] = composition["staff_capacity_hours"].fillna(composition["staff_total_hours"])
        composition["dept_capacity_hours"] = np.where(
            composition["staff_total_hours"] > 0,
            composition["staff_capacity_hours"] * (composition["dept_hours"] / composition["staff_total_hours"]),
            np.nan,
        )

        composition["shared_flag"] = composition["dept_share_of_staff_time"] < 40
        composition["dedicated_flag"] = composition["dept_share_of_staff_time"] >= 70

        dept_summary = composition.groupby("department_final").agg(
            dept_capacity_hours=("dept_capacity_hours", "sum"),
            shared_capacity_hours=("dept_capacity_hours", lambda x: x[composition.loc[x.index, "shared_flag"]].sum()),
            dedicated_capacity_hours=("dept_capacity_hours", lambda x: x[composition.loc[x.index, "dedicated_flag"]].sum()),
        ).reset_index()

        staff_counts = composition.groupby("department_final").agg(
            contributing_staff=("staff_name", "nunique"),
            dedicated_staff=("staff_name", lambda x: x[composition.loc[x.index, "dedicated_flag"]].nunique()),
        ).reset_index()
        dept_summary = dept_summary.merge(staff_counts, on="department_final", how="left")
        dept_summary["shared_capacity_share"] = np.where(
            dept_summary["dept_capacity_hours"] > 0,
            dept_summary["shared_capacity_hours"] / dept_summary["dept_capacity_hours"] * 100,
            np.nan,
        )
        dept_summary["dedicated_capacity_share"] = np.where(
            dept_summary["dept_capacity_hours"] > 0,
            dept_summary["dedicated_capacity_hours"] / dept_summary["dept_capacity_hours"] * 100,
            np.nan,
        )
        dept_summary = dept_summary.sort_values("dept_capacity_hours", ascending=False)

        st.markdown("**Department Mix at a Glance**")
        st.dataframe(
            dept_summary.rename(columns={
                "department_final": "Department",
                "dept_capacity_hours": "Dept Capacity Hours",
                "shared_capacity_share": "Shared Capacity Share",
                "dedicated_capacity_share": "Dedicated Capacity Share",
                "contributing_staff": "Contributing Staff",
                "dedicated_staff": "Dedicated Staff",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Dept Capacity Hours": st.column_config.NumberColumn(format="%.1f"),
                "Shared Capacity Share": st.column_config.NumberColumn(format="%.1f%%"),
                "Dedicated Capacity Share": st.column_config.NumberColumn(format="%.1f%%"),
                "Contributing Staff": st.column_config.NumberColumn(format="%.0f"),
                "Dedicated Staff": st.column_config.NumberColumn(format="%.0f"),
            },
        )

        st.info(
            "Decision lens: departments with high shared‑capacity share require cross‑unit coordination before redeploying people. "
            "Departments with high dedicated‑capacity share can usually absorb new work or release capacity faster."
        )

        st.markdown("**Operational Detail by Department**")
        top_departments = dept_summary["department_final"].head(6).tolist()
        for dept_name in top_departments:
            with st.expander(f"{dept_name} — composition and work footprint", expanded=False):
                dept_comp = composition[composition["department_final"] == dept_name].copy()
                dept_comp = dept_comp.sort_values("dept_capacity_hours", ascending=False).head(12)

                st.markdown("**Effective Make‑Up (Capacity‑Weighted)**")
                st.dataframe(
                    dept_comp.rename(columns={
                        "staff_name": "Staff",
                        "dept_hours": "Dept Hours",
                        "staff_total_hours": "Total Hours (All Depts)",
                        "dept_share_of_staff_time": "Share of Staff Time in Dept",
                        "dept_capacity_hours": "Allocated Capacity Hours",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Dept Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Total Hours (All Depts)": st.column_config.NumberColumn(format="%.1f"),
                        "Share of Staff Time in Dept": st.column_config.NumberColumn(format="%.1f%%"),
                        "Allocated Capacity Hours": st.column_config.NumberColumn(format="%.1f"),
                    },
                )

                dept_df = drill_df[drill_df["department_final"] == dept_name].copy()
                if len(dept_df) > 0:
                    if "task_name" in dept_df.columns:
                        task_roll = dept_df.groupby("task_name")["hours_raw"].sum().reset_index()
                        task_roll = task_roll.sort_values("hours_raw", ascending=False).head(10)
                        st.markdown("**Top Tasks Performed**")
                        st.dataframe(
                            task_roll.rename(columns={"task_name": "Task", "hours_raw": "Hours"}),
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Hours": st.column_config.NumberColumn(format="%.1f")},
                        )

                    if category_col in dept_df.columns:
                        cat_roll = dept_df.groupby(category_col)["hours_raw"].sum().reset_index()
                        cat_roll["share"] = np.where(
                            cat_roll["hours_raw"].sum() > 0,
                            cat_roll["hours_raw"] / cat_roll["hours_raw"].sum() * 100,
                            np.nan,
                        )
                        cat_roll = cat_roll.sort_values("hours_raw", ascending=False).head(8)
                        st.markdown("**Job Categories Supported**")
                        st.dataframe(
                            cat_roll.rename(columns={category_col: "Category", "hours_raw": "Hours", "share": "Share"}),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Hours": st.column_config.NumberColumn(format="%.1f"),
                                "Share": st.column_config.NumberColumn(format="%.1f%%"),
                            },
                        )

                    if "job_no" in dept_df.columns:
                        agg_map = {"hours": ("hours_raw", "sum")}
                        if "rev_alloc" in dept_df.columns:
                            agg_map["revenue"] = ("rev_alloc", "sum")
                        if "base_cost" in dept_df.columns:
                            agg_map["cost"] = ("base_cost", "sum")
                        job_roll = dept_df.groupby("job_no").agg(**agg_map).reset_index()
                        if "revenue" in job_roll.columns and "cost" in job_roll.columns:
                            job_roll["margin"] = job_roll["revenue"] - job_roll["cost"]
                            job_roll["margin_pct"] = np.where(
                                job_roll["revenue"] > 0,
                                job_roll["margin"] / job_roll["revenue"] * 100,
                                np.nan,
                            )
                        if "client" in dept_df.columns:
                            job_roll = job_roll.merge(
                                dept_df[["job_no", "client"]].drop_duplicates(),
                                on="job_no",
                                how="left",
                            )
                        job_roll = job_roll.sort_values("hours", ascending=False).head(10)
                        job_roll = job_roll.rename(columns={
                            "job_no": "Job",
                            "client": "Client",
                            "hours": "Hours",
                            "revenue": "Revenue",
                            "cost": "Cost",
                            "margin": "Margin",
                            "margin_pct": "Margin %",
                        })
                        cols = ["Job", "Hours"]
                        if "Client" in job_roll.columns:
                            cols.insert(1, "Client")
                        if "Revenue" in job_roll.columns:
                            cols.append("Revenue")
                        if "Cost" in job_roll.columns:
                            cols.append("Cost")
                        if "Margin" in job_roll.columns:
                            cols.append("Margin")
                        if "Margin %" in job_roll.columns:
                            cols.append("Margin %")
                        st.markdown("**Top Jobs Touched**")
                        st.dataframe(
                            job_roll[cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Hours": st.column_config.NumberColumn(format="%.1f"),
                                "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                                "Cost": st.column_config.NumberColumn(format="$%.0f"),
                                "Margin": st.column_config.NumberColumn(format="$%.0f"),
                                "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                            },
                        )

                st.info(
                    "Resourcing lens: redeploy capacity into tasks and job types already dominant in this department to minimize ramp time. "
                    "If the footprint is heavy in low‑margin work, prioritize shifting effort toward higher‑margin jobs before adding headcount."
                )

        st.markdown("**Shared Capacity Demand — Cross‑Department Reliance**")
        st.caption(
            "This isolates work delivered by shared staff (people contributing <40% of their time to a department). "
            "Use this to spot categories, tasks, or jobs that depend on cross‑department capacity."
        )
        shared_staff = composition.loc[composition["shared_flag"], "staff_name"].dropna().unique().tolist()
        shared_slice = drill_df[drill_df["staff_name"].isin(shared_staff)].copy()
        if len(shared_slice) > 0:
            shared_slice["shared_hours"] = shared_slice["hours_raw"]
            base_hours = drill_df["hours_raw"].sum()
            shared_hours_total = shared_slice["shared_hours"].sum()
            shared_fte_equiv = shared_hours_total / (4.33 * config.CAPACITY_HOURS_PER_WEEK) if base_hours > 0 else np.nan

            kpi_cols = st.columns(3)
            with kpi_cols[0]:
                st.metric("Shared Hours (All Depts)", fmt_hours(shared_hours_total))
            with kpi_cols[1]:
                st.metric("Shared Share of Total Hours", fmt_percent(shared_hours_total / base_hours * 100) if base_hours > 0 else "—")
            with kpi_cols[2]:
                st.metric("Shared FTE Equiv", f"{shared_fte_equiv:.2f}" if pd.notna(shared_fte_equiv) else "—")

            shared_category = None
            if category_col in shared_slice.columns:
                shared_category = shared_slice.groupby(category_col)["shared_hours"].sum().reset_index()
                total_category = drill_df.groupby(category_col)["hours_raw"].sum().reset_index().rename(columns={"hours_raw": "total_hours"})
                shared_category = shared_category.merge(total_category, on=category_col, how="left")
                shared_category["shared_share"] = np.where(
                    shared_category["total_hours"] > 0,
                    shared_category["shared_hours"] / shared_category["total_hours"] * 100,
                    np.nan,
                )
                shared_category["shared_fte_equiv"] = shared_category["shared_hours"] / (4.33 * config.CAPACITY_HOURS_PER_WEEK)
                shared_category = shared_category.sort_values("shared_hours", ascending=False).head(10)

                st.markdown("**Categories Reliant on Shared Capacity**")
                st.dataframe(
                    shared_category.rename(columns={
                        category_col: "Category",
                        "shared_hours": "Shared Hours",
                        "total_hours": "Total Hours",
                        "shared_share": "Shared Share",
                        "shared_fte_equiv": "Shared FTE Equiv",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Shared Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Total Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Shared Share": st.column_config.NumberColumn(format="%.1f%%"),
                        "Shared FTE Equiv": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

            if "task_name" in shared_slice.columns:
                shared_task = shared_slice.groupby("task_name")["shared_hours"].sum().reset_index()
                total_task = drill_df.groupby("task_name")["hours_raw"].sum().reset_index().rename(columns={"hours_raw": "total_hours"})
                shared_task = shared_task.merge(total_task, on="task_name", how="left")
                shared_task["shared_share"] = np.where(
                    shared_task["total_hours"] > 0,
                    shared_task["shared_hours"] / shared_task["total_hours"] * 100,
                    np.nan,
                )
                shared_task["shared_fte_equiv"] = shared_task["shared_hours"] / (4.33 * config.CAPACITY_HOURS_PER_WEEK)
                shared_task = shared_task.sort_values("shared_hours", ascending=False).head(10)

                st.markdown("**Tasks Reliant on Shared Capacity**")
                st.dataframe(
                    shared_task.rename(columns={
                        "task_name": "Task",
                        "shared_hours": "Shared Hours",
                        "total_hours": "Total Hours",
                        "shared_share": "Shared Share",
                        "shared_fte_equiv": "Shared FTE Equiv",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Shared Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Total Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Shared Share": st.column_config.NumberColumn(format="%.1f%%"),
                        "Shared FTE Equiv": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

            if "job_no" in shared_slice.columns:
                shared_jobs = shared_slice.groupby("job_no")["shared_hours"].sum().reset_index()
                total_jobs = drill_df.groupby("job_no")["hours_raw"].sum().reset_index().rename(columns={"hours_raw": "total_hours"})
                shared_jobs = shared_jobs.merge(total_jobs, on="job_no", how="left")
                shared_jobs["shared_share"] = np.where(
                    shared_jobs["total_hours"] > 0,
                    shared_jobs["shared_hours"] / shared_jobs["total_hours"] * 100,
                    np.nan,
                )
                shared_jobs["shared_fte_equiv"] = shared_jobs["shared_hours"] / (4.33 * config.CAPACITY_HOURS_PER_WEEK)
                if "client" in shared_slice.columns:
                    shared_jobs = shared_jobs.merge(
                        shared_slice[["job_no", "client"]].drop_duplicates(),
                        on="job_no",
                        how="left",
                    )
                shared_jobs = shared_jobs.sort_values("shared_hours", ascending=False).head(10)
                st.markdown("**Jobs Reliant on Shared Capacity**")
                shared_jobs = shared_jobs.rename(columns={
                    "job_no": "Job",
                    "client": "Client",
                    "shared_hours": "Shared Hours",
                    "total_hours": "Total Hours",
                    "shared_share": "Shared Share",
                    "shared_fte_equiv": "Shared FTE Equiv",
                })
                cols = ["Job", "Shared Hours", "Total Hours", "Shared Share", "Shared FTE Equiv"]
                if "Client" in shared_jobs.columns:
                    cols.insert(1, "Client")
                st.dataframe(
                    shared_jobs[cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Shared Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Total Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Shared Share": st.column_config.NumberColumn(format="%.1f%%"),
                        "Shared FTE Equiv": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

            st.info(
                "Resourcing lens: if a category or task has high shared share and high shared FTE, treat it as a cross‑department "
                "service line and plan capacity centrally. If shared reliance is high but total hours are low, prioritize simplification "
                "or automation. Use this view to decide where to build dedicated pods versus maintain flexible shared pools."
            )
        else:
            st.info("No shared‑capacity staff detected in the current selection.")
    else:
        st.info("Department composition requires department and staff data to be present in the current selection.")
    _section_end()


if __name__ == "__main__":
    main()
