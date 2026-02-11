"""
Job mix & demand metrics pack.

Single source of truth for: job intake, portfolio mix, implied FTE demand.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, List, Dict

from src.data.semantic import safe_quote_job_task, get_category_col
from src.data.job_lifecycle import get_job_first_activity, get_job_first_revenue
from src.config import config


@st.cache_data(show_spinner=False)
def assign_job_cohort(df: pd.DataFrame, 
                      cohort_definition: str = "first_activity") -> pd.DataFrame:
    """
    Assign cohort month to each job based on definition.
    
    Cohort definitions:
    - first_activity: first month with timesheet activity
    - first_revenue: first month with revenue
    
    Returns DataFrame with job_no and cohort_month.
    """
    if "job_no" not in df.columns:
        return pd.DataFrame()
    
    if cohort_definition == "first_activity":
        cohort = get_job_first_activity(df)
        cohort = cohort.rename(columns={"first_activity_month": "cohort_month"})
        cohort = cohort[["job_no", "cohort_month"]]
    
    elif cohort_definition == "first_revenue":
        cohort = get_job_first_revenue(df)
        cohort = cohort.rename(columns={"first_revenue_month": "cohort_month"})
    
    else:
        # Default to first activity
        cohort = get_job_first_activity(df)
        cohort = cohort.rename(columns={"first_activity_month": "cohort_month"})
        cohort = cohort[["job_no", "cohort_month"]]
    
    return cohort


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_job_mix(df: pd.DataFrame,
                    group_keys: Optional[tuple[str, ...]] = None,
                    cohort_definition: str = "first_activity") -> pd.DataFrame:
    """
    Compute job mix metrics by cohort month.
    
    Returns DataFrame with:
    - job_count
    - total_quoted_amount, avg_quoted_amount
    - total_quoted_hours, avg_quoted_hours
    - value_per_quoted_hour
    """
    if "job_no" not in df.columns:
        return pd.DataFrame()
    
    # Assign cohorts
    cohort = assign_job_cohort(df, cohort_definition)
    
    # Get job-level attributes
    category_col = get_category_col(df)
    job_attrs = df.groupby("job_no").agg(
        department_final=("department_final", "first"),
        job_category=(category_col, "first"),
    ).reset_index()
    
    job_attrs = job_attrs.merge(cohort, on="job_no", how="left")
    
    # Safe quote totals per job
    job_task = safe_quote_job_task(df)
    
    if len(job_task) > 0:
        job_quotes = job_task.groupby("job_no").agg(
            quoted_hours=("quoted_time_total", "sum") if "quoted_time_total" in job_task.columns else ("job_no", "count"),
            quoted_amount=("quoted_amount_total", "sum") if "quoted_amount_total" in job_task.columns else ("job_no", "count"),
        ).reset_index()
    else:
        job_quotes = pd.DataFrame({"job_no": df["job_no"].unique()})
        job_quotes["quoted_hours"] = 0
        job_quotes["quoted_amount"] = 0
    
    jobs = job_attrs.merge(job_quotes, on="job_no", how="left")
    
    # Build group keys
    agg_keys = ["cohort_month"]
    if group_keys:
        agg_keys = [*group_keys, "cohort_month"]
    
    # Aggregate
    result = jobs.groupby(agg_keys).agg(
        job_count=("job_no", "nunique"),
        total_quoted_hours=("quoted_hours", "sum"),
        total_quoted_amount=("quoted_amount", "sum"),
        avg_quoted_hours=("quoted_hours", "mean"),
        avg_quoted_amount=("quoted_amount", "mean"),
    ).reset_index()
    
    # Derived metrics
    result["value_per_quoted_hour"] = np.where(
        result["total_quoted_hours"] > 0,
        result["total_quoted_amount"] / result["total_quoted_hours"],
        np.nan
    )
    
    return result


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_implied_fte_demand(df: pd.DataFrame,
                               group_keys: Optional[tuple[str, ...]] = None,
                               cohort_definition: str = "first_activity",
                               weeks_per_month: float = 4.33) -> pd.DataFrame:
    """
    Compute implied FTE demand from job mix.
    
    FTE = demand_hours / (38 * weeks)
    
    Returns DataFrame with:
    - demand_hours (total quoted hours)
    - implied_fte_quoted
    - Also actual hours and implied FTE from actuals
    """
    mix = compute_job_mix(df, group_keys, cohort_definition)
    
    if len(mix) == 0:
        return pd.DataFrame()
    
    # Monthly capacity per FTE
    monthly_capacity = config.CAPACITY_HOURS_PER_WEEK * weeks_per_month
    
    mix["implied_fte_quoted"] = mix["total_quoted_hours"] / monthly_capacity
    
    # Also compute from actuals
    agg_keys = ["cohort_month"]
    if group_keys:
        agg_keys = [*group_keys, "cohort_month"]
    
    # Join cohort to fact
    cohort = assign_job_cohort(df, cohort_definition)
    df_with_cohort = df.merge(cohort, on="job_no", how="left")
    
    actuals = df_with_cohort.groupby(agg_keys).agg(
        actual_hours=("hours_raw", "sum"),
    ).reset_index()
    
    actuals["implied_fte_actual"] = actuals["actual_hours"] / monthly_capacity
    
    mix = mix.merge(actuals, on=agg_keys, how="left")
    
    return mix


@st.cache_data(show_spinner=False)
def compute_demand_vs_supply(df: pd.DataFrame,
                             weeks: int = 4) -> pd.DataFrame:
    """
    Compare implied demand vs supply capacity.
    
    Returns DataFrame with:
    - supply_capacity
    - demand_hours (from quote or actual)
    - implied_utilisation
    - slack_pct
    """
    from src.metrics.capacity import compute_capacity_summary
    
    # Get supply
    supply = compute_capacity_summary(df, weeks)
    
    # Get demand from recent months
    if "month_key" not in df.columns:
        return pd.DataFrame()
    
    recent_months = sorted(df["month_key"].dropna().unique())[-3:]  # Last 3 months
    df_recent = df[df["month_key"].isin(recent_months)]
    
    # Safe quote totals
    from src.data.semantic import safe_quote_rollup
    quote = safe_quote_rollup(df_recent, ())
    
    monthly_capacity = config.CAPACITY_HOURS_PER_WEEK * 4.33
    
    result = pd.DataFrame([{
        "supply_total_capacity": supply.get("total_supply", 0),
        "demand_quoted_hours": quote["quoted_hours"].iloc[0] if len(quote) > 0 else 0,
        "demand_actual_hours": df_recent["hours_raw"].sum(),
        "n_staff": supply.get("total_staff", 0),
    }])
    
    result["implied_utilisation_quoted"] = np.where(
        result["supply_total_capacity"] > 0,
        result["demand_quoted_hours"] / result["supply_total_capacity"] * 100,
        0
    )
    
    result["implied_utilisation_actual"] = np.where(
        result["supply_total_capacity"] > 0,
        result["demand_actual_hours"] / result["supply_total_capacity"] * 100,
        0
    )
    
    result["slack_pct_quoted"] = 100 - result["implied_utilisation_quoted"]
    result["slack_pct_actual"] = 100 - result["implied_utilisation_actual"]
    
    return result


@st.cache_data(show_spinner=False)
def compute_job_quadrant(df: pd.DataFrame,
                         cohort_definition: str = "first_activity") -> pd.DataFrame:
    """
    Get job-level data for quadrant scatter.
    
    Returns DataFrame with:
    - job_no, department_final, job_category
    - quoted_hours, quoted_amount
    - cohort_month
    - quadrant assignment
    """
    # Assign cohorts
    cohort = assign_job_cohort(df, cohort_definition)
    
    # Get job attributes
    job_attrs = df.groupby("job_no").agg(
        department_final=("department_final", "first"),
        job_category=(get_category_col(df), "first"),
        client=("client", "first") if "client" in df.columns else ("job_no", "first"),
    ).reset_index()
    
    job_attrs = job_attrs.merge(cohort, on="job_no", how="left")
    
    # Safe quote totals
    job_task = safe_quote_job_task(df)
    
    if len(job_task) > 0:
        job_quotes = job_task.groupby("job_no").agg(
            quoted_hours=("quoted_time_total", "sum") if "quoted_time_total" in job_task.columns else ("job_no", "count"),
            quoted_amount=("quoted_amount_total", "sum") if "quoted_amount_total" in job_task.columns else ("job_no", "count"),
        ).reset_index()
    else:
        job_quotes = pd.DataFrame({"job_no": df["job_no"].unique()})
        job_quotes["quoted_hours"] = 0
        job_quotes["quoted_amount"] = 0
    
    jobs = job_attrs.merge(job_quotes, on="job_no", how="left")
    
    # Compute medians for quadrant assignment
    hours_median = jobs["quoted_hours"].median()
    amount_median = jobs["quoted_amount"].median()
    
    high_value = jobs["quoted_amount"] >= amount_median
    high_effort = jobs["quoted_hours"] >= hours_median
    jobs["quadrant"] = np.select(
        [
            high_value & (~high_effort),
            high_value & high_effort,
            (~high_value) & (~high_effort),
        ],
        [
            "High Value / Low Effort",
            "High Value / High Effort",
            "Low Value / Low Effort",
        ],
        default="Low Value / High Effort",
    )
    jobs["hours_median"] = hours_median
    jobs["amount_median"] = amount_median
    
    return jobs


@st.cache_data(show_spinner=False)
def get_job_mix_summary(df: pd.DataFrame,
                        cohort_definition: str = "first_activity") -> Dict[str, float]:
    """
    Get job mix summary as a dictionary.
    """
    mix = compute_job_mix(df, cohort_definition=cohort_definition)
    
    if len(mix) == 0:
        return {}
    
    # Aggregate across all months
    return {
        "total_jobs": mix["job_count"].sum(),
        "total_quoted_hours": mix["total_quoted_hours"].sum(),
        "total_quoted_amount": mix["total_quoted_amount"].sum(),
        "avg_quoted_hours_per_job": mix["total_quoted_hours"].sum() / mix["job_count"].sum() if mix["job_count"].sum() > 0 else 0,
        "avg_quoted_amount_per_job": mix["total_quoted_amount"].sum() / mix["job_count"].sum() if mix["job_count"].sum() > 0 else 0,
        "value_per_quoted_hour": mix["total_quoted_amount"].sum() / mix["total_quoted_hours"].sum() if mix["total_quoted_hours"].sum() > 0 else 0,
    }
