"""
Job lifecycle management: active job definitions, first activity, completion status.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from src.config import config


def get_job_first_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get first activity date for each job.
    
    Returns DataFrame with job_no, first_activity_date, first_activity_month
    """
    if "job_no" not in df.columns:
        return pd.DataFrame()
    
    result = df.groupby("job_no").agg(
        first_activity_date=("work_date", "min") if "work_date" in df.columns else ("month_key", "min"),
        first_activity_month=("month_key", "min"),
    ).reset_index()
    
    return result


def get_job_first_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get first revenue month for each job.
    
    Returns DataFrame with job_no, first_revenue_month
    """
    if "job_no" not in df.columns or "rev_alloc" not in df.columns:
        return pd.DataFrame()
    
    # Only consider rows with positive revenue
    df_with_rev = df[df["rev_alloc"] > 0]
    
    if len(df_with_rev) == 0:
        return pd.DataFrame(columns=["job_no", "first_revenue_month"])
    
    result = df_with_rev.groupby("job_no").agg(
        first_revenue_month=("month_key", "min"),
    ).reset_index()
    
    return result


def get_job_completion_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get completion status for each job.
    
    Returns DataFrame with job_no, is_completed, completed_date, job_status
    """
    if "job_no" not in df.columns:
        return pd.DataFrame()
    
    job_info = df.groupby("job_no").agg(
        job_status=("job_status", "first") if "job_status" in df.columns else ("job_no", "first"),
        completed_date=("job_completed_date", "first") if "job_completed_date" in df.columns else ("job_no", "first"),
    ).reset_index()
    
    # Determine completion
    job_info["is_completed"] = False
    
    if "job_completed_date" in df.columns:
        job_info["is_completed"] |= job_info["completed_date"].notna()
    
    if "job_status" in df.columns:
        job_info["is_completed"] |= job_info["job_status"].str.lower().str.contains("completed", na=False)
    
    return job_info


def is_job_active(df: pd.DataFrame, 
                  recency_days: Optional[int] = None,
                  reference_date: Optional[datetime] = None) -> pd.Series:
    """
    Determine if jobs are active based on:
    1. Not completed (completed_date is null OR status != 'Completed')
    2. Has recent activity within recency_days
    
    Returns Series with job_no index and boolean values.
    """
    if "job_no" not in df.columns:
        return pd.Series(dtype=bool)
    
    if recency_days is None:
        recency_days = config.active_job_recency_days
    
    if reference_date is None:
        if "work_date" in df.columns:
            reference_date = df["work_date"].max()
        elif "month_key" in df.columns:
            reference_date = df["month_key"].max()
        else:
            reference_date = datetime.now()
    
    cutoff_date = reference_date - timedelta(days=recency_days)
    
    # Get completion status
    completion = get_job_completion_status(df)
    
    # Get last activity
    if "work_date" in df.columns:
        last_activity = df.groupby("job_no")["work_date"].max()
    else:
        last_activity = df.groupby("job_no")["month_key"].max()
    
    last_activity = last_activity.reset_index()
    last_activity.columns = ["job_no", "last_activity"]
    
    # Merge
    job_status = completion.merge(last_activity, on="job_no", how="left")
    
    # Active = not completed AND recent activity
    job_status["is_active"] = (~job_status["is_completed"]) & (job_status["last_activity"] >= cutoff_date)
    
    return job_status.set_index("job_no")["is_active"]


def get_active_jobs_with_metrics(df: pd.DataFrame,
                                  recency_days: Optional[int] = None,
                                  reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Get active jobs with key metrics for delivery tracking.
    
    Returns DataFrame with:
    - job_no, department_final, job_category, client
    - quoted_hours, quoted_amount
    - actual_hours, actual_cost, actual_revenue
    - pct_quote_consumed, scope_creep_pct
    - rate_variance
    - risk_flag
    """
    from src.data.semantic import safe_quote_job_task
    
    if "job_no" not in df.columns:
        return pd.DataFrame()
    
    # Get active job list
    active_mask = is_job_active(df, recency_days, reference_date)
    active_jobs = active_mask[active_mask].index.tolist()
    
    df_active = df[df["job_no"].isin(active_jobs)].copy()
    
    if len(df_active) == 0:
        return pd.DataFrame()
    
    # Job-level aggregations
    job_agg = df_active.groupby("job_no").agg(
        department_final=("department_final", "first"),
        job_category=("job_category", "first"),
        client=("client", "first") if "client" in df_active.columns else ("job_no", "first"),
        job_status=("job_status", "first") if "job_status" in df_active.columns else ("job_no", "first"),
        job_due_date=("job_due_date", "first") if "job_due_date" in df_active.columns else ("job_no", "first"),
        actual_hours=("hours_raw", "sum"),
        actual_cost=("base_cost", "sum"),
        actual_revenue=("rev_alloc", "sum"),
    ).reset_index()
    
    # Safe quote rollup
    job_task = safe_quote_job_task(df_active)
    
    if len(job_task) > 0:
        job_quotes = job_task.groupby("job_no").agg(
            quoted_hours=("quoted_time_total", "sum") if "quoted_time_total" in job_task.columns else ("job_no", "count"),
            quoted_amount=("quoted_amount_total", "sum") if "quoted_amount_total" in job_task.columns else ("job_no", "count"),
        ).reset_index()
        
        job_agg = job_agg.merge(job_quotes, on="job_no", how="left")
    else:
        job_agg["quoted_hours"] = 0
        job_agg["quoted_amount"] = 0
    
    # Scope creep
    if "quote_match_flag" in df_active.columns:
        scope_by_job = df_active.groupby("job_no").apply(
            lambda x: x[x["quote_match_flag"] != "matched"]["hours_raw"].sum() / x["hours_raw"].sum() * 100
            if x["hours_raw"].sum() > 0 else 0
        ).reset_index()
        scope_by_job.columns = ["job_no", "scope_creep_pct"]
        job_agg = job_agg.merge(scope_by_job, on="job_no", how="left")
    else:
        job_agg["scope_creep_pct"] = 0
    
    # Compute derived metrics
    job_agg["pct_quote_consumed"] = np.where(
        job_agg["quoted_hours"] > 0,
        job_agg["actual_hours"] / job_agg["quoted_hours"] * 100,
        np.nan
    )
    
    job_agg["remaining_hours"] = np.maximum(
        job_agg["quoted_hours"] - job_agg["actual_hours"], 0
    )
    
    # Rate metrics
    job_agg["realised_rate"] = np.where(
        job_agg["actual_hours"] > 0,
        job_agg["actual_revenue"] / job_agg["actual_hours"],
        np.nan
    )
    
    job_agg["quote_rate"] = np.where(
        job_agg["quoted_hours"] > 0,
        job_agg["quoted_amount"] / job_agg["quoted_hours"],
        np.nan
    )
    
    job_agg["rate_variance"] = job_agg["realised_rate"] - job_agg["quote_rate"]
    
    # Risk flag
    job_agg["risk_flag"] = "on_track"
    job_agg.loc[job_agg["pct_quote_consumed"] > 80, "risk_flag"] = "watch"
    job_agg.loc[job_agg["pct_quote_consumed"] > 100, "risk_flag"] = "at_risk"
    
    return job_agg


def get_job_burn_rate(df: pd.DataFrame, job_no: str, 
                      weeks: int = 2) -> Dict[str, float]:
    """
    Calculate burn rate for a specific job over recent weeks.
    
    Returns dict with:
    - hours_per_week
    - cost_per_week
    - weeks_remaining (based on quoted hours)
    """
    job_df = df[df["job_no"] == job_no].copy()
    
    if len(job_df) == 0 or "work_date" not in job_df.columns:
        return {"hours_per_week": 0, "cost_per_week": 0, "weeks_remaining": np.nan}
    
    max_date = job_df["work_date"].max()
    cutoff = max_date - timedelta(weeks=weeks)
    
    recent = job_df[job_df["work_date"] >= cutoff]
    
    if len(recent) == 0:
        return {"hours_per_week": 0, "cost_per_week": 0, "weeks_remaining": np.nan}
    
    total_hours = recent["hours_raw"].sum()
    total_cost = recent["base_cost"].sum()
    
    hours_per_week = total_hours / weeks
    cost_per_week = total_cost / weeks
    
    # Estimate remaining weeks
    if "quoted_time_total" in job_df.columns:
        quoted = job_df["quoted_time_total"].iloc[0] if not pd.isna(job_df["quoted_time_total"].iloc[0]) else 0
        actual = job_df["hours_raw"].sum()
        remaining = max(quoted - actual, 0)
        weeks_remaining = remaining / hours_per_week if hours_per_week > 0 else np.nan
    else:
        weeks_remaining = np.nan
    
    return {
        "hours_per_week": hours_per_week,
        "cost_per_week": cost_per_week,
        "weeks_remaining": weeks_remaining,
    }


def get_job_task_attribution(df: pd.DataFrame, job_no: str) -> pd.DataFrame:
    """
    Get task-level attribution for a job showing overrun sources.
    
    Returns DataFrame with task_name, quoted_hours, actual_hours, variance, variance_pct
    """
    from src.data.semantic import safe_quote_job_task
    
    job_df = df[df["job_no"] == job_no].copy()
    
    if len(job_df) == 0:
        return pd.DataFrame()
    
    # Get quote data
    job_task = safe_quote_job_task(job_df)
    
    # Get actual hours
    actuals = job_df.groupby("task_name")["hours_raw"].sum().reset_index()
    actuals.columns = ["task_name", "actual_hours"]
    
    if len(job_task) > 0:
        result = job_task[["task_name", "quoted_time_total"]].copy()
        result.columns = ["task_name", "quoted_hours"]
        result = result.merge(actuals, on="task_name", how="outer")
    else:
        result = actuals.copy()
        result["quoted_hours"] = 0
    
    result["quoted_hours"] = result["quoted_hours"].fillna(0)
    result["actual_hours"] = result["actual_hours"].fillna(0)
    
    result["variance"] = result["actual_hours"] - result["quoted_hours"]
    result["variance_pct"] = np.where(
        result["quoted_hours"] > 0,
        result["variance"] / result["quoted_hours"] * 100,
        np.nan
    )
    
    return result.sort_values("variance", ascending=False)


def get_job_staff_attribution(df: pd.DataFrame, job_no: str) -> pd.DataFrame:
    """
    Get staff-level attribution for a job.
    
    Returns DataFrame with staff_name, hours, cost, tasks_worked
    """
    job_df = df[df["job_no"] == job_no].copy()
    
    if len(job_df) == 0:
        return pd.DataFrame()
    
    result = job_df.groupby("staff_name").agg(
        hours=("hours_raw", "sum"),
        cost=("base_cost", "sum"),
        tasks_worked=("task_name", "nunique"),
    ).reset_index()
    
    return result.sort_values("hours", ascending=False)
