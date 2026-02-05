"""
Active projects metrics pack.

Single source of truth for: active job identification, risk flags, attribution.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta

from src.data.job_lifecycle import (
    get_active_jobs_with_metrics,
    get_job_task_attribution,
    get_job_staff_attribution,
    get_job_burn_rate
)
from src.data.semantic import get_category_col
from src.config import config


def get_active_jobs_table(df: pd.DataFrame,
                          recency_days: Optional[int] = None,
                          department: Optional[str] = None,
                          category: Optional[str] = None) -> pd.DataFrame:
    """
    Get active jobs table with risk metrics.
    
    Returns DataFrame suitable for delivery control tower display.
    """
    df_filtered = df.copy()
    
    if department:
        df_filtered = df_filtered[df_filtered["department_final"] == department]
    if category:
        category_col = get_category_col(df_filtered)
        df_filtered = df_filtered[df_filtered[category_col] == category]
    
    return get_active_jobs_with_metrics(df_filtered, recency_days)


def get_at_risk_jobs(df: pd.DataFrame,
                     risk_threshold: float = 100,
                     recency_days: Optional[int] = None) -> pd.DataFrame:
    """
    Get jobs flagged as at-risk (% quote consumed > threshold).
    """
    active = get_active_jobs_with_metrics(df, recency_days)
    
    if len(active) == 0:
        return pd.DataFrame()
    
    at_risk = active[active["pct_quote_consumed"] > risk_threshold]
    
    return at_risk.sort_values("pct_quote_consumed", ascending=False)


def get_jobs_by_risk_status(df: pd.DataFrame,
                            recency_days: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Categorize active jobs by risk status.
    
    Returns dict with 'on_track', 'watch', 'at_risk' DataFrames.
    """
    active = get_active_jobs_with_metrics(df, recency_days)
    
    if len(active) == 0:
        return {"on_track": pd.DataFrame(), "watch": pd.DataFrame(), "at_risk": pd.DataFrame()}
    
    return {
        "on_track": active[active["risk_flag"] == "on_track"],
        "watch": active[active["risk_flag"] == "watch"],
        "at_risk": active[active["risk_flag"] == "at_risk"],
    }


def get_risk_summary(df: pd.DataFrame,
                     recency_days: Optional[int] = None) -> Dict[str, int]:
    """
    Get summary counts by risk status.
    """
    by_risk = get_jobs_by_risk_status(df, recency_days)
    
    return {
        "total_active": sum(len(v) for v in by_risk.values()),
        "on_track": len(by_risk["on_track"]),
        "watch": len(by_risk["watch"]),
        "at_risk": len(by_risk["at_risk"]),
    }


def get_job_detail(df: pd.DataFrame, job_no: str) -> Dict:
    """
    Get detailed information for a single job.
    
    Returns dict with:
    - summary metrics
    - task attribution
    - staff attribution
    - burn rate
    """
    job_df = df[df["job_no"] == job_no]
    
    if len(job_df) == 0:
        return {}
    
    # Summary
    active = get_active_jobs_with_metrics(job_df)
    summary = active.iloc[0].to_dict() if len(active) > 0 else {}
    
    # Task attribution
    task_attr = get_job_task_attribution(df, job_no)
    
    # Staff attribution
    staff_attr = get_job_staff_attribution(df, job_no)
    
    # Burn rate
    burn = get_job_burn_rate(df, job_no)
    
    return {
        "summary": summary,
        "task_attribution": task_attr,
        "staff_attribution": staff_attr,
        "burn_rate": burn,
    }


def get_overrun_attribution(df: pd.DataFrame,
                            job_no: str,
                            n_tasks: int = 5,
                            n_staff: int = 5) -> Dict:
    """
    Get attribution for job overrun.
    
    Returns dict with:
    - top_overrun_tasks
    - top_contributing_staff
    """
    task_attr = get_job_task_attribution(df, job_no)
    staff_attr = get_job_staff_attribution(df, job_no)
    
    # Top overrun tasks
    overrun_tasks = task_attr[task_attr["variance"] > 0].nlargest(n_tasks, "variance")
    
    # For staff, we need to identify who worked on overrun tasks
    if len(overrun_tasks) > 0:
        overrun_task_names = overrun_tasks["task_name"].tolist()
        job_df = df[df["job_no"] == job_no]
        overrun_work = job_df[job_df["task_name"].isin(overrun_task_names)]
        
        overrun_staff = overrun_work.groupby("staff_name").agg(
            hours_on_overrun=("hours_raw", "sum"),
            tasks_worked=("task_name", "nunique"),
        ).reset_index().nlargest(n_staff, "hours_on_overrun")
    else:
        overrun_staff = pd.DataFrame()
    
    return {
        "top_overrun_tasks": overrun_tasks,
        "top_contributing_staff": overrun_staff,
    }


def get_jobs_due_soon(df: pd.DataFrame,
                      days: int = 14,
                      recency_days: Optional[int] = None) -> pd.DataFrame:
    """
    Get active jobs with due dates in the next N days.
    """
    if "job_due_date" not in df.columns:
        return pd.DataFrame()
    
    active = get_active_jobs_with_metrics(df, recency_days)
    
    if len(active) == 0 or "job_due_date" not in active.columns:
        return pd.DataFrame()
    
    reference = datetime.now()
    cutoff = reference + timedelta(days=days)
    
    # Filter to jobs with due dates in range
    due_soon = active[
        (active["job_due_date"].notna()) &
        (active["job_due_date"] <= cutoff)
    ]
    
    return due_soon.sort_values("job_due_date")


def compute_active_jobs_trend(df: pd.DataFrame,
                              time_key: str = "month_key") -> pd.DataFrame:
    """
    Compute trend of active jobs over time.
    
    Shows how many jobs were active in each period.
    """
    if time_key not in df.columns or "job_no" not in df.columns:
        return pd.DataFrame()
    
    # Get unique months
    months = sorted(df[time_key].dropna().unique())
    
    trend_data = []
    for month in months:
        # Jobs with activity in this month
        month_jobs = df[df[time_key] == month]["job_no"].unique()
        
        # Of those, how many are not completed by end of month
        # Simplified: just count jobs with activity
        trend_data.append({
            time_key: month,
            "active_jobs": len(month_jobs),
        })
    
    return pd.DataFrame(trend_data)


def get_delivery_control_view(df: pd.DataFrame,
                              recency_days: Optional[int] = None,
                              department: Optional[str] = None,
                              category: Optional[str] = None) -> pd.DataFrame:
    """
    Get delivery control view with centralized risk metrics and forecasts.
    """
    from src.metrics.delivery_control import compute_delivery_control_view

    return compute_delivery_control_view(
        df,
        recency_days=recency_days or config.active_job_recency_days,
        department=department,
        category=category,
    )
