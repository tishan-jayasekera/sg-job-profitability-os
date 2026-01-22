"""
Quote → Delivery metrics pack.

Single source of truth for: quote totals, hours variance, scope creep, overrun rates.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

from src.data.semantic import safe_quote_job_task, safe_quote_rollup, get_category_col
from src.config import config


def compute_quote_totals(df: pd.DataFrame,
                         group_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Safely compute quote totals with proper deduplication.
    
    Returns DataFrame with:
    - quoted_hours: safe sum of quoted_time_total
    - quoted_amount: safe sum of quoted_amount_total
    - quote_rate: quoted_amount / quoted_hours
    - job_task_count: number of unique job-tasks
    - job_count: number of unique jobs
    """
    return safe_quote_rollup(df, group_keys if group_keys else [])


def compute_hours_variance(df: pd.DataFrame,
                           group_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute hours variance between actual and quoted.
    
    Returns DataFrame with:
    - quoted_hours, actual_hours
    - hours_variance, hours_variance_pct
    """
    # Safe quote totals
    quote = compute_quote_totals(df, group_keys)
    
    # Actual hours
    if group_keys:
        actuals = df.groupby(group_keys).agg(
            actual_hours=("hours_raw", "sum")
        ).reset_index()
    else:
        actuals = pd.DataFrame([{"actual_hours": df["hours_raw"].sum()}])
    
    # Merge
    if group_keys:
        result = quote.merge(actuals, on=group_keys, how="outer")
    else:
        result = quote.copy()
        result["actual_hours"] = actuals["actual_hours"].iloc[0]
    
    result["quoted_hours"] = result["quoted_hours"].fillna(0)
    result["actual_hours"] = result["actual_hours"].fillna(0)
    
    # Variance
    result["hours_variance"] = result["actual_hours"] - result["quoted_hours"]
    result["hours_variance_pct"] = np.where(
        result["quoted_hours"] > 0,
        result["hours_variance"] / result["quoted_hours"] * 100,
        np.nan
    )
    
    return result


def compute_scope_creep(df: pd.DataFrame,
                        group_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute scope creep metrics.
    
    Scope creep = hours spent on tasks/jobs without quote match.
    
    Returns DataFrame with:
    - total_hours, unquoted_hours
    - unquoted_share (%)
    """
    if "quote_match_flag" not in df.columns:
        if group_keys:
            result = df.groupby(group_keys).agg(
                total_hours=("hours_raw", "sum")
            ).reset_index()
        else:
            result = pd.DataFrame([{"total_hours": df["hours_raw"].sum()}])
        result["unquoted_hours"] = 0
        result["unquoted_share"] = 0
        return result
    
    df = df.copy()
    df["is_unquoted"] = df["quote_match_flag"] != "matched"
    df["unquoted_hours"] = np.where(df["is_unquoted"], df["hours_raw"], 0)
    
    if group_keys:
        result = df.groupby(group_keys).agg(
            total_hours=("hours_raw", "sum"),
            unquoted_hours=("unquoted_hours", "sum"),
        ).reset_index()
    else:
        result = pd.DataFrame([{
            "total_hours": df["hours_raw"].sum(),
            "unquoted_hours": df["unquoted_hours"].sum(),
        }])
    
    result["unquoted_share"] = np.where(
        result["total_hours"] > 0,
        result["unquoted_hours"] / result["total_hours"] * 100,
        0
    )
    
    return result


def compute_overrun_rates(df: pd.DataFrame,
                          group_keys: Optional[List[str]] = None,
                          severe_threshold: float = None) -> pd.DataFrame:
    """
    Compute overrun rates at job-task level.
    
    Returns DataFrame with:
    - n_job_tasks
    - n_overrun (actual > quoted)
    - n_severe_overrun (actual > quoted * threshold)
    - overrun_rate (%), severe_overrun_rate (%)
    """
    if severe_threshold is None:
        severe_threshold = config.severe_overrun_threshold
    
    # Get job-task level data
    job_task = safe_quote_job_task(df)
    
    if len(job_task) == 0:
        cols = (group_keys or []) + [
            "n_job_tasks", "n_overrun", "n_severe_overrun",
            "overrun_rate", "severe_overrun_rate"
        ]
        return pd.DataFrame(columns=cols)
    
    # Merge actual hours
    actuals = df.groupby(["job_no", "task_name"])["hours_raw"].sum().reset_index()
    actuals.columns = ["job_no", "task_name", "actual_hours"]
    
    job_task = job_task.merge(actuals, on=["job_no", "task_name"], how="left")
    job_task["actual_hours"] = job_task["actual_hours"].fillna(0)
    
    # Flags
    if "quoted_time_total" in job_task.columns:
        job_task["is_overrun"] = job_task["actual_hours"] > job_task["quoted_time_total"]
        job_task["is_severe_overrun"] = (
            job_task["actual_hours"] > job_task["quoted_time_total"] * severe_threshold
        )
    else:
        job_task["is_overrun"] = False
        job_task["is_severe_overrun"] = False
    
    # Merge group keys if needed
    if group_keys:
        key_mapping = df[["job_no", "task_name"] + group_keys].drop_duplicates()
        job_task = job_task.merge(key_mapping, on=["job_no", "task_name"], how="left")
        
        result = job_task.groupby(group_keys).agg(
            n_job_tasks=("job_no", "count"),
            n_overrun=("is_overrun", "sum"),
            n_severe_overrun=("is_severe_overrun", "sum"),
        ).reset_index()
    else:
        result = pd.DataFrame([{
            "n_job_tasks": len(job_task),
            "n_overrun": job_task["is_overrun"].sum(),
            "n_severe_overrun": job_task["is_severe_overrun"].sum(),
        }])
    
    result["overrun_rate"] = np.where(
        result["n_job_tasks"] > 0,
        result["n_overrun"] / result["n_job_tasks"] * 100,
        0
    )
    result["severe_overrun_rate"] = np.where(
        result["n_job_tasks"] > 0,
        result["n_severe_overrun"] / result["n_job_tasks"] * 100,
        0
    )
    
    return result


def compute_quote_delivery_full(df: pd.DataFrame,
                                group_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Full quote → delivery analysis combining all metrics.
    """
    # Hours variance
    variance = compute_hours_variance(df, group_keys)
    
    # Scope creep
    scope = compute_scope_creep(df, group_keys)
    
    # Overrun rates
    overrun = compute_overrun_rates(df, group_keys)
    
    # Merge
    if group_keys:
        result = variance.merge(
            scope[[c for c in scope.columns if c not in variance.columns or c in group_keys]],
            on=group_keys,
            how="outer"
        )
        result = result.merge(
            overrun[[c for c in overrun.columns if c not in result.columns or c in group_keys]],
            on=group_keys,
            how="outer"
        )
    else:
        result = variance.copy()
        for col in scope.columns:
            if col not in result.columns:
                result[col] = scope[col].iloc[0]
        for col in overrun.columns:
            if col not in result.columns:
                result[col] = overrun[col].iloc[0]
    
    return result


def get_quote_delivery_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get summary quote delivery metrics as a dictionary.
    """
    full = compute_quote_delivery_full(df)
    
    return {
        "quoted_hours": full["quoted_hours"].iloc[0] if "quoted_hours" in full.columns else 0,
        "actual_hours": full["actual_hours"].iloc[0] if "actual_hours" in full.columns else 0,
        "hours_variance": full["hours_variance"].iloc[0] if "hours_variance" in full.columns else 0,
        "hours_variance_pct": full["hours_variance_pct"].iloc[0] if "hours_variance_pct" in full.columns else 0,
        "unquoted_share": full["unquoted_share"].iloc[0] if "unquoted_share" in full.columns else 0,
        "overrun_rate": full["overrun_rate"].iloc[0] if "overrun_rate" in full.columns else 0,
        "severe_overrun_rate": full["severe_overrun_rate"].iloc[0] if "severe_overrun_rate" in full.columns else 0,
    }


def get_top_overrun_tasks(df: pd.DataFrame, 
                          n: int = 10,
                          department: Optional[str] = None,
                          category: Optional[str] = None) -> pd.DataFrame:
    """
    Get top tasks by hours variance (overruns).
    """
    df_filtered = df.copy()
    
    if department:
        df_filtered = df_filtered[df_filtered["department_final"] == department]
    if category:
        category_col = get_category_col(df_filtered)
        df_filtered = df_filtered[df_filtered[category_col] == category]
    
    # Get task-level variance
    task_variance = compute_hours_variance(df_filtered, ["task_name"])
    
    # Filter to overruns only
    overruns = task_variance[task_variance["hours_variance"] > 0]
    
    return overruns.nlargest(n, "hours_variance")
