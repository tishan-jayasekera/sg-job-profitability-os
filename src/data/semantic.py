"""
Semantic layer: canonical hierarchy, safe rollups, and aggregation helpers.

CRITICAL: All aggregations must use these helpers to ensure consistency.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


# =============================================================================
# CANONICAL HIERARCHY
# =============================================================================
# Company → department_final → category_rev_job → staff_name → breakdown → task_name

HIERARCHY_LEVELS = [
    "company",
    "department_final",
    "category_rev_job",
    "staff_name",
    "breakdown",
    "task_name",
]
DRILL_DIMENSIONS = {
    "company": [],
    "department": ["department_final"],
    "category": ["department_final", "category_rev_job"],
    "staff": ["department_final", "category_rev_job", "staff_name"],
    "breakdown": ["department_final", "category_rev_job", "staff_name", "breakdown"],
    "task": ["department_final", "category_rev_job", "staff_name", "breakdown", "task_name"],
}


# =============================================================================
# LEAVE EXCLUSION
# =============================================================================

def leave_exclusion_mask(df: pd.DataFrame) -> pd.Series:
    """
    Returns boolean mask where True = row should be EXCLUDED (is leave).
    Usage: df_filtered = df[~leave_exclusion_mask(df)]
    """
    if "task_name" not in df.columns:
        return pd.Series(False, index=df.index)
    
    return df["task_name"].str.contains("leave", case=False, na=False)


def exclude_leave(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with leave tasks excluded."""
    return df[~leave_exclusion_mask(df)].copy()


def get_category_col(df: pd.DataFrame) -> str:
    """Return the best available category column."""
    if "category_rev_job" in df.columns and df["category_rev_job"].notna().any():
        return "category_rev_job"
    if "job_category" in df.columns and df["job_category"].notna().any():
        return "job_category"
    if "category_rev_job_month" in df.columns and df["category_rev_job_month"].notna().any():
        return "category_rev_job_month"
    return "job_category"


# =============================================================================
# SAFE QUOTE ROLLUPS (CRITICAL)
# =============================================================================
# Quote fields repeat on every row for the same (job_no, task_name).
# We MUST dedupe at job-task level before summing to avoid inflation.

def safe_quote_job_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce to unique (job_no, task_name) with quote constants.
    This is the ONLY safe way to aggregate quote fields.
    
    Returns DataFrame with one row per job-task containing:
    - job_no, task_name
    - quoted_time_total, quoted_amount_total
    - quote_rate (derived)
    - quote_match_flag
    """
    if "job_no" not in df.columns or "task_name" not in df.columns:
        return pd.DataFrame()
    
    quote_cols = ["quoted_time_total", "quoted_amount_total", "quote_match_flag"]
    available_cols = ["job_no", "task_name"] + [c for c in quote_cols if c in df.columns]
    
    # Take first value per job-task (they should all be the same)
    job_task = df[available_cols].drop_duplicates(subset=["job_no", "task_name"])
    
    # Compute quote rate
    if "quoted_time_total" in job_task.columns and "quoted_amount_total" in job_task.columns:
        job_task["quote_rate"] = np.where(
            job_task["quoted_time_total"] > 0,
            job_task["quoted_amount_total"] / job_task["quoted_time_total"],
            np.nan
        )
    
    return job_task


def safe_quote_rollup(df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    """
    Safely roll up quote totals after deduping at job-task level.
    
    Args:
        df: Raw fact dataframe
        group_keys: Columns to group by (e.g., ['department_final'])
    
    Returns:
        DataFrame with safe quote totals per group
    """
    # First dedupe at job-task
    job_task = safe_quote_job_task(df)
    
    if len(job_task) == 0:
        return pd.DataFrame()
    
    # Now we need to join group_keys back from original df
    # Get unique job_no + task_name + group_keys mapping
    if group_keys:
        key_cols = ["job_no", "task_name"] + group_keys
        key_cols = list(dict.fromkeys(key_cols))
        key_mapping = df[key_cols].drop_duplicates()
        job_task = job_task.merge(key_mapping, on=["job_no", "task_name"], how="left")
    
    # Aggregate
    agg_dict = {}
    if "quoted_time_total" in job_task.columns:
        agg_dict["quoted_time_total"] = "sum"
    if "quoted_amount_total" in job_task.columns:
        agg_dict["quoted_amount_total"] = "sum"
    
    # Count job-tasks
    agg_dict["job_no"] = "nunique"
    
    if not group_keys:
        result = pd.DataFrame([{
            "quoted_hours": job_task["quoted_time_total"].sum() if "quoted_time_total" in job_task.columns else 0,
            "quoted_amount": job_task["quoted_amount_total"].sum() if "quoted_amount_total" in job_task.columns else 0,
            "job_task_count": len(job_task),
            "job_count": job_task["job_no"].nunique(),
        }])
    else:
        result = job_task.groupby(group_keys).agg(
            quoted_hours=("quoted_time_total", "sum") if "quoted_time_total" in job_task.columns else ("job_no", "count"),
            quoted_amount=("quoted_amount_total", "sum") if "quoted_amount_total" in job_task.columns else ("job_no", "count"),
            job_task_count=("task_name", "count"),
            job_count=("job_no", "nunique"),
        ).reset_index()
        
        # Fix if quoted columns didn't exist
        if "quoted_time_total" not in job_task.columns:
            result["quoted_hours"] = 0
        if "quoted_amount_total" not in job_task.columns:
            result["quoted_amount"] = 0
    
    # Compute safe quote rate
    result["quote_rate"] = np.where(
        result["quoted_hours"] > 0,
        result["quoted_amount"] / result["quoted_hours"],
        np.nan
    )
    
    return result


# =============================================================================
# PROFITABILITY ROLLUP
# =============================================================================

def profitability_rollup(df: pd.DataFrame, group_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Standard profitability metrics rollup.
    
    Returns: hours, cost, revenue, margin, margin_pct, realised_rate
    """
    agg_dict = {
        "hours": ("hours_raw", "sum"),
        "cost": ("base_cost", "sum"),
        "revenue": ("rev_alloc", "sum"),
    }
    
    if group_keys:
        result = df.groupby(group_keys).agg(**agg_dict).reset_index()
    else:
        result = pd.DataFrame([{
            "hours": df["hours_raw"].sum(),
            "cost": df["base_cost"].sum(),
            "revenue": df["rev_alloc"].sum(),
        }])
    
    result["margin"] = result["revenue"] - result["cost"]
    result["margin_pct"] = np.where(
        result["revenue"] != 0,
        result["margin"] / result["revenue"] * 100,
        np.nan
    )
    result["realised_rate"] = np.where(
        result["hours"] != 0,
        result["revenue"] / result["hours"],
        np.nan
    )
    
    return result


# =============================================================================
# RATE ROLLUPS
# =============================================================================

def rate_rollups(df: pd.DataFrame, group_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute realised rate and quote rate (safely) for groups.
    """
    # Profitability for realised rate
    prof = profitability_rollup(df, group_keys)
    
    # Safe quote rollup for quote rate
    quote = safe_quote_rollup(df, group_keys if group_keys else [])
    
    if group_keys and len(quote) > 0:
        result = prof.merge(
            quote[group_keys + ["quoted_hours", "quoted_amount", "quote_rate"]],
            on=group_keys,
            how="left"
        )
    else:
        result = prof.copy()
        if len(quote) > 0:
            result["quoted_hours"] = quote["quoted_hours"].iloc[0]
            result["quoted_amount"] = quote["quoted_amount"].iloc[0]
            result["quote_rate"] = quote["quote_rate"].iloc[0]
        else:
            result["quoted_hours"] = 0
            result["quoted_amount"] = 0
            result["quote_rate"] = np.nan
    
    result["rate_variance"] = result["realised_rate"] - result["quote_rate"]
    
    return result


# =============================================================================
# SCOPE CREEP / QUOTE DELIVERY
# =============================================================================

def scope_creep_metrics(df: pd.DataFrame, group_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute scope creep metrics.
    
    Returns: unquoted_hours, unquoted_share
    """
    # Hours where quote_match_flag != 'matched'
    if "quote_match_flag" not in df.columns:
        if group_keys:
            return pd.DataFrame(columns=group_keys + ["unquoted_hours", "unquoted_share"])
        return pd.DataFrame([{"unquoted_hours": 0, "unquoted_share": 0}])
    
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


def quote_delivery_metrics(df: pd.DataFrame, group_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Full quote vs delivery analysis.
    """
    # Safe quote totals
    quote = safe_quote_rollup(df, group_keys if group_keys else [])
    
    # Actual hours
    if group_keys:
        actuals = df.groupby(group_keys).agg(
            actual_hours=("hours_raw", "sum")
        ).reset_index()
    else:
        actuals = pd.DataFrame([{"actual_hours": df["hours_raw"].sum()}])
    
    # Scope creep
    scope = scope_creep_metrics(df, group_keys)
    
    # Merge all
    if group_keys:
        result = quote.merge(actuals, on=group_keys, how="outer")
        result = result.merge(scope[group_keys + ["unquoted_hours", "unquoted_share"]], on=group_keys, how="left")
    else:
        result = quote.copy()
        result["actual_hours"] = actuals["actual_hours"].iloc[0]
        result["unquoted_hours"] = scope["unquoted_hours"].iloc[0]
        result["unquoted_share"] = scope["unquoted_share"].iloc[0]
    
    # Hours variance
    result["hours_variance"] = result["actual_hours"] - result["quoted_hours"]
    result["hours_variance_pct"] = np.where(
        result["quoted_hours"] > 0,
        result["hours_variance"] / result["quoted_hours"] * 100,
        np.nan
    )
    
    return result


# =============================================================================
# UTILISATION
# =============================================================================

def utilisation_metrics(df: pd.DataFrame, group_keys: Optional[List[str]] = None, 
                        exclude_leave: bool = True) -> pd.DataFrame:
    """
    Compute descriptive utilisation metrics (no targets).
    """
    df = df.copy()
    
    if exclude_leave:
        df = df[~leave_exclusion_mask(df)]
    
    if "is_billable" not in df.columns:
        df["is_billable"] = True  # Assume all billable if field missing
    
    df["billable_hours"] = np.where(df["is_billable"], df["hours_raw"], 0)
    
    if group_keys:
        result = df.groupby(group_keys).agg(
            total_hours=("hours_raw", "sum"),
            billable_hours=("billable_hours", "sum"),
        ).reset_index()
    else:
        total_hrs = df["hours_raw"].sum()
        result = pd.DataFrame([{
            "total_hours": total_hrs,
            "billable_hours": df["billable_hours"].sum(),
        }])
    
    result["utilisation"] = np.where(
        result["total_hours"] > 0,
        result["billable_hours"] / result["total_hours"] * 100,
        0
    )
    return result


# =============================================================================
# FULL METRIC PACK
# =============================================================================

def full_metric_pack(df: pd.DataFrame, group_keys: Optional[List[str]] = None,
                     exclude_leave: bool = True) -> pd.DataFrame:
    """
    Compute all standard metrics for a grouping.
    """
    # Profitability + rates
    rates = rate_rollups(df, group_keys)
    
    # Quote delivery
    delivery = quote_delivery_metrics(df, group_keys)
    
    # Utilisation
    util = utilisation_metrics(df, group_keys, exclude_leave)
    
    # Merge
    if group_keys:
        result = rates.merge(
            delivery[[c for c in delivery.columns if c not in rates.columns or c in group_keys]],
            on=group_keys,
            how="left"
        )
        result = result.merge(
            util[[c for c in util.columns if c not in result.columns or c in group_keys]],
            on=group_keys,
            how="left"
        )
    else:
        result = rates.copy()
        for col in delivery.columns:
            if col not in result.columns:
                result[col] = delivery[col].iloc[0]
        for col in util.columns:
            if col not in result.columns:
                result[col] = util[col].iloc[0]
    
    return result
