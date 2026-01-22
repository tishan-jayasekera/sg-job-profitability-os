"""
Time window handling, recency weighting, and cohort definitions.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

from src.config import config


# =============================================================================
# TIME WINDOWS
# =============================================================================

def get_time_window_dates(window: str, reference_date: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    Get start and end dates for a time window.
    
    Args:
        window: One of '3m', '6m', '12m', '24m', 'fytd', 'all'
        reference_date: Reference date (defaults to today)
    
    Returns:
        (start_date, end_date) tuple
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    end_date = reference_date
    
    if window == "3m":
        start_date = end_date - timedelta(days=90)
    elif window == "6m":
        start_date = end_date - timedelta(days=180)
    elif window == "12m":
        start_date = end_date - timedelta(days=365)
    elif window == "24m":
        start_date = end_date - timedelta(days=730)
    elif window == "fytd":
        # Australian FY starts July 1
        if end_date.month >= 7:
            start_date = datetime(end_date.year, 7, 1)
        else:
            start_date = datetime(end_date.year - 1, 7, 1)
    elif window == "all":
        start_date = datetime(2000, 1, 1)
    else:
        # Default to 12 months
        start_date = end_date - timedelta(days=365)
    
    return start_date, end_date


def filter_by_time_window(df: pd.DataFrame, window: str, 
                          date_col: str = "month_key",
                          reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Filter dataframe by time window.
    """
    if date_col not in df.columns:
        return df
    
    start_date, end_date = get_time_window_dates(window, reference_date)
    
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    return df[mask].copy()


def get_available_months(df: pd.DataFrame, date_col: str = "month_key") -> List[datetime]:
    """Get sorted list of available months in data."""
    if date_col not in df.columns:
        return []
    
    months = df[date_col].dropna().unique()
    return sorted(pd.to_datetime(months))


# =============================================================================
# RECENCY WEIGHTING
# =============================================================================

def compute_recency_weights(df: pd.DataFrame, 
                            date_col: str = "month_key",
                            half_life_months: Optional[int] = None,
                            reference_date: Optional[datetime] = None) -> pd.Series:
    """
    Compute exponential decay weights by month_key.
    
    Args:
        df: DataFrame with date column
        date_col: Column to use for recency
        half_life_months: Decay half-life in months (default from config)
        reference_date: Reference date (defaults to max date in data)
    
    Returns:
        Series of weights (0-1) indexed like df
    """
    if date_col not in df.columns:
        return pd.Series(1.0, index=df.index)
    
    if half_life_months is None:
        half_life_months = config.recency_half_life_months
    
    dates = pd.to_datetime(df[date_col])
    
    if reference_date is None:
        reference_date = dates.max()
    else:
        reference_date = pd.Timestamp(reference_date)
    
    # Months ago
    months_ago = ((reference_date.year - dates.dt.year) * 12 + 
                  (reference_date.month - dates.dt.month))
    months_ago = months_ago.clip(lower=0)
    
    # Exponential decay: weight = 0.5^(months_ago / half_life)
    decay_rate = np.log(0.5) / half_life_months
    weights = np.exp(decay_rate * months_ago)
    
    return weights


def apply_recency_weighting(df: pd.DataFrame, value_col: str,
                            date_col: str = "month_key",
                            half_life_months: Optional[int] = None) -> pd.DataFrame:
    """
    Apply recency weighting to a value column.
    
    Returns DataFrame with additional columns:
    - recency_weight
    - {value_col}_weighted
    """
    df = df.copy()
    df["recency_weight"] = compute_recency_weights(df, date_col, half_life_months)
    df[f"{value_col}_weighted"] = df[value_col] * df["recency_weight"]
    return df


def effective_sample_size(weights: pd.Series) -> float:
    """
    Compute effective sample size given weights.
    ESS = (sum(w))^2 / sum(w^2)
    """
    w_sum = weights.sum()
    w_sq_sum = (weights ** 2).sum()
    
    if w_sq_sum == 0:
        return 0
    
    return (w_sum ** 2) / w_sq_sum


# =============================================================================
# ACTIVE STAFF COHORT
# =============================================================================

def get_active_staff(df: pd.DataFrame, 
                     recency_months: Optional[int] = None,
                     min_hours: float = 1.0,
                     reference_date: Optional[datetime] = None) -> List[str]:
    """
    Get list of active staff (those with recent activity).
    
    Args:
        df: Fact dataframe with staff_name and month_key
        recency_months: How many months back to look (default from config)
        min_hours: Minimum hours to be considered active
        reference_date: Reference date
    
    Returns:
        List of active staff names
    """
    if "staff_name" not in df.columns or "month_key" not in df.columns:
        return df["staff_name"].unique().tolist() if "staff_name" in df.columns else []
    
    if recency_months is None:
        recency_months = config.active_staff_recency_months
    
    if reference_date is None:
        reference_date = df["month_key"].max()
    
    cutoff_date = reference_date - pd.DateOffset(months=recency_months)
    
    recent = df[df["month_key"] >= cutoff_date]
    
    staff_hours = recent.groupby("staff_name")["hours_raw"].sum()
    active = staff_hours[staff_hours >= min_hours].index.tolist()
    
    return active


def filter_active_staff(df: pd.DataFrame,
                        recency_months: Optional[int] = None,
                        min_hours: float = 1.0,
                        reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Filter dataframe to only include active staff.
    """
    active = get_active_staff(df, recency_months, min_hours, reference_date)
    
    if "staff_name" not in df.columns:
        return df
    
    return df[df["staff_name"].isin(active)].copy()


# =============================================================================
# ACTIVE JOBS
# =============================================================================

def get_active_jobs(df: pd.DataFrame,
                    recency_days: Optional[int] = None,
                    reference_date: Optional[datetime] = None) -> List[str]:
    """
    Get list of active jobs (not completed + recent activity).
    
    Active job definition:
    - Not completed (completed_date is null OR status != 'Completed')
    - Has timesheet activity in last N days
    """
    if "job_no" not in df.columns:
        return []
    
    if recency_days is None:
        recency_days = config.active_job_recency_days
    
    if reference_date is None:
        if "work_date" in df.columns:
            reference_date = df["work_date"].max()
        else:
            reference_date = datetime.now()
    
    cutoff_date = reference_date - timedelta(days=recency_days)
    
    df = df.copy()
    
    # Check completion status
    is_not_completed = pd.Series(True, index=df.index)
    
    if "job_completed_date" in df.columns:
        is_not_completed &= df["job_completed_date"].isna()
    
    if "job_status" in df.columns:
        is_not_completed &= df["job_status"].str.lower() != "completed"
    
    # Check recent activity
    has_recent_activity = pd.Series(True, index=df.index)
    
    if "work_date" in df.columns:
        has_recent_activity = df["work_date"] >= cutoff_date
    elif "month_key" in df.columns:
        has_recent_activity = df["month_key"] >= cutoff_date
    
    active_mask = is_not_completed & has_recent_activity
    
    return df.loc[active_mask, "job_no"].unique().tolist()


def filter_active_jobs(df: pd.DataFrame,
                       recency_days: Optional[int] = None,
                       reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Filter dataframe to only include active jobs.
    """
    active = get_active_jobs(df, recency_days, reference_date)
    
    if "job_no" not in df.columns:
        return df
    
    return df[df["job_no"].isin(active)].copy()


# =============================================================================
# BENCHMARK METADATA
# =============================================================================

def get_benchmark_metadata(df: pd.DataFrame, 
                           date_col: str = "month_key",
                           recency_weighted: bool = False) -> dict:
    """
    Get metadata about the benchmark sample.
    """
    meta = {
        "n_rows": len(df),
        "n_jobs": df["job_no"].nunique() if "job_no" in df.columns else 0,
        "n_staff": df["staff_name"].nunique() if "staff_name" in df.columns else 0,
        "date_min": None,
        "date_max": None,
        "recency_weighted": recency_weighted,
    }
    
    if date_col in df.columns and len(df) > 0:
        meta["date_min"] = df[date_col].min()
        meta["date_max"] = df[date_col].max()
    
    if recency_weighted and "recency_weight" in df.columns:
        meta["effective_sample_size"] = effective_sample_size(df["recency_weight"])
    
    return meta
