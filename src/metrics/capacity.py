"""
Capacity metrics pack.

Single source of truth for: weekly capacity, load, headroom.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta

from src.data.semantic import leave_exclusion_mask
from src.config import config


def compute_staff_capacity(df: pd.DataFrame,
                           weeks: int = 4,
                           reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Compute capacity metrics per staff member.
    
    Returns DataFrame with:
    - weekly_capacity: 38 * fte_hours_scaling
    - period_capacity: weekly_capacity * weeks
    - billable_load: billable hours in trailing window
    - total_load: total hours in trailing window
    - headroom: period_capacity - total_load
    """
    if "staff_name" not in df.columns:
        return pd.DataFrame()
    
    # Get staff attributes
    if "fte_hours_scaling" in df.columns:
        staff_info = df.groupby("staff_name").agg(
            fte_scaling=("fte_hours_scaling", lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else np.nan),
            department=("department_final", "first") if "department_final" in df.columns else ("staff_name", "first"),
        ).reset_index()
    else:
        staff_info = df.groupby("staff_name").agg(
            fte_scaling=("staff_name", lambda x: 1.0),
            department=("department_final", "first") if "department_final" in df.columns else ("staff_name", "first"),
        ).reset_index()
    
    # Handle missing FTE scaling
    staff_info["fte_scaling"] = staff_info["fte_scaling"].fillna(config.DEFAULT_FTE_SCALING)
    
    # Calculate capacity
    staff_info["weekly_capacity"] = config.CAPACITY_HOURS_PER_WEEK * staff_info["fte_scaling"]
    staff_info["period_capacity"] = staff_info["weekly_capacity"] * weeks
    
    # Calculate trailing load
    if reference_date is None:
        if "work_date" in df.columns:
            reference_date = df["work_date"].max()
        elif "month_key" in df.columns:
            reference_date = df["month_key"].max()
        else:
            reference_date = datetime.now()
    
    cutoff = reference_date - timedelta(weeks=weeks)
    
    # Filter to trailing window and exclude leave
    df_trailing = df.copy()
    if "work_date" in df_trailing.columns:
        df_trailing = df_trailing[df_trailing["work_date"] >= cutoff]
    elif "month_key" in df_trailing.columns:
        df_trailing = df_trailing[df_trailing["month_key"] >= cutoff]
    
    df_trailing = df_trailing[~leave_exclusion_mask(df_trailing)]
    
    # Billable load
    if "is_billable" in df_trailing.columns:
        billable_load = df_trailing[df_trailing["is_billable"]].groupby("staff_name")["hours_raw"].sum()
    else:
        billable_load = df_trailing.groupby("staff_name")["hours_raw"].sum()
    
    total_load = df_trailing.groupby("staff_name")["hours_raw"].sum()
    
    staff_info = staff_info.merge(
        billable_load.reset_index().rename(columns={"hours_raw": "billable_load"}),
        on="staff_name", how="left"
    )
    staff_info = staff_info.merge(
        total_load.reset_index().rename(columns={"hours_raw": "total_load"}),
        on="staff_name", how="left"
    )
    
    staff_info["billable_load"] = staff_info["billable_load"].fillna(0)
    staff_info["total_load"] = staff_info["total_load"].fillna(0)
    
    # Headroom
    staff_info["headroom"] = staff_info["period_capacity"] - staff_info["total_load"]
    
    # Utilisation in trailing period
    staff_info["trailing_utilisation"] = np.where(
        staff_info["period_capacity"] > 0,
        staff_info["billable_load"] / staff_info["period_capacity"] * 100,
        0
    )
    
    # Active jobs count
    if "job_no" in df_trailing.columns:
        active_jobs = df_trailing.groupby("staff_name")["job_no"].nunique().reset_index()
        active_jobs.columns = ["staff_name", "active_job_count"]
        staff_info = staff_info.merge(active_jobs, on="staff_name", how="left")
        staff_info["active_job_count"] = staff_info["active_job_count"].fillna(0).astype(int)
    else:
        staff_info["active_job_count"] = 0
    
    return staff_info


def compute_capacity_summary(df: pd.DataFrame,
                             weeks: int = 4,
                             reference_date: Optional[datetime] = None) -> Dict[str, float]:
    """
    Compute aggregate capacity summary.
    """
    staff = compute_staff_capacity(df, weeks, reference_date)
    
    if len(staff) == 0:
        return {}
    
    return {
        "total_staff": len(staff),
        "total_supply": staff["period_capacity"].sum(),
        "billable_load": staff["billable_load"].sum(),
        "total_load": staff["total_load"].sum(),
        "headroom": staff["headroom"].sum(),
        "avg_utilisation": staff["trailing_utilisation"].mean(),
    }


def compute_department_capacity(df: pd.DataFrame,
                                weeks: int = 4,
                                reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Compute capacity rolled up by department.
    """
    staff = compute_staff_capacity(df, weeks, reference_date)
    
    if len(staff) == 0 or "department" not in staff.columns:
        return pd.DataFrame()
    
    result = staff.groupby("department").agg(
        staff_count=("staff_name", "count"),
        period_capacity=("period_capacity", "sum"),
        billable_load=("billable_load", "sum"),
        total_load=("total_load", "sum"),
        headroom=("headroom", "sum"),
    ).reset_index()
    
    result["utilisation"] = np.where(
        result["period_capacity"] > 0,
        result["billable_load"] / result["period_capacity"] * 100,
        0
    )
    
    return result


def get_staff_with_headroom(df: pd.DataFrame,
                            min_headroom: float = 10,
                            weeks: int = 4) -> pd.DataFrame:
    """
    Get staff members with available headroom.
    """
    staff = compute_staff_capacity(df, weeks)
    
    return staff[staff["headroom"] >= min_headroom].sort_values("headroom", ascending=False)


def get_overloaded_staff(df: pd.DataFrame,
                         weeks: int = 4) -> pd.DataFrame:
    """
    Get staff members with negative headroom (overloaded).
    """
    staff = compute_staff_capacity(df, weeks)
    
    return staff[staff["headroom"] < 0].sort_values("headroom")


def compute_capacity_forecast(df: pd.DataFrame,
                              forecast_weeks: int = 8,
                              trailing_weeks: int = 4) -> pd.DataFrame:
    """
    Simple capacity forecast based on trailing load trends.
    
    Assumes load continues at trailing rate.
    """
    staff = compute_staff_capacity(df, trailing_weeks)
    
    if len(staff) == 0:
        return pd.DataFrame()
    
    # Project load forward
    staff["weekly_load_rate"] = staff["billable_load"] / trailing_weeks
    staff["forecast_load"] = staff["weekly_load_rate"] * forecast_weeks
    staff["forecast_capacity"] = staff["weekly_capacity"] * forecast_weeks
    staff["forecast_headroom"] = staff["forecast_capacity"] - staff["forecast_load"]
    
    return staff
