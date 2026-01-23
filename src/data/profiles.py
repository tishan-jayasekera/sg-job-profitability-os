"""
Staff capacity and capability profiles - fully empirical, no targets.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from src.config import config
from src.data.semantic import leave_exclusion_mask, get_category_col


def _get_date_col(df: pd.DataFrame) -> str:
    return "work_date" if "work_date" in df.columns else "month_key"


def _reference_date(df: pd.DataFrame) -> pd.Timestamp:
    date_col = _get_date_col(df)
    return pd.to_datetime(df[date_col]).max()


def _filter_training_window(df: pd.DataFrame, months: int) -> pd.DataFrame:
    date_col = _get_date_col(df)
    if date_col not in df.columns:
        return df.copy()
    ref = _reference_date(df)
    cutoff = ref - pd.DateOffset(months=months)
    return df[pd.to_datetime(df[date_col]) >= cutoff].copy()


def compute_staff_capacity(df: pd.DataFrame, window_weeks: int = 4) -> pd.DataFrame:
    """
    Compute pure supply capacity per staff.
    
    Formula: capacity_hours = 38 × fte_hours_scaling × window_weeks
    
    Returns DataFrame with:
        staff_name, department_final, fte_scaling,
        capacity_hours_week, capacity_hours_window
    """
    if "staff_name" not in df.columns:
        return pd.DataFrame()
    
    staff_info = df.groupby("staff_name").agg(
        department_final=("department_final", "first") if "department_final" in df.columns else ("staff_name", "first"),
        fte_scaling=("fte_hours_scaling", "first") if "fte_hours_scaling" in df.columns else ("staff_name", lambda x: 1.0),
    ).reset_index()
    
    staff_info["fte_scaling"] = staff_info["fte_scaling"].fillna(config.DEFAULT_FTE_SCALING)
    staff_info["capacity_hours_week"] = config.CAPACITY_HOURS_PER_WEEK * staff_info["fte_scaling"]
    staff_info["capacity_hours_window"] = staff_info["capacity_hours_week"] * window_weeks
    
    return staff_info


def compute_staff_load(df: pd.DataFrame, window_weeks: int = 4) -> pd.DataFrame:
    """
    Compute empirical load from timesheet data.
    
    Excludes leave tasks. Uses work_date if available, else month_key.
    
    Returns DataFrame with:
        staff_name,
        load_total_hours, load_billable_hours, load_nonbillable_hours,
        load_active_jobs_hours, active_jobs_count,
        avg_hours_per_week, billable_ratio
    """
    if "staff_name" not in df.columns or "hours_raw" not in df.columns:
        return pd.DataFrame()
    
    date_col = _get_date_col(df)
    ref = _reference_date(df)
    cutoff = ref - pd.DateOffset(weeks=window_weeks)
    
    df_window = df.copy()
    if date_col in df_window.columns:
        df_window = df_window[pd.to_datetime(df_window[date_col]) >= cutoff]
    
    df_window = df_window[~leave_exclusion_mask(df_window)]
    
    if "is_billable" not in df_window.columns:
        df_window["is_billable"] = False
    
    df_window["billable_hours"] = np.where(df_window["is_billable"], df_window["hours_raw"], 0)
    df_window["nonbillable_hours"] = np.where(~df_window["is_billable"], df_window["hours_raw"], 0)
    
    agg = df_window.groupby("staff_name").agg(
        load_total_hours=("hours_raw", "sum"),
        load_billable_hours=("billable_hours", "sum"),
        load_nonbillable_hours=("nonbillable_hours", "sum"),
        load_active_jobs_hours=("hours_raw", "sum"),
        active_jobs_count=("job_no", "nunique") if "job_no" in df_window.columns else ("staff_name", "count"),
    ).reset_index()
    
    agg["avg_hours_per_week"] = agg["load_total_hours"] / window_weeks
    agg["billable_ratio"] = np.where(
        agg["load_total_hours"] > 0,
        agg["load_billable_hours"] / agg["load_total_hours"],
        0
    )
    
    # Review-task flag for archetype assignment
    if "task_name" in df_window.columns:
        review_flag = df_window["task_name"].str.contains("review|qa|quality", case=False, na=False)
        review_by_staff = df_window[review_flag].groupby("staff_name").size().rename("has_review_tasks").reset_index()
        review_by_staff["has_review_tasks"] = True
        agg = agg.merge(review_by_staff, on="staff_name", how="left")
        agg["has_review_tasks"] = agg["has_review_tasks"].fillna(False)
    else:
        agg["has_review_tasks"] = False
    
    return agg


def compute_expected_load(df: pd.DataFrame,
                          trailing_weeks: int = 4,
                          forecast_weeks: int = 4) -> pd.DataFrame:
    """
    Forecast expected load based on trailing average.
    
    Formula: expected_load = avg_weekly_hours (trailing) × forecast_weeks
    
    Returns DataFrame with:
        staff_name, avg_weekly_hours, expected_load_hours
    """
    load = compute_staff_load(df, window_weeks=trailing_weeks)
    if len(load) == 0:
        return pd.DataFrame()
    
    load["avg_weekly_hours"] = load["load_total_hours"] / trailing_weeks
    load["expected_load_hours"] = load["avg_weekly_hours"] * forecast_weeks
    
    return load[["staff_name", "avg_weekly_hours", "expected_load_hours"]]


def compute_headroom(capacity_df: pd.DataFrame,
                     expected_load_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute headroom (slack) per staff.
    
    Formula: headroom = capacity_hours_window - expected_load_hours
    
    Returns DataFrame with:
        staff_name, capacity_hours_window, expected_load_hours,
        headroom_hours, headroom_pct
    """
    if len(capacity_df) == 0:
        return pd.DataFrame()
    
    merged = capacity_df.merge(expected_load_df, on="staff_name", how="left")
    merged["expected_load_hours"] = merged["expected_load_hours"].fillna(0)
    merged["headroom_hours"] = merged["capacity_hours_window"] - merged["expected_load_hours"]
    merged["headroom_pct"] = np.where(
        merged["capacity_hours_window"] > 0,
        merged["headroom_hours"] / merged["capacity_hours_window"] * 100,
        0
    )
    
    return merged[[
        "staff_name",
        "capacity_hours_window",
        "expected_load_hours",
        "headroom_hours",
        "headroom_pct",
    ]]


def _months_since_last(date_series: pd.Series, reference: pd.Timestamp) -> float:
    if date_series.isna().all():
        return np.nan
    last_date = pd.to_datetime(date_series).max()
    months = (reference.year - last_date.year) * 12 + (reference.month - last_date.month)
    return max(months, 0)


def compute_task_expertise(df: pd.DataFrame,
                           training_months: int = 12,
                           recency_half_life: int = 6) -> pd.DataFrame:
    """
    Build staff × task capability matrix.
    
    Recency weighting: weight = 0.5^(months_ago / half_life)
    
    Returns DataFrame with:
        staff_name, task_name,
        hours_total, hours_weighted,
        job_count, months_since_last,
        share_of_time, capability_score (0-100 normalized vs peers)
    """
    if "staff_name" not in df.columns or "task_name" not in df.columns:
        return pd.DataFrame()
    
    df_train = _filter_training_window(df, training_months)
    df_train = df_train[~leave_exclusion_mask(df_train)]
    
    if len(df_train) == 0:
        return pd.DataFrame()
    
    date_col = _get_date_col(df_train)
    ref = _reference_date(df_train)
    
    dates = pd.to_datetime(df_train[date_col])
    months_ago = (ref.year - dates.dt.year) * 12 + (ref.month - dates.dt.month)
    months_ago = months_ago.clip(lower=0)
    decay_rate = np.log(0.5) / recency_half_life
    df_train["recency_weight"] = np.exp(decay_rate * months_ago)
    df_train["hours_weighted"] = df_train["hours_raw"] * df_train["recency_weight"]
    
    agg = df_train.groupby(["staff_name", "task_name"]).agg(
        hours_total=("hours_raw", "sum"),
        hours_weighted=("hours_weighted", "sum"),
        job_count=("job_no", "nunique") if "job_no" in df_train.columns else ("staff_name", "count"),
        months_since_last=(date_col, lambda x: _months_since_last(pd.to_datetime(x), ref)),
    ).reset_index()
    
    total_by_staff = agg.groupby("staff_name")["hours_total"].sum().rename("staff_total_hours")
    agg = agg.merge(total_by_staff, on="staff_name", how="left")
    agg["share_of_time"] = np.where(
        agg["staff_total_hours"] > 0,
        agg["hours_total"] / agg["staff_total_hours"],
        0
    )
    
    max_by_task = agg.groupby("task_name")["hours_weighted"].max().rename("task_max_weighted")
    agg = agg.merge(max_by_task, on="task_name", how="left")
    agg["capability_score"] = np.where(
        agg["task_max_weighted"] > 0,
        agg["hours_weighted"] / agg["task_max_weighted"] * 100,
        0
    )
    
    return agg.drop(columns=["staff_total_hours", "task_max_weighted"])


def compute_category_expertise(df: pd.DataFrame,
                               training_months: int = 12,
                               recency_half_life: int = 6) -> pd.DataFrame:
    """
    Build staff × category capability matrix.
    
    Same logic as task expertise but at category_rev_job level.
    Also compute task_breadth (distinct tasks within category).
    
    Returns DataFrame with:
        staff_name, category_rev_job,
        hours_total, hours_weighted,
        job_count, task_breadth, months_since_last,
        share_of_time, capability_score
    """
    if "staff_name" not in df.columns:
        return pd.DataFrame()
    
    category_col = get_category_col(df)
    if category_col not in df.columns:
        return pd.DataFrame()
    
    df_train = _filter_training_window(df, training_months)
    df_train = df_train[~leave_exclusion_mask(df_train)]
    
    if len(df_train) == 0:
        return pd.DataFrame()
    
    date_col = _get_date_col(df_train)
    ref = _reference_date(df_train)
    
    dates = pd.to_datetime(df_train[date_col])
    months_ago = (ref.year - dates.dt.year) * 12 + (ref.month - dates.dt.month)
    months_ago = months_ago.clip(lower=0)
    decay_rate = np.log(0.5) / recency_half_life
    df_train["recency_weight"] = np.exp(decay_rate * months_ago)
    df_train["hours_weighted"] = df_train["hours_raw"] * df_train["recency_weight"]
    
    group_cols = ["staff_name", category_col]
    agg = df_train.groupby(group_cols).agg(
        hours_total=("hours_raw", "sum"),
        hours_weighted=("hours_weighted", "sum"),
        job_count=("job_no", "nunique") if "job_no" in df_train.columns else ("staff_name", "count"),
        task_breadth=("task_name", "nunique") if "task_name" in df_train.columns else ("staff_name", "count"),
        months_since_last=(date_col, lambda x: _months_since_last(pd.to_datetime(x), ref)),
    ).reset_index().rename(columns={category_col: "category_rev_job"})
    
    total_by_staff = agg.groupby("staff_name")["hours_total"].sum().rename("staff_total_hours")
    agg = agg.merge(total_by_staff, on="staff_name", how="left")
    agg["share_of_time"] = np.where(
        agg["staff_total_hours"] > 0,
        agg["hours_total"] / agg["staff_total_hours"],
        0
    )
    
    max_by_category = agg.groupby("category_rev_job")["hours_weighted"].max().rename("cat_max_weighted")
    agg = agg.merge(max_by_category, on="category_rev_job", how="left")
    agg["capability_score"] = np.where(
        agg["cat_max_weighted"] > 0,
        agg["hours_weighted"] / agg["cat_max_weighted"] * 100,
        0
    )
    
    return agg.drop(columns=["staff_total_hours", "cat_max_weighted"])


def assign_archetype(load_df: pd.DataFrame,
                     category_expertise_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign archetype to each staff based on rules.
    """
    if len(load_df) == 0:
        return pd.DataFrame()
    
    top_cat = category_expertise_df.sort_values("share_of_time", ascending=False).drop_duplicates("staff_name")
    top_cat = top_cat[["staff_name", "category_rev_job", "share_of_time"]].rename(
        columns={"category_rev_job": "top_category", "share_of_time": "top_category_share"}
    )
    
    cat_summary = category_expertise_df.groupby("staff_name").agg(
        categories_touched=("category_rev_job", "nunique"),
    ).reset_index()
    cat_summary = cat_summary.merge(top_cat, on="staff_name", how="left")
    
    merged = load_df.merge(cat_summary, on="staff_name", how="left")
    merged["categories_touched"] = merged["categories_touched"].fillna(0).astype(int)
    merged["top_category_share"] = merged["top_category_share"].fillna(0)
    
    archetypes = []
    for _, row in merged.iterrows():
        archetype = "Balanced"
        detail = ""
        
        if row["billable_ratio"] < config.ARCHETYPE_OPS_HEAVY_THRESHOLD:
            archetype = "Ops-Heavy"
        elif row["top_category_share"] > config.ARCHETYPE_SPECIALIST_THRESHOLD:
            archetype = "Specialist"
        elif (row["categories_touched"] >= config.ARCHETYPE_GENERALIST_MIN_CATEGORIES and
              row["top_category_share"] < config.ARCHETYPE_GENERALIST_MAX_SHARE):
            archetype = "Generalist"
        elif row["avg_hours_per_week"] < config.ARCHETYPE_SENIOR_MAX_HOURS_WEEK and row.get("has_review_tasks", False):
            archetype = "Senior/Reviewer"
        else:
            archetype = "Balanced"
        
        if archetype == "Specialist":
            top_cat = row.get("top_category", "")
            if pd.isna(top_cat):
                top_cat = ""
            detail = f"Specialist - {top_cat}".strip(" -")
        
        archetypes.append((archetype, detail))
    
    merged["archetype"] = [a[0] for a in archetypes]
    merged["archetype_detail"] = [a[1] for a in archetypes]
    
    return merged[["staff_name", "archetype", "archetype_detail"]]


def compute_context_switching(df: pd.DataFrame,
                              window_weeks: int = 4) -> pd.DataFrame:
    """
    Compute context switching indicators.
    """
    if "staff_name" not in df.columns:
        return pd.DataFrame()
    
    date_col = _get_date_col(df)
    ref = _reference_date(df)
    cutoff = ref - pd.DateOffset(weeks=window_weeks)
    
    df_window = df.copy()
    if date_col in df_window.columns:
        df_window = df_window[pd.to_datetime(df_window[date_col]) >= cutoff]
    
    df_window = df_window[~leave_exclusion_mask(df_window)]
    
    category_col = get_category_col(df_window)
    
    agg = df_window.groupby("staff_name").agg(
        jobs_touched=("job_no", "nunique") if "job_no" in df_window.columns else ("staff_name", "count"),
        categories_touched=(category_col, "nunique") if category_col in df_window.columns else ("staff_name", "count"),
        tasks_touched=("task_name", "nunique") if "task_name" in df_window.columns else ("staff_name", "count"),
    ).reset_index()
    
    # HHI by task
    if "task_name" in df_window.columns:
        totals = df_window.groupby("staff_name")["hours_raw"].sum().rename("total_hours")
        task_hours = df_window.groupby(["staff_name", "task_name"])["hours_raw"].sum().reset_index()
        task_hours = task_hours.merge(totals, on="staff_name", how="left")
        task_hours["share"] = np.where(task_hours["total_hours"] > 0, task_hours["hours_raw"] / task_hours["total_hours"], 0)
        hhi = task_hours.groupby("staff_name")["share"].apply(lambda x: (x ** 2).sum()).reset_index()
        hhi.columns = ["staff_name", "hhi_tasks"]
        agg = agg.merge(hhi, on="staff_name", how="left")
    else:
        agg["hhi_tasks"] = np.nan
    
    return agg


def build_staff_profiles(df: pd.DataFrame,
                         window_weeks: int = 4,
                         training_months: int = 12) -> pd.DataFrame:
    """
    Master function: build complete staff profiles.
    """
    capacity = compute_staff_capacity(df, window_weeks)
    load = compute_staff_load(df, window_weeks)
    expected = compute_expected_load(df, trailing_weeks=window_weeks, forecast_weeks=window_weeks)
    headroom = compute_headroom(capacity, expected)
    task_expertise = compute_task_expertise(df, training_months, config.RECENCY_HALF_LIFE_MONTHS)
    category_expertise = compute_category_expertise(df, training_months, config.RECENCY_HALF_LIFE_MONTHS)
    archetype = assign_archetype(load, category_expertise)
    context = compute_context_switching(df, window_weeks)
    
    profile = capacity.merge(load, on="staff_name", how="left")
    profile = profile.merge(expected, on="staff_name", how="left")
    profile = profile.merge(headroom, on="staff_name", how="left")
    profile = profile.merge(archetype, on="staff_name", how="left")
    profile = profile.merge(context, on="staff_name", how="left")
    
    if "expected_load_hours" in profile.columns:
        profile["expected_load_hours"] = profile["expected_load_hours"].fillna(0)
    else:
        profile["expected_load_hours"] = 0
    if "capacity_hours_window" not in profile.columns:
        profile["capacity_hours_window"] = 0
    if "capacity_hours_week" not in profile.columns:
        profile["capacity_hours_week"] = 0
    if "headroom_hours" in profile.columns:
        profile["headroom_hours"] = profile["headroom_hours"].fillna(0)
    else:
        profile["headroom_hours"] = 0
    
    return profile
