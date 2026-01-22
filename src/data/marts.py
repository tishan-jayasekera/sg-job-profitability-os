"""
Mart builders for precomputed aggregations.
"""
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

from src.config import config
from src.data.semantic import (
    full_metric_pack, safe_quote_job_task, leave_exclusion_mask,
    profitability_rollup, utilisation_metrics
)
from src.data.cohorts import get_active_jobs, compute_recency_weights


def build_cube_dept_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build department × month cube.
    
    Grain: department_final × month_key
    """
    group_keys = ["department_final", "month_key"]
    
    # Ensure columns exist
    for col in group_keys:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return full_metric_pack(df, group_keys, exclude_leave=True)


def build_cube_dept_category_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build department × category × month cube.
    
    Grain: department_final × job_category × month_key
    """
    group_keys = ["department_final", "job_category", "month_key"]
    
    for col in group_keys:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return full_metric_pack(df, group_keys, exclude_leave=True)


def build_cube_dept_category_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build department × category × task cube with benchmarks.
    
    Grain: department_final × job_category × task_name
    """
    group_keys = ["department_final", "job_category", "task_name"]
    
    for col in group_keys:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Basic metrics
    result = full_metric_pack(df, group_keys, exclude_leave=True)
    
    # Task-level benchmarks
    # Get job-task level data for proper quote handling
    job_task = safe_quote_job_task(df)
    
    # Map group keys to job-task
    key_mapping = df[["job_no", "task_name"] + group_keys].drop_duplicates()
    job_task = job_task.merge(key_mapping, on=["job_no", "task_name"], how="left")
    
    # Compute inclusion rate (% of jobs containing this task)
    if "job_no" in df.columns:
        jobs_per_group = df.groupby(group_keys[:-1])["job_no"].nunique().reset_index()
        jobs_per_group.columns = group_keys[:-1] + ["total_jobs_in_slice"]
        
        jobs_per_task = df.groupby(group_keys)["job_no"].nunique().reset_index()
        jobs_per_task.columns = group_keys + ["jobs_with_task"]
        
        inclusion = jobs_per_task.merge(jobs_per_group, on=group_keys[:-1], how="left")
        inclusion["inclusion_rate"] = inclusion["jobs_with_task"] / inclusion["total_jobs_in_slice"] * 100
        
        result = result.merge(
            inclusion[group_keys + ["inclusion_rate", "jobs_with_task"]],
            on=group_keys,
            how="left"
        )
    
    # Compute percentiles for quoted hours at job-task level
    if "quoted_time_total" in job_task.columns:
        task_stats = job_task.groupby(group_keys).agg(
            quoted_hours_p25=("quoted_time_total", lambda x: x.quantile(0.25)),
            quoted_hours_p50=("quoted_time_total", lambda x: x.quantile(0.50)),
            quoted_hours_p75=("quoted_time_total", lambda x: x.quantile(0.75)),
        ).reset_index()
        
        result = result.merge(task_stats, on=group_keys, how="left")
    
    # Compute actual hours percentiles
    if "job_no" in df.columns:
        actual_by_job_task = df.groupby(["job_no", "task_name"] + group_keys[:-1])["hours_raw"].sum().reset_index()
        actual_by_job_task["task_name_check"] = actual_by_job_task["task_name"]
        
        actual_stats = actual_by_job_task.groupby(group_keys).agg(
            actual_hours_p25=("hours_raw", lambda x: x.quantile(0.25)),
            actual_hours_p50=("hours_raw", lambda x: x.quantile(0.50)),
            actual_hours_p75=("hours_raw", lambda x: x.quantile(0.75)),
        ).reset_index()
        
        result = result.merge(actual_stats, on=group_keys, how="left")
    
    # Compute overrun probability
    if "quoted_time_total" in job_task.columns and "job_no" in df.columns:
        # Merge actual hours to job_task
        actuals = df.groupby(["job_no", "task_name"])["hours_raw"].sum().reset_index()
        actuals.columns = ["job_no", "task_name", "actual_hours"]
        
        job_task_with_actuals = job_task.merge(actuals, on=["job_no", "task_name"], how="left")
        job_task_with_actuals["is_overrun"] = (
            job_task_with_actuals["actual_hours"] > job_task_with_actuals["quoted_time_total"]
        )
        job_task_with_actuals["is_severe_overrun"] = (
            job_task_with_actuals["actual_hours"] > job_task_with_actuals["quoted_time_total"] * config.severe_overrun_threshold
        )
        
        overrun_stats = job_task_with_actuals.groupby(group_keys).agg(
            overrun_rate=("is_overrun", "mean"),
            severe_overrun_rate=("is_severe_overrun", "mean"),
        ).reset_index()
        overrun_stats["overrun_rate"] *= 100
        overrun_stats["severe_overrun_rate"] *= 100
        
        result = result.merge(overrun_stats, on=group_keys, how="left")
    
    # Median cost/hr
    df_with_rate = df[df["hours_raw"] > 0].copy()
    df_with_rate["cost_per_hour"] = df_with_rate["base_cost"] / df_with_rate["hours_raw"]
    
    cost_stats = df_with_rate.groupby(group_keys).agg(
        cost_per_hour_median=("cost_per_hour", "median"),
    ).reset_index()
    
    result = result.merge(cost_stats, on=group_keys, how="left")
    
    return result


def build_cube_dept_category_staff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build department × category × staff cube.
    
    Grain: department_final × job_category × staff_name
    """
    group_keys = ["department_final", "job_category", "staff_name"]
    
    for col in group_keys:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    result = full_metric_pack(df, group_keys, exclude_leave=True)
    
    # Add recency-weighted hours for capability scoring
    df_weighted = df.copy()
    df_weighted["recency_weight"] = compute_recency_weights(df_weighted)
    df_weighted["hours_weighted"] = df_weighted["hours_raw"] * df_weighted["recency_weight"]
    
    weighted_hours = df_weighted.groupby(group_keys).agg(
        hours_weighted=("hours_weighted", "sum"),
        effective_weight=("recency_weight", "sum"),
    ).reset_index()
    
    result = result.merge(weighted_hours, on=group_keys, how="left")
    
    # Active job count per staff
    if "job_no" in df.columns:
        active_jobs = get_active_jobs(df)
        df_active = df[df["job_no"].isin(active_jobs)]
        
        active_counts = df_active.groupby(group_keys)["job_no"].nunique().reset_index()
        active_counts.columns = group_keys + ["active_job_count"]
        
        result = result.merge(active_counts, on=group_keys, how="left")
        result["active_job_count"] = result["active_job_count"].fillna(0).astype(int)
    
    return result


def build_active_jobs_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build active jobs snapshot with risk metrics.
    
    Grain: job_no
    """
    if "job_no" not in df.columns:
        return pd.DataFrame()
    
    active_job_nos = get_active_jobs(df)
    df_active = df[df["job_no"].isin(active_job_nos)].copy()
    
    if len(df_active) == 0:
        return pd.DataFrame()
    
    # Job-level aggregation
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
    
    # Safe quote totals
    job_task = safe_quote_job_task(df_active)
    job_quotes = job_task.groupby("job_no").agg(
        quoted_hours=("quoted_time_total", "sum") if "quoted_time_total" in job_task.columns else ("job_no", "count"),
        quoted_amount=("quoted_amount_total", "sum") if "quoted_amount_total" in job_task.columns else ("job_no", "count"),
    ).reset_index()
    
    result = job_agg.merge(job_quotes, on="job_no", how="left")
    
    # Compute metrics
    result["pct_quote_consumed"] = np.where(
        result["quoted_hours"] > 0,
        result["actual_hours"] / result["quoted_hours"] * 100,
        np.nan
    )
    
    result["remaining_hours"] = np.maximum(result["quoted_hours"] - result["actual_hours"], 0)
    
    # Scope creep
    scope_by_job = df_active.groupby("job_no").apply(
        lambda x: x[x["quote_match_flag"] != "matched"]["hours_raw"].sum() / x["hours_raw"].sum() * 100
        if x["hours_raw"].sum() > 0 else 0
    ).reset_index()
    scope_by_job.columns = ["job_no", "scope_creep_pct"]
    
    result = result.merge(scope_by_job, on="job_no", how="left")
    
    # Risk flag
    result["risk_flag"] = "on_track"
    result.loc[result["pct_quote_consumed"] > 80, "risk_flag"] = "watch"
    result.loc[result["pct_quote_consumed"] > 100, "risk_flag"] = "at_risk"
    
    return result


def build_job_mix_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build job mix by month (for intake/demand analysis).
    
    Uses first_activity_month as cohort definition.
    
    Grain: department_final × job_category × first_activity_month
    """
    if "job_no" not in df.columns or "month_key" not in df.columns:
        return pd.DataFrame()
    
    # Get first activity month per job
    first_activity = df.groupby("job_no")["month_key"].min().reset_index()
    first_activity.columns = ["job_no", "first_activity_month"]
    
    # Get job-level attributes
    job_attrs = df.groupby("job_no").agg(
        department_final=("department_final", "first"),
        job_category=("job_category", "first"),
    ).reset_index()
    
    job_attrs = job_attrs.merge(first_activity, on="job_no", how="left")
    
    # Safe quote totals per job
    job_task = safe_quote_job_task(df)
    job_quotes = job_task.groupby("job_no").agg(
        quoted_hours=("quoted_time_total", "sum") if "quoted_time_total" in job_task.columns else ("job_no", "count"),
        quoted_amount=("quoted_amount_total", "sum") if "quoted_amount_total" in job_task.columns else ("job_no", "count"),
    ).reset_index()
    
    jobs = job_attrs.merge(job_quotes, on="job_no", how="left")
    
    # Aggregate by cohort month
    group_keys = ["department_final", "job_category", "first_activity_month"]
    
    result = jobs.groupby(group_keys).agg(
        job_count=("job_no", "nunique"),
        total_quoted_hours=("quoted_hours", "sum"),
        total_quoted_amount=("quoted_amount", "sum"),
        avg_quoted_hours=("quoted_hours", "mean"),
        avg_quoted_amount=("quoted_amount", "mean"),
    ).reset_index()
    
    result["value_per_quoted_hour"] = np.where(
        result["total_quoted_hours"] > 0,
        result["total_quoted_amount"] / result["total_quoted_hours"],
        np.nan
    )
    
    return result


def build_all_marts(df: pd.DataFrame, output_dir: Optional[Path] = None) -> dict:
    """
    Build all marts and optionally save to disk.
    
    Returns dict of mart DataFrames.
    """
    if output_dir is None:
        output_dir = config.marts_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    marts = {}
    
    print("Building cube_dept_month...")
    marts["cube_dept_month"] = build_cube_dept_month(df)
    
    print("Building cube_dept_category_month...")
    marts["cube_dept_category_month"] = build_cube_dept_category_month(df)
    
    print("Building cube_dept_category_task...")
    marts["cube_dept_category_task"] = build_cube_dept_category_task(df)
    
    print("Building cube_dept_category_staff...")
    marts["cube_dept_category_staff"] = build_cube_dept_category_staff(df)
    
    print("Building active_jobs_snapshot...")
    marts["active_jobs_snapshot"] = build_active_jobs_snapshot(df)
    
    print("Building job_mix_month...")
    marts["job_mix_month"] = build_job_mix_month(df)
    
    # Save
    for name, mart_df in marts.items():
        filepath = output_dir / f"{name}.parquet"
        mart_df.to_parquet(filepath, index=False)
        print(f"  Saved {name}: {len(mart_df):,} rows")
    
    return marts
