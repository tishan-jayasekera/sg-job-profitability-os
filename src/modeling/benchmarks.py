"""
Benchmark construction for empirical capacity forecasting.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple

from src.data.semantic import get_category_col


def _completed_jobs(df: pd.DataFrame) -> pd.DataFrame:
    if "job_completed_date" in df.columns:
        return df[df["job_completed_date"].notna()].copy()
    if "job_status" in df.columns:
        return df[df["job_status"].str.lower().str.contains("completed", na=False)].copy()
    return df.iloc[0:0].copy()


def build_category_benchmarks(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build benchmark distributions by (department, category).
    Returns:
        summary_df: job_total_hours_p50, job_total_hours_p80, job_count
        task_mix_df: task_share_pct by (department, category, task_name)
    """
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    category_col = get_category_col(df)
    completed = _completed_jobs(df)
    if len(completed) == 0:
        return pd.DataFrame(), pd.DataFrame()

    job_totals = completed.groupby(["job_no", "department_final", category_col]).agg(
        total_hours=("hours_raw", "sum"),
    ).reset_index()

    summary = job_totals.groupby(["department_final", category_col]).agg(
        job_total_hours_p50=("total_hours", "median"),
        job_total_hours_p80=("total_hours", lambda x: x.quantile(0.8)),
        job_count=("job_no", "nunique"),
    ).reset_index().rename(columns={category_col: "category_rev_job"})

    task_totals = completed.groupby(["department_final", category_col, "task_name"]).agg(
        task_hours=("hours_raw", "sum"),
    ).reset_index().rename(columns={category_col: "category_rev_job"})
    group_totals = task_totals.groupby(["department_final", "category_rev_job"])["task_hours"].sum().rename("group_hours")
    task_totals = task_totals.merge(group_totals, on=["department_final", "category_rev_job"], how="left")
    task_totals["task_share_pct"] = np.where(
        task_totals["group_hours"] > 0,
        task_totals["task_hours"] / task_totals["group_hours"] * 100,
        np.nan,
    )

    task_mix = task_totals[[
        "department_final", "category_rev_job", "task_name", "task_share_pct",
    ]].copy()

    return summary, task_mix
