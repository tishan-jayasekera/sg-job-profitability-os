"""
Remaining work forecast based on empirical benchmarks.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple


def forecast_remaining_work(
    df_active: pd.DataFrame,
    benchmarks: pd.DataFrame,
    task_mix: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each active job, compute remaining task hours based on benchmark shape.
    Returns a task-level remaining dataframe.
    """
    if len(df_active) == 0:
        return pd.DataFrame()

    job_meta = df_active.groupby("job_no").agg(
        department_final=("department_final", "first"),
        category_rev_job=("category_rev_job", "first"),
    ).reset_index()

    actual_task = df_active.groupby(["job_no", "task_name"]).agg(
        actual_task_hours=("hours_raw", "sum"),
    ).reset_index()

    job_actual_total = df_active.groupby("job_no")["hours_raw"].sum().rename("actual_job_hours")
    job_meta = job_meta.merge(job_actual_total.reset_index(), on="job_no", how="left")

    bench = benchmarks.rename(columns={"category_rev_job": "category_rev_job"})
    job_meta = job_meta.merge(
        bench[["department_final", "category_rev_job", "job_total_hours_p50"]],
        on=["department_final", "category_rev_job"],
        how="left",
    )
    job_meta["projected_eac_hours"] = job_meta["job_total_hours_p50"].fillna(job_meta["actual_job_hours"])

    task_mix_use = task_mix.copy()
    remaining_rows = []

    for _, job in job_meta.iterrows():
        job_no = job["job_no"]
        dept = job["department_final"]
        cat = job["category_rev_job"]
        projected_total = job["projected_eac_hours"]

        mix = task_mix_use[
            (task_mix_use["department_final"] == dept) &
            (task_mix_use["category_rev_job"] == cat)
        ]
        actual = actual_task[actual_task["job_no"] == job_no]

        if len(mix) == 0:
            # Fallback: use actual task share so far
            actual_sum = actual["actual_task_hours"].sum()
            if actual_sum > 0:
                mix = actual.copy()
                mix["task_share_pct"] = mix["actual_task_hours"] / actual_sum * 100
                mix = mix[["task_name", "task_share_pct"]]
            else:
                continue

        for _, t in mix.iterrows():
            task_name = t["task_name"]
            expected_task = projected_total * (t["task_share_pct"] / 100.0) if pd.notna(t["task_share_pct"]) else np.nan
            actual_task_hours = actual.loc[actual["task_name"] == task_name, "actual_task_hours"]
            actual_task_hours = actual_task_hours.iloc[0] if len(actual_task_hours) > 0 else 0.0
            remaining = max((expected_task or 0) - actual_task_hours, 0.0) if pd.notna(expected_task) else 0.0
            overrun = actual_task_hours > (expected_task or 0) if pd.notna(expected_task) else False

            remaining_rows.append({
                "job_no": job_no,
                "department_final": dept,
                "category_rev_job": cat,
                "task_name": task_name,
                "projected_eac_hours": projected_total,
                "expected_task_hours": expected_task,
                "actual_task_hours": actual_task_hours,
                "remaining_task_hours": remaining,
                "is_overrun": overrun,
            })

    return pd.DataFrame(remaining_rows)


def solve_bottlenecks(
    remaining_df: pd.DataFrame,
    velocity_df: pd.DataFrame,
    df_jobs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge remaining work with team velocity to compute ETAs and bottlenecks.
    Returns:
        task_level: remaining + velocity + ETA + action flags
        job_level: job ETA and status
    """
    if len(remaining_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    merged = remaining_df.merge(
        velocity_df,
        on=["job_no", "task_name"],
        how="left",
    )
    merged["team_velocity_hours_week"] = merged["team_velocity_hours_week"].fillna(0)
    merged["weeks_to_complete"] = np.where(
        merged["team_velocity_hours_week"] > 0,
        merged["remaining_task_hours"] / merged["team_velocity_hours_week"],
        np.inf,
    )
    merged["is_bottleneck"] = (merged["remaining_task_hours"] > 0) & (merged["team_velocity_hours_week"] == 0)

    job_eta = merged.groupby("job_no")["weeks_to_complete"].max().rename("job_eta_weeks")
    job_level = df_jobs[["job_no"]].drop_duplicates().merge(job_eta.reset_index(), on="job_no", how="left")

    if "job_due_date" in df_jobs.columns:
        due = df_jobs.groupby("job_no")["job_due_date"].first()
        job_level = job_level.merge(due.reset_index(), on="job_no", how="left")
        job_level["job_due_date"] = pd.to_datetime(job_level["job_due_date"], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        job_level["due_weeks"] = (job_level["job_due_date"] - now).dt.days / 7
    else:
        job_level["due_weeks"] = np.nan

    def _status(row: pd.Series) -> str:
        if pd.isna(row["job_eta_weeks"]):
            return "Unknown"
        if np.isinf(row["job_eta_weeks"]):
            return "Blocked"
        if pd.notna(row["due_weeks"]) and row["job_eta_weeks"] > row["due_weeks"]:
            return "At Risk"
        return "On Track"

    job_level["status"] = job_level.apply(_status, axis=1)
    return merged, job_level
