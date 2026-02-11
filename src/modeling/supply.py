"""
Supply modeling for team velocity by task.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
from typing import List


def _date_col(df: pd.DataFrame) -> str:
    return "work_date" if "work_date" in df.columns else "month_key"


@st.cache_data(show_spinner=False)
def calculate_team_velocity(df: pd.DataFrame, active_job_no: str, weeks: int = 4) -> pd.DataFrame:
    """
    Calculate empirical team velocity per task for an active job.
    Uses trailing `weeks` from the latest date in that job.
    """
    if "job_no" not in df.columns:
        return pd.DataFrame()
    df_job = df[df["job_no"] == active_job_no].copy()
    if len(df_job) == 0:
        return pd.DataFrame()

    date_col = _date_col(df_job)
    df_job[date_col] = pd.to_datetime(df_job[date_col], errors="coerce")
    latest = df_job[date_col].max()
    cutoff = latest - pd.Timedelta(weeks=weeks)
    recent = df_job[df_job[date_col] >= cutoff]
    if len(recent) == 0:
        recent = df_job

    task_velocity = recent.groupby("task_name")["hours_raw"].sum().rename("recent_hours").reset_index()
    task_velocity["team_velocity_hours_week"] = task_velocity["recent_hours"] / weeks if weeks > 0 else np.nan
    task_velocity["job_no"] = active_job_no
    return task_velocity[["job_no", "task_name", "team_velocity_hours_week"]]


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def build_velocity_for_active_jobs(df: pd.DataFrame, active_jobs: tuple[str, ...], weeks: int = 4) -> pd.DataFrame:
    rows = []
    for job_no in active_jobs:
        vel = calculate_team_velocity(df, job_no, weeks=weeks)
        if len(vel) > 0:
            rows.append(vel)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
