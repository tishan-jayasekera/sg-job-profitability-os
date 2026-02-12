"""
Delivery Control Tower metrics pack (Command Center).

Keeps calculations minimal and focused on the command-center UI.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.data.job_lifecycle import get_active_jobs_with_metrics
from src.data.semantic import get_category_col, safe_quote_job_task
from src.modeling.intervention import compute_intervention_risk_score


@st.cache_data(show_spinner=False)
def compute_delivery_control_view(df: pd.DataFrame, recency_days: int = 28) -> pd.DataFrame:
    """
    Build the delivery control view.

    Returns DataFrame with columns needed for command center:
    - job_no, job_name (if available)
    - department_final, job_category
    - quoted_hours, actual_hours
    - pct_consumed, scope_creep_pct
    - forecast_margin_pct, median_margin_pct
    - risk_score, risk_band
    - primary_driver, recommended_action
    - margin_at_risk
    """
    if len(df) == 0:
        return pd.DataFrame()

    jobs_df = get_active_jobs_with_metrics(df, recency_days)
    if len(jobs_df) == 0:
        return pd.DataFrame()

    jobs_df = _add_job_name(df, jobs_df)

    benchmarks = compute_benchmarks(df)
    if len(benchmarks) > 0:
        jobs_df = jobs_df.merge(
            benchmarks[[
                "department_final",
                "job_category",
                "median_runtime_days",
                "median_margin_pct",
                "median_quoted_hours",
            ]],
            on=["department_final", "job_category"],
            how="left",
        )
    else:
        jobs_df["median_runtime_days"] = np.nan
        jobs_df["median_margin_pct"] = np.nan
        jobs_df["median_quoted_hours"] = np.nan

    jobs_df = _add_runtime_metrics(df, jobs_df)
    jobs_df = _add_forecasts(df, jobs_df)
    jobs_df = _add_burn_rates(df, jobs_df)

    jobs_df["margin_pct_to_date"] = np.where(
        jobs_df["actual_revenue"] > 0,
        (jobs_df["actual_revenue"] - jobs_df["actual_cost"]) / jobs_df["actual_revenue"] * 100,
        np.nan,
    )

    jobs_df["quote_to_revenue"] = np.where(
        jobs_df["quoted_amount"] > 0,
        jobs_df["actual_revenue"] / jobs_df["quoted_amount"],
        np.nan,
    )

    jobs_df["hours_overrun_pct"] = np.where(
        jobs_df["quoted_hours"] > 0,
        (jobs_df["actual_hours"] - jobs_df["quoted_hours"]) / jobs_df["quoted_hours"] * 100,
        np.nan,
    )

    jobs_df["hours_variance_pct"] = jobs_df["hours_overrun_pct"].fillna(0)
    if "pct_quote_consumed" in jobs_df.columns:
        jobs_df["pct_consumed"] = jobs_df["pct_quote_consumed"]
    elif "pct_consumed" not in jobs_df.columns:
        jobs_df["pct_consumed"] = np.nan

    jobs_df["peer_median_runtime_days"] = jobs_df.get("median_runtime_days")

    risk_result = compute_intervention_risk_score(jobs_df)
    if isinstance(risk_result, pd.DataFrame) and "risk_score" in risk_result.columns:
        jobs_df["risk_score"] = risk_result["risk_score"].values
    else:
        jobs_df["risk_score"] = jobs_df.apply(
            lambda row: compute_intervention_risk_score(row)[0],
            axis=1,
        )
    jobs_df["risk_band"] = np.select(
        [jobs_df["risk_score"] >= 70, jobs_df["risk_score"] >= 50],
        ["Red", "Amber"],
        default="Green",
    )

    driver_action = _compute_primary_driver(jobs_df)
    jobs_df["primary_driver"] = driver_action["primary_driver"].values
    jobs_df["recommended_action"] = driver_action["recommended_action"].values

    raw_margin_at_risk = (
        (jobs_df["median_margin_pct"].fillna(0) - jobs_df["forecast_margin_pct"].fillna(0))
        / 100
        * jobs_df["forecast_revenue"].fillna(0)
    ).clip(lower=0)
    pct_through = pd.to_numeric(jobs_df["pct_consumed"], errors="coerce").fillna(0)
    jobs_df["margin_at_risk"] = np.where(pct_through >= 30, raw_margin_at_risk, np.nan)
    jobs_df["margin_at_risk_confidence"] = np.select(
        [pct_through >= 75, pct_through >= 30],
        ["high", "medium"],
        default="low",
    )

    jobs_df["hours_overrun"] = (
        jobs_df["actual_hours"]
        + jobs_df["remaining_hours"].fillna(0)
        - jobs_df["quoted_hours"]
    ).clip(lower=0)

    jobs_df = _add_recent_activity(df, jobs_df)

    keep_cols = [
        "job_no",
        "job_name",
        "department_final",
        "job_category",
        "quoted_hours",
        "actual_hours",
        "actual_cost",
        "actual_revenue",
        "pct_consumed",
        "scope_creep_pct",
        "margin_pct_to_date",
        "forecast_margin_pct",
        "median_margin_pct",
        "risk_score",
        "risk_band",
        "primary_driver",
        "recommended_action",
        "margin_at_risk",
        "margin_at_risk_confidence",
        "hours_overrun",
        "hours_variance_pct",
        "rate_variance",
        "forecast_revenue",
        "remaining_hours",
        "runtime_days",
        "runtime_delta_days",
        "burn_rate_per_day",
        "burn_rate_prev_per_day",
        "last_activity",
    ]

    existing_cols = [col for col in keep_cols if col in jobs_df.columns]

    return jobs_df[existing_cols].sort_values("risk_score", ascending=False)


@st.cache_data(show_spinner=False)
def compute_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dept+category benchmarks from completed jobs."""
    if "job_no" not in df.columns:
        return pd.DataFrame()

    category_col = get_category_col(df)
    job_completion = df.groupby("job_no").agg(
        completed_date=("job_completed_date", "first") if "job_completed_date" in df.columns else ("job_no", "first"),
        job_status=("job_status", "first") if "job_status" in df.columns else ("job_no", "first"),
        department_final=("department_final", "first"),
        job_category=(category_col, "first"),
    ).reset_index()

    if "job_completed_date" in job_completion.columns:
        job_completion["is_completed"] = job_completion["completed_date"].notna()
    elif "job_status" in job_completion.columns:
        job_completion["is_completed"] = job_completion["job_status"].str.lower().str.contains("completed", na=False)
    else:
        job_completion["is_completed"] = False

    completed_jobs = set(job_completion[job_completion["is_completed"] == True]["job_no"].tolist())
    df_completed = df[df["job_no"].isin(completed_jobs)].copy()
    if len(df_completed) == 0:
        return pd.DataFrame()

    date_col = _get_date_col(df_completed)
    df_completed[date_col] = pd.to_datetime(df_completed[date_col], errors="coerce")

    runtime = df_completed.groupby("job_no").agg(
        start_date=(date_col, "min"),
        end_date=(date_col, "max"),
    ).reset_index()
    runtime["runtime_days"] = (runtime["end_date"] - runtime["start_date"]).dt.days + 1
    runtime = runtime.merge(
        job_completion[["job_no", "department_final", "job_category"]],
        on="job_no",
        how="left",
    )

    profit = df_completed.groupby("job_no").agg(
        revenue=("rev_alloc", "sum") if "rev_alloc" in df_completed.columns else ("job_no", "count"),
        cost=("base_cost", "sum") if "base_cost" in df_completed.columns else ("job_no", "count"),
        hours=("hours_raw", "sum"),
    ).reset_index()
    profit["margin_pct"] = np.where(
        profit["revenue"] > 0,
        (profit["revenue"] - profit["cost"]) / profit["revenue"] * 100,
        np.nan,
    )
    profit = profit.merge(
        job_completion[["job_no", "department_final", "job_category"]],
        on="job_no",
        how="left",
    )

    job_task = safe_quote_job_task(df_completed)
    if len(job_task) > 0 and "quoted_time_total" in job_task.columns:
        job_quotes = job_task.groupby("job_no")["quoted_time_total"].sum().reset_index()
        job_quotes = job_quotes.rename(columns={"quoted_time_total": "quoted_hours"})
    else:
        job_quotes = pd.DataFrame(columns=["job_no", "quoted_hours"])

    profit = profit.merge(job_quotes, on="job_no", how="left")

    bench = profit.merge(runtime[["job_no", "runtime_days"]], on="job_no", how="left")
    bench = bench.groupby(["department_final", "job_category"]).agg(
        median_runtime_days=("runtime_days", "median"),
        median_margin_pct=("margin_pct", "median"),
        median_quoted_hours=("quoted_hours", "median"),
    ).reset_index()

    return bench


def compute_root_cause_drivers(
    df: pd.DataFrame,
    job_row: pd.Series,
    benchmarks: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Compute ranked root cause drivers with evidence.

    Returns list of driver dicts with:
    - driver_name
    - score (0-100)
    - evidence_metric
    - evidence_value
    - benchmark_value
    - recommendation
    """
    if job_row is None or len(job_row) == 0:
        return []

    if benchmarks is None:
        benchmarks = compute_benchmarks(df)

    drivers: List[Dict] = []
    job_no = job_row["job_no"]
    job_df = df[df["job_no"] == job_no]

    scope_creep_pct = job_row.get("scope_creep_pct", 0)
    if scope_creep_pct > 10:
        if "quote_match_flag" in job_df.columns:
            unquoted = job_df[job_df["quote_match_flag"] != "matched"]
            top_unquoted = unquoted.groupby("task_name")["hours_raw"].sum().nlargest(3)
            evidence_detail = ", ".join([
                f"{task}: {hours:.0f}hrs" for task, hours in top_unquoted.items()
            ])
        else:
            evidence_detail = "Unquoted hours detected"

        drivers.append({
            "driver_name": "Scope Creep",
            "score": min(scope_creep_pct * 2, 100),
            "evidence_metric": "Unquoted Hours %",
            "evidence_value": f"{scope_creep_pct:.0f}%",
            "benchmark_value": "<10%",
            "evidence_detail": evidence_detail,
            "recommendation": "Initiate change order for unquoted work; review scope with PM",
        })

    hours_var_pct = job_row.get("hours_variance_pct", 0)
    if pd.isna(hours_var_pct):
        hours_var_pct = 0

    if hours_var_pct > 20:
        dept = job_row.get("department_final")
        cat = job_row.get("job_category")
        bench_hours = None
        if benchmarks is not None and len(benchmarks) > 0:
            bench_match = benchmarks[
                (benchmarks["department_final"] == dept)
                & (benchmarks["job_category"] == cat)
            ]
            if len(bench_match) > 0:
                bench_hours = bench_match["median_quoted_hours"].iloc[0]

        drivers.append({
            "driver_name": "Under-Quoting",
            "score": min(hours_var_pct, 100),
            "evidence_metric": "Hours Variance %",
            "evidence_value": f"+{hours_var_pct:.0f}%",
            "benchmark_value": f"Median: {bench_hours:.0f}hrs" if pd.notna(bench_hours) else "N/A",
            "evidence_detail": (
                f"Actual {job_row.get('actual_hours', 0):.0f} vs "
                f"Quoted {job_row.get('quoted_hours', 0):.0f}"
            ),
            "recommendation": "Review estimation benchmarks for this category",
        })

    rate_variance = job_row.get("rate_variance", 0)
    if pd.notna(rate_variance) and rate_variance < -10:
        task_rates = job_df.groupby("task_name").agg(
            hours=("hours_raw", "sum"),
            revenue=("rev_alloc", "sum"),
        )
        task_rates["rate"] = np.where(
            task_rates["hours"] > 0,
            task_rates["revenue"] / task_rates["hours"],
            np.nan,
        )
        quote_rate = job_row.get("quote_rate", np.nan)
        threshold = quote_rate * 0.8 if pd.notna(quote_rate) else np.nan
        if pd.notna(threshold):
            low_rate_tasks = task_rates[task_rates["rate"] < threshold].nlargest(3, "hours")
            evidence_detail = ", ".join([
                f"{task}: ${rate:.0f}/hr" for task, rate in low_rate_tasks["rate"].items()
            ])
        else:
            evidence_detail = "Multiple tasks below quote rate"

        drivers.append({
            "driver_name": "Rate Leakage",
            "score": min(abs(rate_variance) * 2, 100),
            "evidence_metric": "Rate Variance",
            "evidence_value": f"${rate_variance:.0f}/hr",
            "benchmark_value": f"Quote: ${job_row.get('quote_rate', 0):.0f}/hr",
            "evidence_detail": evidence_detail or "Multiple tasks below quote rate",
            "recommendation": "Audit rate capture; check for write-offs or discounts",
        })

    runtime_delta = job_row.get("runtime_delta_days", 0)
    if pd.notna(runtime_delta) and runtime_delta > 14:
        drivers.append({
            "driver_name": "Runtime Drift",
            "score": min(runtime_delta / 30 * 100, 100),
            "evidence_metric": "Days Over Benchmark",
            "evidence_value": f"+{runtime_delta:.0f} days",
            "benchmark_value": f"Median: {job_row.get('median_runtime_days', 0):.0f} days",
            "evidence_detail": f"Job running {job_row.get('runtime_days', 0):.0f} days",
            "recommendation": "Escalate timeline; review blockers",
        })

    drivers = sorted(drivers, key=lambda x: x["score"], reverse=True)
    return drivers[:3]


def _get_date_col(df: pd.DataFrame) -> str:
    return "work_date" if "work_date" in df.columns else "month_key"


def _add_job_name(df: pd.DataFrame, jobs_df: pd.DataFrame) -> pd.DataFrame:
    for col in ["job_name", "job_title", "job_description"]:
        if col in df.columns and df[col].notna().any():
            job_names = df.groupby("job_no")[col].first().reset_index()
            job_names = job_names.rename(columns={col: "job_name"})
            return jobs_df.merge(job_names, on="job_no", how="left")
    jobs_df["job_name"] = jobs_df["job_no"]
    return jobs_df


def _add_runtime_metrics(df: pd.DataFrame, jobs_df: pd.DataFrame) -> pd.DataFrame:
    if len(jobs_df) == 0:
        return jobs_df

    date_col = _get_date_col(df)
    df_dates = df[df["job_no"].isin(jobs_df["job_no"])].copy()
    df_dates[date_col] = pd.to_datetime(df_dates[date_col], errors="coerce")

    runtime = df_dates.groupby("job_no").agg(
        start_date=(date_col, "min"),
        end_date=(date_col, "max"),
    ).reset_index()
    runtime["runtime_days"] = (runtime["end_date"] - runtime["start_date"]).dt.days + 1

    jobs_df = jobs_df.merge(runtime[["job_no", "runtime_days", "end_date"]], on="job_no", how="left")
    jobs_df["runtime_delta_days"] = jobs_df["runtime_days"] - jobs_df.get("median_runtime_days")

    return jobs_df


def _add_forecasts(df: pd.DataFrame, jobs_df: pd.DataFrame) -> pd.DataFrame:
    if len(jobs_df) == 0:
        return jobs_df

    jobs_df = jobs_df.drop(columns=["remaining_hours"], errors="ignore")

    date_col = _get_date_col(df)
    df_active = df[df["job_no"].isin(jobs_df["job_no"])].copy()
    df_active[date_col] = pd.to_datetime(df_active[date_col], errors="coerce")

    quoted_hours_map = jobs_df.set_index("job_no")["quoted_hours"].to_dict()
    quoted_amount_map = jobs_df.set_index("job_no")["quoted_amount"].to_dict()

    forecasts: List[Dict] = []
    for job_no, df_job in df_active.groupby("job_no"):
        if len(df_job) == 0:
            continue

        last_date = df_job[date_col].max()
        cutoff = last_date - timedelta(days=28)
        recent = df_job[df_job[date_col] >= cutoff]
        if len(recent) == 0:
            recent = df_job

        days = max((recent[date_col].max() - recent[date_col].min()).days, 1)
        burn_rate_per_day = recent["hours_raw"].sum() / days if "hours_raw" in recent.columns else np.nan

        actual_hours = df_job["hours_raw"].sum() if "hours_raw" in df_job.columns else 0
        quoted_hours = quoted_hours_map.get(job_no, np.nan)
        quoted_amount = quoted_amount_map.get(job_no, np.nan)
        quoted_hours_base = quoted_hours if pd.notna(quoted_hours) else 0

        if actual_hours >= quoted_hours_base:
            # Already at/over quote: project two weeks of additional burn.
            remaining_hours_est = (
                burn_rate_per_day * 14
                if pd.notna(burn_rate_per_day) and burn_rate_per_day > 0
                else 0
            )
            remaining_hours = remaining_hours_est
        else:
            remaining_hours = quoted_hours_base - actual_hours

        eta_days = (
            remaining_hours / burn_rate_per_day
            if pd.notna(burn_rate_per_day) and burn_rate_per_day > 0
            else np.nan
        )
        eta_date = last_date + timedelta(days=float(eta_days)) if pd.notna(eta_days) else pd.NaT

        revenue_to_date = df_job["rev_alloc"].sum() if "rev_alloc" in df_job.columns else np.nan
        cost_to_date = df_job["base_cost"].sum() if "base_cost" in df_job.columns else np.nan
        avg_cost_per_hour = (cost_to_date / actual_hours) if actual_hours > 0 else np.nan

        if actual_hours >= quoted_hours_base:
            forecast_cost = (
                cost_to_date + (remaining_hours * avg_cost_per_hour)
                if pd.notna(avg_cost_per_hour)
                else cost_to_date
            )
        else:
            forecast_cost = (
                cost_to_date + (remaining_hours * avg_cost_per_hour)
                if pd.notna(avg_cost_per_hour)
                else np.nan
            )

        if pd.notna(quoted_amount) and quoted_amount > 0:
            forecast_revenue = quoted_amount
        else:
            if actual_hours > 0 and pd.notna(burn_rate_per_day) and burn_rate_per_day > 0:
                revenue_per_hour = revenue_to_date / actual_hours if pd.notna(revenue_to_date) else np.nan
                total_est_hours = actual_hours + max(remaining_hours, burn_rate_per_day * 14)
                forecast_revenue = (
                    revenue_per_hour * total_est_hours
                    if pd.notna(revenue_per_hour)
                    else revenue_to_date
                )
            else:
                forecast_revenue = revenue_to_date

        forecast_margin_pct = (
            (forecast_revenue - forecast_cost) / forecast_revenue * 100
            if pd.notna(forecast_revenue) and forecast_revenue > 0 and pd.notna(forecast_cost)
            else np.nan
        )

        forecasts.append({
            "job_no": job_no,
            "remaining_hours": remaining_hours,
            "eta_days": eta_days,
            "eta_date": eta_date,
            "forecast_margin_pct": forecast_margin_pct,
            "forecast_revenue": forecast_revenue,
        })

    forecast_df = pd.DataFrame(forecasts)
    if len(forecast_df) == 0:
        for col in [
            "remaining_hours",
            "eta_days",
            "eta_date",
            "forecast_margin_pct",
            "forecast_revenue",
        ]:
            if col not in jobs_df.columns:
                jobs_df[col] = np.nan
        return jobs_df

    return jobs_df.merge(forecast_df, on="job_no", how="left")


def _add_burn_rates(df: pd.DataFrame, jobs_df: pd.DataFrame, window_days: int = 28) -> pd.DataFrame:
    if len(jobs_df) == 0:
        return jobs_df

    if "work_date" not in df.columns:
        jobs_df["burn_rate_per_day"] = np.nan
        jobs_df["burn_rate_prev_per_day"] = np.nan
        return jobs_df

    df_dates = df[df["job_no"].isin(jobs_df["job_no"])].copy()
    df_dates["work_date"] = pd.to_datetime(df_dates["work_date"], errors="coerce")

    burn_rows = []
    for job_no, df_job in df_dates.groupby("job_no"):
        last_date = df_job["work_date"].max()
        if pd.isna(last_date):
            burn_rows.append({
                "job_no": job_no,
                "burn_rate_per_day": np.nan,
                "burn_rate_prev_per_day": np.nan,
            })
            continue

        current_start = last_date - timedelta(days=window_days)
        current = df_job[df_job["work_date"] >= current_start]
        current_days = max((current["work_date"].max() - current["work_date"].min()).days, 1)
        current_burn = current["hours_raw"].sum() / current_days if len(current) > 0 else np.nan

        prev_end = current_start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=window_days)
        prev = df_job[(df_job["work_date"] >= prev_start) & (df_job["work_date"] <= prev_end)]
        prev_days = max((prev["work_date"].max() - prev["work_date"].min()).days, 1) if len(prev) > 0 else 0
        prev_burn = prev["hours_raw"].sum() / prev_days if prev_days > 0 else np.nan

        burn_rows.append({
            "job_no": job_no,
            "burn_rate_per_day": current_burn,
            "burn_rate_prev_per_day": prev_burn,
        })

    burn_df = pd.DataFrame(burn_rows)
    return jobs_df.merge(burn_df, on="job_no", how="left")


def _add_recent_activity(df: pd.DataFrame, jobs_df: pd.DataFrame) -> pd.DataFrame:
    date_col = _get_date_col(df)
    if date_col not in df.columns:
        jobs_df["last_activity"] = pd.NaT
        return jobs_df

    df_dates = df[df["job_no"].isin(jobs_df["job_no"])].copy()
    df_dates[date_col] = pd.to_datetime(df_dates[date_col], errors="coerce")
    last_activity = df_dates.groupby("job_no")[date_col].max().reset_index()
    last_activity = last_activity.rename(columns={date_col: "last_activity"})
    return jobs_df.merge(last_activity, on="job_no", how="left")


def _compute_risk_band(score: float) -> str:
    """
    Simple risk banding.

    Red: ≥70 (critical, needs action this week)
    Amber: 50-70 (watch, review this week)
    Green: <50 (on track)
    """
    if score >= 70:
        return "Red"
    if score >= 50:
        return "Amber"
    return "Green"


def _compute_primary_driver(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Compute primary driver and recommended action for all rows in a vectorized pass.
    """
    scope_pct = pd.to_numeric(rows.get("scope_creep_pct"), errors="coerce").fillna(0)
    hours_var = pd.to_numeric(rows.get("hours_variance_pct"), errors="coerce").fillna(0)
    rate_var = pd.to_numeric(rows.get("rate_variance"), errors="coerce").fillna(0)
    margin = pd.to_numeric(rows.get("forecast_margin_pct"), errors="coerce")
    bench = pd.to_numeric(rows.get("median_margin_pct"), errors="coerce")
    runtime_delta = pd.to_numeric(rows.get("runtime_delta_days"), errors="coerce").fillna(0)

    c_scope = scope_pct > 20
    c_hours = (~c_scope) & (hours_var > 30)
    c_rate = (~c_scope) & (~c_hours) & (rate_var < -15)
    c_margin = (~c_scope) & (~c_hours) & (~c_rate) & (margin < (bench - 10))
    c_runtime = (~c_scope) & (~c_hours) & (~c_rate) & (~c_margin) & (runtime_delta > 14)

    primary_driver = np.select(
        [c_scope, c_hours, c_rate, c_margin, c_runtime],
        [
            "Scope creep +" + scope_pct.round(0).astype(int).astype(str) + "%",
            "Hours overrun +" + hours_var.round(0).astype(int).astype(str) + "%",
            "Rate leakage $" + rate_var.round(0).astype(int).astype(str) + "/hr",
            "Margin " + margin.round(0).astype("Int64").astype(str) + "% (bench " + bench.round(0).astype("Int64").astype(str) + "%)",
            "Running +" + runtime_delta.round(0).astype(int).astype(str) + "d over",
        ],
        default="Monitor",
    )

    recommended_action = np.select(
        [c_scope, c_hours, c_rate, c_margin, c_runtime],
        [
            "Initiate change order",
            "PM scope review",
            "Rate audit needed",
            "Cost review",
            "Escalate timeline",
        ],
        default="Continue standard review",
    )

    return pd.DataFrame(
        {
            "primary_driver": primary_driver,
            "recommended_action": recommended_action,
        },
        index=rows.index,
    )
