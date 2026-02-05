"""
Delivery Control Tower metrics pack.

Centralizes risk computation, driver analysis, and forecasting logic
for the Active Delivery page.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import timedelta

from src.data.job_lifecycle import get_active_jobs_with_metrics
from src.data.semantic import safe_quote_job_task, get_category_col
from src.modeling.intervention import compute_intervention_risk_score


@dataclass
class JobRiskProfile:
    """Complete risk profile for a job."""
    job_no: str
    risk_score: float
    risk_band: str
    primary_driver: str
    recommended_action: str
    drivers: List[Dict]
    margin_shortfall: float
    hours_overrun: float


def compute_delivery_control_view(
    df: pd.DataFrame,
    recency_days: int = 28,
    department: Optional[str] = None,
    category: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the complete delivery control view with all metrics.

    This is the single source of truth for the Active Delivery page.
    """
    if len(df) == 0:
        return pd.DataFrame()

    df_filtered = df.copy()
    if department:
        df_filtered = df_filtered[df_filtered["department_final"] == department]
    if category:
        category_col = get_category_col(df_filtered)
        df_filtered = df_filtered[df_filtered[category_col] == category]

    jobs_df = get_active_jobs_with_metrics(df_filtered, recency_days)
    if len(jobs_df) == 0:
        return pd.DataFrame()

    benchmarks = _compute_benchmarks(df_filtered)
    if len(benchmarks) > 0:
        jobs_df = jobs_df.merge(
            benchmarks[[
                "department_final",
                "job_category",
                "median_runtime_days",
                "median_margin_pct",
                "median_billable_share",
                "median_quoted_hours",
            ]],
            on=["department_final", "job_category"],
            how="left",
        )
    else:
        jobs_df["median_runtime_days"] = np.nan
        jobs_df["median_margin_pct"] = np.nan
        jobs_df["median_billable_share"] = np.nan
        jobs_df["median_quoted_hours"] = np.nan

    jobs_df = _add_runtime_metrics(df_filtered, jobs_df)
    jobs_df = _add_operational_metrics(jobs_df)
    jobs_df = _add_forecasts(df_filtered, jobs_df)

    if "median_runtime_days" in jobs_df.columns:
        jobs_df["peer_median_runtime_days"] = jobs_df["median_runtime_days"]

    jobs_df = _add_risk_metrics(jobs_df)

    jobs_df["margin_shortfall"] = (
        (jobs_df["median_margin_pct"].fillna(0) - jobs_df["forecast_margin_pct"].fillna(0))
        / 100
        * jobs_df["forecast_revenue"].fillna(0)
    ).clip(lower=0)

    jobs_df["hours_overrun"] = (
        jobs_df["actual_hours"]
        + jobs_df["remaining_hours"].fillna(0)
        - jobs_df["quoted_hours"]
    ).clip(lower=0)

    jobs_df["hours_variance_pct"] = np.where(
        jobs_df["quoted_hours"] > 0,
        (jobs_df["actual_hours"] - jobs_df["quoted_hours"]) / jobs_df["quoted_hours"] * 100,
        0,
    )

    jobs_df["pct_consumed"] = jobs_df.get("pct_quote_consumed", jobs_df.get("pct_consumed"))

    return jobs_df.sort_values("risk_score", ascending=False)


def compute_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Public wrapper to compute dept+category benchmarks."""
    return _compute_benchmarks(df)


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
        benchmarks = _compute_benchmarks(df)

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


def _compute_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
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

    if "is_billable" in df_completed.columns:
        billable = df_completed[df_completed["is_billable"] == True].groupby("job_no")["hours_raw"].sum()
        profit = profit.merge(billable.rename("billable_hours"), on="job_no", how="left")
        profit["billable_hours"] = profit["billable_hours"].fillna(0)
    else:
        profit["billable_hours"] = 0.0

    profit["margin_pct"] = np.where(
        profit["revenue"] > 0,
        (profit["revenue"] - profit["cost"]) / profit["revenue"] * 100,
        np.nan,
    )
    profit["billable_share"] = np.where(
        profit["hours"] > 0,
        profit["billable_hours"] / profit["hours"],
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
        median_billable_share=("billable_share", "median"),
        median_quoted_hours=("quoted_hours", "median"),
    ).reset_index()

    return bench


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

    jobs_df = jobs_df.merge(runtime[["job_no", "runtime_days"]], on="job_no", how="left")
    jobs_df["runtime_delta_days"] = jobs_df["runtime_days"] - jobs_df.get("median_runtime_days")

    return jobs_df


def _add_operational_metrics(jobs_df: pd.DataFrame) -> pd.DataFrame:
    jobs_df = jobs_df.copy()

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

    jobs_df["pct_consumed"] = jobs_df.get("pct_quote_consumed", jobs_df.get("pct_consumed"))

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
        remaining_hours = max((quoted_hours or 0) - actual_hours, 0) if pd.notna(quoted_hours) else 0

        eta_days = remaining_hours / burn_rate_per_day if burn_rate_per_day and burn_rate_per_day > 0 else np.nan
        eta_date = last_date + timedelta(days=float(eta_days)) if pd.notna(eta_days) else pd.NaT

        revenue_to_date = df_job["rev_alloc"].sum() if "rev_alloc" in df_job.columns else np.nan
        cost_to_date = df_job["base_cost"].sum() if "base_cost" in df_job.columns else np.nan
        avg_cost_per_hour = (cost_to_date / actual_hours) if actual_hours > 0 else np.nan
        forecast_cost = cost_to_date + (remaining_hours * avg_cost_per_hour) if pd.notna(avg_cost_per_hour) else np.nan
        forecast_revenue = quoted_amount if pd.notna(quoted_amount) and quoted_amount > 0 else revenue_to_date
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


def _add_risk_metrics(jobs_df: pd.DataFrame) -> pd.DataFrame:
    def _risk_row(row: pd.Series) -> pd.Series:
        score, reasons = compute_intervention_risk_score(row)

        if score >= 70:
            band = "Red"
        elif score >= 50:
            band = "Amber"
        else:
            band = "Green"

        primary = reasons[0] if reasons else "Monitor"
        action = _map_driver_to_action(primary)

        return pd.Series({
            "risk_score": score,
            "risk_band": band,
            "primary_driver": primary,
            "recommended_action": action,
        })

    risk_metrics = jobs_df.apply(_risk_row, axis=1)
    return pd.concat([jobs_df, risk_metrics], axis=1)


def _map_driver_to_action(driver: str) -> str:
    actions = {
        "Low margin %": "Review cost allocation; consider scope reduction",
        "Revenue lagging quote": "Accelerate invoicing; verify revenue recognition",
        "Hours overrun vs quote": "Scope review with PM; consider change order",
        "Realized rate below quote": "Audit rate capture; check write-offs",
        "Runtime exceeds peers": "Escalate timeline; identify blockers",
        "Monitor": "Continue standard monitoring",
    }
    return actions.get(driver, "Review job status")
