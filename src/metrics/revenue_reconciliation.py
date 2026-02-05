"""
Revenue Reconciliation & Leakage Analysis.

Reconciles quoted amounts vs actual revenue at job level,
identifies leakage patterns, and diagnoses root causes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.semantic import safe_quote_rollup, safe_quote_job_task, get_category_col


# =============================================================================
# STATUS DEFINITIONS
# =============================================================================

REVENUE_STATUS_CONFIG = {
    "Over-Recovery": {
        "color": "#28a745",
        "icon": "🟢",
        "threshold": 10,  # >10% above quote
        "description": "Collected more than quoted (scope charges, rate uplifts)",
    },
    "On-Target": {
        "color": "#6c757d",
        "icon": "⚪",
        "threshold_low": -5,
        "threshold_high": 10,
        "description": "Within -5% to +10% of quote",
    },
    "Minor Leakage": {
        "color": "#ffc107",
        "icon": "🟡",
        "threshold_low": -20,
        "threshold_high": -5,
        "description": "5-20% below quote (small write-offs, discounts)",
    },
    "Major Leakage": {
        "color": "#dc3545",
        "icon": "🔴",
        "threshold": -20,  # >20% below quote
        "description": ">20% below quote (significant write-offs)",
    },
    "No Quote": {
        "color": "#17a2b8",
        "icon": "🔵",
        "description": "Revenue exists but no quote to compare",
    },
    "No Revenue": {
        "color": "#6c757d",
        "icon": "⚫",
        "description": "Quote exists but no revenue recorded",
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ReconciliationSummary:
    """Portfolio-level reconciliation summary."""

    total_jobs: int
    jobs_with_quotes: int
    jobs_with_revenue: int
    jobs_matched: int  # Have both quote and revenue

    total_quoted: float
    total_revenue: float
    total_delta: float
    aggregate_capture_pct: float

    # Status counts
    jobs_over_recovery: int
    jobs_on_target: int
    jobs_minor_leakage: int
    jobs_major_leakage: int
    jobs_no_quote: int
    jobs_no_revenue: int

    # Dollar impacts
    delta_over_recovery: float
    delta_minor_leakage: float
    delta_major_leakage: float

    # Concentration
    jobs_for_80pct_leakage: int
    top_5_leakage_total: float


@dataclass
class JobDiagnosis:
    """Diagnosis for a single job with revenue leakage."""

    job_no: str
    job_name: Optional[str]
    department: Optional[str]
    category: Optional[str]
    client: Optional[str]

    # Amounts
    quoted_hours: float
    quoted_amount: float
    actual_hours: float
    actual_revenue: float
    actual_cost: float

    # Rates
    quote_rate: float
    realised_rate: float
    rate_delta: float

    # Variance
    hours_variance: float
    hours_variance_pct: float
    revenue_delta: float
    revenue_delta_pct: float

    # Decomposition
    gap_from_hours: float  # Hours effect at quote rate
    gap_from_rate: float  # Rate effect on actual hours

    # Hypotheses
    hypotheses: List[str]

    # Task breakdown
    task_breakdown: Optional[pd.DataFrame]


# =============================================================================
# CORE RECONCILIATION
# =============================================================================


def compute_job_reconciliation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconcile quoted amount vs actual revenue at job level.

    Returns DataFrame with one row per job containing:
    - job_no
    - quoted_hours, quoted_amount (from safe rollup)
    - actual_hours, actual_revenue, actual_cost
    - revenue_delta, revenue_delta_pct
    - revenue_status
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Safe quote totals per job (deduped at job-task level)
    quote_by_job = safe_quote_rollup(df, ["job_no"])

    # Actuals per job
    agg_dict = {
        "actual_hours": ("hours_raw", "sum"),
        "actual_revenue": ("rev_alloc", "sum"),
        "actual_cost": ("base_cost", "sum"),
    }
    if "department_final" in df.columns:
        agg_dict["department"] = ("department_final", "first")
    category_col = get_category_col(df)
    if category_col in df.columns:
        agg_dict["category"] = (category_col, "first")
    if "client" in df.columns:
        agg_dict["client"] = ("client", "first")
    if "job_name" in df.columns:
        agg_dict["job_name"] = ("job_name", "first")

    actuals_by_job = df.groupby("job_no").agg(**agg_dict).reset_index()

    # Merge
    if quote_by_job is None or len(quote_by_job) == 0 or "job_no" not in quote_by_job.columns:
        result = actuals_by_job.copy()
        result["quoted_hours"] = 0.0
        result["quoted_amount"] = 0.0
    else:
        result = quote_by_job.merge(actuals_by_job, on="job_no", how="outer")

    # Ensure optional fields exist
    for col in ["department", "category", "client", "job_name"]:
        if col not in result.columns:
            result[col] = None

    # Fill nulls
    result["quoted_hours"] = result["quoted_hours"].fillna(0)
    result["quoted_amount"] = result["quoted_amount"].fillna(0)
    for col in ["actual_hours", "actual_revenue", "actual_cost"]:
        if col not in result.columns:
            result[col] = 0
        result[col] = result[col].fillna(0)

    # Compute rates
    result["quote_rate"] = np.where(
        result["quoted_hours"] > 0,
        result["quoted_amount"] / result["quoted_hours"],
        np.nan,
    )
    result["realised_rate"] = np.where(
        result["actual_hours"] > 0,
        result["actual_revenue"] / result["actual_hours"],
        np.nan,
    )
    result["rate_delta"] = result["realised_rate"] - result["quote_rate"]

    # Compute deltas
    result["revenue_delta"] = result["actual_revenue"] - result["quoted_amount"]
    result["revenue_delta_pct"] = np.where(
        result["quoted_amount"] > 0,
        result["revenue_delta"] / result["quoted_amount"] * 100,
        np.nan,
    )

    # Hours variance
    result["hours_variance"] = result["actual_hours"] - result["quoted_hours"]
    result["hours_variance_pct"] = np.where(
        result["quoted_hours"] > 0,
        result["hours_variance"] / result["quoted_hours"] * 100,
        np.nan,
    )

    # Assign status
    result["revenue_status"] = _assign_revenue_status(result)

    return result


def _assign_revenue_status(df: pd.DataFrame) -> pd.Series:
    """Assign revenue status based on delta percentage."""
    conditions = [
        (df["quoted_amount"] == 0) & (df["actual_revenue"] > 0),  # No quote
        (df["actual_revenue"] == 0) & (df["quoted_amount"] > 0),  # No revenue
        df["revenue_delta_pct"] >= 10,  # Over-recovery
        df["revenue_delta_pct"] >= -5,  # On-target
        df["revenue_delta_pct"] >= -20,  # Minor leakage
        df["revenue_delta_pct"] < -20,  # Major leakage
    ]

    choices = [
        "No Quote",
        "No Revenue",
        "Over-Recovery",
        "On-Target",
        "Minor Leakage",
        "Major Leakage",
    ]

    return np.select(conditions, choices, default="Unknown")


def compute_reconciliation_summary(recon_df: pd.DataFrame) -> ReconciliationSummary:
    """
    Compute portfolio-level reconciliation summary.
    """
    total_jobs = len(recon_df)
    jobs_with_quotes = len(recon_df[recon_df["quoted_amount"] > 0])
    jobs_with_revenue = len(recon_df[recon_df["actual_revenue"] > 0])
    jobs_matched = len(
        recon_df[(recon_df["quoted_amount"] > 0) & (recon_df["actual_revenue"] > 0)]
    )

    # Only consider jobs with quotes for capture calculation
    quoted_jobs = recon_df[recon_df["quoted_amount"] > 0]

    total_quoted = quoted_jobs["quoted_amount"].sum()
    total_revenue = quoted_jobs["actual_revenue"].sum()
    total_delta = total_revenue - total_quoted
    aggregate_capture_pct = (total_revenue / total_quoted * 100) if total_quoted > 0 else 0

    # Status counts
    status_counts = recon_df["revenue_status"].value_counts()
    jobs_over_recovery = int(status_counts.get("Over-Recovery", 0))
    jobs_on_target = int(status_counts.get("On-Target", 0))
    jobs_minor_leakage = int(status_counts.get("Minor Leakage", 0))
    jobs_major_leakage = int(status_counts.get("Major Leakage", 0))
    jobs_no_quote = int(status_counts.get("No Quote", 0))
    jobs_no_revenue = int(status_counts.get("No Revenue", 0))

    # Dollar impacts by status
    delta_over_recovery = recon_df[recon_df["revenue_status"] == "Over-Recovery"][
        "revenue_delta"
    ].sum()
    delta_minor_leakage = recon_df[recon_df["revenue_status"] == "Minor Leakage"][
        "revenue_delta"
    ].sum()
    delta_major_leakage = recon_df[recon_df["revenue_status"] == "Major Leakage"][
        "revenue_delta"
    ].sum()

    # Concentration analysis
    leakage_jobs = recon_df[recon_df["revenue_delta"] < 0].sort_values("revenue_delta")
    total_leakage = leakage_jobs["revenue_delta"].sum()

    if total_leakage < 0 and len(leakage_jobs) > 0:
        leakage_jobs["cumulative"] = leakage_jobs["revenue_delta"].cumsum()
        threshold_80 = total_leakage * 0.8
        cumulative = leakage_jobs["cumulative"].values
        jobs_for_80pct = int(np.argmax(cumulative <= threshold_80) + 1)
    else:
        jobs_for_80pct = 0

    top_5_leakage = recon_df.nsmallest(5, "revenue_delta")["revenue_delta"].sum()

    return ReconciliationSummary(
        total_jobs=total_jobs,
        jobs_with_quotes=jobs_with_quotes,
        jobs_with_revenue=jobs_with_revenue,
        jobs_matched=jobs_matched,
        total_quoted=total_quoted,
        total_revenue=total_revenue,
        total_delta=total_delta,
        aggregate_capture_pct=aggregate_capture_pct,
        jobs_over_recovery=jobs_over_recovery,
        jobs_on_target=jobs_on_target,
        jobs_minor_leakage=jobs_minor_leakage,
        jobs_major_leakage=jobs_major_leakage,
        jobs_no_quote=jobs_no_quote,
        jobs_no_revenue=jobs_no_revenue,
        delta_over_recovery=delta_over_recovery,
        delta_minor_leakage=delta_minor_leakage,
        delta_major_leakage=delta_major_leakage,
        jobs_for_80pct_leakage=jobs_for_80pct,
        top_5_leakage_total=top_5_leakage,
    )


# =============================================================================
# PATTERN ANALYSIS
# =============================================================================


def analyze_patterns_by_dimension(recon_df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """
    Analyze reconciliation patterns by a grouping dimension.

    Args:
        dimension: 'department', 'category', 'client', or 'size_bucket'
    """
    df = recon_df[recon_df["quoted_amount"] > 0].copy()

    if dimension == "size_bucket":
        df["size_bucket"] = pd.cut(
            df["quoted_amount"],
            bins=[0, 5000, 20000, 50000, 100000, float("inf")],
            labels=["<$5K", "$5-20K", "$20-50K", "$50-100K", ">$100K"],
        )
        group_col = "size_bucket"
    else:
        group_col = dimension

    if group_col not in df.columns:
        return pd.DataFrame()

    result = df.groupby(group_col).agg(
        job_count=("job_no", "count"),
        total_quoted=("quoted_amount", "sum"),
        total_revenue=("actual_revenue", "sum"),
        total_delta=("revenue_delta", "sum"),
        total_hours=("actual_hours", "sum"),
        leakage_jobs=(
            "revenue_status",
            lambda x: x.isin(["Minor Leakage", "Major Leakage"]).sum(),
        ),
        recovery_jobs=("revenue_status", lambda x: (x == "Over-Recovery").sum()),
    ).reset_index()

    result["capture_pct"] = np.where(
        result["total_quoted"] > 0,
        result["total_revenue"] / result["total_quoted"] * 100,
        0,
    )
    result["leakage_rate"] = np.where(
        result["job_count"] > 0,
        result["leakage_jobs"] / result["job_count"] * 100,
        0,
    )
    result["avg_delta_per_job"] = result["total_delta"] / result["job_count"]

    return result.sort_values("total_delta", ascending=True)


def compute_concentration_curve(recon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative leakage curve for concentration analysis.

    Returns DataFrame with job_rank, revenue_delta, cumulative_leakage, cumulative_pct
    """
    leakage_jobs = recon_df[recon_df["revenue_delta"] < 0].sort_values(
        "revenue_delta"
    ).copy()

    if len(leakage_jobs) == 0:
        return pd.DataFrame()

    total_leakage = leakage_jobs["revenue_delta"].sum()

    leakage_jobs["job_rank"] = range(1, len(leakage_jobs) + 1)
    leakage_jobs["cumulative_leakage"] = leakage_jobs["revenue_delta"].cumsum()
    leakage_jobs["cumulative_pct"] = leakage_jobs["cumulative_leakage"] / total_leakage * 100

    return leakage_jobs[
        ["job_no", "job_rank", "revenue_delta", "cumulative_leakage", "cumulative_pct"]
    ]


# =============================================================================
# JOB DIAGNOSIS
# =============================================================================


def diagnose_job(
    df: pd.DataFrame, recon_df: pd.DataFrame, job_no: str
) -> Optional[JobDiagnosis]:
    """
    Deep dive diagnosis on a specific job.
    """
    # Get job from reconciliation
    job_recon = recon_df[recon_df["job_no"] == job_no]
    if len(job_recon) == 0:
        return None

    job_row = job_recon.iloc[0]
    job_df = df[df["job_no"] == job_no]

    # Basic facts
    quoted_hours = float(job_row["quoted_hours"])
    quoted_amount = float(job_row["quoted_amount"])
    actual_hours = float(job_row["actual_hours"])
    actual_revenue = float(job_row["actual_revenue"])
    actual_cost = float(job_row["actual_cost"])

    # Rates
    quote_rate = float(job_row["quote_rate"]) if pd.notna(job_row["quote_rate"]) else 0
    realised_rate = (
        float(job_row["realised_rate"]) if pd.notna(job_row["realised_rate"]) else 0
    )
    rate_delta = realised_rate - quote_rate

    # Variance
    hours_variance = actual_hours - quoted_hours
    hours_variance_pct = (
        float(job_row["hours_variance_pct"])
        if pd.notna(job_row["hours_variance_pct"])
        else 0
    )
    revenue_delta = float(job_row["revenue_delta"])
    revenue_delta_pct = (
        float(job_row["revenue_delta_pct"])
        if pd.notna(job_row["revenue_delta_pct"])
        else 0
    )

    # Gap decomposition
    # Total gap = (Actual Hours × Realised Rate) - (Quoted Hours × Quote Rate)
    #           = Hours Effect + Rate Effect
    # Hours Effect = (Actual - Quoted) × Quote Rate
    # Rate Effect = (Realised - Quote) × Actual Hours
    gap_from_hours = hours_variance * quote_rate if quote_rate > 0 else 0
    gap_from_rate = rate_delta * actual_hours if actual_hours > 0 else 0

    # Generate hypotheses
    hypotheses = _generate_hypotheses(
        quoted_hours=quoted_hours,
        quoted_amount=quoted_amount,
        actual_hours=actual_hours,
        actual_revenue=actual_revenue,
        hours_variance_pct=hours_variance_pct,
        rate_delta=rate_delta,
        revenue_delta=revenue_delta,
        revenue_delta_pct=revenue_delta_pct,
    )

    # Task breakdown
    task_breakdown = _get_task_breakdown(df, job_no)

    return JobDiagnosis(
        job_no=job_no,
        job_name=job_row.get("job_name"),
        department=job_row.get("department"),
        category=job_row.get("category"),
        client=job_row.get("client"),
        quoted_hours=quoted_hours,
        quoted_amount=quoted_amount,
        actual_hours=actual_hours,
        actual_revenue=actual_revenue,
        actual_cost=actual_cost,
        quote_rate=quote_rate,
        realised_rate=realised_rate,
        rate_delta=rate_delta,
        hours_variance=hours_variance,
        hours_variance_pct=hours_variance_pct,
        revenue_delta=revenue_delta,
        revenue_delta_pct=revenue_delta_pct,
        gap_from_hours=gap_from_hours,
        gap_from_rate=gap_from_rate,
        hypotheses=hypotheses,
        task_breakdown=task_breakdown,
    )


def _generate_hypotheses(
    quoted_hours: float,
    quoted_amount: float,
    actual_hours: float,
    actual_revenue: float,
    hours_variance_pct: float,
    rate_delta: float,
    revenue_delta: float,
    revenue_delta_pct: float,
) -> List[str]:
    """Generate likely causes for revenue leakage."""
    hypotheses: List[str] = []

    # Zero revenue cases
    if actual_revenue == 0 and actual_hours > 0:
        hypotheses.append(
            "🔴 Zero revenue despite hours worked — likely unbilled WIP or complete write-off"
        )
        return hypotheses

    if actual_revenue == 0 and quoted_amount > 0:
        hypotheses.append(
            "🔴 Quoted but never invoiced — job may have been cancelled or written off entirely"
        )
        return hypotheses

    # Partial revenue cases
    if revenue_delta_pct < -50:
        hypotheses.append(
            f"🔴 Less than half of quote collected ({100 + revenue_delta_pct:.0f}%) — significant scope reduction or write-off"
        )

    # Hours-driven leakage
    if hours_variance_pct > 30 and rate_delta < 5:
        hypotheses.append(
            f"⚠️ Overran by {hours_variance_pct:.0f}% but rate held — extra work likely not invoiced (change order gap)"
        )

    if hours_variance_pct > 20 and rate_delta < -10:
        hypotheses.append(
            f"⚠️ Overran by {hours_variance_pct:.0f}% AND rate dropped ${abs(rate_delta):.0f}/hr — double leakage (execution + commercial)"
        )

    # Rate-driven leakage
    if rate_delta < -20 and abs(hours_variance_pct) < 10:
        hypotheses.append(
            f"⚠️ Hours on track but rate ${rate_delta:.0f}/hr below quote — likely discount or partial write-off"
        )

    if rate_delta < -10 and actual_revenue < quoted_amount:
        hypotheses.append(
            f"💰 Realised rate ${abs(rate_delta):.0f}/hr below quote — check for: discounts, write-offs, or revenue recognition timing"
        )

    # Positive cases (for context)
    if revenue_delta > 0 and hours_variance_pct > 10:
        hypotheses.append(
            f"✅ Overran by {hours_variance_pct:.0f}% but still collected more than quote — scope charges working"
        )

    if revenue_delta > 0 and rate_delta > 10:
        hypotheses.append(
            f"✅ Rate ${rate_delta:.0f}/hr above quote — premium captured or rate uplift"
        )

    # Data quality flags
    if quoted_hours == 0 and quoted_amount > 0:
        hypotheses.append("⚠️ Quote has $ but no hours — may be fixed-fee or data issue")

    if quoted_amount == 0 and actual_revenue > 0:
        hypotheses.append("ℹ️ Revenue without quote — T&M work, or quote not in system")

    return hypotheses[:4]


def _get_task_breakdown(df: pd.DataFrame, job_no: str) -> Optional[pd.DataFrame]:
    """Get task-level breakdown for a job."""
    job_df = df[df["job_no"] == job_no]

    if len(job_df) == 0:
        return None

    # Quote by task
    job_task = safe_quote_job_task(job_df)

    # Actuals by task
    task_actuals = job_df.groupby("task_name").agg(
        actual_hours=("hours_raw", "sum"),
        actual_revenue=("rev_alloc", "sum"),
        actual_cost=("base_cost", "sum"),
    ).reset_index()

    # Merge
    if len(job_task) > 0 and "quoted_time_total" in job_task.columns:
        result = job_task[["task_name", "quoted_time_total", "quoted_amount_total"]].copy()
        result.columns = ["task_name", "quoted_hours", "quoted_amount"]
        result = result.merge(task_actuals, on="task_name", how="outer")
    else:
        result = task_actuals.copy()
        result["quoted_hours"] = 0
        result["quoted_amount"] = 0

    result = result.fillna(0)

    # Compute variance
    result["hours_variance"] = result["actual_hours"] - result["quoted_hours"]
    result["revenue_variance"] = result["actual_revenue"] - result["quoted_amount"]

    return result.sort_values("revenue_variance", ascending=True)


# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================


def check_data_quality(recon_df: pd.DataFrame) -> Dict[str, float]:
    """
    Check data quality issues in quote-revenue matching.
    """
    return {
        "jobs_with_quote_no_revenue": len(
            recon_df[(recon_df["quoted_amount"] > 0) & (recon_df["actual_revenue"] == 0)]
        ),
        "jobs_with_revenue_no_quote": len(
            recon_df[(recon_df["actual_revenue"] > 0) & (recon_df["quoted_amount"] == 0)]
        ),
        "jobs_matched": len(
            recon_df[(recon_df["quoted_amount"] > 0) & (recon_df["actual_revenue"] > 0)]
        ),
        "match_rate": len(
            recon_df[(recon_df["quoted_amount"] > 0) & (recon_df["actual_revenue"] > 0)]
        )
        / len(recon_df)
        * 100
        if len(recon_df) > 0
        else 0,
        "revenue_without_quote_total": recon_df[
            (recon_df["actual_revenue"] > 0) & (recon_df["quoted_amount"] == 0)
        ]["actual_revenue"].sum(),
        "quote_without_revenue_total": recon_df[
            (recon_df["quoted_amount"] > 0) & (recon_df["actual_revenue"] == 0)
        ]["quoted_amount"].sum(),
    }
