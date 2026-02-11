"""
Segment profiling for Executive Summary.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional

from src.metrics.quote_delivery import compute_hours_variance, compute_scope_creep
from src.metrics.rate_capture import compute_rate_metrics


SEGMENT_CONFIG = {
    "Subsidiser": {
        "color": "#28a745",
        "icon": "🟢",
        "short_desc": "Value creators",
        "explanation": "Under-run or on-plan hours with rate capture ≥100%",
    },
    "Mixed": {
        "color": "#ffc107",
        "icon": "🟡",
        "short_desc": "Cross-subsidised",
        "explanation": "Overrun but priced OK, OR under-run but rate leakage",
    },
    "Margin-Erosive": {
        "color": "#dc3545",
        "icon": "🔴",
        "short_desc": "Value destroyers",
        "explanation": "Overrun hours AND rate leakage — both execution and commercial failure",
    },
    "Protected Overrun": {
        "color": "#6f42c1",
        "icon": "🟣",
        "short_desc": "Hidden risk",
        "explanation": "Overrun but strong pricing masks delivery issues — watch for scope recovery erosion",
    },
}


@st.cache_data(show_spinner=False)
def assign_job_segment(row: pd.Series) -> str:
    """
    Assign a job to one of four segments based on execution + commercial profile.
    """
    hours_var_pct = row.get("hours_variance_pct", 0) or 0
    rate_capture = row.get("rate_capture_pct", 100) or 100
    scope_creep = row.get("scope_creep_pct", 0) or 0

    OVERRUN_THRESHOLD = 10
    RATE_CAPTURE_OK = 95
    RATE_CAPTURE_PREMIUM = 110

    is_overrun = hours_var_pct > OVERRUN_THRESHOLD
    is_rate_ok = rate_capture >= RATE_CAPTURE_OK
    is_rate_premium = rate_capture >= RATE_CAPTURE_PREMIUM

    if is_overrun and is_rate_premium:
        return "Protected Overrun"
    if is_overrun and not is_rate_ok:
        return "Margin-Erosive"
    if is_overrun and is_rate_ok:
        return "Mixed"
    if (not is_overrun) and (not is_rate_ok):
        return "Mixed"
    return "Subsidiser"


@st.cache_data(show_spinner=False)
def compute_job_segments(df: pd.DataFrame, job_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute segment assignments for all jobs.
    """
    result = job_df.copy()

    if "hours_variance_pct_job" not in result.columns:
        result["hours_variance_pct_job"] = np.where(
            result["quoted_hours_job"] > 0,
            (result["actual_hours_job"] - result["quoted_hours_job"])
            / result["quoted_hours_job"]
            * 100,
            0,
        )

    if "rate_capture_pct_job" not in result.columns:
        result["rate_capture_pct_job"] = np.where(
            result["quote_rate_job"] > 0,
            result["realised_rate_job"] / result["quote_rate_job"] * 100,
            100,
        )

    if "scope_creep_pct_job" not in result.columns:
        scope = compute_scope_creep(df, ("job_no",))
        if len(scope) > 0:
            result = result.merge(
                scope[["job_no", "unquoted_share"]].rename(
                    columns={"unquoted_share": "scope_creep_pct_job"}
                ),
                on="job_no",
                how="left",
            )
            result["scope_creep_pct_job"] = result["scope_creep_pct_job"].fillna(0)
        else:
            result["scope_creep_pct_job"] = 0

    if "duration_days" not in result.columns and "work_date" in df.columns:
        work_df = df[["job_no", "work_date"]].copy()
        work_df["work_date"] = pd.to_datetime(work_df["work_date"])
        durations = work_df.groupby("job_no").agg(
            first_activity=("work_date", "min"),
            last_activity=("work_date", "max"),
        )
        durations["duration_days"] = (
            durations["last_activity"] - durations["first_activity"]
        ).dt.days
        result = result.merge(
            durations[["duration_days"]].reset_index(),
            on="job_no",
            how="left",
        )

    result["segment"] = result.apply(
        lambda row: assign_job_segment(
            {
                "hours_variance_pct": row.get("hours_variance_pct_job", 0),
                "rate_capture_pct": row.get("rate_capture_pct_job", 100),
                "scope_creep_pct": row.get("scope_creep_pct_job", 0),
            }
        ),
        axis=1,
    )

    return result


@st.cache_data(show_spinner=False)
def compute_segment_profile(
    df: pd.DataFrame,
    job_df: pd.DataFrame,
    segment: str,
    benchmark_dept: Optional[str] = None,
    benchmark_category: Optional[str] = None,
) -> Optional[Dict]:
    """
    Compute comprehensive profile for a segment.
    """
    segment_jobs = job_df[job_df["segment"] == segment]["job_no"].tolist()
    segment_df = df[df["job_no"].isin(segment_jobs)].copy()

    if len(segment_df) == 0:
        return None

    profile = {
        "segment": segment,
        "config": SEGMENT_CONFIG.get(segment, {}),
        "composition": _compute_composition(job_df, segment),
        "drivers": _compute_driver_distributions(job_df, segment),
        "task_mix": _compute_task_mix_divergence(
            df, segment_jobs, benchmark_dept, benchmark_category
        ),
        "duration": _compute_duration_profile(df, segment_jobs),
        "overrun_decomposition": _compute_overrun_decomposition(job_df, segment),
    }

    return profile


def _compute_composition(job_df: pd.DataFrame, segment: str) -> Dict:
    """
    Compute segment composition: count, revenue, hours share.
    """
    total_jobs = len(job_df)
    total_revenue = job_df["revenue_job"].sum()
    total_hours = job_df["actual_hours_job"].sum()

    seg_df = job_df[job_df["segment"] == segment]

    return {
        "job_count": len(seg_df),
        "job_share": len(seg_df) / total_jobs * 100 if total_jobs > 0 else 0,
        "revenue": seg_df["revenue_job"].sum(),
        "revenue_share": seg_df["revenue_job"].sum() / total_revenue * 100
        if total_revenue > 0
        else 0,
        "hours": seg_df["actual_hours_job"].sum(),
        "hours_share": seg_df["actual_hours_job"].sum() / total_hours * 100
        if total_hours > 0
        else 0,
        "avg_margin_pct": seg_df["margin_pct_job"].median(),
        "by_department": seg_df.groupby("department_final").agg(
            count=("job_no", "count"),
            revenue=("revenue_job", "sum"),
        ).to_dict("index"),
        "by_category": seg_df.groupby("job_category").agg(
            count=("job_no", "count"),
            revenue=("revenue_job", "sum"),
        ).to_dict("index")
        if "job_category" in seg_df.columns
        else {},
    }


def _compute_driver_distributions(job_df: pd.DataFrame, segment: str) -> Dict:
    """
    Compute distribution stats for key drivers.
    """
    seg_df = job_df[job_df["segment"] == segment]

    def dist_stats(series: pd.Series) -> Dict:
        valid = series.dropna()
        if len(valid) == 0:
            return {"median": None, "p25": None, "p75": None, "mean": None}
        return {
            "median": valid.median(),
            "p25": valid.quantile(0.25),
            "p75": valid.quantile(0.75),
            "mean": valid.mean(),
        }

    return {
        "hours_variance_pct": dist_stats(seg_df["hours_variance_pct_job"]),
        "rate_variance": dist_stats(seg_df["rate_variance_job"]),
        "rate_capture_pct": dist_stats(seg_df["rate_capture_pct_job"]),
        "scope_creep_pct": dist_stats(
            seg_df.get("scope_creep_pct_job", pd.Series(dtype=float))
        ),
        "margin_pct": dist_stats(seg_df["margin_pct_job"]),
        "leakage_incidence": (seg_df["rate_variance_job"] < -10).sum()
        / len(seg_df)
        * 100
        if len(seg_df) > 0
        else 0,
    }


def _compute_task_mix_divergence(
    df: pd.DataFrame,
    segment_jobs: List[str],
    benchmark_dept: Optional[str] = None,
    benchmark_category: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute task mix divergence vs benchmark.
    """
    seg_df = df[df["job_no"].isin(segment_jobs)]
    seg_hours_total = seg_df["hours_raw"].sum()
    seg_task_mix = seg_df.groupby("task_name")["hours_raw"].sum()
    seg_task_pct = (seg_task_mix / seg_hours_total * 100).reset_index()
    seg_task_pct.columns = ["task_name", "segment_share"]

    bench_df = df.copy()
    if benchmark_dept:
        bench_df = bench_df[bench_df["department_final"] == benchmark_dept]
    if benchmark_category:
        bench_df = bench_df[bench_df["job_category"] == benchmark_category]

    bench_df = bench_df[~bench_df["job_no"].isin(segment_jobs)]
    if len(bench_df) == 0:
        bench_df = df[~df["job_no"].isin(segment_jobs)]

    bench_hours_total = bench_df["hours_raw"].sum()
    bench_task_mix = bench_df.groupby("task_name")["hours_raw"].sum()
    bench_task_pct = (bench_task_mix / bench_hours_total * 100).reset_index()
    bench_task_pct.columns = ["task_name", "benchmark_share"]

    result = seg_task_pct.merge(bench_task_pct, on="task_name", how="outer").fillna(0)
    result["divergence_pp"] = result["segment_share"] - result["benchmark_share"]
    result["abs_divergence"] = result["divergence_pp"].abs()

    return result.sort_values("abs_divergence", ascending=False).drop(columns=["abs_divergence"])


def _compute_duration_profile(df: pd.DataFrame, segment_jobs: List[str]) -> Dict:
    """
    Compute duration profile for segment jobs.
    """
    seg_df = df[df["job_no"].isin(segment_jobs)]
    if "work_date" not in seg_df.columns:
        return {"median_days": None, "p75_days": None, "long_tail_pct": None}

    seg_df = seg_df.copy()
    seg_df["work_date"] = pd.to_datetime(seg_df["work_date"])

    job_durations = seg_df.groupby("job_no").agg(
        first_activity=("work_date", "min"),
        last_activity=("work_date", "max"),
    )
    job_durations["duration_days"] = (
        job_durations["last_activity"] - job_durations["first_activity"]
    ).dt.days

    durations = job_durations["duration_days"]

    return {
        "median_days": durations.median(),
        "p75_days": durations.quantile(0.75),
        "mean_days": durations.mean(),
        "long_tail_pct": (durations > durations.quantile(0.75)).sum()
        / len(durations)
        * 100
        if len(durations) > 0
        else 0,
    }


def _compute_overrun_decomposition(job_df: pd.DataFrame, segment: str) -> Dict:
    """
    Decompose overrun into in-scope vs scope creep.
    """
    seg_df = job_df[job_df["segment"] == segment]

    if len(seg_df) == 0:
        return {"inscope_overrun_share": 0, "scope_creep_share": 0}

    total_variance = seg_df["hours_variance_job"].sum()
    scope_creep_hours = (
        seg_df["scope_creep_pct_job"] / 100 * seg_df["actual_hours_job"]
    ).sum()

    if total_variance <= 0:
        return {"inscope_overrun_share": 0, "scope_creep_share": 0}

    inscope_overrun = max(0, total_variance - scope_creep_hours)

    return {
        "total_variance_hours": total_variance,
        "inscope_overrun_hours": inscope_overrun,
        "scope_creep_hours": scope_creep_hours,
        "inscope_overrun_share": inscope_overrun / total_variance * 100
        if total_variance > 0
        else 0,
        "scope_creep_share": scope_creep_hours / total_variance * 100
        if total_variance > 0
        else 0,
    }


@st.cache_data(show_spinner=False)
def generate_reason_codes(row: pd.Series, benchmark_duration_p75: float = 30) -> List[str]:
    """
    Generate human-readable reason codes for why a job is in its segment.
    """
    reasons = []

    hours_var = row.get("hours_variance_pct_job", 0)
    rate_var = row.get("rate_variance_job", 0)
    scope_creep = row.get("scope_creep_pct_job", 0)
    duration = row.get("duration_days", 0) or 0

    if hours_var > 20:
        reasons.append(f"Severe overrun (+{hours_var:.0f}% hours)")
    elif hours_var > 10:
        reasons.append(f"Moderate overrun (+{hours_var:.0f}% hours)")
    elif hours_var < -10:
        reasons.append(f"Efficient delivery ({hours_var:.0f}% under)")

    if scope_creep > 30:
        reasons.append(f"High scope creep ({scope_creep:.0f}% unquoted)")
    elif scope_creep > 15:
        reasons.append(f"Moderate scope creep ({scope_creep:.0f}% unquoted)")

    if rate_var < -20:
        reasons.append(f"Severe rate leakage (${rate_var:+.0f}/hr)")
    elif rate_var < -10:
        reasons.append(f"Rate leakage (${rate_var:+.0f}/hr)")
    elif rate_var > 20:
        reasons.append(f"Strong premium captured (${rate_var:+.0f}/hr)")

    if duration > benchmark_duration_p75:
        reasons.append(f"Long-running ({duration:.0f} days)")

    return reasons[:3]


@st.cache_data(show_spinner=False)
def compute_job_shortlist(
    job_df: pd.DataFrame,
    segment: str,
    n: int = 10,
    sort_by: str = "margin_at_risk",
) -> pd.DataFrame:
    """
    Get top jobs by impact within a segment.
    """
    seg_df = job_df[job_df["segment"] == segment].copy()
    if len(seg_df) == 0:
        return pd.DataFrame()

    avg_cost_rate = (
        seg_df["cost_job"].sum() / seg_df["actual_hours_job"].sum()
        if seg_df["actual_hours_job"].sum() > 0
        else 0
    )
    seg_df["margin_at_risk"] = (
        seg_df["hours_variance_job"].abs() * avg_cost_rate
        + (seg_df["rate_variance_job"].abs() * seg_df["actual_hours_job"])
    )

    sort_col = {
        "margin_at_risk": "margin_at_risk",
        "hours_variance": "hours_variance_job",
        "leakage": "rate_variance_job",
        "revenue": "revenue_job",
    }.get(sort_by, "margin_at_risk")

    ascending = sort_by == "leakage"

    shortlist = seg_df.nsmallest(n, sort_col) if ascending else seg_df.nlargest(n, sort_col)

    benchmark_p75 = (
        seg_df["duration_days"].quantile(0.75)
        if "duration_days" in seg_df.columns and len(seg_df) > 0
        else 30
    )
    shortlist["reason_codes"] = shortlist.apply(
        lambda row: generate_reason_codes(row, benchmark_duration_p75=benchmark_p75),
        axis=1,
    )

    return shortlist
