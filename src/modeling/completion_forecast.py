"""Computation helpers for job completion timeline and forecast analysis."""
from __future__ import annotations

from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.data.semantic import get_category_col, safe_quote_job_task


_LIFECYCLE_COLUMNS = [
    "task_name",
    "decile_bucket",
    "median_hours_share",
    "p25_hours_share",
    "p75_hours_share",
    "peer_job_count",
]

_TASK_SUMMARY_COLUMNS = [
    "task_name",
    "median_total_share",
    "p25_total_share",
    "p75_total_share",
    "peer_median_total_hours",
    "peer_job_count",
]

_TIMELINE_COLUMNS = [
    "period_start",
    "period_label",
    "task_name",
    "hours",
    "cost",
    "cumulative_hours",
    "cumulative_cost",
]

_REMAINING_COLUMNS = [
    "task_name",
    "actual_hours",
    "expected_hours_p25",
    "expected_hours_median",
    "expected_hours_p75",
    "remaining_hours_p25",
    "remaining_hours_median",
    "remaining_hours_p75",
    "outlier_flag",
    "unmodeled_flag",
    "peer_job_count",
]


def _empty_lifecycle_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_LIFECYCLE_COLUMNS)


def _empty_task_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_TASK_SUMMARY_COLUMNS)


def _empty_timeline_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_TIMELINE_COLUMNS)


def _empty_remaining_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_REMAINING_COLUMNS)


def _completed_mask(df: pd.DataFrame) -> pd.Series:
    """Return completed-job mask using the existing project pattern."""
    if "job_completed_date" in df.columns:
        return df["job_completed_date"].notna()
    if "job_status" in df.columns:
        return df["job_status"].astype(str).str.lower().str.contains("completed", na=False)
    return pd.Series(False, index=df.index)


def _resolve_date_col(df: pd.DataFrame) -> str | None:
    if "work_date" in df.columns and pd.to_datetime(df["work_date"], errors="coerce").notna().any():
        return "work_date"
    if "month_key" in df.columns and pd.to_datetime(df["month_key"], errors="coerce").notna().any():
        return "month_key"
    return None


def _peer_scope(
    df_all: pd.DataFrame,
    department: str,
    category: str,
    exclude_job_no: str,
) -> pd.DataFrame:
    if len(df_all) == 0 or "job_no" not in df_all.columns:
        return df_all.iloc[0:0].copy()

    cat_col = get_category_col(df_all)
    completed = _completed_mask(df_all)
    peers = df_all[
        completed
        & (df_all["department_final"] == department)
        & (df_all[cat_col] == category)
    ].copy()
    if len(peers) == 0:
        return peers

    peers = peers[peers["job_no"].astype(str) != str(exclude_job_no)].copy()
    if len(peers) == 0:
        return peers

    peers["hours_raw"] = pd.to_numeric(peers.get("hours_raw"), errors="coerce").fillna(0.0)
    peers["task_name"] = peers.get("task_name", "Unspecified").fillna("Unspecified").astype(str)

    date_col = _resolve_date_col(peers)
    if date_col is None:
        return peers.iloc[0:0].copy()

    peers[date_col] = pd.to_datetime(peers[date_col], errors="coerce")
    peers = peers[peers[date_col].notna()].copy()
    peers = peers[peers["hours_raw"] > 0].copy()
    peers["_date"] = peers[date_col]
    return peers


def _compute_deciles(df_job: pd.DataFrame, date_col: str = "_date") -> pd.DataFrame:
    job = df_job.copy()
    start_date = pd.to_datetime(job[date_col], errors="coerce").min()
    end_date = pd.to_datetime(job[date_col], errors="coerce").max()
    if pd.isna(start_date) or pd.isna(end_date):
        return job.iloc[0:0].copy()

    duration = max((end_date - start_date).days + 1, 1)
    rel = (job[date_col] - start_date).dt.days / duration
    rel = rel.clip(lower=0, upper=1)
    decile = np.floor(rel * 10).astype(int).clip(0, 9)

    job["decile_bucket"] = decile
    return job


def _peer_job_totals(
    df_all: pd.DataFrame,
    department: str,
    category: str,
    exclude_job_no: str,
) -> pd.Series:
    peers = _peer_scope(df_all, department, category, exclude_job_no)
    if len(peers) == 0:
        return pd.Series(dtype=float)
    return peers.groupby("job_no")["hours_raw"].sum()


def _peer_decile_curve(
    df_all: pd.DataFrame,
    department: str,
    category: str,
    exclude_job_no: str,
) -> pd.DataFrame:
    """Return peer decile hour-share curve (share of total job hours per decile)."""
    peers = _peer_scope(df_all, department, category, exclude_job_no)
    if len(peers) == 0:
        return pd.DataFrame(columns=["decile_bucket", "median_share", "p25_share", "p75_share", "peer_job_count"])

    rows = []
    for job_no, job_df in peers.groupby("job_no"):
        deciled = _compute_deciles(job_df, date_col="_date")
        if len(deciled) == 0:
            continue
        total_hours = float(deciled["hours_raw"].sum())
        if total_hours <= 0:
            continue
        decile_share = (
            deciled.groupby("decile_bucket")["hours_raw"].sum().reindex(range(10), fill_value=0.0) / total_hours
        )
        for decile_bucket, share in decile_share.items():
            rows.append(
                {
                    "job_no": str(job_no),
                    "decile_bucket": int(decile_bucket),
                    "share": float(share),
                }
            )

    if len(rows) == 0:
        return pd.DataFrame(columns=["decile_bucket", "median_share", "p25_share", "p75_share", "peer_job_count"])

    curve = (
        pd.DataFrame(rows)
        .groupby("decile_bucket")
        .agg(
            median_share=("share", "median"),
            p25_share=("share", lambda s: s.quantile(0.25)),
            p75_share=("share", lambda s: s.quantile(0.75)),
            peer_job_count=("job_no", "nunique"),
        )
        .reindex(range(10), fill_value=0.0)
        .reset_index()
    )
    total_median = float(curve["median_share"].sum())
    if total_median > 0:
        curve["median_share"] = curve["median_share"] / total_median
    return curve


def _runtime_progress_ratio(job_df: pd.DataFrame, peers_df: pd.DataFrame) -> float:
    """Estimate runtime progress vs peer median runtime."""
    if len(job_df) == 0 or len(peers_df) == 0:
        return np.nan

    date_col_job = _resolve_date_col(job_df)
    if date_col_job is None:
        return np.nan

    job_dates = pd.to_datetime(job_df[date_col_job], errors="coerce")
    if job_dates.notna().sum() == 0:
        return np.nan
    job_runtime_days = max((job_dates.max() - job_dates.min()).days + 1, 1)

    peer_runtime = peers_df.groupby("job_no")["_date"].agg(["min", "max"])
    peer_runtime_days = (peer_runtime["max"] - peer_runtime["min"]).dt.days + 1
    peer_median_runtime_days = pd.to_numeric(peer_runtime_days, errors="coerce").median()
    if pd.isna(peer_median_runtime_days) or peer_median_runtime_days <= 0:
        return np.nan

    return float(job_runtime_days / float(peer_median_runtime_days))


def _cumulative_share_at_progress(decile_curve: pd.DataFrame, progress_ratio: float) -> float:
    """Interpolate expected cumulative-hour share at a progress ratio in [0, 1]."""
    if len(decile_curve) == 0 or pd.isna(progress_ratio):
        return np.nan

    p = float(np.clip(progress_ratio, 0.0, 1.0))
    bucket_pos = p * 10.0
    full_buckets = int(np.floor(bucket_pos))
    frac = bucket_pos - full_buckets

    shares = (
        decile_curve.set_index("decile_bucket")
        .reindex(range(10), fill_value=0.0)["median_share"]
        .astype(float)
        .values
    )

    cumulative = float(shares[:full_buckets].sum()) if full_buckets > 0 else 0.0
    if full_buckets < 10:
        cumulative += float(shares[full_buckets] * frac)
    return float(np.clip(cumulative, 0.0, 1.0))


def _safe_quote_totals_for_job(df_job: pd.DataFrame) -> Tuple[float, float]:
    """Return deduped (quoted_hours, quoted_amount) from a job-level slice."""
    if len(df_job) == 0:
        return np.nan, np.nan

    job_task = safe_quote_job_task(df_job)
    if len(job_task) == 0:
        return np.nan, np.nan

    quoted_hours = (
        pd.to_numeric(job_task.get("quoted_time_total"), errors="coerce").fillna(0).sum()
        if "quoted_time_total" in job_task.columns
        else np.nan
    )
    quoted_amount = (
        pd.to_numeric(job_task.get("quoted_amount_total"), errors="coerce").fillna(0).sum()
        if "quoted_amount_total" in job_task.columns
        else np.nan
    )
    return float(quoted_hours), float(quoted_amount)


@st.cache_data(show_spinner=False)
def build_peer_lifecycle_profiles(
    df_all: pd.DataFrame,
    department: str,
    category: str,
    exclude_job_no: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build lifecycle task-share profiles from completed peer jobs.

    Algorithm:
    1. Keep completed peers in the same department/category, excluding the selected job.
    2. For each peer job, normalize each activity row to lifecycle deciles (0-9).
    3. Compute task share within each decile and each task share of total job hours.
    4. Aggregate across peers to median/p25/p75 and peer coverage counts.

    Returns:
        lifecycle_df: task-by-decile distribution summary.
        task_summary_df: task total-share summary across peers.
    """
    peers = _peer_scope(df_all, department, category, exclude_job_no)
    if len(peers) == 0:
        return _empty_lifecycle_df(), _empty_task_summary_df()

    lifecycle_rows = []
    task_total_rows = []

    for job_no, job_df in peers.groupby("job_no"):
        deciled = _compute_deciles(job_df, date_col="_date")
        if len(deciled) == 0:
            continue

        total_hours = float(deciled["hours_raw"].sum())
        if total_hours <= 0:
            continue

        task_totals = deciled.groupby("task_name")["hours_raw"].sum()
        decile_totals = deciled.groupby("decile_bucket")["hours_raw"].sum()
        task_decile = deciled.groupby(["task_name", "decile_bucket"])["hours_raw"].sum()

        tasks_for_job = sorted(task_totals.index.astype(str).tolist())
        full_index = pd.MultiIndex.from_product(
            [tasks_for_job, list(range(10))],
            names=["task_name", "decile_bucket"],
        )
        task_decile = task_decile.reindex(full_index, fill_value=0.0).reset_index(name="task_hours")
        task_decile["bucket_total"] = task_decile["decile_bucket"].map(decile_totals.to_dict()).fillna(0.0)
        task_decile["hours_share"] = np.where(
            task_decile["bucket_total"] > 0,
            task_decile["task_hours"] / task_decile["bucket_total"],
            0.0,
        )

        lifecycle_rows.extend(
            {
                "job_no": str(job_no),
                "task_name": str(row.task_name),
                "decile_bucket": int(row.decile_bucket),
                "hours_share": float(row.hours_share),
            }
            for row in task_decile.itertuples(index=False)
        )

        for task_name, task_hours in task_totals.items():
            task_total_rows.append(
                {
                    "job_no": str(job_no),
                    "task_name": str(task_name),
                    "total_share": float(task_hours) / total_hours,
                    "task_hours": float(task_hours),
                }
            )

    if len(lifecycle_rows) == 0 or len(task_total_rows) == 0:
        return _empty_lifecycle_df(), _empty_task_summary_df()

    lifecycle_raw = pd.DataFrame(lifecycle_rows)
    lifecycle_df = (
        lifecycle_raw.groupby(["task_name", "decile_bucket"])
        .agg(
            median_hours_share=("hours_share", "median"),
            p25_hours_share=("hours_share", lambda s: s.quantile(0.25)),
            p75_hours_share=("hours_share", lambda s: s.quantile(0.75)),
            peer_job_count=("job_no", "nunique"),
        )
        .reset_index()
    )

    # Keep each decile on a clean 0-1 scale in aggregate.
    for col in ["median_hours_share", "p25_hours_share", "p75_hours_share"]:
        decile_sum = lifecycle_df.groupby("decile_bucket")[col].transform("sum")
        lifecycle_df[col] = np.where(decile_sum > 0, lifecycle_df[col] / decile_sum, 0.0)

    lifecycle_df = lifecycle_df[_LIFECYCLE_COLUMNS].sort_values(["decile_bucket", "task_name"])

    task_totals_raw = pd.DataFrame(task_total_rows)
    task_summary = (
        task_totals_raw.groupby("task_name")
        .agg(
            median_total_share=("total_share", "median"),
            p25_total_share=("total_share", lambda s: s.quantile(0.25)),
            p75_total_share=("total_share", lambda s: s.quantile(0.75)),
            peer_median_total_hours=("task_hours", "median"),
            peer_job_count=("job_no", "nunique"),
        )
        .reset_index()
    )

    return lifecycle_df, task_summary[_TASK_SUMMARY_COLUMNS].sort_values("median_total_share", ascending=False)


@st.cache_data(show_spinner=False)
def compute_job_timeline(
    df_all: pd.DataFrame,
    job_no: str,
    granularity: str = "weekly",
) -> pd.DataFrame:
    """
    Build task-level hours/cost timeline with cumulative totals for one job.

    Args:
        df_all: Full fact-level dataframe.
        job_no: Selected job id.
        granularity: "weekly" or "monthly".

    Returns:
        DataFrame with period/task sums and period-level cumulative totals.
    """
    if len(df_all) == 0 or "job_no" not in df_all.columns:
        return _empty_timeline_df()

    job_df = df_all[df_all["job_no"].astype(str) == str(job_no)].copy()
    if len(job_df) == 0:
        return _empty_timeline_df()

    date_col = _resolve_date_col(job_df)
    if date_col is None:
        return _empty_timeline_df()

    job_df[date_col] = pd.to_datetime(job_df[date_col], errors="coerce")
    job_df = job_df[job_df[date_col].notna()].copy()
    if len(job_df) == 0:
        return _empty_timeline_df()

    job_df["hours_raw"] = pd.to_numeric(job_df.get("hours_raw"), errors="coerce").fillna(0.0)
    job_df["base_cost"] = pd.to_numeric(job_df.get("base_cost"), errors="coerce").fillna(0.0)
    job_df["task_name"] = job_df.get("task_name", "Unspecified").fillna("Unspecified").astype(str)

    use_weekly = granularity == "weekly"
    if use_weekly:
        iso = job_df[date_col].dt.isocalendar()
        job_df["period_start"] = [
            pd.Timestamp.fromisocalendar(int(y), int(w), 1)
            for y, w in zip(iso["year"], iso["week"])
        ]
        job_df["period_label"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    else:
        job_df["period_start"] = job_df[date_col].dt.to_period("M").dt.to_timestamp()
        job_df["period_label"] = job_df["period_start"].dt.strftime("%Y-%m")

    task_period = (
        job_df.groupby(["period_start", "period_label", "task_name"], as_index=False)
        .agg(
            hours=("hours_raw", "sum"),
            cost=("base_cost", "sum"),
        )
        .sort_values(["period_start", "task_name"])
    )

    period_totals = (
        task_period.groupby(["period_start", "period_label"], as_index=False)
        .agg(hours=("hours", "sum"), cost=("cost", "sum"))
        .sort_values("period_start")
    )
    period_totals["cumulative_hours"] = period_totals["hours"].cumsum()
    period_totals["cumulative_cost"] = period_totals["cost"].cumsum()

    timeline = task_period.merge(
        period_totals[["period_start", "period_label", "cumulative_hours", "cumulative_cost"]],
        on=["period_start", "period_label"],
        how="left",
    )

    return timeline[_TIMELINE_COLUMNS].sort_values(["period_start", "task_name"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def estimate_remaining_work(
    df_all: pd.DataFrame,
    job_no: str,
    department: str,
    category: str,
    job_row: pd.Series,
    included_tasks: tuple[str, ...] | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Estimate remaining task hours from peer task-share distributions.

    Algorithm:
    1. Build peer task-share summary from completed jobs in the same cohort.
    2. Scope forecasting to included tasks (default: executed tasks for the job).
    3. Build scoped EAC baseline from quote-by-task and peer median task totals.
    4. Optionally apply lifecycle adjustment using peer decile curve + runtime progress.
    5. Estimate expected total task hours (p25/p50/p75) and subtract actuals.
    6. Apply outlier guardrails and include unmodeled actual tasks.

    Returns:
        task_remaining_df, summary_dict
    """
    summary = {
        "total_remaining_p25": 0.0,
        "total_remaining_median": 0.0,
        "total_remaining_p75": 0.0,
        "eac_baseline": 0.0,
        "eac_pre_adjustment": 0.0,
        "peer_count": 0,
        "outlier_task_count": 0,
        "unmodeled_task_count": 0,
        "lifecycle_adjustment_applied": False,
        "runtime_progress_pct": np.nan,
        "lifecycle_expected_completion_pct": np.nan,
        "actual_completion_pct": np.nan,
        "task_scope_count": 0,
        "included_tasks": tuple(),
        "scoped_actual_hours": np.nan,
        "scoped_actual_cost": np.nan,
        "scoped_actual_revenue": np.nan,
        "scoped_quoted_amount": np.nan,
        "scope_is_full_job": False,
    }

    if len(df_all) == 0 or "job_no" not in df_all.columns:
        return _empty_remaining_df(), summary

    job_df = df_all[df_all["job_no"].astype(str) == str(job_no)].copy()
    if len(job_df) == 0:
        return _empty_remaining_df(), summary

    job_df["hours_raw"] = pd.to_numeric(job_df.get("hours_raw"), errors="coerce").fillna(0.0)
    job_df["task_name"] = job_df.get("task_name", "Unspecified").fillna("Unspecified").astype(str)
    actual_task_hours_all = job_df.groupby("task_name")["hours_raw"].sum()

    if included_tasks is not None and len(included_tasks) > 0:
        include_set = {str(t).strip() for t in included_tasks if str(t).strip()}
    else:
        include_set = set(actual_task_hours_all.index.astype(str).tolist())

    if len(include_set) == 0:
        return _empty_remaining_df(), summary

    actual_task_hours = actual_task_hours_all[
        actual_task_hours_all.index.astype(str).isin(include_set)
    ]
    total_actual = float(actual_task_hours.sum())
    summary["included_tasks"] = tuple(sorted(include_set))
    all_job_tasks = set(job_df["task_name"].astype(str).dropna().unique().tolist())
    summary["scope_is_full_job"] = bool(len(all_job_tasks) > 0 and include_set == all_job_tasks)

    scoped_job_df = job_df[job_df["task_name"].isin(include_set)].copy()
    scoped_job_df["base_cost"] = pd.to_numeric(scoped_job_df.get("base_cost"), errors="coerce").fillna(0.0)
    scoped_job_df["rev_alloc"] = pd.to_numeric(scoped_job_df.get("rev_alloc"), errors="coerce").fillna(0.0)
    summary["scoped_actual_hours"] = float(total_actual)
    summary["scoped_actual_cost"] = float(scoped_job_df["base_cost"].sum())
    summary["scoped_actual_revenue"] = float(scoped_job_df["rev_alloc"].sum())

    peers_df = _peer_scope(df_all, department, category, str(job_no))
    _lifecycle_df, task_summary_df_all = build_peer_lifecycle_profiles(
        df_all=df_all,
        department=department,
        category=category,
        exclude_job_no=str(job_no),
    )
    task_summary_df = (
        task_summary_df_all[
            task_summary_df_all["task_name"].astype(str).isin(include_set)
        ].copy()
        if len(task_summary_df_all) > 0
        else task_summary_df_all
    )

    peer_totals = peers_df.groupby("job_no")["hours_raw"].sum() if len(peers_df) > 0 else pd.Series(dtype=float)
    peer_count = int(peer_totals.index.nunique())

    quoted_hours_total = pd.to_numeric(job_row.get("quoted_hours"), errors="coerce")
    if pd.isna(quoted_hours_total) or quoted_hours_total <= 0:
        quoted_hours_safe, _ = _safe_quote_totals_for_job(job_df)
        quoted_hours_total = quoted_hours_safe

    quote_scope_hours = np.nan
    job_task_quotes = safe_quote_job_task(job_df)
    if len(job_task_quotes) > 0 and "quoted_time_total" in job_task_quotes.columns:
        q = job_task_quotes.copy()
        q["task_name"] = q["task_name"].astype(str)
        q = q[q["task_name"].isin(include_set)]
        quote_scope_hours = pd.to_numeric(q["quoted_time_total"], errors="coerce").fillna(0.0).sum()
    if len(job_task_quotes) > 0 and "quoted_amount_total" in job_task_quotes.columns:
        qa = job_task_quotes.copy()
        qa["task_name"] = qa["task_name"].astype(str)
        qa = qa[qa["task_name"].isin(include_set)]
        summary["scoped_quoted_amount"] = float(
            pd.to_numeric(qa["quoted_amount_total"], errors="coerce").fillna(0.0).sum()
        )

    peer_scope_hours = (
        float(pd.to_numeric(task_summary_df["peer_median_total_hours"], errors="coerce").fillna(0.0).sum())
        if len(task_summary_df) > 0
        else np.nan
    )
    quote_available_scope = bool(pd.notna(quote_scope_hours) and quote_scope_hours > 0)
    peer_available_scope = bool(pd.notna(peer_scope_hours) and peer_scope_hours > 0)

    if quote_available_scope and peer_available_scope:
        eac_baseline = float(max(quote_scope_hours, peer_scope_hours))
    elif quote_available_scope:
        eac_baseline = float(quote_scope_hours)
    elif peer_available_scope:
        eac_baseline = float(peer_scope_hours)
    elif pd.notna(quoted_hours_total) and quoted_hours_total > 0 and len(include_set) == len(actual_task_hours_all):
        eac_baseline = float(quoted_hours_total)
    elif len(peer_totals) > 0 and len(include_set) == len(actual_task_hours_all):
        eac_baseline = float(peer_totals.median())
    else:
        eac_baseline = float(total_actual)
    eac_pre_adjustment = float(eac_baseline)

    # Lifecycle-informed EAC adjustment:
    # if runtime progress implies peers should have consumed X% by now,
    # infer lifecycle EAC = actual_hours / X and blend with baseline.
    runtime_progress_pct = np.nan
    lifecycle_expected_completion_pct = np.nan
    actual_completion_pct = np.nan
    lifecycle_adjustment_applied = False

    if (
        peer_count >= 3
        and total_actual > 0
        and eac_baseline > 0
        and len(peers_df) > 0
    ):
        progress_ratio = _runtime_progress_ratio(job_df, peers_df)
        if pd.notna(progress_ratio):
            stage_progress = float(np.clip(progress_ratio, 0.0, 1.0))
            decile_curve = _peer_decile_curve(df_all, department, category, str(job_no))
            expected_completion = _cumulative_share_at_progress(decile_curve, stage_progress)

            runtime_progress_pct = float(progress_ratio * 100.0)
            lifecycle_expected_completion_pct = (
                float(expected_completion * 100.0) if pd.notna(expected_completion) else np.nan
            )

            if pd.notna(expected_completion) and expected_completion >= 0.10 and stage_progress >= 0.20:
                eac_lifecycle = float(total_actual / expected_completion)
                if quote_available_scope:
                    blended_eac = (0.70 * eac_baseline) + (0.30 * eac_lifecycle)
                else:
                    blended_eac = (0.40 * eac_baseline) + (0.60 * eac_lifecycle)

                lower_bound = max(total_actual, eac_baseline * 0.60)
                upper_bound = max(lower_bound, eac_baseline * 2.00)
                eac_baseline = float(np.clip(blended_eac, lower_bound, upper_bound))
                lifecycle_adjustment_applied = True

    actual_completion_pct = (
        float((total_actual / eac_baseline) * 100.0)
        if eac_baseline > 0
        else np.nan
    )

    rows = []
    modeled_tasks = set()

    if len(task_summary_df) > 0:
        sum_median_share = float(pd.to_numeric(task_summary_df["median_total_share"], errors="coerce").fillna(0.0).sum())
        sum_p25_share = float(pd.to_numeric(task_summary_df["p25_total_share"], errors="coerce").fillna(0.0).sum())
        sum_p75_share = float(pd.to_numeric(task_summary_df["p75_total_share"], errors="coerce").fillna(0.0).sum())
        sum_peer_hours = float(pd.to_numeric(task_summary_df["peer_median_total_hours"], errors="coerce").fillna(0.0).sum())

        for row in task_summary_df.itertuples(index=False):
            task_name = str(row.task_name)
            modeled_tasks.add(task_name)

            if sum_median_share > 0:
                w_median = float(row.median_total_share) / sum_median_share
            elif sum_peer_hours > 0:
                w_median = float(row.peer_median_total_hours) / sum_peer_hours
            else:
                w_median = 0.0

            w_p25 = (
                float(row.p25_total_share) / sum_p25_share
                if sum_p25_share > 0
                else w_median
            )
            w_p75 = (
                float(row.p75_total_share) / sum_p75_share
                if sum_p75_share > 0
                else w_median
            )

            expected_p25 = float(eac_baseline * w_p25)
            expected_median = float(eac_baseline * w_median)
            expected_p75 = float(eac_baseline * w_p75)
            expected_p75_pre = float(eac_pre_adjustment * w_p75)
            actual = float(actual_task_hours.get(task_name, 0.0))

            remaining_p25 = max(expected_p25 - actual, 0.0)
            remaining_median = max(expected_median - actual, 0.0)
            remaining_p75 = max(expected_p75 - actual, 0.0)

            # Anchor outlier detection to pre-adjustment expectation so lifecycle
            # scaling does not hide extreme task blowouts.
            threshold_p75 = expected_p75_pre if expected_p75_pre > 0 else expected_p75
            outlier = bool(threshold_p75 > 0 and actual > (threshold_p75 * 1.5))
            if outlier:
                remaining_p25 = 0.0
                remaining_median = 0.0
                remaining_p75 = 0.0

            rows.append(
                {
                    "task_name": task_name,
                    "actual_hours": actual,
                    "expected_hours_p25": expected_p25,
                    "expected_hours_median": expected_median,
                    "expected_hours_p75": expected_p75,
                    "remaining_hours_p25": remaining_p25,
                    "remaining_hours_median": remaining_median,
                    "remaining_hours_p75": remaining_p75,
                    "outlier_flag": outlier,
                    "unmodeled_flag": False,
                    "peer_job_count": int(row.peer_job_count),
                }
            )

    for task_name, actual in actual_task_hours.items():
        if str(task_name) in modeled_tasks:
            continue
        rows.append(
            {
                "task_name": str(task_name),
                "actual_hours": float(actual),
                "expected_hours_p25": np.nan,
                "expected_hours_median": np.nan,
                "expected_hours_p75": np.nan,
                "remaining_hours_p25": 0.0,
                "remaining_hours_median": 0.0,
                "remaining_hours_p75": 0.0,
                "outlier_flag": False,
                "unmodeled_flag": True,
                "peer_job_count": 0,
            }
        )

    if len(rows) == 0:
        summary.update(
            {
                "eac_baseline": float(eac_baseline),
                "eac_pre_adjustment": float(eac_pre_adjustment),
                "peer_count": peer_count,
                "lifecycle_adjustment_applied": lifecycle_adjustment_applied,
                "runtime_progress_pct": runtime_progress_pct,
                "lifecycle_expected_completion_pct": lifecycle_expected_completion_pct,
                "actual_completion_pct": actual_completion_pct,
                "task_scope_count": len(include_set),
                "included_tasks": tuple(sorted(include_set)),
                "scope_is_full_job": bool(summary["scope_is_full_job"]),
            }
        )
        return _empty_remaining_df(), summary

    remaining_df = pd.DataFrame(rows)
    remaining_df = remaining_df[_REMAINING_COLUMNS].sort_values(
        ["remaining_hours_median", "actual_hours"],
        ascending=[False, False],
    )

    summary.update(
        {
            "total_remaining_p25": float(remaining_df["remaining_hours_p25"].sum()),
            "total_remaining_median": float(remaining_df["remaining_hours_median"].sum()),
            "total_remaining_p75": float(remaining_df["remaining_hours_p75"].sum()),
            "eac_baseline": float(eac_baseline),
            "eac_pre_adjustment": float(eac_pre_adjustment),
            "peer_count": peer_count,
            "outlier_task_count": int(remaining_df["outlier_flag"].sum()),
            "unmodeled_task_count": int(remaining_df["unmodeled_flag"].sum()),
            "lifecycle_adjustment_applied": lifecycle_adjustment_applied,
            "runtime_progress_pct": runtime_progress_pct,
            "lifecycle_expected_completion_pct": lifecycle_expected_completion_pct,
            "actual_completion_pct": actual_completion_pct,
            "task_scope_count": len(include_set),
            "included_tasks": tuple(sorted(include_set)),
            "scope_is_full_job": bool(summary["scope_is_full_job"]),
        }
    )

    return remaining_df.reset_index(drop=True), summary


@st.cache_data(show_spinner=False)
def forecast_completion(
    job_row: pd.Series,
    remaining_summary: Dict,
    df_all: pd.DataFrame,
    job_no: str,
) -> Dict:
    """
    Forecast completion date and margin for optimistic/expected/conservative scenarios.

    Algorithm:
    1. Use job burn-rate from `job_row`; fallback to trailing 28-day burn from scoped fact rows.
    2. Use scoped actual/quote values from `remaining_summary` when available.
    3. For each remaining-hours scenario, compute days to complete, end date, cost, revenue.
    4. Compute margin and margin percentage with stall handling when burn-rate is zero.

    Returns:
        Dictionary with burn-rate, last activity date, scenario metrics, and stall flag.
    """
    job_df = (
        df_all[df_all["job_no"].astype(str) == str(job_no)].copy()
        if len(df_all) > 0 and "job_no" in df_all.columns
        else pd.DataFrame()
    )
    scope_tasks = tuple(str(t) for t in remaining_summary.get("included_tasks", tuple()) if str(t).strip())
    scope_set = set(scope_tasks)

    scoped_df = job_df.copy()
    if len(scoped_df) > 0 and len(scope_set) > 0 and "task_name" in scoped_df.columns:
        scoped_df["task_name"] = scoped_df["task_name"].fillna("Unspecified").astype(str)
        scoped_df = scoped_df[scoped_df["task_name"].isin(scope_set)].copy()
        if len(scoped_df) == 0:
            scoped_df = job_df.copy()

    date_col = _resolve_date_col(scoped_df) if len(scoped_df) > 0 else _resolve_date_col(job_df)
    if date_col is not None:
        if len(scoped_df) > 0 and date_col in scoped_df.columns:
            scoped_df[date_col] = pd.to_datetime(scoped_df[date_col], errors="coerce")
            scoped_df = scoped_df[scoped_df[date_col].notna()].copy()
        if len(job_df) > 0 and date_col in job_df.columns:
            job_df[date_col] = pd.to_datetime(job_df[date_col], errors="coerce")
            job_df = job_df[job_df[date_col].notna()].copy()

    last_activity = (
        scoped_df[date_col].max()
        if date_col is not None and len(scoped_df) > 0 and date_col in scoped_df.columns
        else (job_df[date_col].max() if date_col is not None and len(job_df) > 0 and date_col in job_df.columns else pd.NaT)
    )

    burn_rate = pd.to_numeric(job_row.get("burn_rate_per_day"), errors="coerce")
    if pd.isna(burn_rate) or burn_rate <= 0:
        burn_source = scoped_df if len(scoped_df) > 0 else job_df
        if date_col is not None and len(burn_source) > 0 and date_col in burn_source.columns:
            last_date = burn_source[date_col].max()
            cutoff = last_date - timedelta(days=28)
            recent = burn_source[burn_source[date_col] >= cutoff]
            if len(recent) == 0:
                recent = burn_source
            span_days = max((recent[date_col].max() - recent[date_col].min()).days, 1)
            hours = pd.to_numeric(recent.get("hours_raw"), errors="coerce").fillna(0.0).sum()
            burn_rate = hours / span_days if span_days > 0 else 0.0
        else:
            burn_rate = 0.0

    burn_rate = float(burn_rate) if pd.notna(burn_rate) else 0.0

    actual_cost = pd.to_numeric(remaining_summary.get("scoped_actual_cost"), errors="coerce")
    actual_hours = pd.to_numeric(remaining_summary.get("scoped_actual_hours"), errors="coerce")
    actual_revenue = pd.to_numeric(remaining_summary.get("scoped_actual_revenue"), errors="coerce")

    if pd.isna(actual_cost):
        actual_cost = pd.to_numeric(job_row.get("actual_cost"), errors="coerce")
    if pd.isna(actual_hours):
        actual_hours = pd.to_numeric(job_row.get("actual_hours"), errors="coerce")
    if pd.isna(actual_revenue):
        actual_revenue = pd.to_numeric(job_row.get("actual_revenue"), errors="coerce")

    if len(scoped_df) > 0:
        if pd.isna(actual_cost):
            actual_cost = pd.to_numeric(scoped_df.get("base_cost"), errors="coerce").fillna(0.0).sum()
        if pd.isna(actual_hours):
            actual_hours = pd.to_numeric(scoped_df.get("hours_raw"), errors="coerce").fillna(0.0).sum()
        if pd.isna(actual_revenue):
            actual_revenue = pd.to_numeric(scoped_df.get("rev_alloc"), errors="coerce").fillna(0.0).sum()

    actual_cost = float(actual_cost) if pd.notna(actual_cost) else 0.0
    actual_hours = float(actual_hours) if pd.notna(actual_hours) else 0.0
    actual_revenue = float(actual_revenue) if pd.notna(actual_revenue) else 0.0

    avg_cost_per_hour = (actual_cost / actual_hours) if actual_hours > 0 else np.nan

    scope_is_full_job = bool(remaining_summary.get("scope_is_full_job", len(scope_set) == 0))
    quoted_amount = pd.to_numeric(remaining_summary.get("scoped_quoted_amount"), errors="coerce")
    if (pd.isna(quoted_amount) or quoted_amount <= 0) and scope_is_full_job:
        quoted_amount = pd.to_numeric(job_row.get("quoted_amount"), errors="coerce")
    if (pd.isna(quoted_amount) or quoted_amount <= 0) and len(scoped_df) > 0:
        scoped_quotes = safe_quote_job_task(scoped_df)
        if len(scoped_quotes) > 0 and "quoted_amount_total" in scoped_quotes.columns:
            quoted_amount = pd.to_numeric(
                scoped_quotes["quoted_amount_total"], errors="coerce"
            ).fillna(0.0).sum()
    if (pd.isna(quoted_amount) or quoted_amount <= 0) and len(job_df) > 0:
        _qh, qa = _safe_quote_totals_for_job(job_df)
        if scope_is_full_job:
            quoted_amount = qa

    is_stalled = bool((pd.isna(burn_rate)) or burn_rate <= 0)

    scenario_map = {
        "optimistic": float(pd.to_numeric(remaining_summary.get("total_remaining_p25"), errors="coerce") or 0.0),
        "expected": float(pd.to_numeric(remaining_summary.get("total_remaining_median"), errors="coerce") or 0.0),
        "conservative": float(pd.to_numeric(remaining_summary.get("total_remaining_p75"), errors="coerce") or 0.0),
    }

    scenarios: Dict[str, Dict] = {}
    for scenario_name, remaining_hours in scenario_map.items():
        days_to_complete = (
            float(remaining_hours / burn_rate)
            if not is_stalled and burn_rate > 0
            else np.nan
        )

        if not is_stalled and pd.notna(last_activity) and pd.notna(days_to_complete):
            forecast_end_date = last_activity + timedelta(days=days_to_complete)
        else:
            forecast_end_date = pd.NaT

        forecast_total_cost = (
            actual_cost + (remaining_hours * avg_cost_per_hour)
            if pd.notna(avg_cost_per_hour)
            else np.nan
        )

        if pd.notna(quoted_amount) and float(quoted_amount) > 0:
            forecast_revenue = float(quoted_amount)
        elif actual_hours > 0:
            revenue_per_hour = actual_revenue / actual_hours if actual_hours > 0 else np.nan
            forecast_revenue = (
                revenue_per_hour * (actual_hours + remaining_hours)
                if pd.notna(revenue_per_hour)
                else actual_revenue
            )
        else:
            forecast_revenue = actual_revenue

        forecast_margin = (
            forecast_revenue - forecast_total_cost
            if pd.notna(forecast_revenue) and pd.notna(forecast_total_cost)
            else np.nan
        )
        forecast_margin_pct = (
            (forecast_margin / forecast_revenue) * 100
            if pd.notna(forecast_revenue) and forecast_revenue > 0 and pd.notna(forecast_margin)
            else np.nan
        )

        scenarios[scenario_name] = {
            "remaining_hours": float(remaining_hours),
            "days_to_complete": days_to_complete,
            "forecast_end_date": forecast_end_date,
            "forecast_total_cost": float(forecast_total_cost) if pd.notna(forecast_total_cost) else np.nan,
            "forecast_revenue": float(forecast_revenue) if pd.notna(forecast_revenue) else np.nan,
            "forecast_margin": float(forecast_margin) if pd.notna(forecast_margin) else np.nan,
            "forecast_margin_pct": float(forecast_margin_pct) if pd.notna(forecast_margin_pct) else np.nan,
        }

    return {
        "burn_rate_per_day": float(burn_rate),
        "last_activity_date": last_activity,
        "scenarios": scenarios,
        "is_stalled": is_stalled,
        "avg_cost_per_hour": float(avg_cost_per_hour) if pd.notna(avg_cost_per_hour) else np.nan,
    }
