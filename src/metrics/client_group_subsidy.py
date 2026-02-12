"""
Client-group subsidy context metrics for Active Delivery.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


GROUP_COLUMN_PRECEDENCE = [
    "client_group_rev_job_month",
    "client_group_rev_job",
    "client_group",
    "client",
]

ROLLUP_COLUMNS = [
    "job_no",
    "job_name",
    "department_final",
    "job_category",
    "revenue",
    "cost",
    "hours",
    "margin",
    "margin_pct",
    "last_date",
]

RESULT_JOB_COLUMNS = [
    "is_selected",
    "job_no",
    "job_name",
    "department_final",
    "job_category",
    "risk_band",
    "risk_score",
    "revenue",
    "cost",
    "hours",
    "margin",
    "margin_pct",
    "contribution_pct_to_group_margin",
    "last_date",
]


def _non_empty_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _first_non_empty(series: pd.Series):
    mask = _non_empty_mask(series)
    if mask.any():
        return series[mask].iloc[0]
    return np.nan


def _resolve_date_series(df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series]]:
    for col in ["work_date", "month_key"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return col, parsed
    return None, None


def _empty_summary() -> dict:
    return {
        "selected_margin": 0.0,
        "selected_margin_pct": np.nan,
        "selected_revenue": 0.0,
        "selected_cost": 0.0,
        "group_revenue": 0.0,
        "group_cost": 0.0,
        "group_margin": 0.0,
        "group_margin_pct": np.nan,
        "job_count": 0,
        "active_job_count": 0,
        "loss_job_count": 0,
        "red_job_count": 0,
        "amber_job_count": 0,
        "selected_loss_abs": 0.0,
        "positive_peer_margin_pool": 0.0,
        "coverage_ratio": np.nan,
        "buffer_after_subsidy": 0.0,
        "subsidizer_job_count": 0,
        "subsidy_concentration_pct": np.nan,
        "verdict": "No Data",
    }


def _empty_jobs() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_JOB_COLUMNS)


def resolve_group_column(df: pd.DataFrame, preferred: str = "client_group_rev_job_month") -> str | None:
    """
    Return first existing non-empty grouping column in precedence:
    preferred first, then client_group_rev_job, client_group, client.
    A column counts as usable if it exists and has at least one non-null/non-empty value.
    """
    candidates = [preferred] + [col for col in GROUP_COLUMN_PRECEDENCE if col != preferred]
    for col in candidates:
        if col in df.columns and _non_empty_mask(df[col]).any():
            return col
    return None


def get_selected_group_value(df: pd.DataFrame, selected_job_no: str, group_col: str) -> str | None:
    """
    For selected job, return the most recent non-null group value.
    Recency order:
      - work_date if present
      - else month_key if present
      - else original row order
    If multiple non-null values remain tied, use mode() first value.
    """
    if "job_no" not in df.columns or group_col not in df.columns:
        return None

    selected = df[df["job_no"].astype(str) == str(selected_job_no)].copy()
    if len(selected) == 0:
        return None

    selected = selected[_non_empty_mask(selected[group_col])].copy()
    if len(selected) == 0:
        return None

    if "work_date" in selected.columns:
        recency = pd.to_datetime(selected["work_date"], errors="coerce")
    elif "month_key" in selected.columns:
        recency = pd.to_datetime(selected["month_key"], errors="coerce")
    else:
        recency = pd.Series(np.arange(len(selected)), index=selected.index, dtype=float)

    candidates = selected.copy()
    if recency.notna().any():
        max_recency = recency.max()
        candidates = selected[recency == max_recency].copy()
    elif len(selected) > 0:
        candidates = selected.iloc[[-1]].copy()

    values = candidates[group_col].astype(str).str.strip()
    mode_vals = values.mode(dropna=True)
    if len(mode_vals) > 0:
        return mode_vals.iloc[0]
    if len(values) > 0:
        return values.iloc[0]
    return None


def resolve_job_group_values(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Resolve one group value per job using the most recent non-empty group entry.

    Returns DataFrame with columns: job_no, group_value.
    """
    if (
        not isinstance(df, pd.DataFrame)
        or len(df) == 0
        or "job_no" not in df.columns
        or group_col not in df.columns
    ):
        return pd.DataFrame(columns=["job_no", "group_value"])

    valid_group_mask = _non_empty_mask(df[group_col])
    if not valid_group_mask.any():
        return pd.DataFrame(columns=["job_no", "group_value"])

    row_order = pd.Series(np.arange(len(df), dtype=int), index=df.index)
    _, date_series = _resolve_date_series(df)

    work = df.loc[valid_group_mask, ["job_no", group_col]].copy()
    work["job_no"] = work["job_no"].astype(str)
    work["group_value"] = work[group_col].astype(str).str.strip()
    work["_row_order"] = row_order.loc[work.index].values
    if date_series is None:
        work["_recency"] = pd.NaT
    else:
        work["_recency"] = date_series.loc[work.index]
    work["_has_recency"] = work["_recency"].notna()

    work = work.sort_values(
        ["job_no", "_has_recency", "_recency", "_row_order"],
        ascending=[True, False, False, False],
        na_position="last",
    )
    latest = work.drop_duplicates("job_no", keep="first")
    return latest[["job_no", "group_value"]]


def build_job_rollup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to job-level with columns:
    job_no, job_name, department_final, job_category,
    revenue, cost, hours, margin, margin_pct, last_date
    revenue = sum(rev_alloc)
    cost = sum(base_cost)
    hours = sum(hours_raw)
    margin = revenue - cost
    margin_pct = margin / revenue * 100 (NaN when revenue <= 0)
    last_date = max(work_date or month_key when available)
    """
    if len(df) == 0 or "job_no" not in df.columns:
        return pd.DataFrame(columns=ROLLUP_COLUMNS)

    work = df.copy()
    work["job_no"] = work["job_no"].astype(str)

    if "job_name" not in work.columns:
        work["job_name"] = work["job_no"]
    if "department_final" not in work.columns:
        work["department_final"] = np.nan
    if "job_category" not in work.columns:
        work["job_category"] = work.get("category_rev_job", np.nan)

    for col in ["rev_alloc", "base_cost", "hours_raw"]:
        if col not in work.columns:
            work[col] = 0.0

    date_col, parsed_dates = _resolve_date_series(work)
    if date_col is None or parsed_dates is None:
        work["_last_date"] = pd.NaT
    else:
        work["_last_date"] = parsed_dates

    rollup = work.groupby("job_no", dropna=False).agg(
        job_name=("job_name", _first_non_empty),
        department_final=("department_final", _first_non_empty),
        job_category=("job_category", _first_non_empty),
        revenue=("rev_alloc", "sum"),
        cost=("base_cost", "sum"),
        hours=("hours_raw", "sum"),
        last_date=("_last_date", "max"),
    ).reset_index()

    rollup["revenue"] = pd.to_numeric(rollup["revenue"], errors="coerce").fillna(0.0)
    rollup["cost"] = pd.to_numeric(rollup["cost"], errors="coerce").fillna(0.0)
    rollup["hours"] = pd.to_numeric(rollup["hours"], errors="coerce").fillna(0.0)
    rollup["margin"] = rollup["revenue"] - rollup["cost"]
    rollup["margin_pct"] = np.where(
        rollup["revenue"] > 0,
        rollup["margin"] / rollup["revenue"] * 100,
        np.nan,
    )

    return rollup[ROLLUP_COLUMNS]


def compute_client_group_subsidy_context(
    df_all: pd.DataFrame,
    jobs_df: pd.DataFrame,
    selected_job_no: str,
    lookback_months: int | None = 12,
    scope: str = "all",  # "all" | "active_only"
) -> dict:
    """
    Return dict with stable contract below.
    Must gracefully handle missing columns/data.
    """
    normalized_scope = "active_only" if scope == "active_only" else "all"
    summary = _empty_summary()
    jobs_out = _empty_jobs()

    result = {
        "status": "empty_group",
        "group_col": None,
        "group_value": None,
        "window_months": lookback_months,
        "scope": normalized_scope,
        "summary": summary,
        "jobs": jobs_out,
    }

    if not isinstance(df_all, pd.DataFrame) or len(df_all) == 0:
        return result

    required_cols = {"job_no", "rev_alloc", "base_cost", "hours_raw"}
    if not required_cols.issubset(df_all.columns):
        return result

    group_col = resolve_group_column(df_all)
    result["group_col"] = group_col
    if group_col is None:
        result["status"] = "missing_group_column"
        return result

    window_df = df_all.copy()
    if lookback_months is not None:
        _, date_series = _resolve_date_series(window_df)
        if date_series is not None and date_series.notna().any():
            latest = date_series.max()
            cutoff = latest - pd.DateOffset(months=lookback_months)
            # Use lookback window to identify WHICH JOBS had recent activity,
            # but keep ALL rows for those jobs so margin is computed on full
            # job history (matching margin_pct_to_date in delivery_control).
            recent_jobs = set(
                window_df.loc[date_series >= cutoff, "job_no"].dropna().unique()
            )
            window_df = window_df[window_df["job_no"].isin(recent_jobs)].copy()

    selected_job_key = str(selected_job_no)
    job_groups = resolve_job_group_values(window_df, group_col)
    if len(job_groups) == 0:
        result["status"] = "missing_group_value"
        return result

    selected_group_rows = job_groups[job_groups["job_no"] == selected_job_key]
    if len(selected_group_rows) == 0:
        result["status"] = "missing_group_value"
        return result

    group_value = str(selected_group_rows.iloc[0]["group_value"]).strip()
    result["group_value"] = group_value

    group_jobs = set(
        job_groups.loc[
            job_groups["group_value"].astype(str).str.strip().eq(group_value),
            "job_no",
        ].tolist()
    )
    group_df = window_df[window_df["job_no"].astype(str).isin(group_jobs)].copy()

    rollup = build_job_rollup(group_df)
    if len(rollup) == 0:
        return result

    active_meta = pd.DataFrame(columns=["job_no", "risk_band", "risk_score"])
    active_set: set[str] = set()
    if isinstance(jobs_df, pd.DataFrame) and "job_no" in jobs_df.columns:
        active_meta = jobs_df.copy()
        active_meta["job_no"] = active_meta["job_no"].astype(str)
        active_set = set(active_meta["job_no"].dropna().tolist())
        if "risk_band" not in active_meta.columns:
            active_meta["risk_band"] = np.nan
        if "risk_score" not in active_meta.columns:
            active_meta["risk_score"] = np.nan
        active_meta = active_meta[["job_no", "risk_band", "risk_score"]].drop_duplicates("job_no")

    if normalized_scope == "active_only":
        keep_jobs = active_set | {selected_job_key}
        rollup = rollup[rollup["job_no"].isin(keep_jobs)].copy()
        if len(rollup) == 0:
            return result

    rollup = rollup.merge(active_meta, on="job_no", how="left")
    rollup["risk_band"] = rollup["risk_band"].fillna("N/A")
    rollup["risk_score"] = pd.to_numeric(rollup["risk_score"], errors="coerce")
    rollup["is_selected"] = rollup["job_no"] == selected_job_key

    group_revenue = float(rollup["revenue"].sum())
    group_cost = float(rollup["cost"].sum())
    group_margin = float(rollup["margin"].sum())
    group_margin_pct = (
        float(group_margin / group_revenue * 100) if group_revenue > 0 else np.nan
    )

    rollup["contribution_pct_to_group_margin"] = np.where(
        group_margin != 0,
        rollup["margin"] / group_margin * 100,
        np.nan,
    )

    selected_rows = rollup[rollup["is_selected"]]
    if len(selected_rows) > 0:
        selected = selected_rows.iloc[0]
        selected_margin = float(selected["margin"])
        selected_margin_pct = float(selected["margin_pct"]) if pd.notna(selected["margin_pct"]) else np.nan
        selected_revenue = float(selected["revenue"])
        selected_cost = float(selected["cost"])
    else:
        selected_margin = 0.0
        selected_margin_pct = np.nan
        selected_revenue = 0.0
        selected_cost = 0.0

    peer_df = rollup[~rollup["is_selected"]].copy()
    peer_positive = peer_df["margin"].clip(lower=0)
    positive_pool = float(peer_positive.sum())
    selected_loss_abs = float(max(-selected_margin, 0.0))
    coverage_ratio = (
        float(positive_pool / selected_loss_abs) if selected_loss_abs > 0 else np.nan
    )
    buffer_after_subsidy = float(positive_pool - selected_loss_abs)
    subsidizer_job_count = int((peer_df["margin"] > 0).sum())

    if positive_pool > 0:
        top_subsidizer = float(peer_positive.max())
        subsidy_concentration_pct = float(top_subsidizer / positive_pool * 100)
    else:
        subsidy_concentration_pct = np.nan

    if selected_loss_abs == 0:
        verdict = "No Subsidy Needed"
    elif positive_pool <= 0:
        verdict = "Not Subsidized"
    elif coverage_ratio < 0.5:
        verdict = "Weak Subsidy"
    elif coverage_ratio < 1.0:
        verdict = "Partially Subsidized"
    else:
        verdict = "Fully Subsidized"

    summary = {
        "selected_margin": selected_margin,
        "selected_margin_pct": selected_margin_pct,
        "selected_revenue": selected_revenue,
        "selected_cost": selected_cost,
        "group_revenue": group_revenue,
        "group_cost": group_cost,
        "group_margin": group_margin,
        "group_margin_pct": group_margin_pct,
        "job_count": int(len(rollup)),
        "active_job_count": int(rollup["job_no"].isin(active_set).sum()),
        "loss_job_count": int((rollup["margin"] < 0).sum()),
        "red_job_count": int((rollup["risk_band"] == "Red").sum()),
        "amber_job_count": int((rollup["risk_band"] == "Amber").sum()),
        "selected_loss_abs": selected_loss_abs,
        "positive_peer_margin_pool": positive_pool,
        "coverage_ratio": coverage_ratio,
        "buffer_after_subsidy": buffer_after_subsidy,
        "subsidizer_job_count": subsidizer_job_count,
        "subsidy_concentration_pct": subsidy_concentration_pct,
        "verdict": verdict,
    }

    jobs_out = rollup[RESULT_JOB_COLUMNS].copy()

    result["status"] = "ok"
    result["summary"] = summary
    result["jobs"] = jobs_out
    return result
