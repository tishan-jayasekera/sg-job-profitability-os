"""
Service load and capacity analytics for user-defined service scopes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.config import config
from src.data.semantic import get_category_col


WEEKS_PER_MONTH = 4.33


def _empty_job_index() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["job_no", "job_name", "job_description", "search_text"]
    )


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return np.nan
    return non_null.iloc[0]


def _to_job_set(values: Iterable) -> set[str]:
    if values is None:
        return set()
    series = pd.Series(list(values))
    return set(series.dropna().astype(str))


def _aligned_scope_mask(df: pd.DataFrame, scope_mask: pd.Series | np.ndarray | list) -> pd.Series:
    if isinstance(scope_mask, pd.Series):
        return scope_mask.reindex(df.index).fillna(False).astype(bool)
    return pd.Series(scope_mask, index=df.index).fillna(False).astype(bool)


def _prepare_month_key(df: pd.DataFrame) -> pd.DataFrame:
    df_work = df.copy()
    if "month_key" not in df_work.columns and "work_date" in df_work.columns:
        df_work["month_key"] = pd.to_datetime(df_work["work_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    elif "month_key" in df_work.columns:
        df_work["month_key"] = pd.to_datetime(df_work["month_key"], errors="coerce")
    return df_work


def _resolve_reference_period(df: pd.DataFrame, reference_date=None) -> Optional[pd.Period]:
    if reference_date is not None:
        ref_ts = pd.to_datetime(reference_date, errors="coerce")
        if pd.notna(ref_ts):
            return ref_ts.to_period("M") - 1

    if "month_key" in df.columns:
        month_ts = pd.to_datetime(df["month_key"], errors="coerce")
        if month_ts.notna().any():
            # month_key is usually month-level data, so include max month by default
            return month_ts.max().to_period("M")

    if "work_date" in df.columns:
        work_ts = pd.to_datetime(df["work_date"], errors="coerce")
        if work_ts.notna().any():
            return work_ts.max().to_period("M") - 1

    return None


def _filter_to_lookback_window(
    df: pd.DataFrame,
    lookback_months: int = 3,
    reference_date=None,
) -> Tuple[pd.DataFrame, int]:
    if lookback_months <= 0:
        return df.iloc[0:0].copy(), 0

    df_work = _prepare_month_key(df)
    if "month_key" not in df_work.columns:
        return df_work.iloc[0:0].copy(), 0

    month_period = pd.to_datetime(df_work["month_key"], errors="coerce").dt.to_period("M")
    valid_periods = month_period.dropna()
    if len(valid_periods) == 0:
        return df_work.iloc[0:0].copy(), 0

    end_period = _resolve_reference_period(df_work, reference_date=reference_date)
    if end_period is None:
        end_period = valid_periods.max()

    start_period = end_period - (lookback_months - 1)
    mask = (month_period >= start_period) & (month_period <= end_period)
    filtered = df_work[mask].copy()

    if len(filtered) == 0:
        available = sorted(valid_periods.unique())
        if len(available) == 0:
            return df_work.iloc[0:0].copy(), 0
        chosen = available[-lookback_months:]
        filtered = df_work[month_period.isin(chosen)].copy()

    if len(filtered) == 0:
        return filtered, 0

    months_in_window = int(
        pd.to_datetime(filtered["month_key"], errors="coerce").dt.to_period("M").nunique()
    )
    return filtered, months_in_window


def _mode_or_nan(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return np.nan
    mode = clean.mode(dropna=True)
    if len(mode) == 0:
        return np.nan
    return float(mode.iloc[0])


@st.cache_data(show_spinner=False)
def build_job_description_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a deduplicated job-level search index from job_no + text fields.
    """
    if "job_no" not in df.columns:
        return _empty_job_index()

    idx = df.copy()
    if "job_name" not in idx.columns:
        idx["job_name"] = np.nan
    if "job_description" not in idx.columns:
        idx["job_description"] = np.nan

    job_index = (
        idx.groupby("job_no", dropna=False)
        .agg(
            job_name=("job_name", _first_non_null),
            job_description=("job_description", _first_non_null),
        )
        .reset_index()
    )

    name_text = job_index["job_name"].fillna("").astype(str)
    desc_text = job_index["job_description"].fillna("").astype(str)
    job_index["search_text"] = (name_text + " | " + desc_text).str.lower()

    return job_index[["job_no", "job_name", "job_description", "search_text"]]


@st.cache_data(show_spinner=False)
def define_service_scope(
    df: pd.DataFrame,
    job_index_df: pd.DataFrame,
    description_keywords: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    clients: Optional[list[str]] = None,
    task_names: Optional[list[str]] = None,
    job_nos: Optional[list[str]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Define scope rows using keyword search (primary) with optional refinement filters.
    """
    if "job_no" not in df.columns:
        return pd.Series(False, index=df.index), pd.DataFrame(
            columns=["job_no", "job_name", "job_description", "match_reason"]
        )

    keywords = [str(k).strip().lower() for k in (description_keywords or []) if str(k).strip()]
    categories = [c for c in (categories or []) if str(c).strip()]
    clients = [c for c in (clients or []) if str(c).strip()]
    task_names = [t for t in (task_names or []) if str(t).strip()]
    explicit_jobs = [j for j in (job_nos or []) if str(j).strip()]

    keyword_jobs: set[str] = set()
    secondary_jobs: set[str] = set()
    reasons: defaultdict[str, set[str]] = defaultdict(set)

    # Primary: description/name keyword matching (OR logic).
    if len(keywords) > 0 and len(job_index_df) > 0:
        search_text = job_index_df.get("search_text", pd.Series("", index=job_index_df.index)).fillna("").astype(str)
        for keyword in keywords:
            kw_mask = search_text.str.contains(keyword, na=False, regex=False)
            matched = _to_job_set(job_index_df.loc[kw_mask, "job_no"])
            keyword_jobs |= matched
            for job in matched:
                reasons[job].add(f"keyword: {keyword}")

    # Secondary: category.
    if len(categories) > 0:
        category_col = get_category_col(df)
        if category_col in df.columns:
            cat_df = df[df[category_col].isin(categories)][["job_no", category_col]].dropna(subset=["job_no"])
            matched = _to_job_set(cat_df["job_no"])
            secondary_jobs |= matched
            for _, row in cat_df.drop_duplicates(subset=["job_no", category_col]).iterrows():
                reasons[str(row["job_no"])].add(f"category: {row[category_col]}")

    # Secondary: client.
    if len(clients) > 0 and "client" in df.columns:
        cli_df = df[df["client"].isin(clients)][["job_no", "client"]].dropna(subset=["job_no"])
        matched = _to_job_set(cli_df["job_no"])
        secondary_jobs |= matched
        for _, row in cli_df.drop_duplicates(subset=["job_no", "client"]).iterrows():
            reasons[str(row["job_no"])].add(f"client: {row['client']}")

    # Secondary: task.
    if len(task_names) > 0 and "task_name" in df.columns:
        task_df = df[df["task_name"].isin(task_names)][["job_no", "task_name"]].dropna(subset=["job_no"])
        matched = _to_job_set(task_df["job_no"])
        secondary_jobs |= matched
        for _, row in task_df.drop_duplicates(subset=["job_no", "task_name"]).iterrows():
            reasons[str(row["job_no"])].add(f"task: {row['task_name']}")

    # Secondary: explicit job numbers.
    if len(explicit_jobs) > 0:
        matched = _to_job_set(explicit_jobs)
        secondary_jobs |= matched
        for job in matched:
            reasons[job].add("explicit")

    has_keywords = len(keywords) > 0
    has_secondary = (
        (len(categories) > 0)
        or (len(clients) > 0)
        or (len(task_names) > 0)
        or (len(explicit_jobs) > 0)
    )

    if has_keywords and has_secondary:
        matched_jobs = keyword_jobs & secondary_jobs
    elif has_keywords:
        matched_jobs = keyword_jobs
    elif has_secondary:
        matched_jobs = secondary_jobs
    else:
        scope_mask = pd.Series(True, index=df.index)
        all_jobs = _to_job_set(df["job_no"])
        out = job_index_df.copy() if len(job_index_df) > 0 else pd.DataFrame({"job_no": sorted(all_jobs)})
        if "job_name" not in out.columns:
            out["job_name"] = np.nan
        if "job_description" not in out.columns:
            out["job_description"] = np.nan
        out = out[out["job_no"].astype(str).isin(all_jobs)].copy()
        out["match_reason"] = "all jobs"
        out = out[["job_no", "job_name", "job_description", "match_reason"]]
        return scope_mask, out.sort_values("job_no")

    scope_mask = df["job_no"].notna() & df["job_no"].astype(str).isin(matched_jobs)

    if len(job_index_df) > 0:
        matched_jobs_df = job_index_df[job_index_df["job_no"].astype(str).isin(matched_jobs)].copy()
    else:
        matched_jobs_df = pd.DataFrame({"job_no": sorted(matched_jobs)})
        matched_jobs_df["job_name"] = np.nan
        matched_jobs_df["job_description"] = np.nan

    if len(matched_jobs_df) > 0:
        matched_jobs_df["match_reason"] = matched_jobs_df["job_no"].astype(str).map(
            lambda j: ", ".join(sorted(reasons.get(j, set())))
        )
    else:
        matched_jobs_df["match_reason"] = []

    matched_jobs_df = matched_jobs_df[
        ["job_no", "job_name", "job_description", "match_reason"]
    ].sort_values("job_no")

    return scope_mask, matched_jobs_df


@st.cache_data(show_spinner=False)
def compute_staff_service_load(
    df: pd.DataFrame,
    scope_mask: pd.Series,
    lookback_months: int = 3,
    reference_date=None,
) -> pd.DataFrame:
    """
    Compute in-scope/out-of-scope service load and capacity headroom per staff member.
    """
    columns = [
        "staff_name",
        "total_hours",
        "in_scope_hours",
        "out_of_scope_hours",
        "in_scope_hours_per_month",
        "in_scope_hours_per_week",
        "out_of_scope_hours_per_month",
        "total_hours_per_month",
        "fte_scaling",
        "monthly_capacity",
        "capacity_used_pct",
        "scope_share_pct",
        "headroom_hours_per_month",
        "headroom_hours_per_week",
        "billable_in_scope_pct",
        "months_in_window",
        "fte_assumed",
    ]
    if "staff_name" not in df.columns or "hours_raw" not in df.columns:
        return pd.DataFrame(columns=columns)

    df_work = df.copy()
    df_work["_scope_match"] = _aligned_scope_mask(df_work, scope_mask)
    df_window, months_in_window = _filter_to_lookback_window(
        df_work, lookback_months=lookback_months, reference_date=reference_date
    )
    if len(df_window) == 0 or months_in_window == 0:
        return pd.DataFrame(columns=columns)

    df_window = df_window[df_window["staff_name"].notna()].copy()
    if len(df_window) == 0:
        return pd.DataFrame(columns=columns)

    df_window["hours_raw"] = pd.to_numeric(df_window["hours_raw"], errors="coerce").fillna(0.0)
    if "is_billable" in df_window.columns:
        df_window["is_billable"] = df_window["is_billable"].fillna(False).astype(bool)
    else:
        df_window["is_billable"] = False

    total_hours = df_window.groupby("staff_name")["hours_raw"].sum()
    in_scope_hours = (
        df_window[df_window["_scope_match"]]
        .groupby("staff_name")["hours_raw"]
        .sum()
    )
    billable_in_scope = (
        df_window[df_window["_scope_match"] & df_window["is_billable"]]
        .groupby("staff_name")["hours_raw"]
        .sum()
    )

    result = pd.DataFrame(index=total_hours.index)
    result["total_hours"] = total_hours
    result["in_scope_hours"] = in_scope_hours.reindex(result.index, fill_value=0.0)
    result["out_of_scope_hours"] = result["total_hours"] - result["in_scope_hours"]

    if "fte_hours_scaling" in df_window.columns:
        fte_mode = df_window.groupby("staff_name")["fte_hours_scaling"].agg(_mode_or_nan)
    else:
        fte_mode = pd.Series(np.nan, index=result.index)

    result["fte_scaling"] = fte_mode.reindex(result.index)
    result["fte_assumed"] = result["fte_scaling"].isna()
    result["fte_scaling"] = result["fte_scaling"].fillna(config.DEFAULT_FTE_SCALING)

    result["in_scope_hours_per_month"] = result["in_scope_hours"] / months_in_window
    result["in_scope_hours_per_week"] = result["in_scope_hours_per_month"] / WEEKS_PER_MONTH
    result["out_of_scope_hours_per_month"] = result["out_of_scope_hours"] / months_in_window
    result["total_hours_per_month"] = result["total_hours"] / months_in_window

    result["monthly_capacity"] = config.CAPACITY_HOURS_PER_WEEK * result["fte_scaling"] * WEEKS_PER_MONTH
    result["capacity_used_pct"] = np.where(
        result["monthly_capacity"] > 0,
        result["total_hours_per_month"] / result["monthly_capacity"] * 100,
        np.nan,
    )
    result["scope_share_pct"] = np.where(
        result["total_hours"] > 0,
        result["in_scope_hours"] / result["total_hours"] * 100,
        0.0,
    )
    result["headroom_hours_per_month"] = result["monthly_capacity"] - result["total_hours_per_month"]
    result["headroom_hours_per_week"] = result["headroom_hours_per_month"] / WEEKS_PER_MONTH
    result["billable_in_scope_pct"] = np.where(
        result["in_scope_hours"] > 0,
        billable_in_scope.reindex(result.index, fill_value=0.0) / result["in_scope_hours"] * 100,
        0.0,
    )
    result["months_in_window"] = months_in_window

    result = result[result["total_hours"] > 0].copy()
    result = result.reset_index().rename(columns={"index": "staff_name"})
    result = result.sort_values("in_scope_hours", ascending=False)

    return result[columns]


@st.cache_data(show_spinner=False)
def compute_staff_client_breakdown(
    df: pd.DataFrame,
    staff_name: str,
    scope_mask: pd.Series,
    lookback_months: int = 3,
) -> pd.DataFrame:
    """
    Break down a staff member's hours by client and job, split by scope.
    """
    columns = [
        "client",
        "job_no",
        "job_name",
        "job_category",
        "total_hours",
        "in_scope_hours",
        "out_of_scope_hours",
        "hours_per_month",
        "share_of_staff_total_pct",
        "is_in_scope",
    ]
    if "staff_name" not in df.columns or "hours_raw" not in df.columns or "job_no" not in df.columns:
        return pd.DataFrame(columns=columns)

    df_work = df.copy()
    df_work["_scope_match"] = _aligned_scope_mask(df_work, scope_mask)
    df_window, months_in_window = _filter_to_lookback_window(df_work, lookback_months=lookback_months)
    if len(df_window) == 0 or months_in_window == 0:
        return pd.DataFrame(columns=columns)

    staff_df = df_window[df_window["staff_name"] == staff_name].copy()
    if len(staff_df) == 0:
        return pd.DataFrame(columns=columns)

    staff_df["hours_raw"] = pd.to_numeric(staff_df["hours_raw"], errors="coerce").fillna(0.0)
    if "client" not in staff_df.columns:
        staff_df["client"] = "Unknown"
    if "job_name" not in staff_df.columns:
        staff_df["job_name"] = np.nan
    category_col = get_category_col(staff_df)
    if category_col not in staff_df.columns:
        staff_df[category_col] = np.nan

    keys = ["client", "job_no"]
    grouped = staff_df.groupby(keys, dropna=False)
    total = grouped["hours_raw"].sum().rename("total_hours")
    in_scope = (
        staff_df[staff_df["_scope_match"]]
        .groupby(keys, dropna=False)["hours_raw"]
        .sum()
        .rename("in_scope_hours")
    )
    attrs = grouped.agg(
        job_name=("job_name", _first_non_null),
        job_category=(category_col, _first_non_null),
    )

    result = pd.concat([attrs, total], axis=1).reset_index()
    result = result.merge(
        in_scope.reset_index(),
        on=keys,
        how="left",
    )
    result["in_scope_hours"] = result["in_scope_hours"].fillna(0.0)
    result["out_of_scope_hours"] = result["total_hours"] - result["in_scope_hours"]
    result["hours_per_month"] = result["total_hours"] / months_in_window

    staff_total = float(result["total_hours"].sum())
    result["share_of_staff_total_pct"] = np.where(
        staff_total > 0,
        result["total_hours"] / staff_total * 100,
        0.0,
    )
    result["is_in_scope"] = result["in_scope_hours"] > 0

    result = result.sort_values("total_hours", ascending=False)
    return result[columns]


@st.cache_data(show_spinner=False)
def compute_staff_task_breakdown(
    df: pd.DataFrame,
    staff_name: str,
    job_no: Optional[str] = None,
    scope_mask: Optional[pd.Series] = None,
    lookback_months: int = 3,
) -> pd.DataFrame:
    """
    Break down a staff member's hours by task, optionally constrained to a job.
    """
    columns = ["task_name", "total_hours", "hours_per_month", "cost", "share_pct"]
    if "staff_name" not in df.columns or "hours_raw" not in df.columns:
        return pd.DataFrame(columns=columns)

    df_work = df.copy()
    if scope_mask is not None:
        df_work["_scope_match"] = _aligned_scope_mask(df_work, scope_mask)
    else:
        df_work["_scope_match"] = True

    df_window, months_in_window = _filter_to_lookback_window(df_work, lookback_months=lookback_months)
    if len(df_window) == 0 or months_in_window == 0:
        return pd.DataFrame(columns=columns)

    task_df = df_window[df_window["staff_name"] == staff_name].copy()
    if job_no is not None and "job_no" in task_df.columns:
        task_df = task_df[task_df["job_no"].astype(str) == str(job_no)].copy()
    if len(task_df) == 0:
        return pd.DataFrame(columns=columns)

    if "task_name" not in task_df.columns:
        task_df["task_name"] = "Unspecified"
    if "base_cost" not in task_df.columns:
        task_df["base_cost"] = 0.0

    task_df["hours_raw"] = pd.to_numeric(task_df["hours_raw"], errors="coerce").fillna(0.0)
    task_df["base_cost"] = pd.to_numeric(task_df["base_cost"], errors="coerce").fillna(0.0)

    result = task_df.groupby("task_name", dropna=False).agg(
        total_hours=("hours_raw", "sum"),
        cost=("base_cost", "sum"),
    ).reset_index()

    total_hours = float(result["total_hours"].sum())
    result["hours_per_month"] = result["total_hours"] / months_in_window
    result["share_pct"] = np.where(
        total_hours > 0,
        result["total_hours"] / total_hours * 100,
        0.0,
    )

    result = result.sort_values("total_hours", ascending=False)
    return result[columns]


@st.cache_data(show_spinner=False)
def compute_scope_budget_comparison(
    staff_load_df: pd.DataFrame,
    scope_budget_hours_per_month: float,
) -> pd.DataFrame:
    """
    Compare each staff member's in-scope monthly load against a scope budget.
    """
    result = staff_load_df.copy()
    if len(result) == 0:
        result["scope_budget"] = []
        result["scope_budget_used_pct"] = []
        result["scope_budget_remaining"] = []
        result["scope_status"] = []
        return result

    budget = float(scope_budget_hours_per_month)
    result["scope_budget"] = budget

    if budget <= 0:
        result["scope_budget_used_pct"] = np.nan
        result["scope_budget_remaining"] = np.nan
        result["scope_status"] = "N/A"
        return result

    result["scope_budget_used_pct"] = (
        pd.to_numeric(result["in_scope_hours_per_month"], errors="coerce") / budget * 100
    )
    result["scope_budget_remaining"] = budget - pd.to_numeric(
        result["in_scope_hours_per_month"], errors="coerce"
    )

    used_pct = result["scope_budget_used_pct"]
    result["scope_status"] = np.select(
        [
            used_pct < 80,
            (used_pct >= 80) & (used_pct <= 100),
            (used_pct > 100) & (used_pct <= 120),
            used_pct > 120,
        ],
        ["Under Budget", "On Track", "Over Budget", "Way Over"],
        default="On Track",
    )

    return result


@st.cache_data(show_spinner=False)
def compute_new_client_absorption(
    staff_load_df: pd.DataFrame,
    avg_hours_per_new_client_per_month: float,
) -> pd.DataFrame:
    """
    Estimate how many additional new clients each staff member can absorb.
    """
    result = staff_load_df.copy()
    if len(result) == 0:
        result["additional_clients_headroom"] = []
        result["additional_clients_budget"] = []
        result["absorption_estimate"] = []
        result["overload_hours_if_one_more"] = []
        return result

    avg_hours = float(avg_hours_per_new_client_per_month)
    if avg_hours <= 0:
        result["additional_clients_headroom"] = 0
        result["additional_clients_budget"] = np.nan
        result["absorption_estimate"] = 0
        result["overload_hours_if_one_more"] = np.nan
        return result

    headroom = pd.to_numeric(result.get("headroom_hours_per_month"), errors="coerce").fillna(0.0)
    by_headroom = np.floor(headroom / avg_hours).astype(int)
    by_headroom = by_headroom.clip(lower=0)
    result["additional_clients_headroom"] = by_headroom

    if "scope_budget_remaining" in result.columns and result["scope_budget_remaining"].notna().any():
        budget_remaining = pd.to_numeric(result["scope_budget_remaining"], errors="coerce").fillna(0.0)
        by_budget = np.floor(budget_remaining / avg_hours).astype(int).clip(lower=0)
        result["additional_clients_budget"] = by_budget
        result["absorption_estimate"] = np.minimum(by_headroom, by_budget).clip(lower=0)
    else:
        result["additional_clients_budget"] = np.nan
        result["absorption_estimate"] = by_headroom

    total_hours_month = pd.to_numeric(result.get("total_hours_per_month"), errors="coerce").fillna(0.0)
    monthly_capacity = pd.to_numeric(result.get("monthly_capacity"), errors="coerce").fillna(0.0)
    result["overload_hours_if_one_more"] = total_hours_month + avg_hours - monthly_capacity

    return result


@st.cache_data(show_spinner=False)
def compute_scope_weekly_trend(
    df: pd.DataFrame,
    scope_mask: pd.Series,
    staff_names: Optional[list[str]] = None,
    lookback_months: int = 3,
) -> pd.DataFrame:
    """
    Return in-scope hours by period and staff (weekly where work_date exists).
    """
    if "hours_raw" not in df.columns or "staff_name" not in df.columns:
        return pd.DataFrame()

    df_work = df.copy()
    df_work["_scope_match"] = _aligned_scope_mask(df_work, scope_mask)
    df_window, _ = _filter_to_lookback_window(df_work, lookback_months=lookback_months)
    if len(df_window) == 0:
        return pd.DataFrame()

    scoped = df_window[df_window["_scope_match"]].copy()
    if len(scoped) == 0:
        return pd.DataFrame()

    if staff_names:
        staff_set = set(staff_names)
        scoped = scoped[scoped["staff_name"].isin(staff_set)].copy()
        if len(scoped) == 0:
            return pd.DataFrame()

    scoped["hours_raw"] = pd.to_numeric(scoped["hours_raw"], errors="coerce").fillna(0.0)

    if "work_date" in scoped.columns and pd.to_datetime(scoped["work_date"], errors="coerce").notna().any():
        scoped["work_date"] = pd.to_datetime(scoped["work_date"], errors="coerce")
        scoped = scoped[scoped["work_date"].notna()].copy()
        if len(scoped) == 0:
            return pd.DataFrame()
        scoped["period"] = scoped["work_date"].dt.to_period("W").dt.start_time
    elif "month_key" in scoped.columns:
        scoped["month_key"] = pd.to_datetime(scoped["month_key"], errors="coerce")
        scoped = scoped[scoped["month_key"].notna()].copy()
        if len(scoped) == 0:
            return pd.DataFrame()
        scoped["period"] = scoped["month_key"].dt.to_period("M").dt.to_timestamp()
    else:
        return pd.DataFrame()

    trend = scoped.pivot_table(
        index="period",
        columns="staff_name",
        values="hours_raw",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    trend.index.name = "period"
    return trend


@st.cache_data(show_spinner=False)
def compute_scope_task_breakdown(
    df: pd.DataFrame,
    scope_mask: pd.Series,
    lookback_months: int = 3,
) -> pd.DataFrame:
    """
    Aggregate scoped hours by task to show where service capacity is being consumed.
    """
    columns = [
        "task_name",
        "total_hours",
        "hours_per_month",
        "share_of_scope_pct",
        "staff_count",
        "job_count",
        "client_count",
        "billable_pct",
        "months_in_window",
    ]
    if "hours_raw" not in df.columns:
        return pd.DataFrame(columns=columns)

    df_work = df.copy()
    df_work["_scope_match"] = _aligned_scope_mask(df_work, scope_mask)
    df_window, months_in_window = _filter_to_lookback_window(df_work, lookback_months=lookback_months)
    if len(df_window) == 0 or months_in_window == 0:
        return pd.DataFrame(columns=columns)

    scoped = df_window[df_window["_scope_match"]].copy()
    if len(scoped) == 0:
        return pd.DataFrame(columns=columns)

    if "task_name" not in scoped.columns:
        scoped["task_name"] = "Unspecified"
    scoped["task_name"] = scoped["task_name"].fillna("Unspecified")
    scoped["hours_raw"] = pd.to_numeric(scoped["hours_raw"], errors="coerce").fillna(0.0)

    if "is_billable" not in scoped.columns:
        scoped["is_billable"] = False
    scoped["is_billable"] = scoped["is_billable"].fillna(False).astype(bool)
    scoped["billable_hours"] = np.where(scoped["is_billable"], scoped["hours_raw"], 0.0)

    grouped = scoped.groupby("task_name", dropna=False)
    result = grouped.agg(
        total_hours=("hours_raw", "sum"),
        billable_hours=("billable_hours", "sum"),
    ).reset_index()

    if "staff_name" in scoped.columns:
        result = result.merge(
            grouped["staff_name"].nunique().rename("staff_count").reset_index(),
            on="task_name",
            how="left",
        )
    else:
        result["staff_count"] = 0

    if "job_no" in scoped.columns:
        result = result.merge(
            grouped["job_no"].nunique().rename("job_count").reset_index(),
            on="task_name",
            how="left",
        )
    else:
        result["job_count"] = 0

    if "client" in scoped.columns:
        result = result.merge(
            grouped["client"].nunique().rename("client_count").reset_index(),
            on="task_name",
            how="left",
        )
    else:
        result["client_count"] = 0

    total_scope_hours = float(result["total_hours"].sum())
    result["hours_per_month"] = result["total_hours"] / months_in_window
    result["share_of_scope_pct"] = np.where(
        total_scope_hours > 0,
        result["total_hours"] / total_scope_hours * 100,
        0.0,
    )
    result["billable_pct"] = np.where(
        result["total_hours"] > 0,
        result["billable_hours"] / result["total_hours"] * 100,
        0.0,
    )
    result["months_in_window"] = months_in_window

    result = result.drop(columns=["billable_hours"], errors="ignore")
    result = result.sort_values("total_hours", ascending=False)
    return result[columns]


@st.cache_data(show_spinner=False)
def compute_staff_task_capacity_flow(
    df: pd.DataFrame,
    staff_name: str | list[str],
    scope_mask: pd.Series,
    lookback_months: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute scoped task flow for one or more staff members and link each task back to constituent jobs.

    Returns:
        task_summary_df: one row per task with contribution and top-driving job metadata.
        task_job_df: task x job detail for constituent linkage.
    """
    summary_cols = [
        "task_name",
        "total_hours",
        "hours_per_month",
        "share_of_staff_scope_pct",
        "job_count",
        "client_count",
        "top_job_no",
        "top_job_name",
        "top_job_client",
        "top_job_share_of_task_pct",
    ]
    detail_cols = [
        "task_name",
        "job_no",
        "job_name",
        "client",
        "total_hours",
        "hours_per_month",
        "share_of_task_pct",
        "share_of_job_pct",
        "share_of_staff_scope_pct",
    ]

    if "staff_name" not in df.columns or "hours_raw" not in df.columns:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=detail_cols)

    df_work = df.copy()
    df_work["_scope_match"] = _aligned_scope_mask(df_work, scope_mask)
    df_window, months_in_window = _filter_to_lookback_window(df_work, lookback_months=lookback_months)
    if len(df_window) == 0 or months_in_window == 0:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=detail_cols)

    selected_staff = [staff_name] if isinstance(staff_name, str) else list(staff_name or [])
    selected_staff = [str(s) for s in selected_staff if str(s).strip()]
    if len(selected_staff) == 0:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=detail_cols)

    staff_df = df_window[
        df_window["staff_name"].astype(str).isin(selected_staff) & (df_window["_scope_match"])
    ].copy()
    if len(staff_df) == 0:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=detail_cols)

    if "task_name" not in staff_df.columns:
        staff_df["task_name"] = "Unspecified"
    staff_df["task_name"] = staff_df["task_name"].fillna("Unspecified")
    if "job_no" not in staff_df.columns:
        staff_df["job_no"] = np.nan
    if "job_name" not in staff_df.columns:
        staff_df["job_name"] = np.nan
    if "client" not in staff_df.columns:
        staff_df["client"] = "Unknown"

    staff_df["hours_raw"] = pd.to_numeric(staff_df["hours_raw"], errors="coerce").fillna(0.0)
    total_staff_scope_hours = float(staff_df["hours_raw"].sum())
    if total_staff_scope_hours <= 0:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=detail_cols)

    task_summary = (
        staff_df.groupby("task_name", dropna=False)["hours_raw"]
        .sum()
        .rename("total_hours")
        .reset_index()
    )
    task_summary["hours_per_month"] = task_summary["total_hours"] / months_in_window
    task_summary["share_of_staff_scope_pct"] = np.where(
        total_staff_scope_hours > 0,
        task_summary["total_hours"] / total_staff_scope_hours * 100,
        0.0,
    )

    task_summary = task_summary.merge(
        staff_df.groupby("task_name", dropna=False)["job_no"].nunique().rename("job_count").reset_index(),
        on="task_name",
        how="left",
    )
    task_summary = task_summary.merge(
        staff_df.groupby("task_name", dropna=False)["client"].nunique().rename("client_count").reset_index(),
        on="task_name",
        how="left",
    )

    task_job_df = (
        staff_df.groupby(["task_name", "job_no", "job_name", "client"], dropna=False)["hours_raw"]
        .sum()
        .rename("total_hours")
        .reset_index()
    )
    task_job_df["hours_per_month"] = task_job_df["total_hours"] / months_in_window

    task_totals = task_job_df.groupby("task_name", dropna=False)["total_hours"].sum().rename("task_total")
    task_job_df = task_job_df.merge(task_totals.reset_index(), on="task_name", how="left")
    task_job_df["share_of_task_pct"] = np.where(
        task_job_df["task_total"] > 0,
        task_job_df["total_hours"] / task_job_df["task_total"] * 100,
        0.0,
    )
    task_job_df["share_of_staff_scope_pct"] = np.where(
        total_staff_scope_hours > 0,
        task_job_df["total_hours"] / total_staff_scope_hours * 100,
        0.0,
    )

    job_totals = task_job_df.groupby("job_no", dropna=False)["total_hours"].sum().rename("job_total")
    task_job_df = task_job_df.merge(job_totals.reset_index(), on="job_no", how="left")
    task_job_df["share_of_job_pct"] = np.where(
        task_job_df["job_total"] > 0,
        task_job_df["total_hours"] / task_job_df["job_total"] * 100,
        0.0,
    )

    top_driver = task_job_df.sort_values(["task_name", "total_hours"], ascending=[True, False]).drop_duplicates(
        subset=["task_name"]
    )
    top_driver = top_driver.rename(
        columns={
            "job_no": "top_job_no",
            "job_name": "top_job_name",
            "client": "top_job_client",
            "share_of_task_pct": "top_job_share_of_task_pct",
        }
    )[
        ["task_name", "top_job_no", "top_job_name", "top_job_client", "top_job_share_of_task_pct"]
    ]

    task_summary = task_summary.merge(top_driver, on="task_name", how="left")
    task_summary = task_summary.sort_values("total_hours", ascending=False)
    task_job_df = task_job_df.sort_values(["task_name", "total_hours"], ascending=[True, False])

    task_job_df = task_job_df.drop(columns=["task_total", "job_total"], errors="ignore")
    return task_summary[summary_cols], task_job_df[detail_cols]
