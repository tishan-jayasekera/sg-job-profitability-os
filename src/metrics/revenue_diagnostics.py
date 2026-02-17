"""Revenue decline diagnostics metrics."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.data.job_lifecycle import get_job_first_activity
from src.data.semantic import (
    exclude_leave,
    get_category_col,
    safe_quote_job_task,
    safe_quote_rollup,
)

SERVICE_LINE_DEFAULT_MAPPING: dict[str, list[str]] = {
    "Marketing Automation": [
        "marketing automation",
        "ma",
        "mkt automation",
        "mktg automation",
    ],
    "CRM": ["crm", "customer relationship management"],
    "Landing Pages": ["landing pages", "landing page", "lp"],
}

MONTHLY_COLUMNS = [
    "month_key",
    "service_line",
    "jobs",
    "revenue",
    "hours",
    "cost",
    "margin",
    "margin_pct",
    "avg_rev_per_job",
    "avg_hours_per_job",
    "realised_rate",
    "active_clients",
    "active_staff",
]

YOY_COLUMNS = [
    "service_line",
    "month_curr",
    "month_prev",
    "revenue_curr",
    "revenue_prev",
    "jobs_curr",
    "jobs_prev",
    "arpj_curr",
    "arpj_prev",
    "hours_curr",
    "hours_prev",
    "rev_yoy_pct",
    "jobs_yoy_pct",
    "arpj_yoy_pct",
    "hours_yoy_pct",
    "delta_revenue",
]

DECOMP_COLUMNS = [
    "service_line",
    "delta_revenue",
    "volume_effect",
    "price_effect",
    "interaction_effect",
    "check_total",
]

DEAL_COLUMNS = ["job_no", "service_line", "quoted_amount", "actual_revenue", "cohort_month"]

STAFF_SELL_COLUMNS = [
    "service_line",
    "month_key",
    "active_staff_count",
    "staff_continuity_pct",
    "top3_revenue_share_pct",
    "total_revenue",
    "total_hours",
]

TASK_MIX_COLUMNS = ["task_name", "prior_share_pct", "current_share_pct", "divergence_pp"]

RATE_TREND_COLUMNS = [
    "month_key",
    "service_line",
    "quote_rate",
    "realised_rate",
    "rate_gap",
    "rate_capture_pct",
]

SCORECARD_COLUMNS = [
    "hypothesis",
    "signal_strength",
    "evidence_1",
    "evidence_2",
    "interpretation",
]

BUNDLE_KEYS = [
    "monthly",
    "yoy",
    "decomp",
    "client_bridge",
    "reputation",
    "personnel",
    "market",
    "deal_sizes_curr",
    "deal_sizes_prev",
    "staffing",
    "task_mix",
    "rate_trend",
    "scorecard",
]

SIGNAL_STRONG = "🔴 Strong"
SIGNAL_MODERATE = "🟡 Moderate"
SIGNAL_WEAK = "🟢 Weak"
SIGNAL_INSUFFICIENT = "⚪ Insufficient Data"


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _normalize_month_series(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    return ts.dt.to_period("M").dt.to_timestamp()


def _normalize_month_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "month_key" in out.columns:
        out["month_key"] = _normalize_month_series(out["month_key"])
    return out


def _normalize_month_value(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts.to_period("M").to_timestamp()


def _normalize_months(months: Iterable) -> list[pd.Timestamp]:
    values = list(months or [])
    if len(values) == 0:
        return []
    ts = pd.to_datetime(pd.Series(values), errors="coerce").dropna()
    if len(ts) == 0:
        return []
    return sorted(ts.dt.to_period("M").dt.to_timestamp().drop_duplicates().tolist())


def _month_slice(df: pd.DataFrame, months: Iterable) -> pd.DataFrame:
    month_list = _normalize_months(months)
    if len(df) == 0 or "month_key" not in df.columns or len(month_list) == 0:
        return df.iloc[0:0].copy()
    return df[df["month_key"].isin(month_list)].copy()


def _prepare_fact(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = _normalize_month_key(df)
    out = exclude_leave(out).copy()
    for col in ["rev_alloc", "hours_raw", "base_cost"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _has_any_client_column(df: pd.DataFrame) -> bool:
    for col in ["client_group_rev_job_month", "client_group_rev_job", "client_group", "client"]:
        if col in df.columns and df[col].notna().any():
            return True
    return False


def _filter_service_line(df: pd.DataFrame, service_line: str) -> pd.DataFrame:
    out = df.copy()
    if "service_line" not in out.columns:
        return out.iloc[0:0].copy()
    if service_line in ("", "All", None):
        return out
    return out[out["service_line"] == service_line].copy()


def _safe_pct_change(curr: float, prev: float) -> float:
    if prev == 0:
        return np.nan
    return (curr - prev) / prev * 100


def _jobs_agg(group: pd.DataFrame) -> int:
    if "job_no" in group.columns:
        return int(group["job_no"].nunique())
    return int(len(group))


def _job_count(df: pd.DataFrame) -> int:
    if "job_no" in df.columns:
        return int(df["job_no"].nunique())
    return int(len(df))


def _get_revenue_by_client(df: pd.DataFrame, client_col: str) -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype=float)
    work = df.copy()
    work[client_col] = work[client_col].fillna("UNKNOWN_CLIENT").astype(str)
    return work.groupby(client_col, dropna=False)["rev_alloc"].sum()


def _score_drop_metric(value: float) -> str:
    if pd.isna(value):
        return SIGNAL_INSUFFICIENT
    if value < -30:
        return SIGNAL_STRONG
    if value < -10:
        return SIGNAL_MODERATE
    return SIGNAL_WEAK


def _score_mix_metric(value: float) -> str:
    if pd.isna(value):
        return SIGNAL_INSUFFICIENT
    if value > 20:
        return SIGNAL_STRONG
    if value > 10:
        return SIGNAL_MODERATE
    return SIGNAL_WEAK


def _dict_error(reason: Exception | str) -> dict[str, Any]:
    return {"status": "error", "reason": str(reason)}


@st.cache_data(show_spinner=False)
def get_client_col(df: pd.DataFrame) -> str:
    """
    Resolve preferred client column.

    Preference: client_group_rev_job_month -> client_group_rev_job -> client_group -> client
    """
    if df is None or len(df) == 0:
        return "client"

    for col in ["client_group_rev_job_month", "client_group_rev_job", "client_group", "client"]:
        if col in df.columns and df[col].notna().any():
            return col

    return "client"


@st.cache_data(show_spinner=False)
def normalize_service_line_labels(
    df: pd.DataFrame,
    category_col: str,
    mapping: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Add service_line column from category labels using case-insensitive synonym mapping."""
    if df is None or len(df) == 0:
        out = pd.DataFrame(columns=list(df.columns) + ["service_line"] if df is not None else ["service_line"])
        return out

    out = df.copy()
    local_map = mapping or SERVICE_LINE_DEFAULT_MAPPING

    if category_col not in out.columns:
        out[category_col] = np.nan

    raw = out[category_col].fillna("Unmapped").astype(str)
    norm = raw.str.strip()
    norm_lower = norm.str.lower()

    synonym_to_target: dict[str, str] = {}
    for target, synonyms in local_map.items():
        for synonym in synonyms:
            synonym_to_target[str(synonym).strip().lower()] = target

    out["service_line"] = norm
    mapped_values = norm_lower.map(synonym_to_target)
    out.loc[mapped_values.notna(), "service_line"] = mapped_values[mapped_values.notna()].values

    return out


@st.cache_data(show_spinner=False)
def compute_service_line_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly service-line rollup with exact contract columns."""
    if df is None or len(df) == 0:
        return _empty_df(MONTHLY_COLUMNS)

    work = _prepare_fact(df)
    if len(work) == 0 or "month_key" not in work.columns:
        return _empty_df(MONTHLY_COLUMNS)

    category_col = get_category_col(work)
    work = normalize_service_line_labels(work, category_col)

    client_col = get_client_col(work)
    if client_col not in work.columns:
        work[client_col] = "UNKNOWN_CLIENT"
    work[client_col] = work[client_col].fillna("UNKNOWN_CLIENT").astype(str)

    group_keys = ["month_key", "service_line"]
    grouped = work.groupby(group_keys, dropna=False)

    summary = grouped.agg(
        revenue=("rev_alloc", "sum"),
        hours=("hours_raw", "sum"),
        cost=("base_cost", "sum"),
        active_clients=(client_col, "nunique"),
    ).reset_index()

    if "job_no" in work.columns:
        jobs = grouped["job_no"].nunique().reset_index(name="jobs")
    else:
        jobs = grouped.size().reset_index(name="jobs")
    summary = summary.merge(jobs, on=group_keys, how="left")

    if "staff_name" in work.columns:
        active_staff = grouped["staff_name"].nunique().reset_index(name="active_staff")
        summary = summary.merge(active_staff, on=group_keys, how="left")
    else:
        summary["active_staff"] = np.nan

    summary["jobs"] = pd.to_numeric(summary["jobs"], errors="coerce").fillna(0)
    summary["margin"] = summary["revenue"] - summary["cost"]
    summary["margin_pct"] = np.where(summary["revenue"] != 0, summary["margin"] / summary["revenue"], 0)
    summary["avg_rev_per_job"] = np.where(summary["jobs"] != 0, summary["revenue"] / summary["jobs"], 0)
    summary["avg_hours_per_job"] = np.where(summary["jobs"] != 0, summary["hours"] / summary["jobs"], 0)
    summary["realised_rate"] = np.where(summary["hours"] != 0, summary["revenue"] / summary["hours"], 0)

    for col in MONTHLY_COLUMNS:
        if col not in summary.columns:
            summary[col] = np.nan

    summary = summary[MONTHLY_COLUMNS].sort_values(["service_line", "month_key"]).reset_index(drop=True)
    return summary


@st.cache_data(show_spinner=False)
def compute_yoy_snapshot(monthly_df: pd.DataFrame, as_of_month: pd.Timestamp) -> pd.DataFrame:
    """Compare service-line KPIs against prior-year month."""
    if monthly_df is None or len(monthly_df) == 0:
        return _empty_df(YOY_COLUMNS)

    work = monthly_df.copy()
    if "month_key" not in work.columns or "service_line" not in work.columns:
        return _empty_df(YOY_COLUMNS)

    work["month_key"] = _normalize_month_series(work["month_key"])
    month_curr = _normalize_month_value(as_of_month)
    if pd.isna(month_curr):
        return _empty_df(YOY_COLUMNS)

    month_prev = _normalize_month_value(month_curr - pd.DateOffset(years=1))

    curr = work[work["month_key"] == month_curr].copy()
    prev = work[work["month_key"] == month_prev].copy()

    curr = curr.set_index("service_line")
    prev = prev.set_index("service_line")

    service_lines = sorted(set(curr.index.tolist()) | set(prev.index.tolist()))
    rows: list[dict[str, Any]] = []

    for service_line in service_lines:
        curr_row = curr.loc[service_line] if service_line in curr.index else None
        prev_row = prev.loc[service_line] if service_line in prev.index else None

        revenue_curr = float(curr_row["revenue"]) if curr_row is not None and "revenue" in curr_row else 0.0
        revenue_prev = float(prev_row["revenue"]) if prev_row is not None and "revenue" in prev_row else 0.0
        jobs_curr = float(curr_row["jobs"]) if curr_row is not None and "jobs" in curr_row else 0.0
        jobs_prev = float(prev_row["jobs"]) if prev_row is not None and "jobs" in prev_row else 0.0
        arpj_curr = float(curr_row["avg_rev_per_job"]) if curr_row is not None and "avg_rev_per_job" in curr_row else 0.0
        arpj_prev = float(prev_row["avg_rev_per_job"]) if prev_row is not None and "avg_rev_per_job" in prev_row else 0.0
        hours_curr = float(curr_row["hours"]) if curr_row is not None and "hours" in curr_row else 0.0
        hours_prev = float(prev_row["hours"]) if prev_row is not None and "hours" in prev_row else 0.0

        rev_yoy_pct = _safe_pct_change(revenue_curr, revenue_prev)
        jobs_yoy_pct = _safe_pct_change(jobs_curr, jobs_prev)
        arpj_yoy_pct = _safe_pct_change(arpj_curr, arpj_prev)
        hours_yoy_pct = _safe_pct_change(hours_curr, hours_prev)

        rows.append(
            {
                "service_line": service_line,
                "month_curr": month_curr,
                "month_prev": month_prev,
                "revenue_curr": revenue_curr,
                "revenue_prev": revenue_prev,
                "jobs_curr": jobs_curr,
                "jobs_prev": jobs_prev,
                "arpj_curr": arpj_curr,
                "arpj_prev": arpj_prev,
                "hours_curr": hours_curr,
                "hours_prev": hours_prev,
                "rev_yoy_pct": rev_yoy_pct,
                "jobs_yoy_pct": jobs_yoy_pct,
                "arpj_yoy_pct": arpj_yoy_pct,
                "hours_yoy_pct": hours_yoy_pct,
                "delta_revenue": revenue_curr - revenue_prev,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return _empty_df(YOY_COLUMNS)

    return out[YOY_COLUMNS].sort_values("service_line").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def decompose_revenue_change(yoy_df: pd.DataFrame) -> pd.DataFrame:
    """Decompose revenue delta into volume, price, and interaction effects."""
    if yoy_df is None or len(yoy_df) == 0:
        return _empty_df(DECOMP_COLUMNS)

    required = {"service_line", "delta_revenue", "jobs_curr", "jobs_prev", "arpj_curr", "arpj_prev"}
    if not required.issubset(yoy_df.columns):
        return _empty_df(DECOMP_COLUMNS)

    out = yoy_df.copy()
    out["volume_effect"] = (out["jobs_curr"] - out["jobs_prev"]) * out["arpj_prev"]
    out["price_effect"] = out["jobs_prev"] * (out["arpj_curr"] - out["arpj_prev"])
    out["interaction_effect"] = (out["jobs_curr"] - out["jobs_prev"]) * (out["arpj_curr"] - out["arpj_prev"])
    out["check_total"] = out["volume_effect"] + out["price_effect"] + out["interaction_effect"]

    diff = (out["check_total"] - out["delta_revenue"]).abs()
    if len(diff) > 0 and float(diff.max()) >= 1e-6:
        raise ValueError("Revenue decomposition identity check failed")

    return out[DECOMP_COLUMNS].sort_values("service_line").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_client_bridge(
    df: pd.DataFrame,
    service_line: str,
    current_months: list,
    prior_months: list,
) -> dict:
    """Build retained/new/lost client bridge for a service line."""
    bridge_cols = ["bridge_component", "amount", "description"]
    top_cols = ["client", "delta_revenue", "curr_revenue", "prev_revenue", "status"]

    if df is None or len(df) == 0:
        return {"bridge": _empty_df(bridge_cols), "top_clients": _empty_df(top_cols)}

    work = _prepare_fact(df)
    category_col = get_category_col(work)
    work = normalize_service_line_labels(work, category_col)
    work = _filter_service_line(work, service_line)

    if len(work) == 0:
        return {"bridge": _empty_df(bridge_cols), "top_clients": _empty_df(top_cols)}

    if not _has_any_client_column(work):
        no_client = pd.DataFrame(
            [{"bridge_component": "No Client Data", "amount": 0.0, "description": "client column not available"}],
            columns=bridge_cols,
        )
        return {"bridge": no_client, "top_clients": _empty_df(top_cols)}

    client_col = get_client_col(work)
    if client_col not in work.columns:
        no_client = pd.DataFrame(
            [{"bridge_component": "No Client Data", "amount": 0.0, "description": "client column not available"}],
            columns=bridge_cols,
        )
        return {"bridge": no_client, "top_clients": _empty_df(top_cols)}

    work[client_col] = work[client_col].fillna("UNKNOWN_CLIENT").astype(str)

    curr_df = _month_slice(work, current_months)
    prev_df = _month_slice(work, prior_months)

    curr_rev = _get_revenue_by_client(curr_df, client_col)
    prev_rev = _get_revenue_by_client(prev_df, client_col)

    curr_clients = set(curr_rev.index.tolist())
    prev_clients = set(prev_rev.index.tolist())

    retained = curr_clients & prev_clients
    new_clients = curr_clients - prev_clients
    lost_clients = prev_clients - curr_clients

    retained_growth = float(curr_rev.loc[list(retained)].sum() - prev_rev.loc[list(retained)].sum()) if len(retained) > 0 else 0.0
    new_revenue = float(curr_rev.loc[list(new_clients)].sum()) if len(new_clients) > 0 else 0.0
    lost_revenue = float(prev_rev.loc[list(lost_clients)].sum()) if len(lost_clients) > 0 else 0.0

    total_delta = retained_growth + new_revenue - lost_revenue

    bridge = pd.DataFrame(
        [
            {
                "bridge_component": "Retained Growth",
                "amount": retained_growth,
                "description": "Revenue change from clients active in both periods",
            },
            {
                "bridge_component": "New Clients",
                "amount": new_revenue,
                "description": "Revenue contribution from newly acquired clients",
            },
            {
                "bridge_component": "Lost Clients",
                "amount": -lost_revenue,
                "description": "Revenue lost from prior-period clients no longer active",
            },
            {
                "bridge_component": "Total Delta",
                "amount": total_delta,
                "description": "Net current minus prior period revenue",
            },
        ],
        columns=bridge_cols,
    )

    total_curr = float(curr_rev.sum())
    total_prev = float(prev_rev.sum())
    if abs(total_delta - (total_curr - total_prev)) >= 1e-6:
        raise ValueError("Client bridge consistency check failed")

    all_clients = sorted(curr_clients | prev_clients)
    top_rows: list[dict[str, Any]] = []
    for client in all_clients:
        curr_val = float(curr_rev.get(client, 0.0))
        prev_val = float(prev_rev.get(client, 0.0))
        if client in retained:
            status = "Retained"
        elif client in new_clients:
            status = "New"
        else:
            status = "Lost"

        top_rows.append(
            {
                "client": client,
                "delta_revenue": curr_val - prev_val,
                "curr_revenue": curr_val,
                "prev_revenue": prev_val,
                "status": status,
            }
        )

    top_clients = pd.DataFrame(top_rows, columns=top_cols)
    if len(top_clients) > 0:
        top_clients = top_clients.assign(abs_delta=top_clients["delta_revenue"].abs())
        top_clients = top_clients.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"]).head(10)

    return {"bridge": bridge, "top_clients": top_clients[top_cols] if len(top_clients) > 0 else _empty_df(top_cols)}


@st.cache_data(show_spinner=False)
def compute_reputation_signals(
    df: pd.DataFrame,
    service_line: str,
    current_months: list,
    prior_months: list,
    repeat_window_months: int = 6,
) -> dict:
    """Assess reputation/retention signal via repeat share and client losses."""
    base = {
        "repeat_share_curr": 0.0,
        "repeat_share_prev": 0.0,
        "repeat_share_delta_pp": 0.0,
        "lost_client_count": 0,
        "lost_client_revenue": 0.0,
        "retained_client_revenue_delta": 0.0,
        "status": "insufficient_data",
        "reason": "insufficient data",
    }

    if df is None or len(df) == 0:
        return base

    work = _prepare_fact(df)
    category_col = get_category_col(work)
    work = normalize_service_line_labels(work, category_col)
    work = _filter_service_line(work, service_line)

    if len(work) == 0:
        base["reason"] = "no rows for selected service line"
        return base

    if not _has_any_client_column(work):
        base["reason"] = "client column missing"
        return base

    client_col = get_client_col(work)
    if client_col not in work.columns:
        base["reason"] = "client column missing"
        return base

    work[client_col] = work[client_col].fillna("UNKNOWN_CLIENT").astype(str)

    curr_df = _month_slice(work, current_months)
    prev_df = _month_slice(work, prior_months)

    curr_clients = set(curr_df[client_col].dropna().astype(str).unique().tolist()) if len(curr_df) > 0 else set()
    prev_clients = set(prev_df[client_col].dropna().astype(str).unique().tolist()) if len(prev_df) > 0 else set()

    if len(curr_clients) < 5 or len(prev_clients) < 5:
        base["reason"] = "fewer than 5 clients in current or prior period"
        return base

    curr_rev = _get_revenue_by_client(curr_df, client_col)
    prev_rev = _get_revenue_by_client(prev_df, client_col)

    curr_start = min(_normalize_months(current_months)) if len(_normalize_months(current_months)) > 0 else pd.NaT
    prev_start = min(_normalize_months(prior_months)) if len(_normalize_months(prior_months)) > 0 else pd.NaT

    if pd.isna(curr_start) or pd.isna(prev_start):
        base["reason"] = "missing period months"
        return base

    curr_lookback_start = _normalize_month_value(curr_start - pd.DateOffset(months=repeat_window_months))
    prev_lookback_start = _normalize_month_value(prev_start - pd.DateOffset(months=repeat_window_months))

    curr_lookback = work[(work["month_key"] >= curr_lookback_start) & (work["month_key"] < curr_start)].copy()
    prev_lookback = work[(work["month_key"] >= prev_lookback_start) & (work["month_key"] < prev_start)].copy()

    curr_repeat_clients = curr_clients & set(curr_lookback[client_col].dropna().astype(str).unique().tolist())
    prev_repeat_clients = prev_clients & set(prev_lookback[client_col].dropna().astype(str).unique().tolist())

    curr_total = float(curr_rev.sum())
    prev_total = float(prev_rev.sum())

    repeat_curr_rev = float(curr_rev.loc[list(curr_repeat_clients)].sum()) if len(curr_repeat_clients) > 0 else 0.0
    repeat_prev_rev = float(prev_rev.loc[list(prev_repeat_clients)].sum()) if len(prev_repeat_clients) > 0 else 0.0

    repeat_share_curr = np.where(curr_total != 0, repeat_curr_rev / curr_total * 100, 0.0)
    repeat_share_prev = np.where(prev_total != 0, repeat_prev_rev / prev_total * 100, 0.0)

    lost_clients = prev_clients - curr_clients
    retained_clients = prev_clients & curr_clients

    lost_client_revenue = float(prev_rev.loc[list(lost_clients)].sum()) if len(lost_clients) > 0 else 0.0
    retained_prev = float(prev_rev.loc[list(retained_clients)].sum()) if len(retained_clients) > 0 else 0.0
    retained_curr = float(curr_rev.loc[list(retained_clients)].sum()) if len(retained_clients) > 0 else 0.0
    retained_delta = retained_curr - retained_prev

    delta_revenue = curr_total - prev_total
    repeat_delta_pp = float(repeat_share_curr - repeat_share_prev)

    supported = repeat_delta_pp <= -10 and lost_client_revenue >= 0.25 * abs(delta_revenue)
    status = "supported" if supported else "not_supported"
    reason = (
        "repeat share declined >=10pp and lost-client revenue >=25% of total delta"
        if supported
        else "thresholds not met"
    )

    return {
        "repeat_share_curr": float(repeat_share_curr),
        "repeat_share_prev": float(repeat_share_prev),
        "repeat_share_delta_pp": repeat_delta_pp,
        "lost_client_count": int(len(lost_clients)),
        "lost_client_revenue": float(lost_client_revenue),
        "retained_client_revenue_delta": float(retained_delta),
        "status": status,
        "reason": reason,
    }


@st.cache_data(show_spinner=False)
def compute_personnel_signals(
    df: pd.DataFrame,
    service_line: str,
    current_months: list,
    prior_months: list,
) -> dict:
    """Assess staffing/selling signal from commercial headcount and load."""
    base = {
        "staff_curr": 0,
        "staff_prev": 0,
        "staff_yoy_pct": 0.0,
        "jobs_per_staff_curr": 0.0,
        "jobs_per_staff_prev": 0.0,
        "revenue_per_staff_curr": 0.0,
        "revenue_per_staff_prev": 0.0,
        "proxy_mode": False,
        "status": "insufficient_data",
        "reason": "insufficient data",
    }

    if df is None or len(df) == 0:
        return base

    work = _prepare_fact(df)
    category_col = get_category_col(work)
    work = normalize_service_line_labels(work, category_col)
    work = _filter_service_line(work, service_line)

    if "staff_name" not in work.columns:
        base["reason"] = "staff_name column missing"
        return base

    curr_df = _month_slice(work, current_months)
    prev_df = _month_slice(work, prior_months)

    proxy_mode = False
    commercial_re = re.compile(r"(?:sales|bdm|business development|account manager|growth)", flags=re.IGNORECASE)

    if "role" in work.columns or "function" in work.columns:
        mask = pd.Series(False, index=work.index)
        if "role" in work.columns:
            mask |= work["role"].fillna("").astype(str).str.contains(commercial_re)
        if "function" in work.columns:
            mask |= work["function"].fillna("").astype(str).str.contains(commercial_re)

        if mask.any():
            work_commercial = work[mask].copy()
            curr_df = _month_slice(work_commercial, current_months)
            prev_df = _month_slice(work_commercial, prior_months)
        else:
            proxy_mode = True
    else:
        proxy_mode = True

    staff_curr = int(curr_df["staff_name"].dropna().nunique())
    staff_prev = int(prev_df["staff_name"].dropna().nunique())

    if staff_curr == 0 or staff_prev == 0:
        base["proxy_mode"] = proxy_mode
        base["reason"] = "zero staff in current or prior period"
        return base

    jobs_curr = _job_count(curr_df)
    jobs_prev = _job_count(prev_df)
    revenue_curr = float(curr_df["rev_alloc"].sum())
    revenue_prev = float(prev_df["rev_alloc"].sum())

    jobs_per_staff_curr = np.where(staff_curr != 0, jobs_curr / staff_curr, 0.0)
    jobs_per_staff_prev = np.where(staff_prev != 0, jobs_prev / staff_prev, 0.0)
    revenue_per_staff_curr = np.where(staff_curr != 0, revenue_curr / staff_curr, 0.0)
    revenue_per_staff_prev = np.where(staff_prev != 0, revenue_prev / staff_prev, 0.0)
    staff_yoy_pct = np.where(staff_prev != 0, (staff_curr - staff_prev) / staff_prev * 100, 0.0)

    supported = float(staff_yoy_pct) <= -20 and float(jobs_per_staff_curr) > float(jobs_per_staff_prev)

    return {
        "staff_curr": staff_curr,
        "staff_prev": staff_prev,
        "staff_yoy_pct": float(staff_yoy_pct),
        "jobs_per_staff_curr": float(jobs_per_staff_curr),
        "jobs_per_staff_prev": float(jobs_per_staff_prev),
        "revenue_per_staff_curr": float(revenue_per_staff_curr),
        "revenue_per_staff_prev": float(revenue_per_staff_prev),
        "proxy_mode": proxy_mode,
        "status": "supported" if supported else "not_supported",
        "reason": (
            "staff down >=20% and jobs per staff increased"
            if supported
            else "thresholds not met"
        ),
    }


@st.cache_data(show_spinner=False)
def compute_market_signals(
    df: pd.DataFrame,
    service_line: str,
    current_months: list,
    prior_months: list,
) -> dict:
    """Assess market concentration of decline by state."""
    empty_state = _empty_df(["state", "prev_revenue", "curr_revenue", "delta"])

    if df is None or len(df) == 0:
        return {
            "states_declining": 0,
            "states_total": 0,
            "negative_state_share": 0.0,
            "top2_decline_contribution": 0.0,
            "state_breakdown": empty_state,
            "status": "insufficient_data",
            "reason": "no input rows",
        }

    work = _prepare_fact(df)
    category_col = get_category_col(work)
    work = normalize_service_line_labels(work, category_col)
    work = _filter_service_line(work, service_line)

    if "state" not in work.columns:
        return {
            "states_declining": 0,
            "states_total": 0,
            "negative_state_share": 0.0,
            "top2_decline_contribution": 0.0,
            "state_breakdown": empty_state,
            "status": "insufficient_data",
            "reason": "state column missing",
        }

    work = work.copy()
    work["state"] = work["state"].fillna("UNKNOWN_STATE").astype(str)

    if work["state"].nunique() == 0:
        return {
            "states_declining": 0,
            "states_total": 0,
            "negative_state_share": 0.0,
            "top2_decline_contribution": 0.0,
            "state_breakdown": empty_state,
            "status": "insufficient_data",
            "reason": "state not present in data",
        }

    curr_df = _month_slice(work, current_months)
    prev_df = _month_slice(work, prior_months)

    curr = curr_df.groupby("state", dropna=False)["rev_alloc"].sum().rename("curr_revenue")
    prev = prev_df.groupby("state", dropna=False)["rev_alloc"].sum().rename("prev_revenue")

    breakdown = pd.concat([prev, curr], axis=1).fillna(0).reset_index()
    breakdown["delta"] = breakdown["curr_revenue"] - breakdown["prev_revenue"]
    breakdown = breakdown[["state", "prev_revenue", "curr_revenue", "delta"]]

    states_total = int(len(breakdown))
    states_declining = int((breakdown["delta"] < 0).sum())
    negative_state_share = np.where(states_total != 0, states_declining / states_total * 100, 0.0)

    declines = breakdown[breakdown["delta"] < 0].copy()
    total_decline_abs = float((-declines["delta"]).sum()) if len(declines) > 0 else 0.0
    top2_decline_abs = float((-declines.nsmallest(2, "delta")["delta"]).sum()) if len(declines) > 0 else 0.0
    top2_contribution = np.where(total_decline_abs != 0, top2_decline_abs / total_decline_abs * 100, 0.0)

    supported = float(negative_state_share) >= 60 or float(top2_contribution) >= 50

    return {
        "states_declining": states_declining,
        "states_total": states_total,
        "negative_state_share": float(negative_state_share),
        "top2_decline_contribution": float(top2_contribution),
        "state_breakdown": breakdown.sort_values("delta"),
        "status": "supported" if supported else "not_supported",
        "reason": (
            "decline is broad-based or concentrated in top-2 states"
            if supported
            else "state decline concentration thresholds not met"
        ),
    }


@st.cache_data(show_spinner=False)
def compute_deal_size_distribution(
    df: pd.DataFrame,
    service_line: str,
    period_months: list,
) -> pd.DataFrame:
    """Compute per-job quoted and actual revenue distribution for the period."""
    if df is None or len(df) == 0:
        return _empty_df(DEAL_COLUMNS)

    work = _prepare_fact(df)
    category_col = get_category_col(work)
    work = normalize_service_line_labels(work, category_col)
    work = _filter_service_line(work, service_line)

    if "job_no" not in work.columns:
        return _empty_df(DEAL_COLUMNS)

    scoped = _month_slice(work, period_months)
    if len(scoped) == 0:
        return _empty_df(DEAL_COLUMNS)

    quotes_job_task = safe_quote_job_task(scoped)
    if len(quotes_job_task) > 0 and "quoted_amount_total" in quotes_job_task.columns:
        quotes = (
            quotes_job_task.groupby("job_no", dropna=False)["quoted_amount_total"]
            .sum()
            .reset_index(name="quoted_amount")
        )
    else:
        quotes = pd.DataFrame({"job_no": scoped["job_no"].dropna().unique(), "quoted_amount": 0.0})

    actual = scoped.groupby("job_no", dropna=False)["rev_alloc"].sum().reset_index(name="actual_revenue")

    first_activity = get_job_first_activity(scoped)
    if len(first_activity) > 0:
        if "first_activity_month" in first_activity.columns:
            cohort = first_activity[["job_no", "first_activity_month"]].rename(columns={"first_activity_month": "cohort_month"})
        elif "first_activity_date" in first_activity.columns:
            cohort = first_activity[["job_no", "first_activity_date"]].rename(columns={"first_activity_date": "cohort_month"})
        else:
            cohort = pd.DataFrame(columns=["job_no", "cohort_month"])
    else:
        cohort = pd.DataFrame(columns=["job_no", "cohort_month"])

    out = actual.merge(quotes, on="job_no", how="left").merge(cohort, on="job_no", how="left")
    out["quoted_amount"] = pd.to_numeric(out["quoted_amount"], errors="coerce").fillna(0.0)
    out["actual_revenue"] = pd.to_numeric(out["actual_revenue"], errors="coerce").fillna(0.0)
    out["cohort_month"] = _normalize_month_series(pd.Series(out["cohort_month"]))

    if service_line in ("", "All", None):
        out["service_line"] = "All"
    else:
        out["service_line"] = service_line

    for col in DEAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out[DEAL_COLUMNS].sort_values(["cohort_month", "job_no"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_staff_selling_analysis(df: pd.DataFrame, service_line: str) -> pd.DataFrame:
    """Compute staffing continuity metrics by service line and month."""
    if df is None or len(df) == 0:
        return _empty_df(STAFF_SELL_COLUMNS)

    work = _prepare_fact(df)
    category_col = get_category_col(work)
    work = normalize_service_line_labels(work, category_col)
    work = _filter_service_line(work, service_line)

    if "staff_name" not in work.columns:
        return _empty_df(STAFF_SELL_COLUMNS)

    if len(work) == 0:
        return _empty_df(STAFF_SELL_COLUMNS)

    staff_month = (
        work.groupby(["service_line", "month_key", "staff_name"], dropna=False)
        .agg(hours=("hours_raw", "sum"), revenue=("rev_alloc", "sum"))
        .reset_index()
    )

    if len(staff_month) == 0:
        return _empty_df(STAFF_SELL_COLUMNS)

    totals = (
        staff_month.groupby(["service_line", "month_key"], dropna=False)
        .agg(total_revenue=("revenue", "sum"), total_hours=("hours", "sum"))
        .reset_index()
    )

    active_staff = (
        staff_month[staff_month["hours"] > 0]
        .groupby(["service_line", "month_key"], dropna=False)["staff_name"]
        .nunique()
        .reset_index(name="active_staff_count")
    )

    top3 = (
        staff_month.sort_values(["service_line", "month_key", "revenue"], ascending=[True, True, False])
        .groupby(["service_line", "month_key"], dropna=False)
        .head(3)
    )
    top3 = top3.groupby(["service_line", "month_key"], dropna=False)["revenue"].sum().reset_index(name="top3_revenue")

    out = totals.merge(active_staff, on=["service_line", "month_key"], how="left")
    out = out.merge(top3, on=["service_line", "month_key"], how="left")
    out["top3_revenue_share_pct"] = np.where(out["total_revenue"] != 0, out["top3_revenue"] / out["total_revenue"] * 100, 0)

    continuity_rows: list[dict[str, Any]] = []
    active_sets = (
        staff_month[staff_month["hours"] > 0]
        .groupby(["service_line", "month_key"], dropna=False)["staff_name"]
        .apply(lambda x: set(x.dropna().astype(str).tolist()))
        .reset_index(name="staff_set")
    )

    for line, group in active_sets.groupby("service_line", dropna=False):
        group = group.sort_values("month_key")
        prev_set: Optional[set[str]] = None
        for _, row in group.iterrows():
            cur_set = row["staff_set"] if isinstance(row["staff_set"], set) else set()
            if prev_set is None:
                continuity = np.nan
            else:
                continuity = np.where(len(cur_set) != 0, len(cur_set & prev_set) / len(cur_set) * 100, np.nan)
            continuity_rows.append(
                {
                    "service_line": line,
                    "month_key": row["month_key"],
                    "staff_continuity_pct": float(continuity) if not pd.isna(continuity) else np.nan,
                }
            )
            prev_set = cur_set

    continuity_df = pd.DataFrame(continuity_rows)
    out = out.merge(continuity_df, on=["service_line", "month_key"], how="left")

    out = out.rename(columns={"total_revenue": "total_revenue", "total_hours": "total_hours"})

    for col in STAFF_SELL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out[STAFF_SELL_COLUMNS].sort_values(["service_line", "month_key"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_task_mix_shift(
    df: pd.DataFrame,
    service_line: str,
    current_months: list,
    prior_months: list,
) -> pd.DataFrame:
    """Compute task mix share divergence between prior and current periods."""
    if df is None or len(df) == 0:
        return _empty_df(TASK_MIX_COLUMNS)

    work = _prepare_fact(df)
    category_col = get_category_col(work)
    work = normalize_service_line_labels(work, category_col)
    work = _filter_service_line(work, service_line)

    if "task_name" not in work.columns:
        return _empty_df(TASK_MIX_COLUMNS)

    prior_df = _month_slice(work, prior_months)
    curr_df = _month_slice(work, current_months)

    prior_hours = prior_df.groupby("task_name", dropna=False)["hours_raw"].sum().reset_index(name="prior_hours")
    curr_hours = curr_df.groupby("task_name", dropna=False)["hours_raw"].sum().reset_index(name="current_hours")

    mix = prior_hours.merge(curr_hours, on="task_name", how="outer").fillna(0)

    prior_total = float(mix["prior_hours"].sum())
    curr_total = float(mix["current_hours"].sum())

    mix["prior_share_pct"] = np.where(prior_total != 0, mix["prior_hours"] / prior_total * 100, 0)
    mix["current_share_pct"] = np.where(curr_total != 0, mix["current_hours"] / curr_total * 100, 0)
    mix["divergence_pp"] = mix["current_share_pct"] - mix["prior_share_pct"]

    out = mix[TASK_MIX_COLUMNS].copy()
    out = out.sort_values("divergence_pp", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def compute_quoted_vs_actual_rate_trend(df: pd.DataFrame, service_line: str) -> pd.DataFrame:
    """Compute monthly quote vs realised rate trend by service line."""
    if df is None or len(df) == 0:
        return _empty_df(RATE_TREND_COLUMNS)

    work = _prepare_fact(df)
    category_col = get_category_col(work)
    work = normalize_service_line_labels(work, category_col)
    work = _filter_service_line(work, service_line)

    if len(work) == 0 or "month_key" not in work.columns:
        return _empty_df(RATE_TREND_COLUMNS)

    group_keys = ("month_key", "service_line")

    actual = (
        work.groupby(list(group_keys), dropna=False)
        .agg(revenue=("rev_alloc", "sum"), hours=("hours_raw", "sum"))
        .reset_index()
    )

    quote = safe_quote_rollup(work, group_keys)
    if len(quote) > 0:
        quote_subset = quote[[c for c in ["month_key", "service_line", "quote_rate"] if c in quote.columns]].copy()
    else:
        quote_subset = _empty_df(["month_key", "service_line", "quote_rate"])

    out = actual.merge(quote_subset, on=["month_key", "service_line"], how="left")
    out["quote_rate"] = pd.to_numeric(out["quote_rate"], errors="coerce").fillna(0)
    out["realised_rate"] = np.where(out["hours"] != 0, out["revenue"] / out["hours"], 0)
    out["rate_gap"] = out["realised_rate"] - out["quote_rate"]
    out["rate_capture_pct"] = np.where(out["quote_rate"] != 0, out["realised_rate"] / out["quote_rate"] * 100, np.nan)

    for col in RATE_TREND_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out[RATE_TREND_COLUMNS].sort_values(["service_line", "month_key"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_hypothesis_scorecard(
    yoy_df: pd.DataFrame,
    churn_result: dict,
    reputation: dict,
    personnel: dict,
    market: dict,
) -> pd.DataFrame:
    """Build 5-hypothesis signal scorecard from computed diagnostics."""
    demand_signal = SIGNAL_INSUFFICIENT
    price_signal = SIGNAL_INSUFFICIENT
    demand_ev1 = "No YoY data"
    demand_ev2 = ""
    price_ev1 = "No YoY data"
    price_ev2 = ""

    if yoy_df is not None and len(yoy_df) > 0:
        work = yoy_df.copy()
        jobs_curr = float(work["jobs_curr"].sum()) if "jobs_curr" in work.columns else 0.0
        jobs_prev = float(work["jobs_prev"].sum()) if "jobs_prev" in work.columns else 0.0
        revenue_curr = float(work["revenue_curr"].sum()) if "revenue_curr" in work.columns else 0.0
        revenue_prev = float(work["revenue_prev"].sum()) if "revenue_prev" in work.columns else 0.0

        jobs_yoy_pct = _safe_pct_change(jobs_curr, jobs_prev)
        arpj_curr = np.where(jobs_curr != 0, revenue_curr / jobs_curr, 0.0)
        arpj_prev = np.where(jobs_prev != 0, revenue_prev / jobs_prev, 0.0)
        arpj_yoy_pct = _safe_pct_change(float(arpj_curr), float(arpj_prev))

        demand_signal = _score_drop_metric(jobs_yoy_pct)
        price_signal = _score_drop_metric(arpj_yoy_pct)
        demand_ev1 = f"Jobs YoY: {jobs_yoy_pct:.1f}%"
        demand_ev2 = f"Market breadth signal: {market.get('negative_state_share', 0):.1f}% states declining"
        price_ev1 = f"Avg revenue/job YoY: {arpj_yoy_pct:.1f}%"
        price_ev2 = f"Current ARPJ: {float(arpj_curr):,.0f}"

    rep_status = reputation.get("status", "insufficient_data") if isinstance(reputation, dict) else "insufficient_data"
    if rep_status == "insufficient_data":
        churn_signal = SIGNAL_INSUFFICIENT
    elif rep_status == "supported":
        churn_signal = SIGNAL_STRONG
    elif rep_status == "not_supported" and float(reputation.get("repeat_share_delta_pp", 0.0)) < -5:
        churn_signal = SIGNAL_MODERATE
    else:
        churn_signal = SIGNAL_WEAK

    per_status = personnel.get("status", "insufficient_data") if isinstance(personnel, dict) else "insufficient_data"
    if per_status == "insufficient_data":
        staffing_signal = SIGNAL_INSUFFICIENT
    elif per_status == "supported":
        staffing_signal = SIGNAL_STRONG
    elif float(personnel.get("staff_yoy_pct", 0.0)) < -10:
        staffing_signal = SIGNAL_MODERATE
    else:
        staffing_signal = SIGNAL_WEAK

    pct_small_jobs_delta_pp = np.nan
    if isinstance(churn_result, dict):
        pct_small_jobs_delta_pp = churn_result.get("pct_small_jobs_delta_pp", np.nan)
    mix_signal = _score_mix_metric(float(pct_small_jobs_delta_pp)) if not pd.isna(pct_small_jobs_delta_pp) else SIGNAL_INSUFFICIENT

    rows = [
        {
            "hypothesis": "Demand decline",
            "signal_strength": demand_signal,
            "evidence_1": demand_ev1,
            "evidence_2": demand_ev2,
            "interpretation": "Lower job intake suggests weaker demand if persistent across months.",
        },
        {
            "hypothesis": "Price erosion",
            "signal_strength": price_signal,
            "evidence_1": price_ev1,
            "evidence_2": price_ev2,
            "interpretation": "Revenue per job compression indicates shrinking deal size or discounting.",
        },
        {
            "hypothesis": "Client churn (reputation)",
            "signal_strength": churn_signal,
            "evidence_1": f"Repeat share delta: {float(reputation.get('repeat_share_delta_pp', 0.0)):.1f}pp",
            "evidence_2": f"Lost client revenue: {float(reputation.get('lost_client_revenue', 0.0)):,.0f}",
            "interpretation": "Repeat-share and lost-client shifts indicate whether loyalty is weakening.",
        },
        {
            "hypothesis": "Staffing/selling changes",
            "signal_strength": staffing_signal,
            "evidence_1": f"Staff YoY: {float(personnel.get('staff_yoy_pct', 0.0)):.1f}%",
            "evidence_2": f"Jobs/staff: {float(personnel.get('jobs_per_staff_prev', 0.0)):.2f} -> {float(personnel.get('jobs_per_staff_curr', 0.0)):.2f}",
            "interpretation": "Fewer commercial staff with higher workload can suppress growth.",
        },
        {
            "hypothesis": "Mix shift to low-value",
            "signal_strength": mix_signal,
            "evidence_1": (
                f"Small jobs (<$1k) delta: {float(pct_small_jobs_delta_pp):.1f}pp"
                if not pd.isna(pct_small_jobs_delta_pp)
                else "Small-job mix unavailable"
            ),
            "evidence_2": f"Market concentration: {float(market.get('top2_decline_contribution', 0.0)):.1f}%",
            "interpretation": "Rising small-job share points to lower-value work mix.",
        },
    ]

    return pd.DataFrame(rows, columns=SCORECARD_COLUMNS)


@st.cache_data(show_spinner=False)
def build_diagnostics_bundle(
    df: pd.DataFrame,
    as_of_month: pd.Timestamp,
    service_line: str,
    current_months: list,
    prior_months: list,
) -> dict:
    """Orchestrate all diagnostics outputs for page consumption."""
    bundle: dict[str, Any] = {
        "monthly": _empty_df(MONTHLY_COLUMNS),
        "yoy": _empty_df(YOY_COLUMNS),
        "decomp": _empty_df(DECOMP_COLUMNS),
        "client_bridge": {"status": "error", "reason": "not computed"},
        "reputation": {"status": "error", "reason": "not computed"},
        "personnel": {"status": "error", "reason": "not computed"},
        "market": {"status": "error", "reason": "not computed"},
        "deal_sizes_curr": _empty_df(DEAL_COLUMNS),
        "deal_sizes_prev": _empty_df(DEAL_COLUMNS),
        "staffing": _empty_df(STAFF_SELL_COLUMNS),
        "task_mix": _empty_df(TASK_MIX_COLUMNS),
        "rate_trend": _empty_df(RATE_TREND_COLUMNS),
        "scorecard": _empty_df(SCORECARD_COLUMNS),
    }

    try:
        monthly_all = compute_service_line_monthly(df)
    except Exception as e:
        monthly_all = _empty_df(MONTHLY_COLUMNS)
        bundle["monthly"] = monthly_all
        bundle["scorecard"] = _empty_df(SCORECARD_COLUMNS)
        bundle["client_bridge"] = _dict_error(e)
        return bundle

    if service_line not in ("", "All", None):
        monthly = monthly_all[monthly_all["service_line"] == service_line].copy()
    else:
        monthly = monthly_all.copy()
    bundle["monthly"] = monthly

    try:
        yoy_all = compute_yoy_snapshot(monthly_all, as_of_month)
        bundle["yoy"] = yoy_all[yoy_all["service_line"] == service_line].copy() if service_line not in ("", "All", None) else yoy_all
    except Exception as e:
        bundle["yoy"] = _empty_df(YOY_COLUMNS)
        bundle["decomp"] = _empty_df(DECOMP_COLUMNS)
        bundle["scorecard"] = _empty_df(SCORECARD_COLUMNS)
        bundle["client_bridge"] = _dict_error(e)
        return bundle

    try:
        decomp_all = decompose_revenue_change(yoy_all)
        bundle["decomp"] = decomp_all[decomp_all["service_line"] == service_line].copy() if service_line not in ("", "All", None) else decomp_all
    except Exception:
        bundle["decomp"] = _empty_df(DECOMP_COLUMNS)

    try:
        client_bridge = compute_client_bridge(df, service_line, current_months, prior_months)
        bundle["client_bridge"] = client_bridge
    except Exception as e:
        bundle["client_bridge"] = _dict_error(e)
        client_bridge = {"status": "error", "reason": str(e)}

    try:
        reputation = compute_reputation_signals(df, service_line, current_months, prior_months)
        bundle["reputation"] = reputation
    except Exception as e:
        bundle["reputation"] = _dict_error(e)
        reputation = {"status": "error", "reason": str(e)}

    try:
        personnel = compute_personnel_signals(df, service_line, current_months, prior_months)
        bundle["personnel"] = personnel
    except Exception as e:
        bundle["personnel"] = _dict_error(e)
        personnel = {"status": "error", "reason": str(e)}

    try:
        market = compute_market_signals(df, service_line, current_months, prior_months)
        bundle["market"] = market
    except Exception as e:
        bundle["market"] = _dict_error(e)
        market = {"status": "error", "reason": str(e)}

    try:
        deals_curr = compute_deal_size_distribution(df, service_line, current_months)
        deals_prev = compute_deal_size_distribution(df, service_line, prior_months)
        bundle["deal_sizes_curr"] = deals_curr
        bundle["deal_sizes_prev"] = deals_prev
    except Exception:
        bundle["deal_sizes_curr"] = _empty_df(DEAL_COLUMNS)
        bundle["deal_sizes_prev"] = _empty_df(DEAL_COLUMNS)
        deals_curr = _empty_df(DEAL_COLUMNS)
        deals_prev = _empty_df(DEAL_COLUMNS)

    try:
        staffing = compute_staff_selling_analysis(df, service_line)
        bundle["staffing"] = staffing
    except Exception:
        bundle["staffing"] = _empty_df(STAFF_SELL_COLUMNS)

    try:
        task_mix = compute_task_mix_shift(df, service_line, current_months, prior_months)
        bundle["task_mix"] = task_mix
    except Exception:
        bundle["task_mix"] = _empty_df(TASK_MIX_COLUMNS)

    try:
        rate_trend = compute_quoted_vs_actual_rate_trend(df, service_line)
        bundle["rate_trend"] = rate_trend
    except Exception:
        bundle["rate_trend"] = _empty_df(RATE_TREND_COLUMNS)

    pct_small_jobs_delta_pp = np.nan
    if len(deals_curr) > 0 and len(deals_prev) > 0:
        pct_curr = float(np.where(len(deals_curr) != 0, (deals_curr["quoted_amount"] < 1000).mean() * 100, 0.0))
        pct_prev = float(np.where(len(deals_prev) != 0, (deals_prev["quoted_amount"] < 1000).mean() * 100, 0.0))
        pct_small_jobs_delta_pp = pct_curr - pct_prev

    score_input = {"pct_small_jobs_delta_pp": pct_small_jobs_delta_pp}
    if isinstance(client_bridge, dict):
        score_input.update(client_bridge)

    try:
        yoy_for_score = bundle["yoy"]
        bundle["scorecard"] = build_hypothesis_scorecard(yoy_for_score, score_input, reputation, personnel, market)
    except Exception:
        bundle["scorecard"] = _empty_df(SCORECARD_COLUMNS)

    return bundle
