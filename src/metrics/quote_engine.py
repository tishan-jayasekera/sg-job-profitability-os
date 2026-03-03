"""
Empirical peer-job quote engine utilities.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.data.semantic import get_category_col, safe_quote_job_task


def _completed_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Return completed jobs using completion date or status fallback."""
    if "job_completed_date" in df.columns:
        return df[df["job_completed_date"].notna()]
    if "job_status" in df.columns:
        return df[df["job_status"].astype(str).str.lower().str.contains("completed", na=False)]
    return df.iloc[0:0]


def _empty_peer_jobs_df() -> pd.DataFrame:
    """Return an empty peer-jobs frame with stable output columns."""
    return pd.DataFrame(
        columns=[
            "job_no",
            "department_final",
            "category",
            "client",
            "total_hours",
            "total_cost",
            "total_revenue",
            "margin",
            "margin_pct",
            "realised_rate",
            "cost_per_hour",
            "job_completed_date",
            "quoted_hours",
            "quoted_amount",
            "quote_rate",
        ]
    )


def _safe_num(value: Any, default: float = 0.0) -> float:
    """Best-effort scalar numeric conversion."""
    try:
        if value is None:
            return default
        if isinstance(value, (float, int, np.floating, np.integer)):
            return float(value)
        out = float(pd.to_numeric(value, errors="coerce"))
        if np.isnan(out):
            return default
        return out
    except Exception:
        return default


def _line_rate(line: dict[str, Any]) -> float:
    """Resolve effective line rate with override precedence."""
    override = line.get("rate_override")
    if override is not None:
        return _safe_num(override, 0.0)
    return _safe_num(line.get("rate_suggested", line.get("rate", 0.0)), 0.0)


def _task_hours(task: dict[str, Any], hours_basis: str) -> float:
    """Resolve effective task hours from override or selected basis."""
    override = task.get("hours_override")
    if override is not None:
        return max(_safe_num(override, 0.0), 0.0)

    basis_key = f"hours_{hours_basis}"
    return max(
        _safe_num(
            task.get(
                basis_key,
                task.get("hours_p50", task.get("emp_hrs", task.get("hours", 0.0))),
            ),
            0.0,
        ),
        0.0,
    )


def _line_hours(line: dict[str, Any], hours_basis: str) -> float:
    """Resolve effective total line hours from tasks or line-level values."""
    tasks = line.get("task_mix") or []
    if tasks:
        return float(sum(_task_hours(task, hours_basis) for task in tasks))

    override = line.get("line_hours_override")
    if override is not None:
        return max(_safe_num(override, 0.0), 0.0)

    basis_key = f"hours_{hours_basis}"
    return max(_safe_num(line.get(basis_key, line.get("hours_p50", line.get("hours", 0.0))), 0.0), 0.0)


@st.cache_data(show_spinner=False)
def get_peer_jobs(
    df: pd.DataFrame,
    department: str,
    category: str,
    min_revenue: float | None = None,
    max_revenue: float | None = None,
    recency_months: int = 24,
) -> pd.DataFrame:
    """
    Return completed jobs matching department + category + optional revenue band.

    Output columns:
        job_no, department_final, category, client,
        total_hours, total_cost, total_revenue,
        margin, margin_pct, realised_rate, cost_per_hour,
        job_completed_date, quoted_hours, quoted_amount, quote_rate
    """
    if len(df) == 0:
        return _empty_peer_jobs_df()

    completed = _completed_jobs(df)
    if len(completed) == 0:
        return _empty_peer_jobs_df()

    category_col = get_category_col(completed)
    scoped = completed[
        (completed["department_final"].astype(str) == str(department))
        & (completed[category_col].astype(str) == str(category))
    ].copy()
    if len(scoped) == 0:
        return _empty_peer_jobs_df()

    agg_map: dict[str, tuple[str, str]] = {
        "department_final": ("department_final", "first"),
        "category": (category_col, "first"),
        "total_hours": ("hours_raw", "sum"),
        "total_cost": ("base_cost", "sum"),
        "total_revenue": ("rev_alloc", "sum"),
    }
    if "client" in scoped.columns:
        agg_map["client"] = ("client", "first")
    if "job_completed_date" in scoped.columns:
        agg_map["job_completed_date"] = ("job_completed_date", "max")
    if "month_key" in scoped.columns:
        agg_map["month_key_last"] = ("month_key", "max")

    jobs = scoped.groupby("job_no", dropna=False).agg(**agg_map).reset_index()

    if "client" not in jobs.columns:
        jobs["client"] = ""
    if "job_completed_date" not in jobs.columns:
        jobs["job_completed_date"] = pd.NaT
    if "month_key_last" not in jobs.columns:
        jobs["month_key_last"] = pd.NaT

    quote_job_task = safe_quote_job_task(scoped)
    if len(quote_job_task) > 0:
        quote_agg = quote_job_task.groupby("job_no", dropna=False).agg(
            quoted_hours=("quoted_time_total", "sum") if "quoted_time_total" in quote_job_task.columns else ("job_no", "count"),
            quoted_amount=("quoted_amount_total", "sum") if "quoted_amount_total" in quote_job_task.columns else ("job_no", "count"),
        ).reset_index()
        if "quoted_time_total" not in quote_job_task.columns:
            quote_agg["quoted_hours"] = 0.0
        if "quoted_amount_total" not in quote_job_task.columns:
            quote_agg["quoted_amount"] = 0.0
        jobs = jobs.merge(quote_agg, on="job_no", how="left")
    else:
        jobs["quoted_hours"] = 0.0
        jobs["quoted_amount"] = 0.0

    jobs["quoted_hours"] = pd.to_numeric(jobs["quoted_hours"], errors="coerce").fillna(0.0)
    jobs["quoted_amount"] = pd.to_numeric(jobs["quoted_amount"], errors="coerce").fillna(0.0)

    if min_revenue is not None:
        jobs = jobs[jobs["total_revenue"] >= float(min_revenue)]
    if max_revenue is not None:
        jobs = jobs[jobs["total_revenue"] <= float(max_revenue)]
    if len(jobs) == 0:
        return _empty_peer_jobs_df()

    recency_dt = jobs["job_completed_date"].copy()
    recency_dt = recency_dt.where(recency_dt.notna(), jobs["month_key_last"])

    if recency_months and recency_months > 0 and recency_dt.notna().any():
        anchor = recency_dt.max()
        cutoff = anchor - pd.DateOffset(months=int(recency_months))
        jobs = jobs[recency_dt >= cutoff].copy()
        recency_dt = recency_dt.loc[jobs.index]

    if len(jobs) == 0:
        return _empty_peer_jobs_df()

    jobs["margin"] = jobs["total_revenue"] - jobs["total_cost"]
    jobs["margin_pct"] = np.where(
        jobs["total_revenue"] != 0,
        jobs["margin"] / jobs["total_revenue"] * 100,
        np.nan,
    )
    jobs["realised_rate"] = np.where(
        jobs["total_hours"] > 0,
        jobs["total_revenue"] / jobs["total_hours"],
        np.nan,
    )
    jobs["cost_per_hour"] = np.where(
        jobs["total_hours"] > 0,
        jobs["total_cost"] / jobs["total_hours"],
        np.nan,
    )
    jobs["quote_rate"] = np.where(
        jobs["quoted_hours"] > 0,
        jobs["quoted_amount"] / jobs["quoted_hours"],
        np.nan,
    )

    keep_cols = [
        "job_no",
        "department_final",
        "category",
        "client",
        "total_hours",
        "total_cost",
        "total_revenue",
        "margin",
        "margin_pct",
        "realised_rate",
        "cost_per_hour",
        "job_completed_date",
        "quoted_hours",
        "quoted_amount",
        "quote_rate",
    ]

    out = jobs[keep_cols].copy()
    out = out.sort_values(["job_completed_date", "total_revenue"], ascending=[False, False], na_position="last")
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def get_peer_task_mix(
    df: pd.DataFrame,
    peer_job_nos: list,
) -> pd.DataFrame:
    """
    Compute empirical task distributions for a set of peer jobs.

    Returns columns:
        task_name, share_pct,
        hours_p25, hours_p50, hours_p75,
        median_cost_per_hour, job_count_with_task
    """
    if len(df) == 0 or not peer_job_nos:
        return pd.DataFrame(
            columns=[
                "task_name",
                "share_pct",
                "hours_p25",
                "hours_p50",
                "hours_p75",
                "median_cost_per_hour",
                "job_count_with_task",
            ]
        )

    scoped = df[df["job_no"].isin(peer_job_nos)].copy()
    if len(scoped) == 0:
        return pd.DataFrame(
            columns=[
                "task_name",
                "share_pct",
                "hours_p25",
                "hours_p50",
                "hours_p75",
                "median_cost_per_hour",
                "job_count_with_task",
            ]
        )

    task_job = scoped.groupby(["job_no", "task_name"], dropna=False).agg(
        task_hours=("hours_raw", "sum"),
        task_cost=("base_cost", "sum"),
    ).reset_index()

    job_total = scoped.groupby("job_no", dropna=False).agg(
        job_total_hours=("hours_raw", "sum"),
    ).reset_index()

    task_job = task_job.merge(job_total, on="job_no", how="left")
    task_job = task_job[(task_job["task_hours"] > 0) & (task_job["job_total_hours"] > 0)].copy()
    if len(task_job) == 0:
        return pd.DataFrame(
            columns=[
                "task_name",
                "share_pct",
                "hours_p25",
                "hours_p50",
                "hours_p75",
                "median_cost_per_hour",
                "job_count_with_task",
            ]
        )

    task_job["task_share"] = task_job["task_hours"] / task_job["job_total_hours"]
    task_job["cost_per_hour"] = np.where(
        task_job["task_hours"] > 0,
        task_job["task_cost"] / task_job["task_hours"],
        np.nan,
    )

    total_pool_hours = float(job_total["job_total_hours"].sum())
    grouped = task_job.groupby("task_name", dropna=False)
    task_mix = grouped.agg(
        task_hours_total=("task_hours", "sum"),
        hours_p25=("task_hours", lambda x: x.quantile(0.25)),
        hours_p50=("task_hours", "median"),
        hours_p75=("task_hours", lambda x: x.quantile(0.75)),
        median_cost_per_hour=("cost_per_hour", "median"),
        job_count_with_task=("job_no", "nunique"),
    ).reset_index()

    task_mix["share_pct"] = np.where(
        total_pool_hours > 0,
        task_mix["task_hours_total"] / total_pool_hours * 100,
        0.0,
    )

    out = task_mix[
        [
            "task_name",
            "share_pct",
            "hours_p25",
            "hours_p50",
            "hours_p75",
            "median_cost_per_hour",
            "job_count_with_task",
        ]
    ].sort_values("share_pct", ascending=False)

    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_peer_pool_summary(
    peer_jobs: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute peer pool summary KPIs from peer-jobs output.

    Returns keys:
        n_jobs, hours_p25, hours_p50, hours_p75, hours_p80,
        median_revenue, median_margin_pct,
        blended_cost_per_hour, blended_realised_rate
    """
    if len(peer_jobs) == 0:
        return {
            "n_jobs": 0,
            "hours_p25": 0.0,
            "hours_p50": 0.0,
            "hours_p75": 0.0,
            "hours_p80": 0.0,
            "median_revenue": 0.0,
            "median_margin_pct": np.nan,
            "blended_cost_per_hour": np.nan,
            "blended_realised_rate": np.nan,
        }

    total_hours = float(peer_jobs["total_hours"].sum())
    total_cost = float(peer_jobs["total_cost"].sum())
    total_revenue = float(peer_jobs["total_revenue"].sum())

    return {
        "n_jobs": int(peer_jobs["job_no"].nunique()),
        "hours_p25": float(peer_jobs["total_hours"].quantile(0.25)),
        "hours_p50": float(peer_jobs["total_hours"].median()),
        "hours_p75": float(peer_jobs["total_hours"].quantile(0.75)),
        "hours_p80": float(peer_jobs["total_hours"].quantile(0.80)),
        "median_revenue": float(peer_jobs["total_revenue"].median()),
        "median_margin_pct": float(peer_jobs["margin_pct"].median()),
        "blended_cost_per_hour": (total_cost / total_hours) if total_hours > 0 else np.nan,
        "blended_realised_rate": (total_revenue / total_hours) if total_hours > 0 else np.nan,
    }


def compute_quote_economics(
    quote_lines: list[dict],
    hours_basis: str = "p50",
) -> dict[str, Any]:
    """
    Compute quote-level economics for configured line items.

    Returns:
        total_hours, total_value, blended_rate,
        est_labour_cost, est_margin, est_margin_pct,
        lines (per-line economics)
    """
    line_outputs: list[dict[str, Any]] = []
    total_hours = 0.0
    total_value = 0.0
    est_labour_cost = 0.0

    for idx, line in enumerate(quote_lines):
        line_hours = _line_hours(line, hours_basis)
        line_rate = _line_rate(line)
        line_value = line_hours * line_rate

        cost_per_hour = _safe_num(
            line.get("cost_per_hour_peer", line.get("median_cost_per_hour", 0.0)),
            0.0,
        )
        line_cost = line_hours * cost_per_hour
        line_margin = line_value - line_cost
        line_margin_pct = (line_margin / line_value * 100) if line_value > 0 else np.nan

        total_hours += line_hours
        total_value += line_value
        est_labour_cost += line_cost

        line_outputs.append(
            {
                "line_index": idx,
                "department": line.get("department", ""),
                "category": line.get("category", ""),
                "hours": line_hours,
                "rate": line_rate,
                "value": line_value,
                "cost": line_cost,
                "margin": line_margin,
                "margin_pct": line_margin_pct,
                "peer_count": int(_safe_num(line.get("peer_job_count", 0), 0)),
                "peer_rate": _safe_num(line.get("rate_suggested", np.nan), np.nan),
            }
        )

    blended_rate = (total_value / total_hours) if total_hours > 0 else np.nan
    est_margin = total_value - est_labour_cost
    est_margin_pct = (est_margin / total_value * 100) if total_value > 0 else np.nan

    return {
        "total_hours": total_hours,
        "total_value": total_value,
        "blended_rate": blended_rate,
        "est_labour_cost": est_labour_cost,
        "est_margin": est_margin,
        "est_margin_pct": est_margin_pct,
        "lines": line_outputs,
    }


def compute_risk_indicators(
    quote_lines: list[dict],
    hours_basis: str = "p50",
) -> list[dict[str, Any]]:
    """
    Build risk indicators for quote lines.

    Returns list of dicts:
        {"level": "warning"|"danger"|"ok", "message": str, "line_index": int|None}
    """
    risks: list[dict[str, Any]] = []

    for idx, line in enumerate(quote_lines):
        peer_count = int(_safe_num(line.get("peer_job_count", 0), 0))
        line_label = f"Line {idx + 1}"

        if peer_count < 5:
            risks.append(
                {
                    "level": "warning",
                    "line_index": idx,
                    "message": f"{line_label}: {peer_count} peer jobs - low confidence (fewer than 5).",
                }
            )
        elif peer_count < 15:
            risks.append(
                {
                    "level": "warning",
                    "line_index": idx,
                    "message": f"{line_label}: {peer_count} peer jobs - moderate confidence.",
                }
            )
        else:
            risks.append(
                {
                    "level": "ok",
                    "line_index": idx,
                    "message": f"{line_label}: {peer_count} peer jobs - good confidence.",
                }
            )

        line_rate = _line_rate(line)
        peer_rate = _safe_num(line.get("rate_suggested", np.nan), np.nan)
        if not np.isnan(peer_rate) and peer_rate > 0:
            if line_rate < peer_rate * 0.9:
                pct = (peer_rate - line_rate) / peer_rate * 100
                risks.append(
                    {
                        "level": "warning",
                        "line_index": idx,
                        "message": f"{line_label}: rate is {pct:.1f}% below peer median.",
                    }
                )
            elif line_rate > peer_rate * 1.2:
                pct = (line_rate - peer_rate) / peer_rate * 100
                risks.append(
                    {
                        "level": "warning",
                        "line_index": idx,
                        "message": f"{line_label}: rate is {pct:.1f}% above peer median - check competitiveness.",
                    }
                )

        line_hours = _line_hours(line, hours_basis)
        hours_p25 = _safe_num(line.get("hours_p25", np.nan), np.nan)
        if not np.isnan(hours_p25) and hours_p25 > 0 and line_hours < hours_p25:
            risks.append(
                {
                    "level": "danger",
                    "line_index": idx,
                    "message": f"{line_label}: hours {line_hours:.1f} are below P25 ({hours_p25:.1f}) - high overrun risk.",
                }
            )

    return risks
