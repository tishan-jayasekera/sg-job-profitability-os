"""
Recurring Quote Overruns (Task Margin Erosion)

Standalone diagnostic page for repeated task-level quote overruns.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.data.cohorts import filter_by_time_window
from src.data.semantic import get_category_col, safe_quote_job_task
from src.metrics.quote_delivery import (
    compute_task_overrun_consistency,
    get_overrun_jobs_for_task,
)
from src.ui.formatting import fmt_currency


st.set_page_config(
    page_title="Recurring Quote Overruns",
    page_icon="📉",
    layout="wide",
)


COLORS = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "border": "#dbe3ee",
    "panel": "#f8fafc",
    "accent": "#0f766e",
    "accent_soft": "#ecfeff",
    "warn": "#b91c1c",
    "warn_soft": "#fef2f2",
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return load_fact_timesheet()


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .rq-hero {
            border: 1px solid #dbe3ee;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            background: linear-gradient(140deg, #f0fdfa 0%, #ffffff 52%, #f8fafc 100%);
            margin-bottom: 0.9rem;
        }
        .rq-eyebrow {
            font-size: 0.72rem;
            font-weight: 700;
            color: #0f766e;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .rq-title {
            font-size: clamp(1.45rem, 2vw, 2rem);
            font-weight: 760;
            color: #0f172a;
            margin: 0;
            line-height: 1.15;
        }
        .rq-sub {
            color: #334155;
            margin-top: 0.3rem;
            font-size: 0.93rem;
        }
        .rq-note {
            border-left: 4px solid #0f766e;
            background: #f0fdfa;
            padding: 0.65rem 0.8rem;
            border-radius: 8px;
            color: #0f172a;
            margin: 0.55rem 0 0.7rem 0;
            font-size: 0.9rem;
        }
        .rq-action {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.85rem;
            background: #fafbfc;
            min-height: 172px;
        }
        .rq-action-title {
            font-weight: 650;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }
        .rq-action-body {
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.38;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _is_uncategorised_value(category: Optional[str]) -> bool:
    if category is None:
        return False
    cat = str(category).strip().lower()
    return cat in {"(uncategorised)", "(uncategorized)", "uncategorised", "uncategorized", "__null__"}


def _scope_key(scope_label: str, department: Optional[str], category: Optional[str]) -> str:
    raw = f"{scope_label}|{department}|{category}"
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in raw)


def _filter_scope(
    df_scope: pd.DataFrame,
    department: Optional[str],
    category: Optional[str],
    time_window: str,
) -> pd.DataFrame:
    scoped = df_scope

    if department and "department_final" in scoped.columns:
        scoped = scoped[scoped["department_final"].astype(str) == str(department)]

    if category is not None and len(scoped) > 0:
        category_col = get_category_col(scoped)
        if category_col in scoped.columns:
            if _is_uncategorised_value(category):
                scoped = scoped[scoped[category_col].isna()]
            else:
                scoped = scoped[scoped[category_col] == category]

    date_col = "month_key" if "month_key" in scoped.columns else "work_date" if "work_date" in scoped.columns else None
    if date_col and len(scoped) > 0:
        scoped = scoped.copy()
        scoped[date_col] = pd.to_datetime(scoped[date_col], errors="coerce")
        scoped = filter_by_time_window(scoped, window=time_window, date_col=date_col)

    return scoped


def _format_signed_pct(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value * 100:+.0f}%"


def _build_task_leakage_chart(task_overruns: pd.DataFrame, has_cost: bool):
    chart_df = task_overruns.head(12).copy()
    chart_df["task_label"] = chart_df["task_name"].fillna("(Unspecified task)").astype(str)

    if has_cost and chart_df["total_overrun_cost"].notna().any():
        value_col = "total_overrun_cost"
        x_label = "Estimated Margin Erosion ($)"
        hover_fmt = ":$,.0f"
    else:
        value_col = "total_overrun_hours"
        x_label = "Total Overrun Hours"
        hover_fmt = ":,.1f"

    chart_df = chart_df.sort_values(value_col, ascending=True)

    fig = px.bar(
        chart_df,
        x=value_col,
        y="task_label",
        orientation="h",
        color="overrun_rate",
        color_continuous_scale="Reds",
        labels={value_col: x_label, "task_label": "Task", "overrun_rate": "Overrun Rate"},
        hover_data={
            "overrun_rate": ":.0%",
            "avg_overrun_pct": ":.0%",
            value_col: hover_fmt,
            "jobs_with_quote": True,
            "overrun_jobs": True,
            "task_label": False,
        },
    )
    fig.update_layout(
        title="Top Recurring Leakage Tasks",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar_title="Overrun rate",
        yaxis_title="",
    )
    return fig


def _build_task_pattern_chart(task_overruns: pd.DataFrame, min_overrun_rate: float, has_cost: bool):
    bubble_df = task_overruns.copy()
    bubble_df["task_label"] = bubble_df["task_name"].fillna("(Unspecified task)").astype(str)
    bubble_df["overrun_rate_pct"] = bubble_df["overrun_rate"] * 100
    bubble_df["avg_overrun_pct_pct"] = bubble_df["avg_overrun_pct"] * 100

    size_col = "total_overrun_cost" if has_cost and bubble_df["total_overrun_cost"].notna().any() else "total_overrun_hours"

    fig = px.scatter(
        bubble_df,
        x="overrun_rate_pct",
        y="avg_overrun_pct_pct",
        size=size_col,
        color="total_overrun_hours",
        color_continuous_scale="Teal",
        hover_name="task_label",
        hover_data={
            "overrun_rate_pct": ":.0f",
            "avg_overrun_pct_pct": ":.0f",
            "jobs_with_quote": True,
            "overrun_jobs": True,
            "total_overrun_hours": ":,.1f",
            "total_overrun_cost": ":$,.0f",
            "task_label": False,
        },
        labels={
            "overrun_rate_pct": "Overrun Rate (%)",
            "avg_overrun_pct_pct": "Avg Overrun per Quoted Job (%)",
            "total_overrun_hours": "Total Overrun Hours",
        },
    )
    fig.add_vline(x=min_overrun_rate * 100, line_dash="dash", line_color="#b91c1c", opacity=0.6)
    fig.add_hline(y=20, line_dash="dot", line_color="#0f766e", opacity=0.6)
    fig.update_layout(
        title="Pattern View: Frequency vs Severity",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_overrun_trend(
    df: pd.DataFrame,
    department: Optional[str] = None,
    category: Optional[str] = None,
    lookback_months: int = 12,
    rolling_months: int = 3,
    min_jobs_with_quote: int = 8,
    min_overrun_rate: float = 0.30,
) -> pd.DataFrame:
    scoped = _filter_scope(
        df_scope=df,
        department=department,
        category=category,
        time_window="all",
    )
    if scoped.empty:
        return pd.DataFrame()

    date_col = "month_key" if "month_key" in scoped.columns else "work_date" if "work_date" in scoped.columns else None
    if not date_col:
        return pd.DataFrame()

    scoped = scoped.copy()
    scoped[date_col] = pd.to_datetime(scoped[date_col], errors="coerce")
    scoped = scoped[scoped[date_col].notna()].copy()
    if scoped.empty:
        return pd.DataFrame()

    scoped["trend_month"] = scoped[date_col].dt.to_period("M").dt.to_timestamp()
    months = sorted(scoped["trend_month"].dropna().unique().tolist())
    if not months:
        return pd.DataFrame()

    months = months[-max(lookback_months, 1):]

    rows = []
    for month in months:
        month_ts = pd.Timestamp(month)
        window_start = (month_ts - pd.DateOffset(months=max(rolling_months, 1) - 1)).replace(day=1)
        window_end = month_ts + pd.offsets.MonthEnd(0)

        window_df = scoped[(scoped[date_col] >= window_start) & (scoped[date_col] <= window_end)].copy()
        if window_df.empty:
            continue

        task_overruns = compute_task_overrun_consistency(
            window_df,
            department=None,
            category=None,
            time_window="all",
            min_jobs_with_quote=min_jobs_with_quote,
            min_overrun_rate=min_overrun_rate,
        )

        if task_overruns.empty:
            rows.append(
                {
                    "month": month_ts,
                    "tasks_flagged": 0,
                    "weighted_overrun_rate": 0.0,
                    "total_overrun_hours": 0.0,
                    "total_overrun_cost": np.nan,
                    "leakage_score": np.nan,
                }
            )
            continue

        jobs_with_quote = float(task_overruns["jobs_with_quote"].sum())
        overrun_jobs = float(task_overruns["overrun_jobs"].sum())
        weighted_overrun_rate = overrun_jobs / jobs_with_quote if jobs_with_quote > 0 else 0.0

        rows.append(
            {
                "month": month_ts,
                "tasks_flagged": int(len(task_overruns)),
                "weighted_overrun_rate": weighted_overrun_rate,
                "total_overrun_hours": float(task_overruns["total_overrun_hours"].sum()),
                "total_overrun_cost": float(task_overruns["total_overrun_cost"].sum())
                if task_overruns["total_overrun_cost"].notna().any()
                else np.nan,
                "leakage_score": float(task_overruns["leakage_score"].sum())
                if task_overruns["leakage_score"].notna().any()
                else np.nan,
            }
        )

    trend_df = pd.DataFrame(rows).sort_values("month")
    if trend_df.empty:
        return trend_df

    trend_df["weighted_overrun_rate_pct"] = trend_df["weighted_overrun_rate"] * 100
    trend_df["month_label"] = trend_df["month"].dt.strftime("%b %Y")
    return trend_df


def _build_trend_rate_chart(trend_df: pd.DataFrame):
    fig = px.line(
        trend_df,
        x="month",
        y="weighted_overrun_rate_pct",
        markers=True,
        labels={"month": "Month", "weighted_overrun_rate_pct": "Weighted Overrun Rate (%)"},
    )
    fig.update_traces(line_color="#b91c1c", marker_size=8)
    fig.update_layout(
        title="Trend: Overrun Rate",
        height=320,
        margin=dict(l=10, r=10, t=42, b=10),
        yaxis_ticksuffix="%",
    )
    return fig


def _build_trend_leakage_chart(trend_df: pd.DataFrame, has_cost: bool):
    use_cost = has_cost and "total_overrun_cost" in trend_df.columns and trend_df["total_overrun_cost"].notna().any()
    y_col = "total_overrun_cost" if use_cost else "total_overrun_hours"
    y_label = "Estimated Margin Erosion ($)" if use_cost else "Total Overrun Hours"

    fig = px.bar(
        trend_df,
        x="month",
        y=y_col,
        color=y_col,
        color_continuous_scale="Teal",
        labels={"month": "Month", y_col: y_label},
    )
    fig.update_layout(
        title="Trend: Leakage Magnitude",
        height=320,
        margin=dict(l=10, r=10, t=42, b=10),
    )
    return fig


def _trend_signal_text(trend_df: pd.DataFrame, has_cost: bool) -> tuple[str, str]:
    if len(trend_df) < 2:
        return ("Not enough trend history", "Need at least 2 monthly points to assess direction.")

    first = trend_df.iloc[0]
    last = trend_df.iloc[-1]
    rate_delta_pp = (last["weighted_overrun_rate"] - first["weighted_overrun_rate"]) * 100

    if has_cost and "total_overrun_cost" in trend_df.columns and pd.notna(first.get("total_overrun_cost")) and pd.notna(last.get("total_overrun_cost")):
        base = float(first["total_overrun_cost"])
        delta = float(last["total_overrun_cost"] - first["total_overrun_cost"])
        delta_pct = (delta / base) * 100 if base > 0 else np.nan
        metric_label = "erosion $"
        metric_value = f"{delta_pct:+.0f}%" if pd.notna(delta_pct) else f"{fmt_currency(delta)}"
    else:
        base = float(first["total_overrun_hours"])
        delta = float(last["total_overrun_hours"] - first["total_overrun_hours"])
        delta_pct = (delta / base) * 100 if base > 0 else np.nan
        metric_label = "overrun hours"
        metric_value = f"{delta_pct:+.0f}%" if pd.notna(delta_pct) else f"{delta:+.1f}h"

    if rate_delta_pp <= -3 and (pd.isna(delta_pct) or delta_pct <= -10):
        return (
            "Improving",
            f"Overrun rate moved {rate_delta_pp:+.1f} pp and {metric_label} moved {metric_value} vs trend start.",
        )
    if rate_delta_pp >= 3 and (pd.isna(delta_pct) or delta_pct >= 10):
        return (
            "Worsening",
            f"Overrun rate moved {rate_delta_pp:+.1f} pp and {metric_label} moved {metric_value} vs trend start.",
        )
    return (
        "Mixed",
        f"Overrun rate moved {rate_delta_pp:+.1f} pp and {metric_label} moved {metric_value} vs trend start.",
    )


def _show_actions(task_scope: pd.DataFrame, selected_task_row: pd.Series):
    overrun_rate = float(selected_task_row.get("overrun_rate", 0) or 0)
    avg_overrun_pct = float(selected_task_row.get("avg_overrun_pct", 0) or 0)
    rule_1_trigger = overrun_rate >= 0.60 and avg_overrun_pct >= 0.20

    mismatch_share = np.nan
    rule_2_evaluable = "quote_match_flag" in task_scope.columns and not task_scope.empty
    rule_2_trigger = False
    if rule_2_evaluable:
        mismatch_share = task_scope["quote_match_flag"].astype(str).str.lower().ne("matched").mean()
        rule_2_trigger = mismatch_share >= 0.20

    quote_rate = np.nan
    realised_rate = np.nan
    rate_gap_pct = np.nan
    rule_3_evaluable = False
    rule_3_trigger = False
    needed_for_rate = {"quoted_time_total", "quoted_amount_total", "rev_alloc", "hours_raw", "job_no", "task_name"}
    if needed_for_rate.issubset(task_scope.columns) and not task_scope.empty:
        task_quotes = safe_quote_job_task(task_scope)
        if (
            not task_quotes.empty
            and "quoted_time_total" in task_quotes.columns
            and "quoted_amount_total" in task_quotes.columns
        ):
            quote_hours = pd.to_numeric(task_quotes["quoted_time_total"], errors="coerce").fillna(0).sum()
            quote_amount = pd.to_numeric(task_quotes["quoted_amount_total"], errors="coerce").sum(min_count=1)
            actual_hours = pd.to_numeric(task_scope["hours_raw"], errors="coerce").sum()
            actual_revenue = pd.to_numeric(task_scope["rev_alloc"], errors="coerce").sum(min_count=1)

            if quote_hours > 0 and pd.notna(quote_amount):
                quote_rate = quote_amount / quote_hours
            if actual_hours > 0 and pd.notna(actual_revenue):
                realised_rate = actual_revenue / actual_hours

            if pd.notna(quote_rate) and quote_rate > 0 and pd.notna(realised_rate):
                rate_gap_pct = (quote_rate - realised_rate) / quote_rate
                rule_3_evaluable = True
                rule_3_trigger = rate_gap_pct >= 0.15

    card_1_body = (
        f"Overruns hit {overrun_rate * 100:.0f}% of quoted jobs with {avg_overrun_pct * 100:.0f}% average overshoot. "
        "Increase standard hours for this task, add complexity drivers, and update quote builder defaults."
        if rule_1_trigger
        else "Refresh baseline hours from latest delivery evidence and tighten estimate guardrails in the quote template."
    )

    if rule_2_evaluable and rule_2_trigger:
        card_2_body = (
            f"{mismatch_share * 100:.0f}% of task rows are not quote-matched. Enforce variation approval before extra "
            "delivery effort and track exceptions weekly."
        )
    elif rule_2_evaluable:
        card_2_body = (
            f"Quote mismatch share is {mismatch_share * 100:.0f}%. Keep change-control gates active and review mismatches monthly."
        )
    else:
        card_2_body = "Introduce scope variation control with mandatory approval and a weekly scope-change log."

    if rule_3_evaluable and rule_3_trigger:
        card_3_body = (
            f"Realised rate ({fmt_currency(realised_rate)}/hr) is {rate_gap_pct * 100:.0f}% below quote rate "
            f"({fmt_currency(quote_rate)}/hr). Rebalance staffing mix and tighten write-down control."
        )
    elif rule_3_evaluable:
        card_3_body = (
            f"Realised rate ({fmt_currency(realised_rate)}/hr) is close to quote rate ({fmt_currency(quote_rate)}/hr). "
            "Maintain current staffing mix and monitor rate slippage."
        )
    else:
        card_3_body = "Review execution playbook and QA gates for this task to improve delivery efficiency and rate capture."

    actions = [
        {"title": "Fix the quote baseline", "body": card_1_body},
        {"title": "Stop scope creep", "body": card_2_body},
        {"title": "Fix staffing / rate capture", "body": card_3_body},
    ]

    st.markdown("**Recommended actions**")
    cols = st.columns(3)
    for col, action in zip(cols, actions):
        with col:
            st.markdown(
                (
                    "<div class='rq-action'>"
                    f"<div class='rq-action-title'>{action['title']}</div>"
                    f"<div class='rq-action-body'>{action['body']}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_recurring_task_overruns_section(
    df_scope: pd.DataFrame,
    scope_label: str,
    department: Optional[str],
    category: Optional[str],
) -> None:
    st.markdown(f"#### {scope_label}")

    if "quoted_time_total" not in df_scope.columns:
        st.info("Quoted hours not available — cannot compute quote overruns.")
        return

    missing_core = [c for c in ["job_no", "task_name", "hours_raw"] if c not in df_scope.columns]
    if missing_core:
        st.info(f"Required columns missing: {', '.join(missing_core)}")
        return

    has_cost = "base_cost" in df_scope.columns
    scope_key = _scope_key(scope_label, department, category)

    with st.container(border=True):
        ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.0, 1.0])
        with ctrl1:
            time_window = st.selectbox(
                "Time window",
                options=["3m", "6m", "12m", "24m", "fytd", "all"],
                index=2,
                key=f"overrun_window_{scope_key}",
            )
        with ctrl2:
            min_jobs_with_quote = st.slider(
                "Min jobs with quote",
                min_value=3,
                max_value=30,
                value=8,
                step=1,
                key=f"overrun_min_jobs_{scope_key}",
            )
        with ctrl3:
            min_overrun_rate = st.slider(
                "Min overrun rate",
                min_value=0.0,
                max_value=0.9,
                value=0.30,
                step=0.05,
                key=f"overrun_min_rate_{scope_key}",
            )

    task_overruns = compute_task_overrun_consistency(
        df_scope,
        department=department,
        category=category,
        time_window=time_window,
        min_jobs_with_quote=min_jobs_with_quote,
        min_overrun_rate=min_overrun_rate,
    )

    if task_overruns.empty:
        st.info("No recurring task overruns meet the selected thresholds.")
        return

    scoped_for_detail = _filter_scope(
        df_scope,
        department=department,
        category=category,
        time_window=time_window,
    )

    top_rows = task_overruns.head(2)
    if has_cost and top_rows["total_overrun_cost"].notna().any():
        top_summary = [
            (
                f"**{r['task_name'] if pd.notna(r['task_name']) else '(Unspecified task)'}** leaks "
                f"{fmt_currency(r['total_overrun_cost'])} at {r['overrun_rate'] * 100:.0f}% overrun frequency"
            )
            for _, r in top_rows.iterrows()
        ]
    else:
        top_summary = [
            (
                f"**{r['task_name'] if pd.notna(r['task_name']) else '(Unspecified task)'}** overruns in "
                f"{r['overrun_rate'] * 100:.0f}% of quoted jobs ({r['total_overrun_hours']:,.1f}h total overrun)"
            )
            for _, r in top_rows.iterrows()
        ]

    st.markdown(f"<div class='rq-note'><b>So what:</b> {'; '.join(top_summary)}.</div>", unsafe_allow_html=True)

    total_jobs_with_quote = float(task_overruns["jobs_with_quote"].sum())
    weighted_overrun_rate = (
        float(task_overruns["overrun_jobs"].sum()) / total_jobs_with_quote if total_jobs_with_quote > 0 else 0.0
    )
    total_overrun_hours = float(task_overruns["total_overrun_hours"].sum())

    if has_cost and task_overruns["total_overrun_cost"].notna().any():
        total_erosion_display = fmt_currency(task_overruns["total_overrun_cost"].sum())
    else:
        total_erosion_display = "—"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Recurring tasks flagged", f"{len(task_overruns):,}")
    with m2:
        st.metric("Overrun jobs", f"{int(task_overruns['overrun_jobs'].sum()):,}")
    with m3:
        st.metric("Weighted overrun rate", f"{weighted_overrun_rate * 100:.0f}%")
    with m4:
        if has_cost:
            st.metric("Est. margin erosion", total_erosion_display)
        else:
            st.metric("Total overrun hours", f"{total_overrun_hours:,.1f}h")

    st.markdown("**Trend analysis: are we getting better or worse?**")
    t1, t2 = st.columns([1.0, 1.0])
    with t1:
        trend_lookback_months = st.slider(
            "Trend horizon (months)",
            min_value=6,
            max_value=24,
            value=12,
            step=1,
            key=f"trend_horizon_{scope_key}",
        )
    with t2:
        trend_basis = st.selectbox(
            "Trend basis",
            options=["3m rolling", "6m rolling", "12m rolling"],
            index=0,
            key=f"trend_basis_{scope_key}",
        )

    rolling_map = {"3m rolling": 3, "6m rolling": 6, "12m rolling": 12}
    rolling_months = rolling_map.get(trend_basis, 3)

    trend_df = compute_overrun_trend(
        df=df_scope,
        department=department,
        category=category,
        lookback_months=trend_lookback_months,
        rolling_months=rolling_months,
        min_jobs_with_quote=min_jobs_with_quote,
        min_overrun_rate=min_overrun_rate,
    )

    if trend_df.empty:
        st.info("Trend view unavailable for this scope/time selection.")
    else:
        signal_title, signal_text = _trend_signal_text(trend_df, has_cost=has_cost)
        if signal_title == "Improving":
            st.success(f"{signal_title}: {signal_text}")
        elif signal_title == "Worsening":
            st.error(f"{signal_title}: {signal_text}")
        else:
            st.warning(f"{signal_title}: {signal_text}")

        first_point = trend_df.iloc[0]
        last_point = trend_df.iloc[-1]
        rate_delta_pp = (last_point["weighted_overrun_rate"] - first_point["weighted_overrun_rate"]) * 100

        tm1, tm2, tm3 = st.columns(3)
        with tm1:
            st.metric(
                f"Overrun rate ({last_point['month_label']})",
                f"{last_point['weighted_overrun_rate_pct']:.0f}%",
                delta=f"{rate_delta_pp:+.1f} pp",
                delta_color="inverse",
            )
        with tm2:
            st.metric(
                f"Tasks flagged ({last_point['month_label']})",
                f"{int(last_point['tasks_flagged']):,}",
                delta=f"{int(last_point['tasks_flagged'] - first_point['tasks_flagged']):+d}",
                delta_color="inverse",
            )
        with tm3:
            if has_cost and trend_df["total_overrun_cost"].notna().any():
                cost_delta = last_point["total_overrun_cost"] - first_point["total_overrun_cost"]
                st.metric(
                    f"Erosion ({last_point['month_label']})",
                    fmt_currency(last_point["total_overrun_cost"]),
                    delta=fmt_currency(cost_delta),
                    delta_color="inverse",
                )
            else:
                hours_delta = last_point["total_overrun_hours"] - first_point["total_overrun_hours"]
                st.metric(
                    f"Overrun hours ({last_point['month_label']})",
                    f"{last_point['total_overrun_hours']:,.1f}h",
                    delta=f"{hours_delta:+,.1f}h",
                    delta_color="inverse",
                )

        tc1, tc2 = st.columns([1.0, 1.0])
        with tc1:
            st.plotly_chart(_build_trend_rate_chart(trend_df), use_container_width=True)
            st.caption(
                f"How calculated: each point is a {rolling_months}-month rolling window. "
                "Weighted overrun rate = sum(overrun_jobs) / sum(jobs_with_quote) across tasks passing filters."
            )
        with tc2:
            st.plotly_chart(_build_trend_leakage_chart(trend_df, has_cost=has_cost), use_container_width=True)
            if has_cost and trend_df["total_overrun_cost"].notna().any():
                st.caption(
                    "How calculated: for each job-task, overrun_hours = max(actual_hours - quoted_hours, 0) "
                    "when quoted_hours > 0; effective_cost_rate = actual_cost / actual_hours; "
                    "margin erosion ($) = overrun_hours × effective_cost_rate. "
                    f"The chart shows the sum across tasks per {rolling_months}-month rolling window."
                )
            else:
                st.caption(
                    f"How calculated: total overrun hours are summed across tasks in each "
                    f"{rolling_months}-month rolling window."
                )

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        st.plotly_chart(_build_task_leakage_chart(task_overruns, has_cost), use_container_width=True)
        if has_cost:
            st.caption(
                "How to read: bars rank tasks by total estimated margin erosion ($); color shows overrun frequency "
                "(overrun_jobs / jobs_with_quote)."
            )
        else:
            st.caption(
                "How to read: bars rank tasks by total overrun hours; color shows overrun frequency "
                "(overrun_jobs / jobs_with_quote)."
            )
    with c2:
        st.plotly_chart(
            _build_task_pattern_chart(task_overruns, min_overrun_rate=min_overrun_rate, has_cost=has_cost),
            use_container_width=True,
        )
        st.caption(
            "How to read: x-axis = overrun frequency (% of quoted jobs); y-axis = average overrun per quoted job "
            "((actual_hours - quoted_hours) / quoted_hours); bubble size = leakage magnitude."
        )

    show_revenue_at_risk = (
        "total_revenue_at_risk" in task_overruns.columns
        and task_overruns["total_revenue_at_risk"].notna().any()
    )

    table_df = task_overruns.copy()
    table_df["Task"] = table_df["task_name"].fillna("(Unspecified task)").astype(str)
    table_df["Overrun rate"] = table_df["overrun_rate"]
    table_df["Avg overrun"] = table_df["avg_overrun_pct"]

    display_cols = [
        "Task",
        "Overrun rate",
        "overrun_jobs",
        "jobs_with_quote",
        "total_overrun_hours",
        "total_overrun_cost",
        "Avg overrun",
    ]
    rename_map = {
        "overrun_jobs": "Overrun jobs",
        "jobs_with_quote": "Jobs with quote",
        "total_overrun_hours": "Total overrun hours",
        "total_overrun_cost": "Est. margin erosion ($)",
    }

    if show_revenue_at_risk:
        display_cols.append("total_revenue_at_risk")
        rename_map["total_revenue_at_risk"] = "Revenue at risk ($)"

    task_table = table_df[display_cols].rename(columns=rename_map)

    if not has_cost:
        task_table["Est. margin erosion ($)"] = np.nan

    task_formatters = {
        "Overrun rate": "{:.0%}",
        "Avg overrun": "{:+.0%}",
        "Total overrun hours": "{:,.1f}",
        "Est. margin erosion ($)": lambda x: fmt_currency(x) if pd.notna(x) else "—",
    }
    if "Revenue at risk ($)" in task_table.columns:
        task_formatters["Revenue at risk ($)"] = lambda x: fmt_currency(x) if pd.notna(x) else "—"

    styled_task = task_table.style.format(task_formatters)

    st.markdown("**Recurring Margin-Leak Tasks**")
    st.dataframe(styled_task, use_container_width=True, hide_index=True)

    task_selector_df = task_overruns.reset_index(drop=True).copy()
    task_selector_df["task_label"] = task_selector_df["task_name"].fillna("(Unspecified task)").astype(str)
    selected_label = st.selectbox(
        "Deep dive a task",
        options=task_selector_df["task_label"].tolist(),
        key=f"overrun_task_{scope_key}",
    )
    selected_idx = task_selector_df["task_label"].tolist().index(selected_label)
    selected_task_row = task_selector_df.iloc[selected_idx]
    selected_task = selected_task_row["task_name"]

    if pd.isna(selected_task):
        task_scope = scoped_for_detail[scoped_for_detail["task_name"].isna()].copy()
    else:
        task_scope = scoped_for_detail[scoped_for_detail["task_name"] == selected_task].copy()

    top_jobs = get_overrun_jobs_for_task(
        df_scope,
        task_name=selected_task,
        department=department,
        category=category,
        time_window=time_window,
        n=15,
    )

    st.markdown("**Where it happens: top offending jobs**")
    if top_jobs.empty:
        st.info("No overrun jobs found for this task in the selected scope/window.")
    else:
        client_col = None
        if "client_name" in top_jobs.columns and top_jobs["client_name"].notna().any():
            client_col = "client_name"
        elif "client_group" in top_jobs.columns and top_jobs["client_group"].notna().any():
            client_col = "client_group"

        chart_col_metric = "overrun_cost" if has_cost and top_jobs["overrun_cost"].notna().any() else "overrun_hours"
        chart_label = "Overrun cost ($)" if chart_col_metric == "overrun_cost" else "Overrun hours"

        jobs_chart_df = top_jobs.head(10).copy()
        jobs_chart_df["job_label"] = jobs_chart_df["job_no"].astype(str)
        if client_col:
            jobs_chart_df["job_label"] = jobs_chart_df["job_label"] + " | " + jobs_chart_df[client_col].fillna("—").astype(str)

        jobs_chart_df = jobs_chart_df.sort_values(chart_col_metric, ascending=True)
        fig_jobs = px.bar(
            jobs_chart_df,
            x=chart_col_metric,
            y="job_label",
            orientation="h",
            color=chart_col_metric,
            color_continuous_scale="Reds",
            labels={chart_col_metric: chart_label, "job_label": "Job"},
            height=360,
        )
        fig_jobs.update_layout(margin=dict(l=10, r=10, t=35, b=10), title="Highest-impact jobs")

        left, right = st.columns([1.0, 1.25])
        with left:
            st.plotly_chart(fig_jobs, use_container_width=True)
            st.caption(
                "How calculated: jobs are ranked for the selected task by overrun cost "
                "(or overrun hours when cost is unavailable)."
            )

        jobs_table = pd.DataFrame({
            "Job": top_jobs["job_no"].astype(str),
            "Quoted hours": top_jobs["quoted_hours"],
            "Actual hours": top_jobs["actual_hours"],
            "Overrun hours": top_jobs["overrun_hours"],
            "Overrun cost ($)": top_jobs["overrun_cost"] if has_cost else np.nan,
            "Avg cost rate": top_jobs["avg_cost_rate"],
        })
        if client_col:
            jobs_table.insert(1, "Client", top_jobs[client_col].fillna("—").astype(str))
        if "department_final" in top_jobs.columns and top_jobs["department_final"].notna().any():
            jobs_table.insert(2 if client_col else 1, "Department", top_jobs["department_final"].fillna("—").astype(str))
        if "category" in top_jobs.columns and top_jobs["category"].notna().any():
            jobs_table.insert(3 if client_col else 2, "Category", top_jobs["category"].fillna("—").astype(str))

        if "quote_rate" in top_jobs.columns and top_jobs["quote_rate"].notna().any():
            jobs_table["Quote rate"] = top_jobs["quote_rate"]
        if "revenue_at_risk" in top_jobs.columns and top_jobs["revenue_at_risk"].notna().any():
            jobs_table["Revenue at risk ($)"] = top_jobs["revenue_at_risk"]

        job_formatters = {
            "Quoted hours": "{:,.1f}",
            "Actual hours": "{:,.1f}",
            "Overrun hours": "{:,.1f}",
            "Overrun cost ($)": lambda x: fmt_currency(x) if pd.notna(x) else "—",
            "Avg cost rate": lambda x: f"${x:,.0f}/hr" if pd.notna(x) else "—",
        }
        if "Quote rate" in jobs_table.columns:
            job_formatters["Quote rate"] = lambda x: f"${x:,.0f}/hr" if pd.notna(x) else "—"
        if "Revenue at risk ($)" in jobs_table.columns:
            job_formatters["Revenue at risk ($)"] = lambda x: fmt_currency(x) if pd.notna(x) else "—"

        styled_jobs = jobs_table.style.format(job_formatters)

        with right:
            st.dataframe(styled_jobs, use_container_width=True, hide_index=True)

    if "staff_name" in scoped_for_detail.columns and not top_jobs.empty:
        staff_scope = scoped_for_detail.copy()
        if pd.isna(selected_task):
            staff_scope = staff_scope[staff_scope["task_name"].isna()]
        else:
            staff_scope = staff_scope[staff_scope["task_name"] == selected_task]
        staff_scope = staff_scope[staff_scope["job_no"].isin(top_jobs["job_no"].tolist())]

        if not staff_scope.empty:
            if has_cost:
                staff_table = staff_scope.groupby("staff_name", dropna=False).agg(
                    hours=("hours_raw", "sum"),
                    cost=("base_cost", "sum"),
                ).reset_index()
            else:
                staff_table = staff_scope.groupby("staff_name", dropna=False).agg(
                    hours=("hours_raw", "sum"),
                ).reset_index()
                staff_table["cost"] = np.nan

            staff_table = staff_table.sort_values("hours", ascending=False).head(12)
            staff_table["Staff"] = staff_table["staff_name"].fillna("(Unassigned)").astype(str)

            st.markdown("**Who drives it: top contributing staff**")
            s1, s2 = st.columns([1.0, 1.2])
            with s1:
                fig_staff = px.bar(
                    staff_table.sort_values("hours", ascending=True),
                    x="hours",
                    y="Staff",
                    orientation="h",
                    color="hours",
                    color_continuous_scale="Teal",
                    labels={"hours": "Hours", "Staff": "Staff"},
                    height=320,
                )
                fig_staff.update_layout(margin=dict(l=10, r=10, t=35, b=10), title="Staff effort concentration")
                st.plotly_chart(fig_staff, use_container_width=True)
                st.caption(
                    "How to read: shows who contributes most delivery effort to the selected task across the top offending jobs."
                )

            with s2:
                staff_display = staff_table[["Staff", "hours", "cost"]].rename(columns={"hours": "Hours", "cost": "Cost"})
                styled_staff = staff_display.style.format(
                    {
                        "Hours": "{:,.1f}",
                        "Cost": lambda x: fmt_currency(x) if pd.notna(x) else "—",
                    }
                )
                st.dataframe(styled_staff, use_container_width=True, hide_index=True)

    _show_actions(task_scope=task_scope, selected_task_row=selected_task_row)


def main() -> None:
    inject_theme()

    st.markdown(
        """
        <div class="rq-hero">
            <div class="rq-eyebrow">Delivery Assurance</div>
            <h1 class="rq-title">Recurring Quote Overruns</h1>
            <div class="rq-sub">
                Identify tasks that repeatedly exceed quoted budgets, quantify margin erosion,
                and drill straight to the jobs and people driving leakage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    if df.empty:
        st.error("No data available.")
        st.stop()

    if "department_final" not in df.columns or df["department_final"].dropna().empty:
        st.info("Department data is not available in this dataset.")
        st.stop()

    dept_options = sorted(df["department_final"].dropna().astype(str).unique().tolist())

    top_left, top_right = st.columns([1.4, 1.0])
    with top_left:
        dept = st.selectbox("Department", options=dept_options, index=0)
    with top_right:
        st.metric("Rows in selected department", f"{int((df['department_final'].astype(str) == str(dept)).sum()):,}")

    df_dept = df[df["department_final"].astype(str) == str(dept)].copy()

    tabs = st.tabs([
        "Level 1: Department Diagnostic",
        "Level 2: Category Diagnostic",
    ])

    with tabs[0]:
        render_recurring_task_overruns_section(
            df_scope=df_dept,
            scope_label=f"Department: {dept}",
            department=dept,
            category=None,
        )

    with tabs[1]:
        if df_dept.empty:
            st.info("No rows for this department.")
            return

        category_col = get_category_col(df_dept)
        category_map: Dict[str, object] = {}
        if category_col in df_dept.columns:
            non_null_values = sorted(df_dept[category_col].dropna().unique().tolist(), key=lambda x: str(x))
            for value in non_null_values:
                category_map[str(value)] = value
            if df_dept[category_col].isna().any():
                category_map["(Uncategorised)"] = "(Uncategorised)"

        if not category_map:
            st.info("Category data is not available in this scope.")
            return

        selected_category_label = st.selectbox(
            "Category",
            options=list(category_map.keys()),
            index=0,
        )
        selected_category_value = category_map[selected_category_label]

        render_recurring_task_overruns_section(
            df_scope=df_dept,
            scope_label=f"Category: {dept} : {selected_category_label}",
            department=dept,
            category=selected_category_value,
        )


if __name__ == "__main__":
    main()
