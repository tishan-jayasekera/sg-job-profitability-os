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


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return load_fact_timesheet()


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


def render_recurring_task_overruns_section(
    df_scope: pd.DataFrame,
    scope_label: str,
    department: Optional[str],
    category: Optional[str],
) -> None:
    st.subheader("Recurring Quote Overruns (Task Margin Erosion)")
    st.caption(f"{scope_label} | Tasks ranked by repeated margin leakage, not one-off variance.")

    if "quoted_time_total" not in df_scope.columns:
        st.info("Quoted hours not available — cannot compute quote overruns.")
        return

    missing_core = [c for c in ["job_no", "task_name", "hours_raw"] if c not in df_scope.columns]
    if missing_core:
        st.info(f"Required columns missing: {', '.join(missing_core)}")
        return

    has_cost = "base_cost" in df_scope.columns
    scope_key = _scope_key(scope_label, department, category)

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
                f"**{r['task_name'] if pd.notna(r['task_name']) else '(Unspecified task)'}** "
                f"leaks {fmt_currency(r['total_overrun_cost'])} "
                f"at {r['overrun_rate'] * 100:.0f}% overrun frequency"
            )
            for _, r in top_rows.iterrows()
        ]
    else:
        top_summary = [
            (
                f"**{r['task_name'] if pd.notna(r['task_name']) else '(Unspecified task)'}** "
                f"overruns in {r['overrun_rate'] * 100:.0f}% of quoted jobs "
                f"({r['total_overrun_hours']:,.1f}h total overrun)"
            )
            for _, r in top_rows.iterrows()
        ]
    st.markdown(f"**So what:** {'; '.join(top_summary)}.")

    show_revenue_at_risk = (
        "total_revenue_at_risk" in task_overruns.columns
        and task_overruns["total_revenue_at_risk"].notna().any()
    )

    recurring_table = pd.DataFrame({
        "Task": task_overruns["task_name"].fillna("(Unspecified task)").astype(str),
        "Overrun rate (%)": task_overruns["overrun_rate"].apply(lambda v: f"{v * 100:.0f}%"),
        "Overrun jobs": task_overruns["overrun_jobs"].fillna(0).astype(int),
        "Jobs with quote": task_overruns["jobs_with_quote"].fillna(0).astype(int),
        "Total overrun hours": task_overruns["total_overrun_hours"].apply(lambda v: f"{v:,.1f}"),
        "Est. margin erosion ($)": (
            task_overruns["total_overrun_cost"].apply(fmt_currency)
            if has_cost
            else "—"
        ),
        "Avg overrun (%)": task_overruns["avg_overrun_pct"].apply(_format_signed_pct),
    })
    if show_revenue_at_risk:
        recurring_table["Revenue at risk ($)"] = task_overruns["total_revenue_at_risk"].apply(fmt_currency)

    st.markdown("**Recurring Margin-Leak Tasks**")
    st.dataframe(recurring_table, use_container_width=True, hide_index=True)

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

    top_jobs = get_overrun_jobs_for_task(
        df_scope,
        task_name=selected_task,
        department=department,
        category=category,
        time_window=time_window,
        n=15,
    )

    st.markdown("**Top Offending Jobs**")
    if top_jobs.empty:
        st.info("No overrun jobs found for this task in the selected scope/window.")
    else:
        client_col = None
        if "client_name" in top_jobs.columns and top_jobs["client_name"].notna().any():
            client_col = "client_name"
        elif "client_group" in top_jobs.columns and top_jobs["client_group"].notna().any():
            client_col = "client_group"

        jobs_table = pd.DataFrame({
            "Job": top_jobs["job_no"].astype(str),
            "Quoted hours": top_jobs["quoted_hours"].apply(lambda v: f"{v:,.1f}"),
            "Actual hours": top_jobs["actual_hours"].apply(lambda v: f"{v:,.1f}"),
            "Overrun hours": top_jobs["overrun_hours"].apply(lambda v: f"{v:,.1f}"),
            "Overrun cost ($)": top_jobs["overrun_cost"].apply(fmt_currency) if has_cost else "—",
            "Avg cost rate": (
                top_jobs["avg_cost_rate"].apply(lambda v: f"${v:,.0f}/hr" if pd.notna(v) else "—")
                if has_cost
                else "—"
            ),
        })
        if client_col:
            jobs_table.insert(1, "Client", top_jobs[client_col].fillna("—").astype(str))
        if "department_final" in top_jobs.columns and top_jobs["department_final"].notna().any():
            jobs_table.insert(2 if client_col else 1, "Department", top_jobs["department_final"].fillna("—").astype(str))
        if "category" in top_jobs.columns and top_jobs["category"].notna().any():
            jobs_table.insert(3 if client_col else 2, "Category", top_jobs["category"].fillna("—").astype(str))
        if "quote_rate" in top_jobs.columns and top_jobs["quote_rate"].notna().any():
            jobs_table["Quote rate"] = top_jobs["quote_rate"].apply(lambda v: f"${v:,.0f}/hr" if pd.notna(v) else "—")
        if "revenue_at_risk" in top_jobs.columns and top_jobs["revenue_at_risk"].notna().any():
            jobs_table["Revenue at risk ($)"] = top_jobs["revenue_at_risk"].apply(fmt_currency)

        st.dataframe(jobs_table, use_container_width=True, hide_index=True)

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
                staff_table = staff_table.sort_values("hours", ascending=False).head(15)
                staff_display = pd.DataFrame({
                    "Staff": staff_table["staff_name"].fillna("(Unassigned)").astype(str),
                    "Hours": staff_table["hours"].apply(lambda v: f"{v:,.1f}"),
                    "Cost": staff_table["cost"].apply(fmt_currency),
                })
            else:
                staff_table = staff_scope.groupby("staff_name", dropna=False).agg(
                    hours=("hours_raw", "sum"),
                ).reset_index()
                staff_table = staff_table.sort_values("hours", ascending=False).head(15)
                staff_display = pd.DataFrame({
                    "Staff": staff_table["staff_name"].fillna("(Unassigned)").astype(str),
                    "Hours": staff_table["hours"].apply(lambda v: f"{v:,.1f}"),
                    "Cost": "—",
                })

            st.markdown("**Top Contributing Staff (Selected Task + Top Jobs)**")
            st.dataframe(staff_display, use_container_width=True, hide_index=True)

    if pd.isna(selected_task):
        task_scope = scoped_for_detail[scoped_for_detail["task_name"].isna()].copy()
    else:
        task_scope = scoped_for_detail[scoped_for_detail["task_name"] == selected_task].copy()

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

    card_1 = {
        "title": "Fix the quote baseline",
        "body": (
            f"Overruns hit {overrun_rate * 100:.0f}% of quoted jobs with {avg_overrun_pct * 100:.0f}% average overshoot. "
            "Increase standard hours for this task, add complexity drivers, and update quote builder defaults."
            if rule_1_trigger
            else "Refresh baseline hours from the latest delivery data and tighten estimate guardrails in the quote template."
        ),
    }

    if rule_2_evaluable and rule_2_trigger:
        card_2_body = (
            f"{mismatch_share * 100:.0f}% of task rows are not quote-matched. Enforce variation approval before any extra "
            "delivery effort and track exceptions weekly."
        )
    elif rule_2_evaluable:
        card_2_body = (
            f"Quote mismatch share is {mismatch_share * 100:.0f}%. Keep change-control gates active and review mismatches monthly."
        )
    else:
        card_2_body = "Introduce scope variation control on this task with mandatory approval and a weekly scope-change log."

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
        {"title": "Fix the quote baseline", "body": card_1["body"]},
        {"title": "Stop scope creep", "body": card_2_body},
        {"title": "Fix staffing / rate capture", "body": card_3_body},
    ]

    st.markdown("**Recommended actions**")
    action_cols = st.columns(3)
    for col, action in zip(action_cols, actions):
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid #e9ecef;border-radius:10px;padding:0.85rem;background:#fafbfc;min-height:170px;">
                    <div style="font-weight:600;color:#1a1a1a;">{action['title']}</div>
                    <div style="font-size:0.86rem;color:#555;margin-top:0.4rem;line-height:1.35;">{action['body']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.title("📉 Recurring Quote Overruns")
    st.caption("Find tasks that repeatedly exceed quoted budgets and erode margin.")

    df = load_data()
    if df.empty:
        st.error("No data available.")
        st.stop()

    if "department_final" not in df.columns or df["department_final"].dropna().empty:
        st.info("Department data is not available in this dataset.")
        st.stop()

    dept_options = sorted(df["department_final"].dropna().astype(str).unique().tolist())
    dept = st.selectbox("Department", options=dept_options, index=0)

    st.markdown("### Level 1: Department Diagnostic")
    df_dept = df[df["department_final"].astype(str) == str(dept)].copy()
    render_recurring_task_overruns_section(
        df_scope=df_dept,
        scope_label=f"Department: {dept}",
        department=dept,
        category=None,
    )

    st.divider()
    st.markdown(f"### Level 2: Category Diagnostic — {dept}")

    if df_dept.empty:
        st.info("No rows for this department.")
        st.stop()

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
