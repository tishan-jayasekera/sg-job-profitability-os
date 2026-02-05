"""
Active Delivery Control Tower

Three-tab workflow for portfolio triage, job diagnosis, and interventions.
"""
from __future__ import annotations

import streamlit as st
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state
from src.ui.formatting import build_job_name_lookup
from src.data.loader import load_fact_timesheet
from src.data.semantic import get_category_col
from src.data.job_lifecycle import get_job_task_attribution
from src.metrics.delivery_control import (
    compute_benchmarks,
    compute_delivery_control_view,
    compute_root_cause_drivers,
)
from src.ui.delivery_control_components import (
    render_portfolio_kpi_strip,
    render_portfolio_risk_table,
    render_risk_map,
    render_weekly_focus,
    render_job_health_card,
    render_root_cause_drivers,
    render_task_breakdown,
    render_staff_attribution,
    build_staff_attribution_df,
    render_intervention_builder,
    render_next_7_days_plan,
    render_export_section,
)


st.set_page_config(page_title="Active Delivery", page_icon="🚦", layout="wide")

init_state()


def _ensure_selected_job(jobs_df: pd.DataFrame) -> Optional[str]:
    if len(jobs_df) == 0:
        return None

    selected = st.session_state.get("selected_job")
    if selected in jobs_df["job_no"].values:
        return selected

    top_job = jobs_df.sort_values("risk_score", ascending=False).iloc[0]["job_no"]
    st.session_state.selected_job = top_job
    return top_job


def main() -> None:
    st.markdown("## Active Delivery Control Tower")
    st.caption("Identify risk, diagnose root causes, and execute interventions.")

    df = load_fact_timesheet()

    # Filters
    departments = ["All"] + sorted(df["department_final"].dropna().unique().tolist())
    dept_col, cat_col = st.columns([1, 1])

    with dept_col:
        selected_dept = st.selectbox("Department", departments, key="dc_dept")

    df_dept = df if selected_dept == "All" else df[df["department_final"] == selected_dept]
    category_col = get_category_col(df_dept)
    categories = ["All"] + sorted(df_dept[category_col].dropna().unique().tolist())

    with cat_col:
        selected_category = st.selectbox("Category", categories, key="dc_category")

    df_scope = df_dept if selected_category == "All" else df_dept[df_dept[category_col] == selected_category]

    jobs_df = compute_delivery_control_view(df_scope, recency_days=28)
    if len(jobs_df) == 0:
        st.warning("No active jobs found for the selected filters.")
        return

    job_name_lookup = build_job_name_lookup(df_scope)

    # Ensure selection is always valid for downstream tabs
    _ensure_selected_job(jobs_df)

    tab1, tab2, tab3 = st.tabs([
        "🚦 Portfolio Triage",
        "🔎 Job Diagnosis",
        "🛠️ Interventions & Execution",
    ])

    with tab1:
        render_portfolio_kpi_strip(jobs_df)
        render_portfolio_risk_table(jobs_df, job_name_lookup)
        render_risk_map(jobs_df, job_name_lookup)
        render_weekly_focus(jobs_df, job_name_lookup)

    with tab2:
        selected_job = _ensure_selected_job(jobs_df)
        if not selected_job:
            st.info("Select a job in the Portfolio Triage tab to view diagnosis.")
            return

        job_row = jobs_df[jobs_df["job_no"] == selected_job].iloc[0]
        benchmarks = compute_benchmarks(df_scope)
        drivers = compute_root_cause_drivers(df_scope, job_row, benchmarks)

        render_job_health_card(job_row, job_name_lookup)
        render_root_cause_drivers(drivers)
        render_task_breakdown(df_scope, selected_job)
        render_staff_attribution(df_scope, selected_job, jobs_df)

    with tab3:
        selected_job = _ensure_selected_job(jobs_df)
        if not selected_job:
            st.info("Select a job in the Portfolio Triage tab to view interventions.")
            return

        job_row = jobs_df[jobs_df["job_no"] == selected_job].iloc[0]
        benchmarks = compute_benchmarks(df_scope)
        drivers = compute_root_cause_drivers(df_scope, job_row, benchmarks)

        interventions = render_intervention_builder(selected_job, drivers)
        plan_md = render_next_7_days_plan(selected_job, job_row, drivers, interventions)

        task_df = get_job_task_attribution(df_scope, selected_job)
        staff_df = build_staff_attribution_df(df_scope, selected_job, jobs_df)

        render_export_section(
            selected_job,
            job_row,
            task_df,
            staff_df,
            drivers,
            interventions,
            plan_md,
        )


if __name__ == "__main__":
    main()
