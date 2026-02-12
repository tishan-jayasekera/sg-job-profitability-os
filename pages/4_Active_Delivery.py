"""
Active Delivery Control Tower (Command Center)

Single-page master-detail layout for delivery risk management.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.data.semantic import get_category_col
from src.metrics.delivery_control import compute_delivery_control_view
from src.ui.delivery_control_components import (
    inject_delivery_control_theme,
    render_job_queue,
    render_selected_job_panel,
    summarize_alerts,
)
from src.ui.formatting import build_job_name_lookup
from src.ui.state import init_state


st.set_page_config(page_title="Active Delivery", page_icon="🚦", layout="wide")

init_state()


def _resolve_selected_job(jobs_df) -> Optional[str]:
    """Return selected job if valid, else first job. Always sets state."""
    if len(jobs_df) == 0:
        return None

    selected = st.session_state.get("selected_job")
    selected_str = str(selected) if selected is not None else None
    job_values = jobs_df["job_no"].astype(str).values
    if selected_str and selected_str in job_values:
        return selected_str

    first_job = str(jobs_df.iloc[0]["job_no"])
    st.session_state.selected_job = first_job
    return first_job


def _render_master_detail(df_scope, df_all, jobs_df, job_name_lookup) -> None:
    left_col, right_col = st.columns([1, 2])

    with left_col:
        include_green = st.session_state.get("include_green_jobs", False)
        selected_job = render_job_queue(jobs_df, job_name_lookup, include_green)

    with right_col:
        if selected_job is None:
            st.info("← No jobs available for the current queue filter")
            return
        selected_job = selected_job or _resolve_selected_job(jobs_df)
        if selected_job and selected_job in jobs_df["job_no"].astype(str).values:
            render_selected_job_panel(
                df_scope,
                jobs_df,
                selected_job,
                job_name_lookup,
                df_all=df_all,
            )
        else:
            st.info("← Select a job from the queue")


def main() -> None:
    inject_delivery_control_theme()
    st.markdown('<div class="dc-page-title">Active Delivery Control Tower</div>', unsafe_allow_html=True)

    df_all = load_fact_timesheet()
    with st.container(border=True):
        col1, col2, col3, col4, col5 = st.columns([1.1, 1.1, 1.2, 0.8, 0.8], gap="small")

        with col1:
            st.markdown('<div class="dc-filter-label">Department</div>', unsafe_allow_html=True)
            depts = ["All"] + sorted(df_all["department_final"].dropna().unique().tolist())
            dept = st.selectbox(
                "Department",
                depts,
                key="dc_dept",
                label_visibility="collapsed",
            )

        df_scope = df_all if dept == "All" else df_all[df_all["department_final"] == dept]

        with col2:
            st.markdown('<div class="dc-filter-label">Category</div>', unsafe_allow_html=True)
            cat_col = get_category_col(df_scope)
            cats = ["All"] + sorted(df_scope[cat_col].dropna().unique().tolist())
            cat = st.selectbox(
                "Category",
                cats,
                key="dc_cat",
                label_visibility="collapsed",
            )

        with col3:
            st.markdown('<div class="dc-filter-label">Recency</div>', unsafe_allow_html=True)
            recency_label = st.selectbox(
                "Recency",
                ["Last 14d", "Last 28d", "Last 60d", "Last 90d"],
                index=1,
                key="dc_recency",
                label_visibility="collapsed",
            )

        recency_days = int(recency_label.split()[1].replace("d", ""))
        if cat != "All":
            df_scope = df_scope[df_scope[cat_col] == cat]

        jobs_df = compute_delivery_control_view(df_scope, recency_days=recency_days)
        summary = summarize_alerts(jobs_df) if len(jobs_df) > 0 else {"red_count": 0, "amber_count": 0, "margin_at_risk": 0}

        with col4:
            st.markdown(
                f"""
                <div class="dc-pill dc-pill-critical">
                    <div class="dc-pill-title">Critical Exposure</div>
                    <div class="dc-pill-value">{int(summary["red_count"])} jobs</div>
                    <div class="dc-pill-sub">${summary["margin_at_risk"]:,.0f} margin at risk</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col5:
            st.markdown(
                f"""
                <div class="dc-pill dc-pill-watch">
                    <div class="dc-pill-title">Watchlist</div>
                    <div class="dc-pill-value">{int(summary["amber_count"])} jobs</div>
                    <div class="dc-pill-sub">Require review this cycle</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="dc-command-divider"></div>', unsafe_allow_html=True)

    if len(jobs_df) == 0:
        st.info("No active jobs found for the selected filters.")
        return

    job_name_lookup = build_job_name_lookup(df_all)
    _render_master_detail(df_scope, df_all, jobs_df, job_name_lookup)


if __name__ == "__main__":
    main()
