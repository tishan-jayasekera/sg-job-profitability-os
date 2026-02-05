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
    render_alert_strip,
    render_job_queue,
    render_selected_job_panel,
)
from src.ui.formatting import build_job_name_lookup
from src.ui.state import init_state


st.set_page_config(page_title="Active Delivery", page_icon="🚦", layout="wide")

init_state()


def _resolve_selected_job(jobs_df) -> Optional[str]:
    if len(jobs_df) == 0:
        return None

    selected = st.session_state.get("selected_job")
    if selected in jobs_df["job_no"].values:
        return selected

    top_job = jobs_df.iloc[0]["job_no"]
    st.session_state.selected_job = top_job
    return top_job


def main() -> None:
    st.markdown("## Active Delivery Control Tower")

    df = load_fact_timesheet()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        depts = ["All"] + sorted(df["department_final"].dropna().unique().tolist())
        dept = st.selectbox(
            "Department",
            depts,
            key="dc_dept",
            label_visibility="collapsed",
        )

    df_filtered = df if dept == "All" else df[df["department_final"] == dept]

    with col2:
        cat_col = get_category_col(df_filtered)
        cats = ["All"] + sorted(df_filtered[cat_col].dropna().unique().tolist())
        cat = st.selectbox(
            "Category",
            cats,
            key="dc_cat",
            label_visibility="collapsed",
        )

    with col3:
        recency_label = st.selectbox(
            "Recency",
            ["Last 14d", "Last 28d", "Last 60d", "Last 90d"],
            index=1,
            key="dc_recency",
            label_visibility="collapsed",
        )

    recency_days = int(recency_label.split()[1].replace("d", ""))

    if dept != "All":
        df = df[df["department_final"] == dept]
    if cat != "All":
        df = df[df[cat_col] == cat]

    jobs_df = compute_delivery_control_view(df, recency_days=recency_days)

    if len(jobs_df) == 0:
        st.info("No active jobs found for the selected filters.")
        return

    job_name_lookup = build_job_name_lookup(df)

    render_alert_strip(jobs_df)
    st.divider()

    left_col, right_col = st.columns([1, 2])

    with left_col:
        include_green = st.session_state.get("include_green_jobs", False)
        selected_job = render_job_queue(jobs_df, job_name_lookup, include_green)

    with right_col:
        selected_job = selected_job or _resolve_selected_job(jobs_df)
        if selected_job and selected_job in jobs_df["job_no"].values:
            render_selected_job_panel(df, jobs_df, selected_job, job_name_lookup)
        else:
            st.info("← Select a job from the queue")


if __name__ == "__main__":
    main()
