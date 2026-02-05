"""
Revenue Reconciliation & Leakage Diagnostics

Standalone page for reconciling quoted amounts vs actual revenue,
identifying leakage patterns, and diagnosing root causes.
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

st.set_page_config(
    page_title="Revenue Reconciliation",
    page_icon="💰",
    layout="wide",
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.ui.state import init_state
from src.ui.formatting import build_job_name_lookup
from src.metrics.revenue_reconciliation import (
    compute_job_reconciliation,
    compute_reconciliation_summary,
)
from src.ui.revenue_recon_components import (
    render_portfolio_summary,
    render_distribution_analysis,
    render_concentration_analysis,
    render_pattern_analysis,
    render_job_deep_dive,
    render_data_quality_section,
)

init_state()


def main():
    # Header
    st.title("💰 Revenue Reconciliation & Leakage Diagnostics")
    st.caption("*Where does revenue vs quote diverge, and why?*")

    st.markdown(
        """
    This page reconciles **quoted amounts** against **actual revenue** at the job level
    to identify commercial leakage and diagnose root causes.

    **Key questions answered:**
    - Are we collecting what we quote?
    - Is leakage concentrated or systemic?
    - Which departments/job types leak most?
    - Why did specific jobs leak?
    """
    )

    st.divider()

    # Load data
    with st.spinner("Loading data..."):
        try:
            df = load_fact_timesheet()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    if len(df) == 0:
        st.warning("No data available")
        return

    # Optional filters
    with st.sidebar:
        st.header("Filters")

        # Time filter
        if "month_key" in df.columns:
            months = sorted(df["month_key"].dropna().unique())
            if len(months) > 1:
                date_range = st.select_slider(
                    "Date Range",
                    options=months,
                    value=(months[0], months[-1]),
                    format_func=lambda x: x.strftime("%b %Y")
                    if hasattr(x, "strftime")
                    else str(x),
                )
                df = df[
                    (df["month_key"] >= date_range[0])
                    & (df["month_key"] <= date_range[1])
                ]

        # Department filter
        if "department_final" in df.columns:
            depts = ["All"] + sorted(df["department_final"].dropna().unique().tolist())
            selected_dept = st.selectbox("Department", depts)
            if selected_dept != "All":
                df = df[df["department_final"] == selected_dept]

        st.divider()
        st.caption(f"Analyzing {len(df):,} rows")

    # Compute reconciliation
    with st.spinner("Computing reconciliation..."):
        recon_df = compute_job_reconciliation(df)
        summary = compute_reconciliation_summary(recon_df)
        job_name_lookup = build_job_name_lookup(df, "job_no", "job_name")

    # Render sections
    render_portfolio_summary(summary)

    st.divider()

    render_distribution_analysis(recon_df, summary)

    st.divider()

    render_concentration_analysis(recon_df, summary, job_name_lookup)

    st.divider()

    render_pattern_analysis(recon_df)

    st.divider()

    render_job_deep_dive(df, recon_df, job_name_lookup)

    st.divider()

    render_data_quality_section(recon_df)


if __name__ == "__main__":
    main()
