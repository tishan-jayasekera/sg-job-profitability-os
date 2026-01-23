"""
Job Mix & Implied FTE Demand Page

Analyze job intake patterns and implied capacity demand.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state, get_state, set_state
from src.ui.layout import section_header
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_count
from src.ui.charts import time_series, quadrant_scatter, horizontal_bar
from src.data.loader import load_fact_timesheet
from src.data.semantic import safe_quote_job_task, get_category_col
from src.data.cohorts import filter_by_time_window
from src.config import config


st.set_page_config(page_title="Job Mix & Demand", page_icon="📊", layout="wide")

init_state()


def calculate_job_cohorts(df: pd.DataFrame, cohort_type: str = "first_activity") -> pd.DataFrame:
    """
    Calculate job cohorts by intake period.
    
    cohort_type: 'first_activity', 'first_revenue', or 'quote_date'
    """
    if "job_no" not in df.columns:
        return pd.DataFrame()
    
    if cohort_type == "first_activity":
        # First month with hours > 0
        cohort = df[df["hours_raw"] > 0].groupby("job_no")["month_key"].min().reset_index()
        cohort.columns = ["job_no", "cohort_month"]
    
    elif cohort_type == "first_revenue":
        # First month with revenue > 0
        cohort = df[df["rev_alloc"] > 0].groupby("job_no")["month_key"].min().reset_index()
        cohort.columns = ["job_no", "cohort_month"]
    
    else:
        # Default to first activity
        cohort = df[df["hours_raw"] > 0].groupby("job_no")["month_key"].min().reset_index()
        cohort.columns = ["job_no", "cohort_month"]
    
    return cohort


def calculate_job_mix_metrics(df: pd.DataFrame, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate job mix metrics by cohort month."""
    
    # Get job-level attributes
    category_col = get_category_col(df)
    job_attrs = df.groupby("job_no").agg(
        department_final=("department_final", "first"),
        job_category=(category_col, "first"),
    ).reset_index()
    
    # Safe quote totals per job
    job_task = safe_quote_job_task(df)
    if len(job_task) > 0:
        job_quotes = job_task.groupby("job_no").agg(
            quoted_hours=("quoted_time_total", "sum"),
            quoted_amount=("quoted_amount_total", "sum"),
        ).reset_index()
    else:
        job_quotes = pd.DataFrame({"job_no": [], "quoted_hours": [], "quoted_amount": []})
    
    # Merge
    jobs = job_attrs.merge(cohort_df, on="job_no", how="inner")
    jobs = jobs.merge(job_quotes, on="job_no", how="left")
    
    # Fill NaN
    jobs["quoted_hours"] = jobs["quoted_hours"].fillna(0)
    jobs["quoted_amount"] = jobs["quoted_amount"].fillna(0)
    
    # Aggregate by cohort month
    monthly = jobs.groupby("cohort_month").agg(
        job_count=("job_no", "nunique"),
        total_quoted_hours=("quoted_hours", "sum"),
        total_quoted_amount=("quoted_amount", "sum"),
        avg_quoted_hours=("quoted_hours", "mean"),
        avg_quoted_amount=("quoted_amount", "mean"),
    ).reset_index()
    
    # Derived metrics
    monthly["value_per_hour"] = np.where(
        monthly["total_quoted_hours"] > 0,
        monthly["total_quoted_amount"] / monthly["total_quoted_hours"],
        np.nan
    )
    
    # Implied FTE demand
    # FTE = quoted_hours / (weeks_in_month * 38)
    weeks_per_month = 4.33
    monthly["implied_fte"] = monthly["total_quoted_hours"] / (weeks_per_month * config.CAPACITY_HOURS_PER_WEEK)
    
    return monthly.sort_values("cohort_month")


def calculate_demand_vs_supply(df: pd.DataFrame, monthly_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate demand vs supply comparison."""
    
    # Get supply capacity by month
    # Count unique staff per month and their capacity
    staff_months = df.groupby(["month_key", "staff_name"]).agg(
        fte_scaling=("fte_hours_scaling", "first"),
    ).reset_index()
    
    supply = staff_months.groupby("month_key").agg(
        staff_count=("staff_name", "nunique"),
        avg_fte=("fte_scaling", "mean"),
    ).reset_index()
    
    # Supply capacity
    weeks_per_month = 4.33
    supply["supply_fte"] = supply["staff_count"] * supply["avg_fte"]
    supply["supply_capacity_hours"] = supply["supply_fte"] * weeks_per_month * config.CAPACITY_HOURS_PER_WEEK
    
    # Merge with demand
    comparison = monthly_metrics.merge(
        supply.rename(columns={"month_key": "cohort_month"}),
        on="cohort_month",
        how="left"
    )
    
    # Calculate slack
    comparison["implied_utilisation"] = np.where(
        comparison["supply_capacity_hours"] > 0,
        comparison["total_quoted_hours"] / comparison["supply_capacity_hours"] * 100,
        np.nan
    )
    
    comparison["slack_pct"] = 100 - comparison["implied_utilisation"]
    
    return comparison


def main():
    st.title("Job Mix & Implied FTE Demand")
    st.caption("Analyze job intake patterns and capacity implications")
    
    # Load data
    df_all = load_fact_timesheet()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    cohort_options = {
        "first_activity": "First Activity Month",
        "first_revenue": "First Revenue Month",
    }
    
    cohort_type = st.sidebar.selectbox(
        "Cohort Definition",
        options=list(cohort_options.keys()),
        format_func=lambda x: cohort_options[x]
    )
    set_state("cohort_definition", cohort_type)
    
    time_window = st.sidebar.selectbox(
        "Time Window",
        options=["6m", "12m", "24m", "all"],
        format_func=lambda x: f"Last {x}" if x != "all" else "All Time",
        index=1
    )
    
    departments = ["All"] + sorted(df_all["department_final"].dropna().unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)
    
    if selected_dept != "All":
        df_all = df_all[df_all["department_final"] == selected_dept]
    
    # Calculate cohorts
    cohort_df = calculate_job_cohorts(df_all, cohort_type)
    
    if len(cohort_df) == 0:
        st.warning("No job data available for analysis.")
        return
    
    # Calculate metrics
    monthly_metrics = calculate_job_mix_metrics(df_all, cohort_df)
    monthly_metrics = filter_by_time_window(monthly_metrics, time_window, date_col="cohort_month")
    demand_supply = calculate_demand_vs_supply(df_all, monthly_metrics)
    
    # =========================================================================
    # SECTION A: KPI STRIP
    # =========================================================================
    section_header("Job Mix Summary")
    
    kpi_cols = st.columns(6)
    
    total_jobs = monthly_metrics["job_count"].sum()
    total_quoted_hours = monthly_metrics["total_quoted_hours"].sum()
    total_quoted_amount = monthly_metrics["total_quoted_amount"].sum()
    avg_hours_per_job = total_quoted_hours / total_jobs if total_jobs > 0 else 0
    avg_amount_per_job = total_quoted_amount / total_jobs if total_jobs > 0 else 0
    value_per_hour = total_quoted_amount / total_quoted_hours if total_quoted_hours > 0 else 0
    
    with kpi_cols[0]:
        st.metric("Jobs", fmt_count(total_jobs))
    
    with kpi_cols[1]:
        st.metric("Total Quoted", fmt_currency(total_quoted_amount))
    
    with kpi_cols[2]:
        st.metric("Avg $/Job", fmt_currency(avg_amount_per_job))
    
    with kpi_cols[3]:
        st.metric("Total Hours", fmt_hours(total_quoted_hours))
    
    with kpi_cols[4]:
        st.metric("Avg Hrs/Job", fmt_hours(avg_hours_per_job))
    
    with kpi_cols[5]:
        st.metric("$/Hour", fmt_currency(value_per_hour))
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION B: TREND CHARTS
    # =========================================================================
    section_header("Monthly Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Job count trend
        fig = time_series(
            monthly_metrics,
            x="cohort_month",
            y="job_count",
            title="Jobs per Month",
            y_title="Job Count"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Avg hours per job
        fig = time_series(
            monthly_metrics,
            x="cohort_month",
            y="avg_quoted_hours",
            title="Avg Quoted Hours per Job",
            y_title="Hours"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Avg value per job
        fig = time_series(
            monthly_metrics,
            x="cohort_month",
            y="avg_quoted_amount",
            title="Avg Quoted Value per Job",
            y_title="$"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Value per hour
        fig = time_series(
            monthly_metrics,
            x="cohort_month",
            y="value_per_hour",
            title="Value per Quoted Hour",
            y_title="$/hr"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION C: JOB QUADRANT
    # =========================================================================
    section_header("Job Value vs Effort", "Identify 'high value / low hours' segment")
    
    # Get job-level data
    job_task = safe_quote_job_task(df_all)
    if len(job_task) > 0:
        job_quotes = job_task.groupby("job_no").agg(
            quoted_hours=("quoted_time_total", "sum"),
            quoted_amount=("quoted_amount_total", "sum"),
        ).reset_index()
        
        job_quotes = job_quotes[(job_quotes["quoted_hours"] > 0) & (job_quotes["quoted_amount"] > 0)]
        
        if len(job_quotes) > 5:
            fig = quadrant_scatter(
                job_quotes,
                x="quoted_hours",
                y="quoted_amount",
                hover_name="job_no",
                title="Job Portfolio: Value vs Effort",
                x_title="Quoted Hours",
                y_title="Quoted Amount ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats by quadrant
            med_hours = job_quotes["quoted_hours"].median()
            med_amount = job_quotes["quoted_amount"].median()
            
            job_quotes["quadrant"] = np.where(
                (job_quotes["quoted_hours"] < med_hours) & (job_quotes["quoted_amount"] > med_amount),
                "High Value / Low Effort",
                np.where(
                    (job_quotes["quoted_hours"] >= med_hours) & (job_quotes["quoted_amount"] > med_amount),
                    "High Value / High Effort",
                    np.where(
                        (job_quotes["quoted_hours"] < med_hours) & (job_quotes["quoted_amount"] <= med_amount),
                        "Low Value / Low Effort",
                        "Low Value / High Effort"
                    )
                )
            )
            
            quadrant_summary = job_quotes.groupby("quadrant").agg(
                jobs=("job_no", "count"),
                avg_hours=("quoted_hours", "mean"),
                avg_value=("quoted_amount", "mean"),
            ).reset_index()
            
            st.dataframe(
                quadrant_summary.rename(columns={
                    "quadrant": "Quadrant",
                    "jobs": "Jobs",
                    "avg_hours": "Avg Hours",
                    "avg_value": "Avg Value"
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Avg Hours": st.column_config.NumberColumn(format="%.0f"),
                    "Avg Value": st.column_config.NumberColumn(format="$%.0f"),
                }
            )
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION D: DEMAND VS SUPPLY
    # =========================================================================
    section_header("Demand vs Supply", "Implied capacity use and slack")
    
    if len(demand_supply) > 0 and "implied_utilisation" in demand_supply.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Implied FTE trend
            fig = time_series(
                demand_supply,
                x="cohort_month",
                y="implied_fte",
                title="Implied FTE Demand",
                y_title="FTE"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Implied capacity use
            demand_supply_valid = demand_supply.dropna(subset=["implied_utilisation"])
            if len(demand_supply_valid) > 0:
                fig = time_series(
                    demand_supply_valid,
                    x="cohort_month",
                    y="implied_utilisation",
                    title="Implied Capacity Use (%)",
                    y_title="%"
                )
                # Add 100% reference line
                fig.add_hline(y=100, line_dash="dash", line_color="red",
                              annotation_text="Full Capacity")
                st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        avg_implied_util = demand_supply["implied_utilisation"].mean()
        avg_slack = demand_supply["slack_pct"].mean()
        
        slack_cols = st.columns(2)
        
        with slack_cols[0]:
            st.metric("Avg Capacity Use", fmt_percent(avg_implied_util))
        
        with slack_cols[1]:
            st.metric("Avg Slack", fmt_percent(avg_slack))
        
        if avg_slack > 20:
            st.info(f"""
            **Slack Analysis:** {avg_slack:.1f}% average slack suggests capacity exceeds demand.
            This could indicate:
            - Fewer jobs coming in
            - Smaller job sizes
            - Higher efficiency / fewer hours per job
            - Opportunity to take on more work or reduce headcount
            """)
        elif avg_slack < 0:
            st.warning(f"""
            **Capacity Constraint:** Negative slack ({avg_slack:.1f}%) indicates demand exceeds capacity.
            Consider:
            - Hiring additional staff
            - Outsourcing or partnerships
            - Prioritizing higher-value work
            """)


if __name__ == "__main__":
    main()
