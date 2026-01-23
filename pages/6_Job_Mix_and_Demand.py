"""
Job Mix & Implied FTE Demand Page

Analyze job intake patterns and implied capacity demand.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

    job_actuals = df.groupby("job_no", dropna=False).agg(
        actual_hours=("hours_raw", "sum"),
    ).reset_index()
    
    # Merge
    jobs = job_attrs.merge(cohort_df, on="job_no", how="inner")
    jobs = jobs.merge(job_quotes, on="job_no", how="left")
    jobs = jobs.merge(job_actuals, on="job_no", how="left")
    
    # Fill NaN
    jobs["quoted_hours"] = jobs["quoted_hours"].fillna(0)
    jobs["quoted_amount"] = jobs["quoted_amount"].fillna(0)
    jobs["actual_hours"] = jobs["actual_hours"].fillna(0)
    
    # Aggregate by cohort month
    monthly = jobs.groupby("cohort_month").agg(
        job_count=("job_no", "nunique"),
        total_quoted_hours=("quoted_hours", "sum"),
        total_quoted_amount=("quoted_amount", "sum"),
        avg_quoted_hours=("quoted_hours", "mean"),
        avg_quoted_amount=("quoted_amount", "mean"),
        total_actual_hours=("actual_hours", "sum"),
        avg_actual_hours=("actual_hours", "mean"),
    ).reset_index()

    # Derived metrics
    monthly["value_per_quoted_hour"] = np.where(
        monthly["total_quoted_hours"] > 0,
        monthly["total_quoted_amount"] / monthly["total_quoted_hours"],
        np.nan
    )

    monthly["quoted_vs_actual_ratio"] = np.where(
        monthly["total_actual_hours"] > 0,
        monthly["total_quoted_hours"] / monthly["total_actual_hours"],
        np.nan
    )
    
    # Implied FTE demand
    # FTE = quoted_hours / (weeks_in_month * 38)
    weeks_per_month = 4.33
    monthly["implied_fte_quoted"] = (
        monthly["total_quoted_hours"] / (weeks_per_month * config.CAPACITY_HOURS_PER_WEEK)
    )
    monthly["implied_fte_actual"] = (
        monthly["total_actual_hours"] / (weeks_per_month * config.CAPACITY_HOURS_PER_WEEK)
    )

    monthly["cohort_month"] = pd.to_datetime(monthly["cohort_month"])
    
    return monthly.sort_values("cohort_month")


def calculate_demand_vs_supply(df: pd.DataFrame, monthly_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate demand vs supply comparison."""
    
    # Get supply capacity by month
    # Count unique staff per month and their capacity
    staff_months = df.groupby(["month_key", "staff_name"]).agg(
        fte_scaling=("fte_hours_scaling", "first"),
    ).reset_index()
    staff_months["fte_scaling"] = staff_months["fte_scaling"].fillna(config.DEFAULT_FTE_SCALING)
    
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
    comparison["implied_utilisation_quoted"] = np.where(
        comparison["supply_capacity_hours"] > 0,
        comparison["total_quoted_hours"] / comparison["supply_capacity_hours"] * 100,
        np.nan
    )

    comparison["implied_utilisation_actual"] = np.where(
        comparison["supply_capacity_hours"] > 0,
        comparison["total_actual_hours"] / comparison["supply_capacity_hours"] * 100,
        np.nan
    )

    comparison["slack_pct_quoted"] = 100 - comparison["implied_utilisation_quoted"]
    comparison["slack_pct_actual"] = 100 - comparison["implied_utilisation_actual"]
    
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
    monthly_metrics["underquoted_hours"] = (
        monthly_metrics["total_actual_hours"] - monthly_metrics["total_quoted_hours"]
    ).clip(lower=0)
    demand_supply = calculate_demand_vs_supply(df_all, monthly_metrics)

    # Job-level data for loss diagnosis and quadrant drill
    category_col = get_category_col(df_all)
    job_task = safe_quote_job_task(df_all)
    if len(job_task) > 0:
        job_quotes = job_task.groupby("job_no").agg(
            quoted_hours=("quoted_time_total", "sum"),
            quoted_amount=("quoted_amount_total", "sum"),
        ).reset_index()
    else:
        job_quotes = pd.DataFrame({"job_no": [], "quoted_hours": [], "quoted_amount": []})

    job_actuals = df_all.groupby("job_no", dropna=False).agg(
        actual_hours=("hours_raw", "sum"),
    ).reset_index()

    job_attrs = df_all.groupby("job_no").agg(
        department_final=("department_final", "first"),
        job_category=(category_col, "first"),
    ).reset_index()

    job_level = job_attrs.merge(job_quotes, on="job_no", how="left")
    job_level = job_level.merge(job_actuals, on="job_no", how="left")
    job_level = job_level.merge(cohort_df, on="job_no", how="left")
    job_level["quoted_hours"] = job_level["quoted_hours"].fillna(0)
    job_level["quoted_amount"] = job_level["quoted_amount"].fillna(0)
    job_level["actual_hours"] = job_level["actual_hours"].fillna(0)
    job_level["underquoted_hours"] = (job_level["actual_hours"] - job_level["quoted_hours"]).clip(lower=0)
    job_level["underquoted_ratio"] = np.where(
        job_level["actual_hours"] > 0,
        job_level["quoted_hours"] / job_level["actual_hours"],
        np.nan,
    )
    job_level["cohort_month"] = pd.to_datetime(job_level["cohort_month"])
    job_level = filter_by_time_window(job_level, time_window, date_col="cohort_month")
    
    # =========================================================================
    # SECTION A: KPI STRIP
    # =========================================================================
    section_header("Job Mix Summary")
    
    kpi_cols = st.columns(8)
    
    total_jobs = monthly_metrics["job_count"].sum()
    total_quoted_hours = monthly_metrics["total_quoted_hours"].sum()
    total_quoted_amount = monthly_metrics["total_quoted_amount"].sum()
    total_actual_hours = monthly_metrics["total_actual_hours"].sum()
    avg_hours_per_job = total_quoted_hours / total_jobs if total_jobs > 0 else 0
    avg_amount_per_job = total_quoted_amount / total_jobs if total_jobs > 0 else 0
    value_per_quoted_hour = total_quoted_amount / total_quoted_hours if total_quoted_hours > 0 else 0
    quoted_vs_actual_ratio = total_quoted_hours / total_actual_hours if total_actual_hours > 0 else np.nan
    
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
        st.metric("Total Actual Hours", fmt_hours(total_actual_hours))

    with kpi_cols[6]:
        st.metric("Quoted/Actual Hrs", f"{quoted_vs_actual_ratio:.2f}x" if pd.notna(quoted_vs_actual_ratio) else "—")

    with kpi_cols[7]:
        st.metric("$/Quoted Hr", fmt_currency(value_per_quoted_hour))

    with st.expander("How these are calculated", expanded=False):
        st.markdown(
            """
- Total Quoted Hours = sum of `quoted_time_total` by job
- Total Actual Hours = sum of `hours_raw` by job
- Avg $/Job = Total Quoted $ / Jobs
- $/Quoted Hr = Total Quoted $ / Total Quoted Hours
- Quoted vs Actual = Total Quoted Hours / Total Actual Hours
            """
        )
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION B: TREND CHARTS
    # =========================================================================
    section_header("Monthly Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_bar(
            x=monthly_metrics["cohort_month"],
            y=monthly_metrics["job_count"],
            name="Jobs",
            marker_color="#4c78a8",
        )
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["avg_quoted_amount"],
                name="Avg Quote Value",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#f58518", width=2),
            )
        )
        fig.update_layout(
            title="Job Volume vs Quote Value",
            yaxis=dict(title="Jobs"),
            yaxis2=dict(title="Avg Quote Value ($)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["avg_quoted_hours"],
                name="Avg Quoted Hours",
                mode="lines+markers",
                line=dict(color="#54a24b", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["avg_actual_hours"],
                name="Avg Actual Hours",
                mode="lines+markers",
                line=dict(color="#e45756", width=2),
            )
        )
        fig.update_layout(
            title="Avg Hours per Job: Quoted vs Actual",
            yaxis=dict(title="Hours"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["total_quoted_hours"],
                name="Quoted Hours",
                mode="lines+markers",
                line=dict(color="#72b7b2", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["total_actual_hours"],
                name="Actual Hours",
                mode="lines+markers",
                line=dict(color="#b279a2", width=2),
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthly_metrics["cohort_month"],
                y=monthly_metrics["underquoted_hours"],
                name="Under-Quoted Hours",
                marker_color="rgba(228,87,86,0.35)",
            )
        )
        fig.update_layout(
            title="Total Hours: Quoted vs Actual",
            yaxis=dict(title="Hours"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = time_series(
            monthly_metrics,
            x="cohort_month",
            y="value_per_quoted_hour",
            title="Value per Quoted Hour",
            y_title="$/hr",
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(job_level) > 0:
        top_depts = (
            job_level.groupby("department_final")["underquoted_hours"]
            .sum()
            .sort_values(ascending=False)
            .head(4)
            .index
            .tolist()
        )
        dept_month = job_level[job_level["department_final"].isin(top_depts)].groupby(
            ["cohort_month", "department_final"]
        )["underquoted_hours"].sum().reset_index()
        if len(dept_month) > 0:
            fig = go.Figure()
            for dept in top_depts:
                dept_slice = dept_month[dept_month["department_final"] == dept]
                fig.add_bar(
                    x=dept_slice["cohort_month"],
                    y=dept_slice["underquoted_hours"],
                    name=dept,
                )
            fig.update_layout(
                title="Under-Quoted Hours by Department (Top 4)",
                yaxis=dict(title="Under-Quoted Hours"),
                barmode="stack",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION C: JOB QUADRANT
    # =========================================================================
    section_header("Job Value vs Effort", "Identify 'high value / low hours' segment")
    
    if len(job_level) > 0:
        job_quotes = job_level[(job_level["quoted_hours"] > 0) & (job_level["quoted_amount"] > 0)].copy()

        if len(job_quotes) > 5:
            fig = quadrant_scatter(
                job_quotes,
                x="quoted_hours",
                y="quoted_amount",
                hover_name="job_no",
                title="Job Portfolio: Value vs Effort",
                x_title="Quoted Hours",
                y_title="Quoted Amount ($)",
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
                        "Low Value / High Effort",
                    ),
                ),
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
                    "avg_value": "Avg Value",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Avg Hours": st.column_config.NumberColumn(format="%.0f"),
                    "Avg Value": st.column_config.NumberColumn(format="$%.0f"),
                },
            )

            quadrant_options = [
                "High Value / Low Effort",
                "High Value / High Effort",
                "Low Value / Low Effort",
                "Low Value / High Effort",
            ]
            selected_quadrant = st.selectbox(
                "Inspect Jobs in Quadrant",
                options=quadrant_options,
                index=0,
            )

            quadrant_jobs = job_quotes[job_quotes["quadrant"] == selected_quadrant].copy()
            quadrant_jobs["quoted_vs_actual"] = np.where(
                quadrant_jobs["actual_hours"] > 0,
                quadrant_jobs["quoted_hours"] / quadrant_jobs["actual_hours"],
                np.nan,
            )

            st.dataframe(
                quadrant_jobs.sort_values("quoted_amount", ascending=False)[
                    [
                        "job_no",
                        "department_final",
                        "job_category",
                        "quoted_amount",
                        "quoted_hours",
                        "actual_hours",
                        "quoted_vs_actual",
                    ]
                ].rename(columns={
                    "job_no": "Job",
                    "department_final": "Department",
                    "job_category": "Category",
                    "quoted_amount": "Quoted $",
                    "quoted_hours": "Quoted Hours",
                    "actual_hours": "Actual Hours",
                    "quoted_vs_actual": "Quoted/Actual",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Quoted $": st.column_config.NumberColumn(format="$%.0f"),
                    "Quoted Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Actual Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Quoted/Actual": st.column_config.NumberColumn(format="%.2fx"),
                },
            )

            st.info(
                """
Operational signals:
- High Value / Low Effort: productize and standardize; protect scope and pricing.
- High Value / High Effort: review estimation, resourcing, and margin guardrails.
- Low Value / Low Effort: automate or bundle; consider off-peak capacity.
- Low Value / High Effort: strong candidates for re-pricing or exit.
                """
            )
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION D: DEMAND VS SUPPLY
    # =========================================================================
    section_header("Demand vs Supply", "Implied capacity use and slack")
    
    if len(demand_supply) > 0 and "implied_utilisation_quoted" in demand_supply.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=demand_supply["cohort_month"],
                    y=demand_supply["implied_fte_quoted"],
                    name="Implied FTE (Quoted)",
                    mode="lines+markers",
                    line=dict(color="#4c78a8", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=demand_supply["cohort_month"],
                    y=demand_supply["implied_fte_actual"],
                    name="Implied FTE (Actual)",
                    mode="lines+markers",
                    line=dict(color="#f58518", width=2),
                )
            )
            fig.update_layout(
                title="Implied FTE Demand",
                yaxis=dict(title="FTE"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            demand_supply_valid = demand_supply.dropna(
                subset=["implied_utilisation_quoted", "implied_utilisation_actual"]
            )
            if len(demand_supply_valid) > 0:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=demand_supply_valid["cohort_month"],
                        y=demand_supply_valid["implied_utilisation_quoted"],
                        name="Capacity Use (Quoted)",
                        mode="lines+markers",
                        line=dict(color="#54a24b", width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=demand_supply_valid["cohort_month"],
                        y=demand_supply_valid["implied_utilisation_actual"],
                        name="Capacity Use (Actual)",
                        mode="lines+markers",
                        line=dict(color="#e45756", width=2),
                    )
                )
                fig.update_layout(
                    title="Implied Capacity Use (%)",
                    yaxis=dict(title="%"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                # Add 100% reference line
                fig.add_hline(y=100, line_dash="dash", line_color="red",
                              annotation_text="Full Capacity")
                st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        avg_implied_util = demand_supply["implied_utilisation_quoted"].mean()
        avg_actual_util = demand_supply["implied_utilisation_actual"].mean()
        avg_slack = demand_supply["slack_pct_quoted"].mean()
        avg_slack_actual = demand_supply["slack_pct_actual"].mean()
        
        slack_cols = st.columns(4)
        
        with slack_cols[0]:
            st.metric("Avg Capacity Use (Quoted)", fmt_percent(avg_implied_util))
        
        with slack_cols[1]:
            st.metric("Avg Capacity Use (Actual)", fmt_percent(avg_actual_util))

        with slack_cols[2]:
            st.metric("Avg Slack (Quoted)", fmt_percent(avg_slack))

        with slack_cols[3]:
            st.metric("Avg Slack (Actual)", fmt_percent(avg_slack_actual))
        
        if avg_slack > 20:
            st.info(f"""
            **Slack Analysis:** {avg_slack:.1f}% average quoted slack suggests capacity exceeds demand.
            This could indicate:
            - Fewer jobs coming in
            - Smaller job sizes
            - Higher efficiency / fewer hours per job
            - Opportunity to take on more work or reduce headcount
            """)
        elif avg_slack < 0:
            st.warning(f"""
            **Capacity Constraint:** Negative quoted slack ({avg_slack:.1f}%) indicates demand exceeds capacity.
            Consider:
            - Hiring additional staff
            - Outsourcing or partnerships
            - Prioritizing higher-value work
            """)

        with st.expander("Methodology and reconciliation", expanded=False):
            st.markdown(
                """
**Demand (Quoted)**
- Total Quoted Hours = Σ(quoted_time_total) by job in cohort month
- Implied FTE (Quoted) = Total Quoted Hours / (4.33 × 38)

**Demand (Actual)**
- Total Actual Hours = Σ(hours_raw) by job in cohort month
- Implied FTE (Actual) = Total Actual Hours / (4.33 × 38)

**Supply**
- Staff Count = unique staff_name with time in month_key
- Avg FTE = mean fte_hours_scaling (default 1.0 if missing)
- Supply Capacity Hours = Staff Count × Avg FTE × 4.33 × 38

**Utilisation & Slack**
- Capacity Use = Demand Hours / Supply Capacity Hours
- Slack = 100% − Capacity Use

Note: demand uses cohort month (job intake timing) while supply uses staffing in the same calendar month.
Interpret this as intake pressure vs available capacity at that time.
                """
            )

            latest = demand_supply.sort_values("cohort_month").tail(1)
            if len(latest) > 0:
                st.dataframe(
                    latest[[
                        "cohort_month",
                        "total_quoted_hours",
                        "total_actual_hours",
                        "staff_count",
                        "avg_fte",
                        "supply_capacity_hours",
                        "implied_utilisation_quoted",
                        "implied_utilisation_actual",
                    ]].rename(columns={
                        "cohort_month": "Month",
                        "total_quoted_hours": "Quoted Hours",
                        "total_actual_hours": "Actual Hours",
                        "staff_count": "Staff",
                        "avg_fte": "Avg FTE",
                        "supply_capacity_hours": "Supply Capacity Hours",
                        "implied_utilisation_quoted": "Capacity Use (Quoted)",
                        "implied_utilisation_actual": "Capacity Use (Actual)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Month": st.column_config.DateColumn(format="YYYY-MM"),
                        "Quoted Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Actual Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Avg FTE": st.column_config.NumberColumn(format="%.2f"),
                        "Supply Capacity Hours": st.column_config.NumberColumn(format="%.1f"),
                        "Capacity Use (Quoted)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Capacity Use (Actual)": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )

    st.markdown("---")

    # =========================================================================
    # SECTION E: SO WHAT (CONSULTANT VIEW)
    # =========================================================================
    section_header("So What", "What the trend implies about pricing vs delivery")

    monthly_sorted = monthly_metrics.sort_values("cohort_month")
    if len(monthly_sorted) >= 6:
        recent = monthly_sorted.tail(3)
        prior = monthly_sorted.tail(6).head(3)

        recent_jobs = recent["job_count"].mean()
        prior_jobs = prior["job_count"].mean()
        recent_quote = recent["avg_quoted_amount"].mean()
        prior_quote = prior["avg_quoted_amount"].mean()
        recent_ratio = (
            recent["total_quoted_hours"].sum() / recent["total_actual_hours"].sum()
            if recent["total_actual_hours"].sum() > 0 else np.nan
        )
        prior_ratio = (
            prior["total_quoted_hours"].sum() / prior["total_actual_hours"].sum()
            if prior["total_actual_hours"].sum() > 0 else np.nan
        )

        jobs_delta = (recent_jobs - prior_jobs) / prior_jobs if prior_jobs > 0 else np.nan
        quote_delta = (recent_quote - prior_quote) / prior_quote if prior_quote > 0 else np.nan
        ratio_delta = recent_ratio - prior_ratio if pd.notna(recent_ratio) and pd.notna(prior_ratio) else np.nan

        jobs_delta_text = fmt_percent(jobs_delta * 100) if pd.notna(jobs_delta) else "n/a"
        quote_delta_text = fmt_percent(quote_delta * 100) if pd.notna(quote_delta) else "n/a"
        ratio_delta_text = f"{ratio_delta:+.2f}x" if pd.notna(ratio_delta) else "n/a"

        st.info(
            f"""
**Recent 3 months vs prior 3 months**
- Job volume change: {jobs_delta_text}
- Avg quote value change: {quote_delta_text}
- Quoted vs actual hours ratio change: {ratio_delta_text}

**Interpretation**
- If job volume is falling while quote value rises, you are winning fewer, larger jobs.
- If quoted vs actual ratio is < 1.0, delivery is heavier than quoted (margin compression).
- If quoted vs actual ratio is > 1.0, quotes exceed delivery (potential slack/underutilisation).
            """
        )

        loss_dept = job_level.groupby("department_final").agg(
            jobs=("job_no", "nunique"),
            actual_hours=("actual_hours", "sum"),
            underquoted_hours=("underquoted_hours", "sum"),
        ).reset_index()
        loss_dept["underquoted_share"] = np.where(
            loss_dept["actual_hours"] > 0,
            loss_dept["underquoted_hours"] / loss_dept["actual_hours"],
            np.nan,
        )
        loss_dept = loss_dept.sort_values("underquoted_hours", ascending=False).head(5)

        loss_cat = job_level.groupby("job_category").agg(
            jobs=("job_no", "nunique"),
            actual_hours=("actual_hours", "sum"),
            underquoted_hours=("underquoted_hours", "sum"),
        ).reset_index()
        loss_cat["underquoted_share"] = np.where(
            loss_cat["actual_hours"] > 0,
            loss_cat["underquoted_hours"] / loss_cat["actual_hours"],
            np.nan,
        )
        loss_cat = loss_cat.sort_values("underquoted_hours", ascending=False).head(5)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Loss Hotspots by Department")
            st.dataframe(
                loss_dept.rename(columns={
                    "department_final": "Department",
                    "jobs": "Jobs",
                    "actual_hours": "Actual Hours",
                    "underquoted_hours": "Under-Quoted Hours",
                    "underquoted_share": "Under-Quoted Share",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Actual Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Under-Quoted Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Under-Quoted Share": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

        with col2:
            st.subheader("Loss Hotspots by Job Category")
            st.dataframe(
                loss_cat.rename(columns={
                    "job_category": "Category",
                    "jobs": "Jobs",
                    "actual_hours": "Actual Hours",
                    "underquoted_hours": "Under-Quoted Hours",
                    "underquoted_share": "Under-Quoted Share",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Actual Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Under-Quoted Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Under-Quoted Share": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
    else:
        st.caption("Need at least 6 months of data to compute the trend comparison.")


if __name__ == "__main__":
    main()
