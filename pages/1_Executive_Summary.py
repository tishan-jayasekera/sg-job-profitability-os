"""
Executive Summary: Margin & Delivery Diagnosis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Tuple, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.data.semantic import filter_jobs_by_state


st.set_page_config(
    page_title="Executive Summary | Delivery Diagnosis",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the enriched timesheet fact table."""
    try:
        return load_fact_timesheet()
    except Exception:
        data_path = Path(__file__).parent.parent / "data" / "processed" / "fact_timesheet_day_enriched.parquet"
        if not data_path.exists():
            data_path = data_path.with_suffix(".csv")

        if data_path.exists():
            if data_path.suffix == ".parquet":
                return pd.read_parquet(data_path)
            return pd.read_csv(data_path, parse_dates=["work_date"])

        st.error(
            "Could not find fact_timesheet_day_enriched in either `src/data/processed` "
            "or `data/processed`. Check DATA_DIR or place the file in the expected folder."
        )
        st.stop()


def compute_job_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute job-level metrics with safe quote deduplication.

    Returns DataFrame with columns:
    - job_no, department_final, job_category
    - quoted_hours_job, quoted_amount_job, quote_rate_job
    - actual_hours_job, revenue_job, cost_job
    - realised_rate_job, margin_job, margin_pct_job
    - hours_variance_job, hours_variance_pct_job
    - rate_variance_job, rate_capture_pct_job
    - overrun_flag, severe_overrun_flag (>20%)
    """
    actuals = df.groupby("job_no").agg(
        actual_hours_job=("hours_raw", "sum"),
        revenue_job=("rev_alloc", "sum"),
        cost_job=("base_cost", "sum"),
        department_final=("department_final", "first"),
        job_category=("job_category", "first"),
    ).reset_index()

    job_task_quotes = df.groupby(["job_no", "task_name"]).agg(
        quoted_time_total=("quoted_time_total", "first"),
        quoted_amount_total=("quoted_amount_total", "first"),
    ).reset_index()

    quotes = job_task_quotes.groupby("job_no").agg(
        quoted_hours_job=("quoted_time_total", "sum"),
        quoted_amount_job=("quoted_amount_total", "sum"),
    ).reset_index()

    job_df = actuals.merge(quotes, on="job_no", how="left")

    job_df["quoted_hours_job"] = job_df["quoted_hours_job"].fillna(0)
    job_df["quoted_amount_job"] = job_df["quoted_amount_job"].fillna(0)

    job_df["quote_rate_job"] = np.where(
        job_df["quoted_hours_job"] > 0,
        job_df["quoted_amount_job"] / job_df["quoted_hours_job"],
        np.nan,
    )
    job_df["realised_rate_job"] = np.where(
        job_df["actual_hours_job"] > 0,
        job_df["revenue_job"] / job_df["actual_hours_job"],
        np.nan,
    )

    job_df["hours_variance_job"] = job_df["actual_hours_job"] - job_df["quoted_hours_job"]
    job_df["hours_variance_pct_job"] = np.where(
        job_df["quoted_hours_job"] > 0,
        (job_df["hours_variance_job"] / job_df["quoted_hours_job"]) * 100,
        np.where(job_df["actual_hours_job"] > 0, 100, 0),
    )

    job_df["rate_variance_job"] = job_df["realised_rate_job"] - job_df["quote_rate_job"]
    job_df["rate_capture_pct_job"] = np.where(
        job_df["quote_rate_job"] > 0,
        (job_df["realised_rate_job"] / job_df["quote_rate_job"]) * 100,
        np.nan,
    )

    job_df["margin_job"] = job_df["revenue_job"] - job_df["cost_job"]
    job_df["margin_pct_job"] = np.where(
        job_df["revenue_job"] > 0,
        (job_df["margin_job"] / job_df["revenue_job"]) * 100,
        np.nan,
    )

    job_df["overrun_flag"] = job_df["hours_variance_job"] > 0
    job_df["severe_overrun_flag"] = job_df["hours_variance_pct_job"] > 20
    job_df["has_quote"] = job_df["quoted_hours_job"] > 0

    if "quote_match_flag" in df.columns:
        unquoted_hours = df[df["quote_match_flag"] != "matched"].groupby("job_no")["hours_raw"].sum()
        job_df = job_df.merge(unquoted_hours.rename("unquoted_hours_job"), on="job_no", how="left")
        job_df["unquoted_hours_job"] = job_df["unquoted_hours_job"].fillna(0)
    else:
        job_df["unquoted_hours_job"] = 0

    job_df["unquoted_pct_job"] = np.where(
        job_df["actual_hours_job"] > 0,
        (job_df["unquoted_hours_job"] / job_df["actual_hours_job"]) * 100,
        0,
    )

    return job_df


def compute_portfolio_metrics(job_df: pd.DataFrame) -> Dict:
    """
    Compute portfolio-level KPIs.
    All rates are WEIGHTED (total/total), never mean of rates.
    """
    quoted_jobs = job_df[job_df["has_quote"]].copy()

    total_quoted_hours = quoted_jobs["quoted_hours_job"].sum()
    total_quoted_amount = quoted_jobs["quoted_amount_job"].sum()
    total_actual_hours = job_df["actual_hours_job"].sum()
    total_revenue = job_df["revenue_job"].sum()
    total_cost = job_df["cost_job"].sum()

    quote_rate_portfolio = total_quoted_amount / total_quoted_hours if total_quoted_hours > 0 else 0
    realised_rate_portfolio = total_revenue / total_actual_hours if total_actual_hours > 0 else 0

    n_jobs = len(job_df)
    n_quoted = len(quoted_jobs)
    n_overrun = int(quoted_jobs["overrun_flag"].sum())
    n_severe = int(quoted_jobs["severe_overrun_flag"].sum())

    margin = total_revenue - total_cost

    return {
        "quote_rate": quote_rate_portfolio,
        "realised_rate": realised_rate_portfolio,
        "rate_variance": realised_rate_portfolio - quote_rate_portfolio,
        "margin": margin,
        "margin_pct": (margin / total_revenue * 100) if total_revenue > 0 else 0,
        "pct_jobs_overrun": (n_overrun / n_quoted * 100) if n_quoted > 0 else 0,
        "pct_jobs_severe_overrun": (n_severe / n_quoted * 100) if n_quoted > 0 else 0,
        "total_hours_variance": quoted_jobs["hours_variance_job"].sum(),
        "n_jobs": n_jobs,
        "n_quoted_jobs": n_quoted,
    }


def render_kpi_strip(metrics: Dict):
    cols = st.columns(6)

    with cols[0]:
        st.metric("Quote Rate", f"${metrics['quote_rate']:,.0f}/hr")

    with cols[1]:
        st.metric(
            "Realised Rate",
            f"${metrics['realised_rate']:,.0f}/hr",
            delta=f"${metrics['rate_variance']:+,.0f}",
        )

    with cols[2]:
        st.metric("Rate Variance", f"${metrics['rate_variance']:+,.0f}/hr")

    with cols[3]:
        st.metric("Portfolio Margin", f"{metrics['margin_pct']:.1f}%")

    with cols[4]:
        st.metric("% Jobs Overrun", f"{metrics['pct_jobs_overrun']:.0f}%")

    with cols[5]:
        st.metric("% 20%+ Overrun", f"{metrics['pct_jobs_severe_overrun']:.0f}%")


def compute_rate_bridge(job_df: pd.DataFrame, portfolio_metrics: Dict) -> pd.DataFrame:
    """
    Decompose the Quote → Realised rate bridge.
    """
    quoted_jobs = job_df[job_df["has_quote"]].copy()

    under_run = quoted_jobs[quoted_jobs["hours_variance_job"] < 0]
    under_run_hours_saved = abs(under_run["hours_variance_job"].sum())

    overrun = quoted_jobs[quoted_jobs["hours_variance_job"] > 0]
    overrun_hours_excess = overrun["hours_variance_job"].sum()

    total_actual_hours = job_df["actual_hours_job"].sum()
    unquoted_hours = job_df["unquoted_hours_job"].sum()

    quote_rate = portfolio_metrics["quote_rate"]
    realised_rate = portfolio_metrics["realised_rate"]

    if total_actual_hours > 0:
        under_run_impact = under_run_hours_saved * quote_rate / total_actual_hours
        overrun_impact = -overrun_hours_excess * quote_rate / total_actual_hours
        unquoted_impact = -unquoted_hours * quote_rate / total_actual_hours
    else:
        under_run_impact = 0
        overrun_impact = 0
        unquoted_impact = 0

    residual = realised_rate - quote_rate - under_run_impact - overrun_impact - unquoted_impact

    return pd.DataFrame({
        "category": [
            "Quote Rate",
            "Under-run Jobs ↑",
            "Overrun Jobs ↓",
            "Unquoted Hours ↓",
            "Mix/Pricing Effect",
            "Realised Rate",
        ],
        "value": [
            quote_rate,
            under_run_impact,
            overrun_impact,
            unquoted_impact,
            residual,
            realised_rate,
        ],
        "type": ["absolute", "relative", "relative", "relative", "relative", "total"],
    })


def render_rate_bridge(bridge_df: pd.DataFrame):
    """Render waterfall chart using Plotly."""
    fig = go.Figure(
        go.Waterfall(
            name="Rate Bridge",
            orientation="v",
            measure=bridge_df["type"].tolist(),
            x=bridge_df["category"].tolist(),
            y=bridge_df["value"].tolist(),
            textposition="outside",
            text=[f"${v:,.0f}" for v in bridge_df["value"]],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2E7D32"}},
            decreasing={"marker": {"color": "#C62828"}},
            totals={"marker": {"color": "#1565C0"}},
        )
    )

    fig.update_layout(
        title="Quote Rate → Realised Rate Bridge",
        yaxis_title="$/hr",
        showlegend=False,
        height=400,
    )

    return fig


def render_quadrant_scatter(job_df: pd.DataFrame):
    """Render the Winners vs Losers quadrant scatter."""
    plot_df = job_df[
        job_df["has_quote"]
        & job_df["realised_rate_job"].notna()
        & job_df["quote_rate_job"].notna()
    ].copy()

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No quoted jobs with valid rates",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(title="Job Outcome Quadrants: Winners vs Losers", height=500)
        return fig

    plot_df["revenue_size"] = plot_df["revenue_job"].abs()

    fig = px.scatter(
        plot_df,
        x="hours_variance_pct_job",
        y="rate_variance_job",
        size="revenue_size",
        color="department_final",
        hover_data=["job_no", "margin_pct_job", "actual_hours_job"],
        labels={
            "hours_variance_pct_job": "Hours Variance %",
            "rate_variance_job": "Rate Variance ($/hr)",
            "revenue_size": "Revenue",
            "department_final": "Department",
        },
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    y_max = plot_df["rate_variance_job"].max()
    y_min = plot_df["rate_variance_job"].min()
    fig.add_annotation(
        x=-50,
        y=y_max * 0.8 if pd.notna(y_max) else 0,
        text="SUBSIDISERS<br>(Under-run + High Rate)",
        showarrow=False,
        font=dict(color="green", size=10),
    )
    fig.add_annotation(
        x=50,
        y=y_min * 0.8 if pd.notna(y_min) else 0,
        text="EROSIVE<br>(Overrun + Low Rate)",
        showarrow=False,
        font=dict(color="red", size=10),
    )

    fig.update_layout(
        title="Job Outcome Quadrants: Winners vs Losers",
        height=500,
    )

    return fig


def render_overrun_distribution(job_df: pd.DataFrame):
    """Render overrun distribution and contribution charts."""
    quoted_jobs = job_df[job_df["has_quote"]].copy()

    def categorise_overrun(pct):
        if pct < 0:
            return "Under-run"
        if pct <= 20:
            return "Mild Overrun (0-20%)"
        return "Severe Overrun (>20%)"

    if quoted_jobs.empty:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Job Distribution", "Hours Variance Contribution"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )
        fig.update_layout(height=350, title_text="Overrun Distribution & Impact")
        return fig, pd.DataFrame()

    quoted_jobs["overrun_category"] = quoted_jobs["hours_variance_pct_job"].apply(categorise_overrun)

    category_stats = quoted_jobs.groupby("overrun_category").agg(
        job_count=("job_no", "count"),
        hours_variance=("hours_variance_job", "sum"),
        margin=("margin_job", "sum"),
    ).reset_index()
    category_stats["job_pct"] = category_stats["job_count"] / category_stats["job_count"].sum() * 100

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Job Distribution", "Hours Variance Contribution"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    colors = {
        "Under-run": "#2E7D32",
        "Mild Overrun (0-20%)": "#FFA726",
        "Severe Overrun (>20%)": "#C62828",
    }

    for cat in ["Under-run", "Mild Overrun (0-20%)", "Severe Overrun (>20%)"]:
        row = category_stats[category_stats["overrun_category"] == cat]
        if len(row) > 0:
            fig.add_trace(
                go.Bar(
                    name=cat,
                    x=[cat],
                    y=row["job_pct"].values,
                    marker_color=colors[cat],
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    name=cat,
                    x=[cat],
                    y=row["hours_variance"].values,
                    marker_color=colors[cat],
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.update_layout(height=350, title_text="Overrun Distribution & Impact")
    fig.update_yaxes(title_text="% of Jobs", row=1, col=1)
    fig.update_yaxes(title_text="Hours Variance", row=1, col=2)

    category_stats = category_stats.rename(
        columns={
            "overrun_category": "Category",
            "job_count": "Job Count",
            "hours_variance": "Hours Variance",
            "margin": "Margin",
            "job_pct": "Job %",
        }
    )

    return fig, category_stats


def compute_department_scoreboard(job_df: pd.DataFrame) -> pd.DataFrame:
    """Compute department-level scoreboard."""
    if job_df.empty:
        return pd.DataFrame()

    dept_df = job_df.groupby("department_final").agg(
        quoted_hours=("quoted_hours_job", "sum"),
        quoted_amount=("quoted_amount_job", "sum"),
        actual_hours=("actual_hours_job", "sum"),
        revenue=("revenue_job", "sum"),
        cost=("cost_job", "sum"),
        hours_variance=("hours_variance_job", "sum"),
        overrun_jobs=("overrun_flag", "sum"),
        total_jobs=("job_no", "count"),
    ).reset_index()

    dept_df["Quote Rate"] = np.where(
        dept_df["quoted_hours"] > 0,
        dept_df["quoted_amount"] / dept_df["quoted_hours"],
        np.nan,
    )
    dept_df["Realised Rate"] = np.where(
        dept_df["actual_hours"] > 0,
        dept_df["revenue"] / dept_df["actual_hours"],
        np.nan,
    )
    dept_df["Rate Variance"] = dept_df["Realised Rate"] - dept_df["Quote Rate"]
    dept_df["% Overrun"] = dept_df["overrun_jobs"] / dept_df["total_jobs"] * 100

    total_hours = dept_df["actual_hours"].sum()
    avg_cost_rate = dept_df["cost"].sum() / total_hours if total_hours > 0 else 50
    dept_df["Margin Impact Est."] = dept_df["hours_variance"] * avg_cost_rate * -1

    display_cols = [
        "department_final",
        "Quote Rate",
        "Realised Rate",
        "Rate Variance",
        "% Overrun",
        "hours_variance",
        "Margin Impact Est.",
    ]

    dept_df = dept_df[display_cols].rename(
        columns={
            "department_final": "Department",
            "hours_variance": "Hours Variance",
        }
    )

    return dept_df.sort_values("Rate Variance", ascending=True)


def compute_intervention_queue(job_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Compute intervention queue with risk scores and reason codes.
    """
    queue_df = job_df[job_df["has_quote"]].copy()

    if queue_df.empty:
        return pd.DataFrame(
            columns=[
                "Job",
                "Department",
                "Revenue",
                "Hours Overrun %",
                "Rate Leakage %",
                "Risk Score",
                "Reason Codes",
                "Action",
            ]
        )

    queue_df["norm_overrun"] = queue_df["hours_variance_pct_job"].clip(lower=0) / 100
    queue_df["norm_overrun"] = queue_df["norm_overrun"].clip(upper=1)

    queue_df["norm_revenue"] = queue_df["revenue_job"].rank(pct=True)

    queue_df["rate_leakage_pct"] = np.where(
        queue_df["quote_rate_job"] > 0,
        (queue_df["quote_rate_job"] - queue_df["realised_rate_job"]) / queue_df["quote_rate_job"],
        0,
    )
    queue_df["rate_leakage_pct"] = queue_df["rate_leakage_pct"].replace([np.inf, -np.inf], np.nan).fillna(0)
    queue_df["norm_rate_leakage"] = queue_df["rate_leakage_pct"].clip(lower=0, upper=1)

    queue_df["norm_runtime"] = np.where(queue_df["hours_variance_pct_job"] > 20, 0.5, 0.2)

    queue_df["risk_score"] = (
        0.30 * queue_df["norm_overrun"]
        + 0.25 * queue_df["norm_revenue"]
        + 0.25 * queue_df["norm_rate_leakage"]
        + 0.20 * queue_df["norm_runtime"]
    ) * 100

    def get_reason_codes(row):
        codes = []
        if row["hours_variance_pct_job"] > 20:
            codes.append("SEVERE_OVERRUN")
        if row["rate_leakage_pct"] > 0.2:
            codes.append("RATE_LEAKAGE")
        if row["unquoted_pct_job"] > 10:
            codes.append("SCOPE_CREEP")
        if row["margin_job"] < 0:
            codes.append("MARGIN_NEGATIVE")
        if row["norm_revenue"] > 0.8:
            codes.append("HIGH_EXPOSURE")
        return ", ".join(codes) if codes else "MONITOR"

    queue_df["reason_codes"] = queue_df.apply(get_reason_codes, axis=1)

    def get_action(row):
        if row["margin_job"] < 0:
            return "🚨 Escalate billing / re-scope"
        if row["hours_variance_pct_job"] > 50:
            return "⚠️ Right-size staffing"
        if row["rate_leakage_pct"] > 0.3:
            return "💰 Quote adjustment review"
        return "👀 Monitor closely"

    queue_df["recommended_action"] = queue_df.apply(get_action, axis=1)

    queue_df = queue_df.nlargest(top_n, "risk_score")

    display_df = queue_df[[
        "job_no",
        "department_final",
        "revenue_job",
        "hours_variance_pct_job",
        "rate_leakage_pct",
        "risk_score",
        "reason_codes",
        "recommended_action",
    ]].copy()

    display_df.columns = [
        "Job",
        "Department",
        "Revenue",
        "Hours Overrun %",
        "Rate Leakage %",
        "Risk Score",
        "Reason Codes",
        "Action",
    ]

    return display_df


def render_drill_down_panel(df: pd.DataFrame, selected_job: str):
    """Render drill-down panel for selected job."""
    job_data = df[df["job_no"] == selected_job].copy()

    if job_data.empty:
        st.warning(f"No data found for job {selected_job}")
        return

    st.subheader(f"📋 Job Details: {selected_job}")

    with st.expander("Job Brief", expanded=True):
        actual_hours = job_data["hours_raw"].sum()
        revenue = job_data["rev_alloc"].sum()
        cost = job_data["base_cost"].sum()

        task_quotes = job_data.groupby("task_name").agg(
            quoted_time_total=("quoted_time_total", "first"),
            quoted_amount_total=("quoted_amount_total", "first"),
        )
        quoted_hours = task_quotes["quoted_time_total"].sum()
        quoted_amount = task_quotes["quoted_amount_total"].sum()

        quote_rate = quoted_amount / quoted_hours if quoted_hours > 0 else 0
        realised_rate = revenue / actual_hours if actual_hours > 0 else 0
        margin = revenue - cost
        margin_pct = margin / revenue * 100 if revenue > 0 else 0

        job_status = job_data["job_status"].dropna().iloc[0] if "job_status" in job_data.columns and job_data["job_status"].notna().any() else "Unknown"
        department = job_data["department_final"].dropna().iloc[0] if job_data["department_final"].notna().any() else "Unknown"
        category = job_data["job_category"].dropna().iloc[0] if "job_category" in job_data.columns and job_data["job_category"].notna().any() else "Unknown"

        st.markdown(
            f"**Department:** {department}  |  **Category:** {category}  |  **Status:** {job_status}"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quoted Hours", f"{quoted_hours:,.0f}")
            st.metric("Quote Rate", f"${quote_rate:,.0f}/hr")
        with col2:
            st.metric("Actual Hours", f"{actual_hours:,.0f}", delta=f"{actual_hours - quoted_hours:+,.0f}")
            st.metric("Realised Rate", f"${realised_rate:,.0f}/hr", delta=f"${realised_rate - quote_rate:+,.0f}")
        with col3:
            st.metric("Margin", f"${margin:,.0f}")
            st.metric("Margin %", f"{margin_pct:.1f}%")

    with st.expander("Task Drivers", expanded=True):
        task_df = job_data.groupby("task_name").agg(
            hours_raw=("hours_raw", "sum"),
            quoted_time_total=("quoted_time_total", "first"),
            quote_match_flag=("quote_match_flag", "first"),
        ).reset_index()

        task_df["hours_variance"] = task_df["hours_raw"] - task_df["quoted_time_total"].fillna(0)
        task_df["is_unquoted"] = task_df["quote_match_flag"] != "matched"

        total_variance = task_df["hours_variance"].abs().sum()
        if total_variance > 0:
            task_df["contribution_pct"] = task_df["hours_variance"].abs() / total_variance * 100
        else:
            task_df["contribution_pct"] = 0

        display_task = task_df[[
            "task_name",
            "quoted_time_total",
            "hours_raw",
            "hours_variance",
            "is_unquoted",
            "contribution_pct",
        ]].copy()
        display_task.columns = [
            "Task",
            "Quoted Hrs",
            "Actual Hrs",
            "Variance",
            "Unquoted?",
            "Contribution %",
        ]

        st.dataframe(
            display_task.sort_values("Variance", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Staff Drivers", expanded=False):
        staff_df = job_data.groupby("staff_name").agg(
            hours_raw=("hours_raw", "sum"),
            base_cost=("base_cost", "sum"),
        ).reset_index()

        staff_df["pct_of_hours"] = staff_df["hours_raw"] / staff_df["hours_raw"].sum() * 100
        staff_df.columns = ["Staff", "Hours", "Cost", "% of Job Hours"]

        st.dataframe(
            staff_df.sort_values("Hours", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def render_reconciliation_panel(df: pd.DataFrame, job_df: pd.DataFrame, portfolio_metrics: Dict):
    """Render collapsible reconciliation panel."""
    with st.expander("🔍 Data Reconciliation (QA)", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Raw Data Totals**")
            st.write(f"- Total rows: {len(df):,}")
            st.write(f"- Total hours (sum hours_raw): {df['hours_raw'].sum():,.1f}")
            st.write(f"- Total revenue (sum rev_alloc): ${df['rev_alloc'].sum():,.0f}")
            st.write(f"- Unique jobs: {df['job_no'].nunique():,}")

        with col2:
            st.markdown("**Job Aggregation Totals**")
            st.write(f"- Jobs in summary: {len(job_df):,}")
            st.write(f"- Total hours (sum job hours): {job_df['actual_hours_job'].sum():,.1f}")
            st.write(f"- Total revenue (sum job revenue): ${job_df['revenue_job'].sum():,.0f}")

        hours_delta = abs(df["hours_raw"].sum() - job_df["actual_hours_job"].sum())
        rev_delta = abs(df["rev_alloc"].sum() - job_df["revenue_job"].sum())

        if hours_delta < 0.01 and rev_delta < 0.01:
            st.success("✅ Totals reconcile correctly")
        else:
            st.error(
                f"❌ Reconciliation mismatch: Hours delta={hours_delta:.2f}, Revenue delta=${rev_delta:.2f}"
            )


def plotly_chart_with_select(fig):
    try:
        return st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    except TypeError:
        return st.plotly_chart(fig, use_container_width=True)


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters for period under review and job state."""
    st.sidebar.header("Filters")

    df_filtered = df.copy()

    if "work_date" in df_filtered.columns and df_filtered["work_date"].notna().any():
        work_dates = pd.to_datetime(df_filtered["work_date"], errors="coerce")
        min_date = work_dates.min().date()
        max_date = work_dates.max().date()

        start_date, end_date = st.sidebar.date_input(
            "Period under review",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="exec_period_review",
        )

        df_filtered = df_filtered[
            (work_dates.dt.date >= start_date) & (work_dates.dt.date <= end_date)
        ].copy()
    else:
        st.sidebar.caption("No work_date column available for period filtering.")

    st.sidebar.divider()

    if "department_final" in df_filtered.columns:
        departments = ["All"] + sorted(df_filtered["department_final"].dropna().unique().tolist())
        selected_dept = st.sidebar.selectbox(
            "Department",
            options=departments,
            index=0,
            key="exec_department",
        )
        if selected_dept != "All":
            df_filtered = df_filtered[df_filtered["department_final"] == selected_dept].copy()
    else:
        st.sidebar.caption("No department column available for filtering.")

    st.sidebar.divider()

    job_state = st.sidebar.selectbox(
        "Job status",
        options=["All", "Active", "Completed"],
        index=0,
        key="exec_job_state",
    )

    if job_state != "All":
        df_filtered = filter_jobs_by_state(df_filtered, job_state)

    return df_filtered


def main():
    st.title("📊 Executive Summary: Margin & Delivery Diagnosis")

    df = load_data()
    df = apply_sidebar_filters(df)

    job_df = compute_job_metrics(df)
    portfolio_metrics = compute_portfolio_metrics(job_df)

    render_kpi_strip(portfolio_metrics)

    st.divider()

    st.header("Act 1: The Headline & The Contradiction")
    st.markdown(
        """
**What this does**
- Reconciles how portfolio **Realised Rate** can exceed **Quote Rate** even when many jobs overrun.
- Decomposes the gap into under-run savings, overrun dilution, unquoted hours, and residual mix/pricing effects.

**How each metric is calculated**
- **Quote Rate (Base)** = Σ quoted amount ÷ Σ quoted hours (after job-task dedupe). This is the contracted $/hr.
- **Realised Rate** = Σ revenue ÷ Σ actual hours. This is the earned $/hr after delivery.
- **Under-run Jobs ↑** = hours saved on quoted jobs where actual < quoted.  
  - **Hours saved** = Σ(quoted hours − actual hours) for under-run jobs  
  - **Rate impact** = hours saved × quote rate ÷ total actual hours
- **Overrun Jobs ↓** = hours added on quoted jobs where actual > quoted.  
  - **Excess hours** = Σ(actual hours − quoted hours) for overrun jobs  
  - **Rate impact** = −excess hours × quote rate ÷ total actual hours
- **Unquoted Hours ↓** = hours logged on unquoted or unmatched tasks.  
  - **Unquoted hours** = Σ actual hours where `quote_match_flag` ≠ `matched`  
  - **Rate impact** = −unquoted hours × quote rate ÷ total actual hours
- **Mix/Pricing Effect** = residual difference after the above components.  
  - Captures pricing uplift, client mix, task mix, or allocation effects not explained by hours.

**Interpretation tips**
- If **Realised Rate > Quote Rate** but **Overrun Jobs ↓** is large, the portfolio is being “saved” by under-runs or mix/pricing effects.
- A large **Unquoted Hours ↓** suggests scope creep or quoting gaps.
"""
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        bridge_df = compute_rate_bridge(job_df, portfolio_metrics)
        st.plotly_chart(render_rate_bridge(bridge_df), use_container_width=True)

    with col2:
        st.info(
            """
        **Why does Realised Rate exceed Quote Rate despite overruns?**

        This is explained by **mix effects**:
        - Under-run jobs "save" hours, boosting effective rate
        - High-value jobs contribute disproportionately to revenue
        - Revenue-weighted rate ≠ job-count weighted average
        """
        )

    st.divider()

    st.header("Act 2: Winners vs Losers Decomposition")
    st.markdown(
        """
**What this does**
- Classifies each job into a quadrant based on delivery (hours variance %) and pricing (rate variance).
- Highlights “subsidisers” that lift the portfolio and “erosive” jobs that drag it down.

**How each metric is calculated**
- **Hours Variance %** = (actual hours − quoted hours) ÷ quoted hours × 100  
  - Under-run = negative %, Overrun = positive %
- **Rate Variance ($/hr)** = realised rate − quote rate  
  - Positive = pricing captured above quote, Negative = leakage vs quote
- **Bubble size** = absolute revenue for the job (bigger = higher stakes)

**Quadrant meaning**
- **Accretive / Subsidisers**: under-run + rate above quote → lifts portfolio
- **Protected Overruns**: overrun + rate above quote → delivery slipped but pricing protected
- **Under-run Leakage**: under-run + rate below quote → efficient delivery but pricing loss
- **Margin Erosive**: overrun + rate below quote → compounding negative impact

**Interpretation tips**
- Use bubble size to prioritize: a large bubble in the erosive quadrant is an immediate risk.
- Jobs near zero on both axes are stable and low variance.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = render_quadrant_scatter(job_df)
        plotly_chart_with_select(fig_scatter)

    with col2:
        fig_dist, _ = render_overrun_distribution(job_df)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    st.header("Act 3: Operational Intelligence")
    st.markdown(
        """
**What this does**
- Ranks departments on rate capture and overrun impact.
- Surfaces an intervention queue of the riskiest jobs with reason codes and recommended actions.

**Department scoreboard calculations**
- **Department Quote Rate** = Σ quoted amount ÷ Σ quoted hours (weighted)
- **Department Realised Rate** = Σ revenue ÷ Σ actual hours (weighted)
- **Rate Variance** = realised rate − quote rate
- **% Overrun Jobs** = count of jobs with actual > quoted ÷ total jobs
- **Hours Variance** = Σ(actual hours − quoted hours)
- **Margin Impact Est.** = hours variance × blended cost rate (overrun → negative impact)

**Intervention queue calculations**
- **Hours Overrun %** = job-level hours variance ÷ quoted hours
- **Rate Leakage %** = (quote rate − realised rate) ÷ quote rate
- **Risk Score** = 0.30×overrun + 0.25×revenue exposure + 0.25×rate leakage + 0.20×runtime risk
- **Reason Codes** flag why the job is risky (severe overrun, rate leakage, scope creep, negative margin, high exposure)

**Interpretation tips**
- Sort by **Risk Score** to prioritize the next intervention call.
- A high **Rate Leakage %** with low overrun suggests pricing/discounting issues, not delivery.
"""
    )

    tab1, tab2 = st.tabs(["Department Scoreboard", "Intervention Queue"])

    with tab1:
        dept_df = compute_department_scoreboard(job_df)
        if dept_df.empty:
            st.info("No department data available.")
        else:
            st.dataframe(
                dept_df.style.format({
                    "Quote Rate": "${:,.0f}",
                    "Realised Rate": "${:,.0f}",
                    "Rate Variance": "${:+,.0f}",
                    "% Overrun": "{:.0f}%",
                    "Hours Variance": "{:,.0f}",
                    "Margin Impact Est.": "${:,.0f}",
                }).background_gradient(subset=["Rate Variance"], cmap="RdYlGn"),
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        queue_df = compute_intervention_queue(job_df, top_n=15)
        if queue_df.empty:
            st.info("No quoted jobs available for intervention queue.")
        else:
            st.dataframe(
                queue_df.style.format({
                    "Revenue": "${:,.0f}",
                    "Hours Overrun %": "{:.0f}%",
                    "Rate Leakage %": "{:.1%}",
                    "Risk Score": "{:.0f}",
                }).background_gradient(subset=["Risk Score"], cmap="Reds"),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    st.header("🔎 Job Drill-Down")

    job_options = job_df["job_no"].dropna().tolist()
    selected_job = st.selectbox("Select a job to drill into:", options=[""] + job_options)

    if selected_job:
        render_drill_down_panel(df, selected_job)

    st.divider()

    render_reconciliation_panel(df, job_df, portfolio_metrics)


if __name__ == "__main__":
    main()
