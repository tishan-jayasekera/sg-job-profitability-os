"""
Executive Summary: Causal Root-Cause Drill Engine (v2)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet


# =============================================================================
# DATA LOADING
# =============================================================================

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


# =============================================================================
# DRILL STATE
# =============================================================================


def init_drill_state():
    """Initialize session state for drill tracking."""
    if "drill_path" not in st.session_state:
        st.session_state.drill_path = {
            "department": None,
            "category": None,
            "job": None,
            "task": None,
        }


def reset_drill_from(level: str):
    """Reset drill state from a given level downward."""
    levels = ["department", "category", "job", "task"]
    if level not in levels:
        return
    idx = levels.index(level)
    for l in levels[idx:]:
        st.session_state.drill_path[l] = None


def get_drill_breadcrumb() -> str:
    """Generate breadcrumb string for current drill path."""
    path = st.session_state.drill_path
    parts = ["Portfolio"]
    if path["department"]:
        parts.append(path["department"])
    if path["category"] is not None:
        parts.append(path["category"] if path["category"] is not None else "(Uncategorised)")
    if path["job"]:
        parts.append(path["job"])
    if path["task"]:
        parts.append(path["task"])
    return " → ".join(parts)


def render_breadcrumb():
    """Render clickable breadcrumb that allows backtracking."""
    st.markdown(f"**📍 {get_drill_breadcrumb()}**")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("⬅️ Back", disabled=st.session_state.drill_path["department"] is None):
            for level in reversed(["task", "job", "category", "department"]):
                if st.session_state.drill_path[level] is not None:
                    reset_drill_from(level)
                    st.rerun()
                    break
    with col2:
        if st.button("🏠 Reset to Portfolio"):
            reset_drill_from("department")
            st.rerun()


# =============================================================================
# QUOTE DEDUPE
# =============================================================================


def safe_quote_rollup(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Dedupe quotes to job-task, then aggregate to requested grain.
    """
    job_task = df.groupby(["job_no", "task_name"]).agg(
        quoted_time_total=("quoted_time_total", "first"),
        quoted_amount_total=("quoted_amount_total", "first"),
    ).reset_index()

    result = job_task.groupby(group_cols).agg(
        quoted_time_total=("quoted_time_total", "sum"),
        quoted_amount_total=("quoted_amount_total", "sum"),
    ).reset_index()

    return result


# =============================================================================
# METRICS
# =============================================================================


def compute_job_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute job-level metrics with safe quote deduplication."""
    actuals = df.groupby("job_no").agg(
        actual_hours_job=("hours_raw", "sum"),
        revenue_job=("rev_alloc", "sum"),
        cost_job=("base_cost", "sum"),
        department_final=("department_final", "first"),
        job_category=("job_category", "first"),
        job_status=("job_status", "first"),
    ).reset_index()

    quotes = safe_quote_rollup(df, ["job_no"])
    quotes.columns = ["job_no", "quoted_hours_job", "quoted_amount_job"]

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
    job_df["critical_overrun_flag"] = job_df["hours_variance_pct_job"] > 20
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
    """Compute portfolio-level KPIs with weighted rates."""
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
    n_critical = int(quoted_jobs["critical_overrun_flag"].sum())

    margin = total_revenue - total_cost

    return {
        "quote_rate": quote_rate_portfolio,
        "realised_rate": realised_rate_portfolio,
        "rate_variance": realised_rate_portfolio - quote_rate_portfolio,
        "margin": margin,
        "margin_pct": (margin / total_revenue * 100) if total_revenue > 0 else 0,
        "pct_jobs_overrun": (n_overrun / n_quoted * 100) if n_quoted > 0 else 0,
        "pct_jobs_critical_overrun": (n_critical / n_quoted * 100) if n_quoted > 0 else 0,
        "total_hours_variance": quoted_jobs["hours_variance_job"].sum(),
        "n_jobs": n_jobs,
        "n_quoted_jobs": n_quoted,
    }


def compute_rate_bridge(job_df: pd.DataFrame, portfolio_metrics: Dict) -> pd.DataFrame:
    """Decompose the Quote → Realised rate bridge."""
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


# =============================================================================
# PORTFOLIO RENDERING
# =============================================================================


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
        st.metric("% Critical Overrun", f"{metrics['pct_jobs_critical_overrun']:.0f}%")


def render_rate_bridge(bridge_df: pd.DataFrame):
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


# =============================================================================
# ACT 2a: DEPARTMENT
# =============================================================================


def compute_department_contribution(job_df: pd.DataFrame, portfolio_metrics: dict) -> pd.DataFrame:
    """Compute each department's contribution to portfolio realised rate."""
    portfolio_rate = portfolio_metrics["realised_rate"]
    portfolio_revenue = job_df["revenue_job"].sum()

    dept_df = job_df.groupby("department_final").agg(
        revenue=("revenue_job", "sum"),
        actual_hours=("actual_hours_job", "sum"),
        cost=("cost_job", "sum"),
        quoted_hours=("quoted_hours_job", "sum"),
        quoted_amount=("quoted_amount_job", "sum"),
        hours_variance=("hours_variance_job", "sum"),
        overrun_jobs=("overrun_flag", "sum"),
        total_jobs=("job_no", "count"),
    ).reset_index()

    dept_df["realised_rate"] = np.where(
        dept_df["actual_hours"] > 0,
        dept_df["revenue"] / dept_df["actual_hours"],
        np.nan,
    )
    dept_df["quote_rate"] = np.where(
        dept_df["quoted_hours"] > 0,
        dept_df["quoted_amount"] / dept_df["quoted_hours"],
        np.nan,
    )

    dept_df["revenue_weight"] = dept_df["revenue"] / portfolio_revenue if portfolio_revenue > 0 else 0
    dept_df["rate_contribution"] = dept_df["revenue_weight"] * (dept_df["realised_rate"] - portfolio_rate)
    dept_df["impact_type"] = np.where(dept_df["rate_contribution"] >= 0, "Subsidiser", "Erosive")

    dept_df = dept_df.rename(columns={"department_final": "department"})

    return dept_df.sort_values("rate_contribution", ascending=True)


def render_dept_contribution_bars(dept_df: pd.DataFrame):
    colors = {"Subsidiser": "#2E7D32", "Erosive": "#C62828"}
    fig = px.bar(
        dept_df,
        x="rate_contribution",
        y="department",
        orientation="h",
        color="impact_type",
        color_discrete_map=colors,
        labels={"rate_contribution": "Rate Contribution ($/hr)", "department": "Department"},
    )
    fig.update_layout(title="Department Contribution to Portfolio Rate", height=420)
    return fig


def render_dept_efficiency_scatter(dept_df: pd.DataFrame):
    dept_df = dept_df.copy()
    dept_df["hours_variance_pct"] = np.where(
        dept_df["quoted_hours"] > 0,
        dept_df["hours_variance"] / dept_df["quoted_hours"] * 100,
        np.nan,
    )
    dept_df["rate_variance"] = dept_df["realised_rate"] - dept_df["quote_rate"]
    dept_df["size"] = dept_df["revenue"].abs()

    fig = px.scatter(
        dept_df,
        x="hours_variance_pct",
        y="rate_variance",
        size="size",
        color="impact_type",
        color_discrete_map={"Subsidiser": "#2E7D32", "Erosive": "#C62828"},
        hover_data=["department", "revenue", "actual_hours"],
        labels={
            "hours_variance_pct": "Hours Variance %",
            "rate_variance": "Rate Variance ($/hr)",
            "size": "Revenue",
        },
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(title="Department Efficiency vs Pricing", height=420)
    return fig


def render_department_selector(dept_df: pd.DataFrame):
    st.subheader("Select Department to Drill Into")

    dept_options = dept_df["department"].dropna().tolist()
    selected = st.selectbox(
        "Choose department:",
        options=["-- Select --"] + dept_options,
        format_func=lambda x: x if x == "-- Select --" else f"{x} ({dept_df[dept_df['department'] == x]['impact_type'].values[0]})",
    )

    if selected != "-- Select --":
        st.session_state.drill_path["department"] = selected
        st.rerun()
    else:
        st.warning("⬇️ Select a department above to continue root-cause analysis")


# =============================================================================
# ACT 2b: CATEGORY
# =============================================================================


def compute_category_contribution(job_df: pd.DataFrame, department: str) -> pd.DataFrame:
    dept_jobs = job_df[job_df["department_final"] == department].copy()
    if dept_jobs.empty:
        return pd.DataFrame()

    dept_rate = dept_jobs["revenue_job"].sum() / dept_jobs["actual_hours_job"].sum()
    dept_revenue = dept_jobs["revenue_job"].sum()

    cat_df = dept_jobs.groupby("job_category").agg(
        revenue=("revenue_job", "sum"),
        actual_hours=("actual_hours_job", "sum"),
        quoted_hours=("quoted_hours_job", "sum"),
        quoted_amount=("quoted_amount_job", "sum"),
        hours_variance=("hours_variance_job", "sum"),
        overrun_jobs=("overrun_flag", "sum"),
        total_jobs=("job_no", "count"),
    ).reset_index()

    cat_df["realised_rate"] = np.where(
        cat_df["actual_hours"] > 0,
        cat_df["revenue"] / cat_df["actual_hours"],
        np.nan,
    )
    cat_df["quote_rate"] = np.where(
        cat_df["quoted_hours"] > 0,
        cat_df["quoted_amount"] / cat_df["quoted_hours"],
        np.nan,
    )
    cat_df["rate_variance"] = cat_df["realised_rate"] - cat_df["quote_rate"]
    cat_df["hours_variance_pct"] = np.where(
        cat_df["quoted_hours"] > 0,
        cat_df["hours_variance"] / cat_df["quoted_hours"] * 100,
        np.nan,
    )

    cat_df["revenue_weight"] = cat_df["revenue"] / dept_revenue if dept_revenue > 0 else 0
    cat_df["rate_contribution"] = cat_df["revenue_weight"] * (cat_df["realised_rate"] - dept_rate)
    cat_df["impact_type"] = np.where(cat_df["rate_contribution"] >= 0, "Subsidiser", "Erosive")

    return cat_df.sort_values("rate_contribution", ascending=True)


def render_category_contribution_bars(cat_df: pd.DataFrame):
    colors = {"Subsidiser": "#2E7D32", "Erosive": "#C62828"}
    fig = px.bar(
        cat_df,
        x="rate_contribution",
        y="job_category",
        orientation="h",
        color="impact_type",
        color_discrete_map=colors,
        labels={"rate_contribution": "Rate Contribution ($/hr)", "job_category": "Category"},
    )
    fig.update_layout(title="Category Contribution to Department Rate", height=420)
    return fig


def render_category_efficiency_scatter(cat_df: pd.DataFrame):
    cat_df = cat_df.copy()
    cat_df["size"] = cat_df["revenue"].abs()

    fig = px.scatter(
        cat_df,
        x="hours_variance_pct",
        y="rate_variance",
        size="size",
        color="impact_type",
        color_discrete_map={"Subsidiser": "#2E7D32", "Erosive": "#C62828"},
        hover_data=["job_category", "revenue", "actual_hours"],
        labels={
            "hours_variance_pct": "Hours Variance %",
            "rate_variance": "Rate Variance ($/hr)",
            "size": "Revenue",
        },
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(title="Category Efficiency vs Pricing", height=420)
    return fig


def render_category_selector(cat_df: pd.DataFrame):
    st.subheader(f"Select Category within {st.session_state.drill_path['department']}")

    cat_df = cat_df.copy()
    cat_df["category_display"] = cat_df["job_category"].fillna("(Uncategorised)")
    display_options = cat_df["category_display"].tolist()

    selected = st.selectbox(
        "Choose category:",
        options=["-- Select --"] + display_options,
    )

    if selected != "-- Select --":
        actual_val = None if selected == "(Uncategorised)" else selected
        st.session_state.drill_path["category"] = actual_val
        st.rerun()
    else:
        st.warning("⬇️ Select a category above to see job-level breakdown")


# =============================================================================
# ACT 2c: JOB QUADRANT
# =============================================================================


def render_job_quadrant_scatter(job_df: pd.DataFrame, impact_mode: str):
    if job_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No jobs available",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(title="Job Outcome Quadrant", height=500)
        return fig

    total_revenue = job_df["revenue_job"].sum()
    category_rate = total_revenue / job_df["actual_hours_job"].sum() if job_df["actual_hours_job"].sum() > 0 else 0
    cost_rate = job_df["cost_job"].sum() / job_df["actual_hours_job"].sum() if job_df["actual_hours_job"].sum() > 0 else 0

    job_df = job_df.copy()
    job_df["revenue_weight"] = job_df["revenue_job"] / total_revenue if total_revenue > 0 else 0
    job_df["rate_contribution"] = job_df["revenue_weight"] * (job_df["realised_rate_job"] - category_rate)
    job_df["impact_type"] = np.where(job_df["rate_contribution"] >= 0, "Subsidiser", "Erosive")

    job_df["abs_rate_contribution"] = job_df["rate_contribution"].abs()
    job_df["abs_margin_impact"] = (job_df["hours_variance_job"] * cost_rate).abs()

    if impact_mode == "Revenue Exposure":
        size_col = "revenue_job"
    elif impact_mode == "Rate Impact":
        size_col = "abs_rate_contribution"
    else:
        size_col = "abs_margin_impact"

    job_df["size"] = job_df[size_col].abs()

    fig = px.scatter(
        job_df,
        x="hours_variance_pct_job",
        y="rate_variance_job",
        size="size",
        color="impact_type",
        color_discrete_map={"Subsidiser": "#2E7D32", "Erosive": "#C62828"},
        hover_data=["job_no", "revenue_job", "margin_pct_job"],
        labels={
            "hours_variance_pct_job": "Hours Variance %",
            "rate_variance_job": "Rate Variance ($/hr)",
            "size": impact_mode,
        },
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title="Job Outcome Quadrant", height=500)

    return fig


def render_job_selector(job_df: pd.DataFrame):
    st.markdown("**Select job to drill into:**")
    sort_df = job_df.copy()
    total_revenue = sort_df["revenue_job"].sum()
    category_rate = total_revenue / sort_df["actual_hours_job"].sum() if sort_df["actual_hours_job"].sum() > 0 else 0
    sort_df["revenue_weight"] = sort_df["revenue_job"] / total_revenue if total_revenue > 0 else 0
    sort_df["rate_contribution"] = sort_df["revenue_weight"] * (sort_df["realised_rate_job"] - category_rate)
    sort_df["priority_score"] = sort_df["rate_contribution"].abs()

    job_options = sort_df.sort_values("priority_score", ascending=False)["job_no"].tolist()
    selected = st.selectbox("Choose job:", options=["-- Select --"] + job_options)

    if selected != "-- Select --":
        st.session_state.drill_path["job"] = selected
        st.rerun()


# =============================================================================
# ACT 3a: JOB DIAGNOSTIC
# =============================================================================


def render_job_diagnostic_card(df: pd.DataFrame, job_no: str):
    job_data = df[df["job_no"] == job_no]

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

    hours_variance = actual_hours - quoted_hours
    hours_variance_pct = (hours_variance / quoted_hours * 100) if quoted_hours > 0 else 0
    rate_variance = realised_rate - quote_rate

    margin = revenue - cost
    margin_pct = (margin / revenue * 100) if revenue > 0 else 0

    unquoted_tasks = job_data[job_data["quote_match_flag"] != "matched"] if "quote_match_flag" in job_data.columns else pd.DataFrame()
    unquoted_hours = unquoted_tasks["hours_raw"].sum() if not unquoted_tasks.empty else 0
    unquoted_pct = (unquoted_hours / actual_hours * 100) if actual_hours > 0 else 0

    st.subheader(f"📋 Job Diagnostic: {job_no}")

    st.markdown("### A. Volume Efficiency (Time)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quoted Hours", f"{quoted_hours:,.0f}")
    col2.metric("Actual Hours", f"{actual_hours:,.0f}", delta=f"{hours_variance:+,.0f}")
    col3.metric("Hours Variance", f"{hours_variance_pct:+.0f}%")
    col4.metric("Scope Creep", f"{unquoted_pct:.0f}%", help="Unquoted hours as % of actual hours")

    st.markdown("### B. Rate Integrity (Commercial)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Quote Rate", f"${quote_rate:,.0f}/hr")
    col2.metric("Realised Rate", f"${realised_rate:,.0f}/hr", delta=f"${rate_variance:+,.0f}")
    col3.metric("Rate Capture", f"{(realised_rate / quote_rate * 100) if quote_rate > 0 else 0:.0f}%")

    st.markdown("### C. Margin Outcome")
    cost_rate = cost / actual_hours if actual_hours > 0 else 0
    cost_delta_from_hours = (actual_hours - quoted_hours) * cost_rate

    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue", f"${revenue:,.0f}")
    col2.metric("Cost", f"${cost:,.0f}")
    col3.metric("Margin", f"${margin:,.0f}", delta=f"{margin_pct:.1f}%")

    st.caption(f"**Margin delta from hours overrun:** ${cost_delta_from_hours:,.0f}")


# =============================================================================
# ACT 3b: TASK WATERFALL
# =============================================================================


def compute_task_waterfall(df: pd.DataFrame, job_no: str) -> pd.DataFrame:
    job_data = df[df["job_no"] == job_no]

    task_df = job_data.groupby("task_name").agg(
        hours_raw=("hours_raw", "sum"),
        quoted_time_total=("quoted_time_total", "first"),
        quote_match_flag=("quote_match_flag", "first"),
        base_cost=("base_cost", "sum"),
    ).reset_index()

    task_df["quoted_hours"] = task_df["quoted_time_total"].fillna(0)
    task_df["actual_hours"] = task_df["hours_raw"]
    task_df["hours_variance"] = task_df["actual_hours"] - task_df["quoted_hours"]

    task_df["variance_type"] = np.where(
        task_df["quote_match_flag"] != "matched",
        "Scope Creep (Unquoted)",
        np.where(
            task_df["hours_variance"] > 0,
            "Overrun (Inefficiency)",
            np.where(
                task_df["hours_variance"] < 0,
                "Under-run (Efficiency)",
                "On Target",
            ),
        ),
    )

    total_abs_variance = task_df["hours_variance"].abs().sum()
    task_df["contribution_pct"] = (
        task_df["hours_variance"].abs() / total_abs_variance * 100 if total_abs_variance > 0 else 0
    )

    return task_df.sort_values("hours_variance", ascending=False)


def render_task_waterfall(task_df: pd.DataFrame, quoted_hours_total: float):
    waterfall_data = [{"task": "Quoted Hours", "value": quoted_hours_total, "type": "absolute"}]

    for _, row in task_df.iterrows():
        if row["hours_variance"] != 0:
            waterfall_data.append({
                "task": row["task_name"][:25],
                "value": row["hours_variance"],
                "type": "relative",
            })

    waterfall_data.append({
        "task": "Actual Hours",
        "value": quoted_hours_total + task_df["hours_variance"].sum(),
        "type": "total",
    })

    wf_df = pd.DataFrame(waterfall_data)

    fig = go.Figure(
        go.Waterfall(
            name="Hours",
            orientation="v",
            measure=wf_df["type"].tolist(),
            x=wf_df["task"].tolist(),
            y=wf_df["value"].tolist(),
            textposition="outside",
            text=[
                f"{v:+,.0f}" if t == "relative" else f"{v:,.0f}"
                for v, t in zip(wf_df["value"], wf_df["type"])
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#C62828"}},
            decreasing={"marker": {"color": "#2E7D32"}},
            totals={"marker": {"color": "#1565C0"}},
        )
    )

    fig.update_layout(
        title="Task Contribution to Hours Variance",
        yaxis_title="Hours",
        showlegend=False,
        height=400,
    )

    return fig


def render_task_table(task_df: pd.DataFrame):
    display_df = task_df[[
        "task_name",
        "quoted_hours",
        "actual_hours",
        "hours_variance",
        "variance_type",
        "contribution_pct",
        "base_cost",
    ]].copy()

    display_df.columns = [
        "Task",
        "Quoted Hrs",
        "Actual Hrs",
        "Variance",
        "Type",
        "Contribution %",
        "Cost",
    ]

    def highlight_type(row):
        if "Scope Creep" in row["Type"]:
            return ["background-color: #FFCDD2"] * len(row)
        if "Overrun" in row["Type"]:
            return ["background-color: #FFE0B2"] * len(row)
        if "Under-run" in row["Type"]:
            return ["background-color: #C8E6C9"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style
            .apply(highlight_type, axis=1)
            .format({
                "Quoted Hrs": "{:,.0f}",
                "Actual Hrs": "{:,.0f}",
                "Variance": "{:+,.0f}",
                "Contribution %": "{:.0f}%",
                "Cost": "${:,.0f}",
            }),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# ACT 3c: FTE ATTRIBUTION
# =============================================================================


def render_fte_attribution(df: pd.DataFrame, job_no: str, task_name: Optional[str] = None):
    job_data = df[df["job_no"] == job_no]

    if task_name:
        job_data = job_data[job_data["task_name"] == task_name]
        st.subheader(f"👥 Staff Attribution: {task_name}")
    else:
        st.subheader("👥 Staff Attribution (All Tasks)")

    staff_df = job_data.groupby("staff_name").agg(
        hours_raw=("hours_raw", "sum"),
        base_cost=("base_cost", "sum"),
    ).reset_index()

    staff_df.columns = ["Staff", "Hours", "Cost"]

    total_hours = staff_df["Hours"].sum()
    total_cost = staff_df["Cost"].sum()

    staff_df["Hours %"] = staff_df["Hours"] / total_hours * 100 if total_hours > 0 else 0
    staff_df["Cost %"] = staff_df["Cost"] / total_cost * 100 if total_cost > 0 else 0
    staff_df["Effective Rate"] = np.where(
        staff_df["Hours"] > 0,
        staff_df["Cost"] / staff_df["Hours"],
        np.nan,
    )

    staff_df = staff_df.sort_values("Hours", ascending=False)

    st.dataframe(
        staff_df.style.format({
            "Hours": "{:,.1f}",
            "Cost": "${:,.0f}",
            "Hours %": "{:.0f}%",
            "Cost %": "{:.0f}%",
            "Effective Rate": "${:,.0f}/hr",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if not staff_df.empty:
        top_staff = staff_df.iloc[0]
        st.info(
            f"**{top_staff['Staff']}** contributed {top_staff['Hours %']:.0f}% of hours "
            f"at ${top_staff['Effective Rate']:,.0f}/hr effective rate"
        )


# =============================================================================
# RECONCILIATION
# =============================================================================


def render_reconciliation_panel(df: pd.DataFrame, job_df: pd.DataFrame, portfolio_metrics: Dict):
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


# =============================================================================
# MAIN
# =============================================================================


def main():
    st.set_page_config(
        page_title="Executive Summary | Root Cause Engine",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Executive Summary: Causal Root-Cause Engine")

    init_drill_state()

    df = load_data()
    job_df = compute_job_metrics(df)
    portfolio_metrics = compute_portfolio_metrics(job_df)

    render_breadcrumb()
    st.divider()

    # ========================================
    # ACT 1: PORTFOLIO
    # ========================================
    st.header("Act 1: Portfolio Headline")
    render_kpi_strip(portfolio_metrics)

    col1, col2 = st.columns([2, 1])
    with col1:
        bridge_df = compute_rate_bridge(job_df, portfolio_metrics)
        st.plotly_chart(render_rate_bridge(bridge_df), use_container_width=True)
    with col2:
        st.info(
            """
**The Contradiction Explained**

Realised rate exceeds quote rate despite overruns because of **mix effects**.

**Click a department below to trace the root cause.**
"""
        )

    st.divider()

    # ========================================
    # ACT 2a: DEPARTMENT
    # ========================================
    if st.session_state.drill_path["department"] is None:
        st.header("Act 2a: Department Diagnostic")
        st.caption("Which department is driving the portfolio outcome?")

        dept_df = compute_department_contribution(job_df, portfolio_metrics)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(render_dept_contribution_bars(dept_df), use_container_width=True)
        with col2:
            st.plotly_chart(render_dept_efficiency_scatter(dept_df), use_container_width=True)

        render_department_selector(dept_df)
        return

    # ========================================
    # ACT 2b: CATEGORY
    # ========================================
    if st.session_state.drill_path["category"] is None:
        st.header(f"Act 2b: Category Diagnostic — {st.session_state.drill_path['department']}")
        st.caption("Is this a category problem or a job-level problem?")

        cat_df = compute_category_contribution(job_df, st.session_state.drill_path["department"])

        if cat_df.empty:
            st.warning("No category data available for this department")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(render_category_contribution_bars(cat_df), use_container_width=True)
            with col2:
                st.plotly_chart(render_category_efficiency_scatter(cat_df), use_container_width=True)

            render_category_selector(cat_df)
        return

    # ========================================
    # ACT 2c: JOB QUADRANT
    # ========================================
    if st.session_state.drill_path["job"] is None:
        dept = st.session_state.drill_path["department"]
        cat = st.session_state.drill_path["category"]

        st.header(f"Act 2c: Job Quadrant — {cat or '(Uncategorised)'}")
        st.caption("Which jobs are Subsidisers and which are Erosive?")

        filtered_jobs = job_df[
            (job_df["department_final"] == dept)
            & ((job_df["job_category"] == cat) | (job_df["job_category"].isna() & (cat is None)))
        ].copy()

        impact_mode = st.radio(
            "Size bubbles by:",
            options=["Revenue Exposure", "Rate Impact", "Margin Impact"],
            horizontal=True,
        )

        st.plotly_chart(render_job_quadrant_scatter(filtered_jobs, impact_mode), use_container_width=True)
        render_job_selector(filtered_jobs)
        return

    # ========================================
    # ACT 3: JOB → TASK → FTE
    # ========================================
    job_no = st.session_state.drill_path["job"]

    st.header(f"Act 3: Root Cause — Job {job_no}")
    render_job_diagnostic_card(df, job_no)

    st.divider()

    st.subheader("Task Contribution Waterfall")
    task_df = compute_task_waterfall(df, job_no)
    quoted_hours = task_df["quoted_hours"].sum()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(render_task_waterfall(task_df, quoted_hours), use_container_width=True)
    with col2:
        render_task_table(task_df)

    st.divider()

    task_selection = st.selectbox(
        "View staff for:",
        options=["All Tasks"] + task_df["task_name"].tolist(),
    )

    selected_task = None if task_selection == "All Tasks" else task_selection
    render_fte_attribution(df, job_no, selected_task)

    st.divider()
    render_reconciliation_panel(df, job_df, portfolio_metrics)


if __name__ == "__main__":
    main()
