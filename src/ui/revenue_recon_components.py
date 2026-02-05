"""
UI components for Revenue Reconciliation page.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.metrics.revenue_reconciliation import (
    REVENUE_STATUS_CONFIG,
    ReconciliationSummary,
    analyze_patterns_by_dimension,
    check_data_quality,
    compute_concentration_curve,
    diagnose_job,
)
from src.ui.formatting import format_job_label


# =============================================================================
# SECTION 1: PORTFOLIO SUMMARY
# =============================================================================


def render_portfolio_summary(summary: ReconciliationSummary):
    """Render the portfolio-level reconciliation summary."""

    st.subheader("1️⃣ Portfolio Reconciliation")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Quoted",
            f"${summary.total_quoted:,.0f}",
            help="Sum of quoted amounts for all jobs with quotes",
        )

    with col2:
        st.metric(
            "Total Revenue",
            f"${summary.total_revenue:,.0f}",
            help="Sum of actual revenue for jobs with quotes",
        )

    with col3:
        delta_color = "normal" if summary.total_delta >= 0 else "inverse"
        st.metric(
            "Net Delta",
            f"${summary.total_delta:+,.0f}",
            delta=f"{summary.aggregate_capture_pct:.0f}% capture",
            delta_color=delta_color,
        )

    with col4:
        st.metric(
            "Jobs Analyzed",
            f"{summary.jobs_matched:,}",
            help=f"Jobs with both quote and revenue ({summary.jobs_with_quotes} quoted, {summary.jobs_with_revenue} with revenue)",
        )

    # Interpretation
    if summary.aggregate_capture_pct >= 100:
        st.success(
            f"✅ Portfolio is collecting **{summary.aggregate_capture_pct:.0f}%** of quoted amounts — net over-recovery"
        )
    elif summary.aggregate_capture_pct >= 90:
        st.info(
            f"ℹ️ Portfolio capture at **{summary.aggregate_capture_pct:.0f}%** — within acceptable range"
        )
    elif summary.aggregate_capture_pct >= 80:
        st.warning(
            f"⚠️ Portfolio capture at **{summary.aggregate_capture_pct:.0f}%** — notable leakage"
        )
    else:
        st.error(
            f"🔴 Portfolio capture at **{summary.aggregate_capture_pct:.0f}%** — significant commercial leakage"
        )


# =============================================================================
# SECTION 2: DISTRIBUTION
# =============================================================================


def render_distribution_analysis(recon_df: pd.DataFrame, summary: ReconciliationSummary):
    """Render the job-level distribution analysis."""

    st.subheader("2️⃣ Job-Level Distribution")
    st.caption("*Is this systemic leakage or concentrated in a few jobs?*")

    # Status breakdown
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        config = REVENUE_STATUS_CONFIG["Over-Recovery"]
        st.metric(
            f"{config['icon']} Over-Recovery",
            f"{summary.jobs_over_recovery} jobs",
            f"${summary.delta_over_recovery:+,.0f}",
            help=config["description"],
        )

    with col2:
        config = REVENUE_STATUS_CONFIG["On-Target"]
        st.metric(
            f"{config['icon']} On-Target",
            f"{summary.jobs_on_target} jobs",
            help=config["description"],
        )

    with col3:
        config = REVENUE_STATUS_CONFIG["Minor Leakage"]
        st.metric(
            f"{config['icon']} Minor Leakage",
            f"{summary.jobs_minor_leakage} jobs",
            f"${summary.delta_minor_leakage:+,.0f}",
            help=config["description"],
        )

    with col4:
        config = REVENUE_STATUS_CONFIG["Major Leakage"]
        st.metric(
            f"{config['icon']} Major Leakage",
            f"{summary.jobs_major_leakage} jobs",
            f"${summary.delta_major_leakage:+,.0f}",
            help=config["description"],
        )

    # Histogram
    quoted_jobs = recon_df[recon_df["quoted_amount"] > 0].copy()

    if len(quoted_jobs) > 0:
        fig = px.histogram(
            quoted_jobs,
            x="revenue_delta_pct",
            nbins=40,
            color="revenue_status",
            color_discrete_map={
                status: config["color"] for status, config in REVENUE_STATUS_CONFIG.items()
            },
            category_orders={
                "revenue_status": [
                    "Over-Recovery",
                    "On-Target",
                    "Minor Leakage",
                    "Major Leakage",
                ]
            },
        )

        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
        fig.add_annotation(
            x=0, y=1, yref="paper", text="100% Capture", showarrow=False, yshift=10
        )

        fig.update_layout(
            title="Distribution of Revenue Capture by Job",
            xaxis_title="Revenue Delta % ((Revenue - Quote) / Quote × 100)",
            yaxis_title="Number of Jobs",
            height=400,
            legend_title="Status",
            bargap=0.1,
        )

        st.plotly_chart(fig, use_container_width=True)

    # Key insight
    total_leakage_jobs = summary.jobs_minor_leakage + summary.jobs_major_leakage
    total_leakage_pct = (
        total_leakage_jobs / summary.jobs_matched * 100
        if summary.jobs_matched > 0
        else 0
    )

    if total_leakage_pct > 30:
        st.warning(
            f"⚠️ **{total_leakage_pct:.0f}% of jobs** have leakage — this is systemic, not isolated"
        )
    elif total_leakage_pct > 10:
        st.info(
            f"ℹ️ **{total_leakage_pct:.0f}% of jobs** have leakage — check if concentrated"
        )
    else:
        st.success(
            f"✅ Only **{total_leakage_pct:.0f}% of jobs** have leakage — well controlled"
        )


# =============================================================================
# SECTION 3: CONCENTRATION
# =============================================================================


def render_concentration_analysis(
    recon_df: pd.DataFrame, summary: ReconciliationSummary, job_name_lookup: Dict[str, str]
):
    """Render the concentration analysis."""

    st.subheader("3️⃣ Concentration Analysis")
    st.caption("*How many jobs explain the leakage?*")

    # Top leakage jobs table
    st.markdown("**Top 10 Jobs with Largest Leakage**")

    top_leakage = recon_df.nsmallest(10, "revenue_delta").copy()
    top_leakage["job_label"] = top_leakage["job_no"].apply(
        lambda x: format_job_label(x, job_name_lookup)
    )

    display_cols = [
        "job_label",
        "department",
        "quoted_amount",
        "actual_revenue",
        "revenue_delta",
        "revenue_delta_pct",
    ]
    display_df = top_leakage[[c for c in display_cols if c in top_leakage.columns]].copy()

    st.dataframe(
        display_df,
        column_config={
            "job_label": st.column_config.TextColumn("Job", width="large"),
            "department": "Department",
            "quoted_amount": st.column_config.NumberColumn("Quoted", format="$%.0f"),
            "actual_revenue": st.column_config.NumberColumn("Revenue", format="$%.0f"),
            "revenue_delta": st.column_config.NumberColumn("Delta", format="$%+.0f"),
            "revenue_delta_pct": st.column_config.NumberColumn("Delta %", format="%+.0f%%"),
        },
        hide_index=True,
        use_container_width=True,
    )

    # Cumulative curve
    curve_df = compute_concentration_curve(recon_df)

    if len(curve_df) > 0:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=curve_df["job_rank"],
                y=curve_df["cumulative_pct"],
                mode="lines+markers",
                name="Cumulative Leakage",
                line=dict(color="#dc3545", width=2),
                marker=dict(size=4),
            )
        )

        # Add 80% line
        fig.add_hline(
            y=80,
            line_dash="dot",
            line_color="#666",
            annotation_text="80% of leakage",
            annotation_position="right",
        )

        fig.update_layout(
            title="Cumulative Leakage Curve",
            xaxis_title="Number of Jobs (ranked by leakage severity)",
            yaxis_title="Cumulative % of Total Leakage",
            height=350,
            yaxis_range=[0, 105],
        )

        st.plotly_chart(fig, use_container_width=True)

        # Insight
        if summary.jobs_for_80pct_leakage > 0:
            total_leakage_jobs = len(curve_df)
            concentration_ratio = (
                summary.jobs_for_80pct_leakage / total_leakage_jobs * 100
                if total_leakage_jobs > 0
                else 0
            )

            if concentration_ratio < 20:
                st.success(
                    f"💡 **{summary.jobs_for_80pct_leakage} jobs** ({concentration_ratio:.0f}% of leakage jobs) explain 80% of total leakage — highly concentrated, fix these first"
                )
            elif concentration_ratio < 50:
                st.info(
                    f"💡 **{summary.jobs_for_80pct_leakage} jobs** explain 80% of leakage — moderately concentrated"
                )
            else:
                st.warning(
                    f"💡 **{summary.jobs_for_80pct_leakage} jobs** explain 80% of leakage — broadly distributed, likely systemic issue"
                )


# =============================================================================
# SECTION 4: PATTERNS
# =============================================================================


def render_pattern_analysis(recon_df: pd.DataFrame):
    """Render pattern analysis by various dimensions."""

    st.subheader("4️⃣ Pattern Analysis")
    st.caption("*Who has leakage? Where is it concentrated?*")

    tabs = st.tabs(["By Department", "By Job Size", "By Category", "By Client"])

    # By Department
    with tabs[0]:
        dept_df = analyze_patterns_by_dimension(recon_df, "department")
        if len(dept_df) > 0:
            _render_pattern_chart(dept_df, "department", "Department")
        else:
            st.info("No department data available")

    # By Job Size
    with tabs[1]:
        size_df = analyze_patterns_by_dimension(recon_df, "size_bucket")
        if len(size_df) > 0:
            _render_pattern_chart(size_df, "size_bucket", "Job Size")
            st.caption(
                "💡 Small jobs often have higher write-off rates due to fixed admin costs exceeding job value"
            )
        else:
            st.info("No size data available")

    # By Category
    with tabs[2]:
        cat_df = analyze_patterns_by_dimension(recon_df, "category")
        if len(cat_df) > 0:
            _render_pattern_chart(cat_df, "category", "Category")
        else:
            st.info("No category data available")

    # By Client
    with tabs[3]:
        client_df = analyze_patterns_by_dimension(recon_df, "client")
        if len(client_df) > 0 and client_df["client"].notna().any():
            # Show worst clients
            worst_clients = client_df.nsmallest(10, "capture_pct")
            _render_pattern_chart(worst_clients, "client", "Client (Worst 10)")
        else:
            st.info("No client data available")


def _render_pattern_chart(df: pd.DataFrame, group_col: str, label: str):
    """Render a pattern analysis chart for a dimension."""

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart
        fig = go.Figure()

        colors = ["#dc3545" if x < 0 else "#28a745" for x in df["total_delta"]]

        fig.add_trace(
            go.Bar(
                y=df[group_col].astype(str),
                x=df["total_delta"],
                orientation="h",
                marker_color=colors,
                text=[f"${x:+,.0f}" for x in df["total_delta"]],
                textposition="outside",
            )
        )

        fig.add_vline(x=0, line_dash="dash", line_color="#666")

        fig.update_layout(
            title=f"Revenue Delta by {label}",
            xaxis_title="Revenue Delta ($)",
            yaxis_title="",
            height=max(300, len(df) * 40),
            margin=dict(l=150),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Summary table
        display_df = df[[group_col, "job_count", "capture_pct", "leakage_rate"]].copy()
        display_df.columns = [label, "Jobs", "Capture %", "Leakage Rate"]

        st.dataframe(
            display_df,
            column_config={
                "Capture %": st.column_config.NumberColumn(format="%.0f%%"),
                "Leakage Rate": st.column_config.NumberColumn(format="%.0f%%"),
            },
            hide_index=True,
            use_container_width=True,
        )


# =============================================================================
# SECTION 5: JOB DEEP DIVE
# =============================================================================


def render_job_deep_dive(
    df: pd.DataFrame, recon_df: pd.DataFrame, job_name_lookup: Dict[str, str]
):
    """Render job-level deep dive section."""

    st.subheader("5️⃣ Job Deep Dive")
    st.caption("*Diagnose the root cause of a specific job's leakage*")

    # Job selector
    leakage_jobs = recon_df[
        recon_df["revenue_status"].isin(["Minor Leakage", "Major Leakage"])
    ].nsmallest(30, "revenue_delta")

    if len(leakage_jobs) == 0:
        st.success("✅ No significant leakage jobs to diagnose")
        return

    job_options = leakage_jobs["job_no"].tolist()

    selected_job = st.selectbox(
        "Select a job to diagnose",
        job_options,
        format_func=lambda x: f"{format_job_label(x, job_name_lookup)} — ${recon_df[recon_df['job_no'] == x]['revenue_delta'].iloc[0]:+,.0f}",
    )

    if not selected_job:
        return

    # Get diagnosis
    diagnosis = diagnose_job(df, recon_df, selected_job)

    if not diagnosis:
        st.error("Could not diagnose this job")
        return

    # Job header
    st.markdown(f"### {format_job_label(diagnosis.job_no, job_name_lookup)}")

    if diagnosis.department or diagnosis.category:
        st.caption(f"{diagnosis.department or ''} • {diagnosis.category or ''}")

    # Core metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Quoted**")
        st.metric("Amount", f"${diagnosis.quoted_amount:,.0f}")
        st.metric("Hours", f"{diagnosis.quoted_hours:,.0f}")
        st.metric("Rate", f"${diagnosis.quote_rate:.0f}/hr")

    with col2:
        st.markdown("**Actual**")
        st.metric("Revenue", f"${diagnosis.actual_revenue:,.0f}")
        st.metric("Hours", f"{diagnosis.actual_hours:,.0f}")
        st.metric("Rate", f"${diagnosis.realised_rate:.0f}/hr")

    with col3:
        st.markdown("**Variance**")
        delta_color = "normal" if diagnosis.revenue_delta >= 0 else "inverse"
        st.metric(
            "Revenue Delta",
            f"${diagnosis.revenue_delta:+,.0f}",
            delta=f"{diagnosis.revenue_delta_pct:+.0f}%",
            delta_color=delta_color,
        )
        st.metric(
            "Hours Variance",
            f"{diagnosis.hours_variance:+,.0f}",
            delta=f"{diagnosis.hours_variance_pct:+.0f}%",
        )
        st.metric("Rate Delta", f"${diagnosis.rate_delta:+.0f}/hr")

    st.divider()

    # Gap decomposition
    st.markdown("**Gap Decomposition**")
    st.markdown(
        f"""
    The ${diagnosis.revenue_delta:+,.0f} gap breaks down as:

    | Component | Amount | Explanation |
    |-----------|--------|-------------|
    | Hours Effect | ${diagnosis.gap_from_hours:+,.0f} | {diagnosis.hours_variance:+,.0f} hours × ${diagnosis.quote_rate:.0f}/hr quote rate |
    | Rate Effect | ${diagnosis.gap_from_rate:+,.0f} | ${diagnosis.rate_delta:+.0f}/hr × {diagnosis.actual_hours:,.0f} actual hours |
    | **Total** | **${diagnosis.gap_from_hours + diagnosis.gap_from_rate:+,.0f}** | |
    """
    )

    # Hypotheses
    if diagnosis.hypotheses:
        st.markdown("**Likely Causes**")
        for hyp in diagnosis.hypotheses:
            st.markdown(f"- {hyp}")

    # Task breakdown
    if diagnosis.task_breakdown is not None and len(diagnosis.task_breakdown) > 0:
        with st.expander("Task-Level Breakdown", expanded=False):
            st.dataframe(
                diagnosis.task_breakdown,
                column_config={
                    "task_name": "Task",
                    "quoted_hours": st.column_config.NumberColumn(
                        "Quoted Hrs", format="%.0f"
                    ),
                    "actual_hours": st.column_config.NumberColumn(
                        "Actual Hrs", format="%.0f"
                    ),
                    "hours_variance": st.column_config.NumberColumn(
                        "Hrs Var", format="%+.0f"
                    ),
                    "quoted_amount": st.column_config.NumberColumn(
                        "Quoted $", format="$%.0f"
                    ),
                    "actual_revenue": st.column_config.NumberColumn(
                        "Revenue", format="$%.0f"
                    ),
                    "revenue_variance": st.column_config.NumberColumn(
                        "Rev Var", format="$%+.0f"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )


# =============================================================================
# SECTION 6: DATA QUALITY
# =============================================================================


def render_data_quality_section(recon_df: pd.DataFrame):
    """Render data quality checks."""

    st.subheader("6️⃣ Data Quality Check")
    st.caption("*Are there data issues affecting this analysis?*")

    quality = check_data_quality(recon_df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Quote-Revenue Match Rate",
            f"{quality['match_rate']:.0f}%",
            help="Jobs that have both a quote and revenue",
        )

    with col2:
        st.metric(
            "Revenue w/o Quote",
            f"{quality['jobs_with_revenue_no_quote']} jobs",
            f"${quality['revenue_without_quote_total']:,.0f}",
            help="Jobs with revenue but no quote in system",
        )

    with col3:
        st.metric(
            "Quote w/o Revenue",
            f"{quality['jobs_with_quote_no_revenue']} jobs",
            f"${quality['quote_without_revenue_total']:,.0f}",
            help="Jobs with quote but no revenue recorded",
        )

    # Flags
    if quality["match_rate"] < 70:
        st.error(
            "🔴 **Low match rate** — many jobs missing quotes or revenue. This analysis may be incomplete."
        )

    if quality["revenue_without_quote_total"] > quality["quote_without_revenue_total"] * 0.5:
        st.warning(
            "⚠️ **Significant revenue without quotes** — check if T&M work or quotes not entered"
        )

    if quality["jobs_with_quote_no_revenue"] > 10:
        st.warning(
            f"⚠️ **{quality['jobs_with_quote_no_revenue']} jobs** have quotes but no revenue — may be WIP, cancelled, or unbilled"
        )
