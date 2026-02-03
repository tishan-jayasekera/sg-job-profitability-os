"""
UI Components for the Operational Intervention Engine.

These components render the 6-section cockpit for delivery leaders:
1. Quadrant Health Summary
2. Intervention Queue
3. Selected Job Brief
4. Driver Analysis
5. Peer Context
6. Quadrant Trend
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from src.ui.formatting import (
    fmt_currency,
    fmt_hours,
    fmt_percent,
    fmt_rate,
    build_job_name_lookup,
    format_job_label,
)
from src.modeling.intervention import (
    compute_quadrant_health_summary,
    build_intervention_queue,
    get_peer_context,
)


def render_quadrant_health_summary(
    quadrant_jobs: pd.DataFrame,
    quadrant_name: str,
) -> None:
    """
    Section 1: Render quadrant health KPI cards.
    
    Args:
        quadrant_jobs: DataFrame of all jobs in the quadrant
        quadrant_name: Display name of quadrant
    """
    st.subheader("📊 Quadrant Health Summary")
    
    if len(quadrant_jobs) == 0:
        st.info(f"No jobs in {quadrant_name}.")
        return
    
    health = compute_quadrant_health_summary(quadrant_jobs)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Active Jobs",
            f"{health['job_count']:,}",
            help="Number of active jobs in this quadrant"
        )
        st.metric(
            "Quoted Revenue",
            fmt_currency(health['quoted_revenue_exposure']),
            help="Total quoted revenue exposure"
        )
    
    with col2:
        median_margin = health['median_margin_pct']
        quoted_margin = health['median_margin_pct_quote']
        delta = median_margin - quoted_margin if pd.notna(median_margin) and pd.notna(quoted_margin) else None
        
        st.metric(
            "Median Margin % (Actual vs Quote)",
            f"{median_margin:.1f}%" if pd.notna(median_margin) else "—",
            delta=f"{delta:.1f}pp" if delta is not None else None,
            help="Median margin percentage to date vs original quote"
        )
        
        st.metric(
            "Median Realized Rate (vs Quote)",
            fmt_rate(health['median_realized_rate']),
            help="Median realized rate vs quoted rate"
        )
    
    with col3:
        pct_at_risk = health['pct_breaching_guardrails']
        st.metric(
            "% Jobs Breaching Guardrails",
            f"{pct_at_risk:.0f}%" if pd.notna(pct_at_risk) else "—",
            help="Percentage of jobs with risk score > 50"
        )
        
        st.metric(
            "Avg Risk Score",
            f"{health['avg_risk_score']:.1f}" if pd.notna(health['avg_risk_score']) else "—",
            help="Average risk score across active jobs (0-100)"
        )
    
    st.markdown("---")


def render_intervention_queue(
    quadrant_jobs: pd.DataFrame,
    max_rows: int = 10,
    on_job_select: Optional[callable] = None,
) -> Optional[str]:
    """
    Section 2: Render intervention queue (ranked worklist).
    
    Args:
        quadrant_jobs: DataFrame of active jobs in quadrant
        max_rows: Number of rows to display by default
        on_job_select: Callback when user selects a job
        
    Returns:
        Selected job_no (or None if no selection)
    """
    st.subheader("🎯 Active Jobs Requiring Intervention")
    
    if len(quadrant_jobs) == 0:
        st.info("No active jobs to review.")
        return None
    
    # Build queue
    queue = build_intervention_queue(quadrant_jobs, active_only=True, top_n=max_rows * 2)
    
    if len(queue) == 0:
        st.info("No high-risk jobs identified.")
        return None
    
    # Show slider for filtering top N
    col_slider, col_filter = st.columns([2, 3])
    
    with col_slider:
        show_n = st.slider(
            "Show top N jobs",
            min_value=3,
            max_value=min(20, len(queue)),
            value=max_rows,
            step=1,
            key="intervention_slider"
        )
        queue_display = queue.head(show_n)
    
    with col_filter:
        all_issues = set()
        queue["primary_issue"].apply(lambda x: all_issues.update(x.split("; ")))
        issue_options = ["All Issues"] + sorted(list(all_issues))
        
        selected_issue = st.selectbox(
            "Filter by Primary Issue",
            options=issue_options,
            key="intervention_filter"
        )
        
        if selected_issue != "All Issues":
            queue_display = queue_display[
                queue_display["primary_issue"].str.contains(selected_issue, na=False)
            ]
    
    st.caption(f"Showing {len(queue_display)} of {len(queue)} high-risk jobs (ranked by risk score)")
    
    # Prepare display columns (max 8)
    display_cols = [
        "job_no",
        "risk_score",
        "primary_issue",
        "margin_delta",
        "hours_delta",
        "rate_delta",
        "job_age_days",
        "owner",
    ]
    
    # Filter to columns that exist
    display_cols = [c for c in display_cols if c in queue_display.columns]
    
    # Rename for readability
    col_rename = {
        "job_no": "Job ID",
        "risk_score": "Risk Score",
        "primary_issue": "Primary Issue",
        "margin_delta": "Margin Δ (%pp)",
        "hours_delta": "Hours Δ (%)",
        "rate_delta": "Rate Δ ($/hr)",
        "job_age_days": "Age (days)",
        "owner": "Owner",
    }
    
    display_df = queue_display[display_cols].rename(columns=col_rename)
    
    # Format numeric columns
    if "Risk Score" in display_df.columns:
        display_df["Risk Score"] = display_df["Risk Score"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "—"
        )
    if "Margin Δ (%pp)" in display_df.columns:
        display_df["Margin Δ (%pp)"] = display_df["Margin Δ (%pp)"].apply(
            lambda x: f"{x:+.1f}" if pd.notna(x) else "—"
        )
    if "Rate Δ ($/hr)" in display_df.columns:
        display_df["Rate Δ ($/hr)"] = display_df["Rate Δ ($/hr)"].apply(
            lambda x: f"{x:+.2f}" if pd.notna(x) else "—"
        )
    if "Age (days)" in display_df.columns:
        display_df["Age (days)"] = display_df["Age (days)"].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "—"
        )
    
    # Display table
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Optional: row selection (if callback provided)
    job_name_lookup = build_job_name_lookup(quadrant_jobs)
    selected_job = st.selectbox(
        "Select a job for detailed review",
        options=["—"] + queue_display["job_no"].tolist(),
        key="intervention_job_select",
        label_visibility="collapsed",
        format_func=lambda j: format_job_label(j, job_name_lookup),
    )
    
    if selected_job != "—":
        if on_job_select:
            on_job_select(selected_job)
        return selected_job
    
    return None


def render_selected_job_brief(
    job_row: pd.DataFrame,
    peer_segment: Optional[pd.DataFrame] = None,
) -> None:
    """
    Section 3: Render job brief for selected job.
    
    Args:
        job_row: Single job row
        peer_segment: Optional peer segment for context
    """
    st.subheader("📋 Selected Job Brief")
    
    job_id = job_row.get("job_no", "Unknown")
    client = job_row.get("client", "—")
    
    st.write(f"**Job #{job_id}** | Client: *{client}*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Quoted vs Actual**")
        st.markdown(f"- Margin: {job_row.get('margin_pct_quote', np.nan):.1f}% → {job_row.get('margin_pct_to_date', np.nan):.1f}%")
        st.markdown(f"- Hours: {job_row.get('quoted_hours', np.nan):.0f} → {job_row.get('actual_hours', np.nan):.0f}")
        st.markdown(f"- Rate: {fmt_rate(job_row.get('quote_rate', np.nan))} → {fmt_rate(job_row.get('realised_rate', np.nan))}")
    
    with col2:
        st.markdown("**Timeline**")
        runtime = job_row.get('runtime_days', np.nan)
        peer_median = job_row.get('peer_median_runtime_days', np.nan)
        
        if pd.notna(runtime):
            st.markdown(f"- Runtime: {int(runtime)} days")
        if pd.notna(peer_median):
            st.markdown(f"- Peer Median: {int(peer_median)} days")
            if pd.notna(runtime) and runtime > peer_median * 1.5:
                st.markdown(f"- ⚠️ **50% longer than peers**")
    
    with col3:
        st.markdown("**Risk Assessment**")
        risk_score = job_row.get('risk_score', np.nan)
        primary_issue = job_row.get('primary_issue', "—")
        
        if pd.notna(risk_score):
            if risk_score > 70:
                color = "🔴"
            elif risk_score > 50:
                color = "🟡"
            else:
                color = "🟢"
            st.markdown(f"- Risk Score: {color} {risk_score:.0f}/100")
        
        st.markdown(f"- Top Issue: {primary_issue}")
    
    st.markdown("---")


def render_driver_analysis(
    job_row: pd.DataFrame,
    task_data: Optional[pd.DataFrame] = None,
    staffing_data: Optional[pd.DataFrame] = None,
) -> None:
    """
    Section 4: Render driver analysis tabs (Tasks + Staffing).
    
    Args:
        job_row: Single job row
        task_data: Task-level breakdowns for this job
        staffing_data: Staffing mix data for this job
    """
    st.subheader("🔍 Driver Analysis")
    
    tab1, tab2 = st.tabs(["Task Drivers", "Staffing Mix Drivers"])
    
    with tab1:
        if task_data is not None and len(task_data) > 0:
            st.markdown("**Task Share: Job vs Peer Median**")
            
            # Show task overruns (positive deltas)
            task_data_copy = task_data.copy()
            task_data_copy['delta'] = (
                task_data_copy.get('job_task_pct', 0) - 
                task_data_copy.get('peer_median_task_pct', 0)
            )
            
            # Filter to positive deltas (contributors to overrun)
            task_data_copy = task_data_copy[task_data_copy['delta'] > 0].sort_values('delta', ascending=False)
            
            if len(task_data_copy) > 0:
                st.dataframe(
                    task_data_copy[['task_name', 'job_task_pct', 'peer_median_task_pct', 'delta']].rename(
                        columns={
                            'task_name': 'Task',
                            'job_task_pct': 'Job %',
                            'peer_median_task_pct': 'Peer Median %',
                            'delta': 'Overrun %',
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No task overruns identified vs peers.")
        else:
            st.info("Task data not available.")
    
    with tab2:
        if staffing_data is not None and len(staffing_data) > 0:
            st.markdown("**Staffing Mix: Seniority / Role / Geography**")
            
            # Show mix comparison
            staffing_display = staffing_data.copy()
            
            if 'mix_delta' in staffing_display.columns:
                staffing_display = staffing_display.sort_values('mix_delta', ascending=False)
            
            st.dataframe(
                staffing_display,
                use_container_width=True,
                hide_index=True,
            )
            
            st.markdown("**Recommendation:** " + staffing_data.get('recommendation', 'Review staffing mix for rate uplift opportunity'))
        else:
            st.info("Staffing data not available.")
    
    st.markdown("---")


def render_peer_context(
    job_row: pd.DataFrame,
    peer_segment: Optional[pd.DataFrame] = None,
) -> None:
    """
    Section 5: Render peer context (sanity check).
    
    Args:
        job_row: Single job row
        peer_segment: All peers in the segment
    """
    st.subheader("📍 Peer Context")
    
    if peer_segment is not None and len(peer_segment) > 0:
        context = get_peer_context(job_row, peer_segment)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'runtime_percentile' in context:
                pct = context['runtime_percentile']
                st.metric(
                    "Runtime Percentile",
                    f"{pct:.0f}th",
                    help="Job runtime vs peers (higher = longer)"
                )
        
        with col2:
            if 'margin_percentile' in context:
                pct = context['margin_percentile']
                st.metric(
                    "Margin Percentile",
                    f"{pct:.0f}th",
                    help="Job margin vs peers (lower = worse)"
                )
        
        with col3:
            if 'rate_percentile' in context:
                pct = context['rate_percentile']
                st.metric(
                    "Realized Rate Percentile",
                    f"{pct:.0f}th",
                    help="Job realized rate vs peers (lower = worse)"
                )
        
        # Interpretation
        st.markdown("---")
        st.markdown("**Interpretation**")
        
        is_unique_issue = False
        
        if 'runtime_percentile' in context and context['runtime_percentile'] > 75:
            st.markdown("- 🔴 **Runtime is significantly higher than peers** → May indicate scope or staffing issues")
            is_unique_issue = True
        
        if 'margin_percentile' in context and context['margin_percentile'] < 25:
            st.markdown("- 🔴 **Margin is structurally lower than peers** → Check pricing or resource costs")
            is_unique_issue = True
        
        if 'rate_percentile' in context and context['rate_percentile'] < 25:
            st.markdown("- 🔴 **Realized rate is lower than peers** → Staffing mix or scope creep issue")
            is_unique_issue = True
        
        if not is_unique_issue:
            st.markdown("- 🟢 Job metrics are within peer range. Issue may be category-wide.")
    else:
        st.info("No peer segment available for context.")
    
    st.markdown("---")


def render_quadrant_trend(
    quadrant_data: pd.DataFrame,
    quadrant_name: str,
    metric_columns: Optional[List[str]] = None,
) -> None:
    """
    Section 6: Render quadrant trend (time-series).
    
    Args:
        quadrant_data: Time-series data with columns like:
                      - period: time period (e.g., month)
                      - median_margin_pct
                      - median_realized_rate
                      - pct_jobs_breaching_guardrails
        quadrant_name: Display name
        metric_columns: Which metrics to plot (default: margin, rate, guardrails)
    """
    st.subheader("📈 Quadrant Trend (Last 6 months)")
    
    if quadrant_data is None or len(quadrant_data) == 0:
        st.info("Insufficient trend data.")
        return
    
    if metric_columns is None:
        metric_columns = [
            "median_margin_pct",
            "median_realized_rate",
            "pct_jobs_breaching_guardrails",
        ]
    
    # Plot time-series for each metric
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    if "median_margin_pct" in quadrant_data.columns and "median_margin_pct" in metric_columns:
        fig.add_trace(go.Scatter(
            x=quadrant_data.get("period", range(len(quadrant_data))),
            y=quadrant_data["median_margin_pct"],
            mode="lines+markers",
            name="Median Margin %",
            yaxis="y1",
        ))
    
    if "median_realized_rate" in quadrant_data.columns and "median_realized_rate" in metric_columns:
        fig.add_trace(go.Scatter(
            x=quadrant_data.get("period", range(len(quadrant_data))),
            y=quadrant_data["median_realized_rate"],
            mode="lines+markers",
            name="Median Realized Rate",
            yaxis="y2",
        ))
    
    if "pct_jobs_breaching_guardrails" in quadrant_data.columns and "pct_jobs_breaching_guardrails" in metric_columns:
        fig.add_trace(go.Scatter(
            x=quadrant_data.get("period", range(len(quadrant_data))),
            y=quadrant_data["pct_jobs_breaching_guardrails"],
            mode="lines+markers",
            name="% Jobs Breaching Guardrails",
            yaxis="y3",
        ))
    
    fig.update_layout(
        title=f"{quadrant_name} - Trend",
        hovermode="x unified",
        yaxis1=dict(title="Margin %", position=0.0),
        yaxis2=dict(title="Rate ($/hr)", overlaying="y", side="right"),
        yaxis3=dict(title="% At Risk", overlaying="y", side="far right"),
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_methodology_expander() -> None:
    """
    Render methodology and definitions expander.
    """
    with st.expander("📖 Methodology & Definitions", expanded=False):
        st.markdown("""
        ### Risk Score Model
        
        The risk score combines 5 factors on a 0-100 scale:
        
        1. **Margin Erosion (0-30 points)**
           - Alert if margin % to date < 15%
           - Weighted by how far below threshold
        
        2. **Revenue Lag (0-25 points)**
           - Alert if revenue earned / quoted < 70%
           - Indicates slow billing or scope loss
        
        3. **Scope Creep (0-25 points)**
           - Alert if hours overrun > 10%
           - Reflects ability to execute to budget
        
        4. **Rate Leakage (0-20 points)**
           - Alert if realized rate < 85% of quoted rate
           - Indicates staffing mix or scope issues
        
        5. **Runtime Risk (0-20 points)**
           - Alert if runtime > 1.5x peer median
           - "Zombie job" indicator
        
        ### Reason Codes
        
        Top 2-3 reason codes are shown per job. These are:
        - Low margin %
        - Revenue lagging quote
        - Hours overrun vs quote
        - Realized rate below quote
        - Runtime exceeds peers
        
        ### Quartile Thresholds
        
        - **Good (🟢)**: Risk Score < 30
        - **Watch (🟡)**: Risk Score 30-60
        - **At Risk (🔴)**: Risk Score > 60
        """)
