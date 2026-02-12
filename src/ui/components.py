"""
Reusable UI components and blocks.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_rate


def kpi_strip(metrics: Dict[str, Any], 
              format_map: Optional[Dict[str, str]] = None):
    """
    Render horizontal strip of KPI cards.
    
    Args:
        metrics: Dict of {label: value}
        format_map: Dict of {label: format_type} where format_type is
                    'currency', 'hours', 'percent', 'rate', 'count'
    """
    if format_map is None:
        format_map = {}
    
    cols = st.columns(len(metrics))
    
    formatters = {
        "currency": fmt_currency,
        "hours": fmt_hours,
        "percent": fmt_percent,
        "rate": fmt_rate,
        "count": lambda x: "—" if (pd.isna(x) or np.isinf(x)) else f"{int(x):,}",
        "text": lambda x: str(x) if pd.notna(x) else "—",
    }
    
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            fmt_type = format_map.get(label, "currency")
            formatter = formatters.get(fmt_type, str)
            
            formatted = formatter(value) if pd.notna(value) else "—"
            st.metric(label=label, value=formatted)


def risk_badge(status: str) -> str:
    """
    Return colored risk badge HTML.
    """
    colors = {
        "on_track": ("🟢", "#28a745"),
        "watch": ("🟡", "#ffc107"),
        "at_risk": ("🔴", "#dc3545"),
    }
    
    emoji, color = colors.get(status.lower(), ("⚪", "#6c757d"))
    return emoji


def status_indicator(value: float, 
                     good_threshold: float, 
                     bad_threshold: float,
                     invert: bool = False) -> str:
    """
    Return status indicator based on thresholds.
    
    Args:
        value: The value to check
        good_threshold: Above this is good (or bad if inverted)
        bad_threshold: Below this is bad (or good if inverted)
        invert: If True, lower is better
    """
    if pd.isna(value):
        return "⚪"
    
    if invert:
        if value <= good_threshold:
            return "🟢"
        elif value >= bad_threshold:
            return "🔴"
        return "🟡"
    else:
        if value >= good_threshold:
            return "🟢"
        elif value <= bad_threshold:
            return "🔴"
        return "🟡"


def metric_card_with_trend(label: str, 
                           value: str,
                           trend_value: Optional[float] = None,
                           trend_is_good: bool = True):
    """
    Render metric card with trend indicator.
    """
    delta = None
    delta_color = "normal"
    
    if trend_value is not None:
        if trend_value > 0:
            delta = f"+{trend_value:.1f}%"
            delta_color = "normal" if trend_is_good else "inverse"
        else:
            delta = f"{trend_value:.1f}%"
            delta_color = "inverse" if trend_is_good else "normal"
    
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def info_panel(title: str, items: List[Dict[str, str]]):
    """
    Render information panel with key-value pairs.
    """
    st.markdown(f"**{title}**")
    
    for item in items:
        label = item.get("label", "")
        value = item.get("value", "")
        st.markdown(f"- {label}: {value}")


def action_card(title: str,
                description: str,
                action_label: str,
                on_click: callable,
                key: str):
    """
    Render action card with button.
    """
    with st.container():
        st.markdown(f"**{title}**")
        st.caption(description)
        st.button(action_label, on_click=on_click, key=key)


def expandable_section(title: str, 
                       content_func: callable,
                       expanded: bool = False):
    """
    Render expandable section.
    """
    with st.expander(title, expanded=expanded):
        content_func()


def tab_container(tab_labels: List[str]) -> List:
    """
    Create tabs and return tab objects for content.
    """
    return st.tabs(tab_labels)


def progress_bar(label: str,
                 value: float,
                 max_value: float = 100,
                 color: str = "blue"):
    """
    Render labeled progress bar.
    """
    st.markdown(f"**{label}**")
    progress = min(value / max_value, 1.0) if max_value > 0 else 0
    st.progress(progress)
    st.caption(f"{value:.1f} / {max_value:.1f}")


def benchmark_badge(n_jobs: int, 
                    date_min: Optional[str] = None,
                    date_max: Optional[str] = None,
                    recency_weighted: bool = False):
    """
    Display benchmark sample metadata.
    """
    parts = [f"{n_jobs} jobs"]
    
    if date_min and date_max:
        parts.append(f"{date_min} - {date_max}")
    
    if recency_weighted:
        parts.append("recency-weighted")
    
    st.caption(" | ".join(parts))


def empty_state(message: str, 
                icon: str = "📭",
                action_label: Optional[str] = None,
                on_action: Optional[callable] = None,
                key: str = "empty_state"):
    """
    Render empty state with optional action.
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"### {icon}")
        st.markdown(f"**{message}**")
        
        if action_label and on_action:
            st.button(action_label, on_click=on_action, key=key)


def filter_chips(filters: Dict[str, Any]):
    """
    Display active filters as chips.
    """
    active = []
    
    for key, value in filters.items():
        if value is not None and value != "" and value != "All":
            if isinstance(value, bool):
                if value:
                    active.append(key.replace("_", " ").title())
            else:
                active.append(f"{key}: {value}")
    
    if active:
        chips = " | ".join([f"`{f}`" for f in active])
        st.caption(f"Active filters: {chips}")


# =============================================================================
# CLIENT PROFITABILITY COMPONENTS
# =============================================================================

def render_client_portfolio_health(summary: Dict[str, float]):
    st.subheader("Executive Portfolio Health")
    metrics = {
        "Total Clients": summary.get("total_clients"),
        "Portfolio Revenue": summary.get("portfolio_revenue"),
        "Portfolio Profit": summary.get("portfolio_profit"),
        "Median Margin %": summary.get("median_margin_pct"),
        "% Unprofitable Clients": summary.get("unprofitable_share"),
    }
    format_map = {
        "Total Clients": "count",
        "Portfolio Revenue": "currency",
        "Portfolio Profit": "currency",
        "Median Margin %": "percent",
        "% Unprofitable Clients": "percent",
    }
    kpi_strip(metrics, format_map=format_map)
    if summary.get("top5_profit_share") is not None:
        st.caption(
            f"Concentration risk: Top 5 clients contribute {fmt_percent(summary.get('top5_profit_share'))} of total profit."
        )


def render_client_quadrant_scatter(fig):
    st.subheader("Portfolio Quadrant Scatter")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_client_intervention_queue(df: pd.DataFrame, shortlist_size: int, client_col: str = "client"):
    st.subheader("Intervention Queue")
    if len(df) == 0:
        st.info("No clients found for the selected quadrant.")
        return
    queue = df.head(shortlist_size).rename(columns={
        client_col: "Client",
        "margin": "Profit",
        "margin_pct": "Margin %",
        "revenue": "Revenue",
        "realised_rate": "Avg Realised Rate",
        "primary_driver": "Primary Driver",
    })
    st.dataframe(
        queue[["Client", "Profit", "Margin %", "Revenue", "Avg Realised Rate", "Primary Driver"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Profit": st.column_config.NumberColumn(format="$%.0f"),
            "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
            "Revenue": st.column_config.NumberColumn(format="$%.0f"),
            "Avg Realised Rate": st.column_config.NumberColumn(format="$%.0f"),
        },
    )


def render_client_deep_dive(client_name: str, metrics: Dict[str, float], grade: str,
                            ledger: pd.DataFrame, dept_fig):
    st.subheader("Selected Client Deep-Dive")
    st.markdown(f"**Client:** {client_name}")
    metrics = metrics.copy()
    metrics["Health Grade"] = grade
    format_map = {
        "Revenue": "currency",
        "Profit": "currency",
        "Margin %": "percent",
        "Realised Rate": "rate",
        "Jobs": "count",
        "Quoted Revenue": "currency",
        "Quoted Cost": "currency",
        "Quoted Margin %": "percent",
        "Quoted Hours": "hours",
        "Health Grade": "text",
    }
    kpi_strip(metrics, format_map=format_map)
    st.markdown("**Job Ledger**")
    if len(ledger) == 0:
        st.info("No jobs found for this client in the selected window.")
    else:
        st.dataframe(
            ledger.rename(columns={
                "job_no": "Job",
                "department_final": "Department",
                "job_category": "Category",
                "hours": "Hours",
                "revenue": "Revenue",
                "cost": "Cost",
                "margin": "Margin",
                "margin_pct": "Margin %",
                "quoted_amount": "Quoted Revenue",
                "quoted_cost": "Quoted Cost",
                "quoted_margin_pct": "Quoted Margin %",
                "realised_rate": "Realised Rate",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Hours": st.column_config.NumberColumn(format="%.1f"),
                "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                "Cost": st.column_config.NumberColumn(format="$%.0f"),
                "Margin": st.column_config.NumberColumn(format="$%.0f"),
                "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                "Quoted Revenue": st.column_config.NumberColumn(format="$%.0f"),
                "Quoted Cost": st.column_config.NumberColumn(format="$%.0f"),
                "Quoted Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                "Realised Rate": st.column_config.NumberColumn(format="$%.0f"),
            },
        )
    st.markdown("**Department Contribution (Profit)**")
    st.plotly_chart(dept_fig, use_container_width=True)


def render_client_driver_forensics(task_table: pd.DataFrame, staffing_table: pd.DataFrame,
                                   senior_flag: bool, task_time_fig=None,
                                   staff_cost_time_fig=None,
                                   task_benchmark_fig=None, delivery_burn_fig=None,
                                   erosion_table: Optional[pd.DataFrame] = None):
    st.subheader("Driver Forensics")
    summary_lines = []
    if task_table is not None and len(task_table) > 0:
        top_task = task_table.iloc[0]
        summary_lines.append(
            f"Task mix skew: {top_task.get('task_name', 'Top task')} is +{top_task.get('delta_pp', 0):.1f}pp vs global median."
        )
    if senior_flag:
        summary_lines.append("Staffing mix: blended cost rate is >20% above company average.")
    if erosion_table is not None and len(erosion_table) > 0:
        worst_margin = pd.to_numeric(erosion_table.get("Margin %", pd.Series(dtype=float)), errors="coerce").min()
        worst_overrun = pd.to_numeric(erosion_table.get("Hours Overrun %", pd.Series(dtype=float)), errors="coerce").max()
        erosion_note = f"Erosion watchlist: {len(erosion_table)} jobs below 10% margin or >10% hours overrun."
        if pd.notna(worst_margin):
            erosion_note += f" Worst margin: {worst_margin:.1f}%."
        if pd.notna(worst_overrun):
            erosion_note += f" Worst overrun: {worst_overrun:.1f}%."
        summary_lines.append(erosion_note)

    if summary_lines:
        st.markdown("**Signal Summary**")
        st.markdown("\n".join([f"- {line}" for line in summary_lines]))
    else:
        st.info("No material driver anomalies detected for this slice.")

    tabs = st.tabs(["Delivery Drivers", "Staffing Mix"])
    with tabs[0]:
        st.markdown("Task mix deltas, burn-rate vs quote, and job-level erosion flags for this client slice.")
        if task_benchmark_fig is not None:
            st.plotly_chart(task_benchmark_fig, use_container_width=True)
        if task_time_fig is not None:
            st.plotly_chart(task_time_fig, use_container_width=True)
        if delivery_burn_fig is not None:
            st.plotly_chart(delivery_burn_fig, use_container_width=True)
        if erosion_table is not None and len(erosion_table) > 0:
            st.markdown("**Constituent Jobs Driving Erosion**")
            st.dataframe(
                erosion_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Hours Overrun %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                    "Cost": st.column_config.NumberColumn(format="$%.0f"),
                    "Margin": st.column_config.NumberColumn(format="$%.0f"),
                },
            )
        if len(task_table) == 0:
            st.info("No task mix variance detected.")
        else:
            st.dataframe(
                task_table.rename(columns={
                    "task_name": "Task",
                    "client_share_pct": "Client Share %",
                    "global_share_pct": "Global Median %",
                    "delta_pp": "Delta (pp)",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Client Share %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Global Median %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Delta (pp)": st.column_config.NumberColumn(format="%.1f"),
                },
            )
    with tabs[1]:
        if senior_flag:
            st.warning("Senior-heavy delivery detected (blended cost rate >20% above company average).")
        if staff_cost_time_fig is not None:
            st.plotly_chart(staff_cost_time_fig, use_container_width=True)
        if len(staffing_table) == 0:
            st.info("No staffing mix data available for this slice.")
        else:
            st.markdown("Top contributors by hours and cost rate (use to validate staffing mix).")
            st.dataframe(
                staffing_table.rename(columns={
                    "staff_name": "Staff",
                    "role": "Role",
                    "hours": "Hours",
                    "cost": "Cost",
                    "cost_rate": "Cost Rate",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Cost": st.column_config.NumberColumn(format="$%.0f"),
                    "Cost Rate": st.column_config.NumberColumn(format="$%.0f"),
                },
            )


def render_client_ltv_section(cumulative_fig, margin_fig, tenure_months: int):
    st.subheader("Empirical LTV & Trends")
    st.metric("Client Lifetime (months)", tenure_months)
    st.plotly_chart(cumulative_fig, use_container_width=True)
    st.plotly_chart(margin_fig, use_container_width=True)


def render_client_ltv_methodology_expander():
    with st.expander("LTV Methodology", expanded=False):
        st.markdown(
            """
**LTV is empirical and historical.** It represents cumulative profit captured to date,
not a forward-looking or projected lifetime value. The trend view uses the full, unfiltered
history to preserve tenure and cumulative economics.
            """
        )



def download_button(df: pd.DataFrame,
                    filename: str,
                    label: str = "Download CSV",
                    key: str = "download"):
    """
    Render download button for dataframe.
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key
    )


def render_data_quality_panel(panel: Dict[str, Any], warnings: Optional[List[str]] = None):
    """
    Render data quality transparency panel.

    Args:
        panel: Dict of key->value strings to display.
        warnings: List of warning strings to show.
    """
    with st.expander("Data Quality & Transparency"):
        for key, value in panel.items():
            st.markdown(f"- **{key}**: {value}")
        if warnings:
            st.markdown("---")
            st.markdown("**Warnings**")
            for w in warnings:
                st.warning(w)
        st.page_link("pages/8_Glossary_Method.py", label="Glossary & Method", icon="📖")


# =============================================================================
# ENHANCED KPI STRIP WITH SPARKLINES (NEW FOR PHASE 1)
# =============================================================================

def render_kpi_strip_with_sparklines(
    metrics: Dict[str, Any],
    sparklines: Optional[Dict[str, List[float]]] = None,
    format_map: Optional[Dict[str, str]] = None,
    status_indicators: Optional[Dict[str, str]] = None,  # Dict of label -> "green"/"yellow"/"red"
):
    """
    Render enhanced KPI strip with status indicators and optional trend sparklines.
    
    Args:
        metrics: Dict of {label: value}
        sparklines: Dict of {label: [trend_values]} for mini-charts
        format_map: Dict of {label: format_type}
        status_indicators: Dict of {label: status_color}
    """
    if format_map is None:
        format_map = {}
    if status_indicators is None:
        status_indicators = {}
    
    cols = st.columns(len(metrics))
    
    formatters = {
        "currency": fmt_currency,
        "hours": fmt_hours,
        "percent": fmt_percent,
        "rate": fmt_rate,
        "count": lambda x: "—" if (pd.isna(x) or np.isinf(x)) else f"{int(x):,}",
        "weeks": lambda x: "—" if (pd.isna(x) or np.isinf(x)) else f"{x:.1f}w",
        "text": lambda x: str(x) if pd.notna(x) else "—",
    }
    
    status_emoji = {
        "green": "🟢",
        "yellow": "🟡",
        "red": "🔴",
    }
    
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            fmt_type = format_map.get(label, "currency")
            formatter = formatters.get(fmt_type, str)
            formatted = formatter(value) if pd.notna(value) else "—"
            
            # Add status emoji
            status = status_indicators.get(label, "")
            emoji = f" {status_emoji.get(status, '')}" if status in status_emoji else ""
            
            st.metric(label=label, value=f"{formatted}{emoji}")
            
            # Add sparkline if provided
            if sparklines and label in sparklines and len(sparklines[label]) > 0:
                try:
                    import plotly.graph_objects as go
                    sparkline_data = [v for v in sparklines[label] if pd.notna(v)]
                    if len(sparkline_data) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=sparkline_data,
                            mode="lines",
                            fill="tozeroy",
                            line=dict(color="#1f77b4", width=1),
                            hovertemplate="%{y:.1f}<extra></extra>",
                        ))
                        fig.update_layout(
                            showlegend=False,
                            hovermode="x unified",
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=60,
                            xaxis=dict(showticklabels=False),
                            yaxis=dict(showticklabels=False),
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                except Exception:
                    pass  # Fail silently if sparkline rendering fails


# =============================================================================
# SMART TABLE RENDERING WITH SORTING/FILTERING
# =============================================================================

def render_sortable_table(
    df: pd.DataFrame,
    sort_column: Optional[str] = None,
    sort_ascending: bool = False,
    filter_columns: Optional[Dict[str, List[str]]] = None,
    color_map: Optional[Dict[str, Dict[str, str]]] = None,
    export_label: str = "Export CSV",
):
    """
    Render a sortable, filterable table with optional row color-coding.
    
    Args:
        df: DataFrame to display
        sort_column: Column to sort by (default None = use input order)
        sort_ascending: Sort ascending if True, descending if False
        filter_columns: Dict of {column_name: [allowed_values]} for filtering
        color_map: Dict of {column_name: {value: color_hex}}
        export_label: Label for CSV export button
    """
    df = df.copy()
    
    # Apply filters
    if filter_columns:
        for col, allowed_values in filter_columns.items():
            if col in df.columns and allowed_values:
                df = df[df[col].isin(allowed_values)]
    
    # Apply sorting
    if sort_column and sort_column in df.columns:
        df = df.sort_values(sort_column, ascending=sort_ascending)
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
    
    # Export button
    csv_data = df.to_csv(index=False)
    st.download_button(
        label=export_label,
        data=csv_data,
        file_name="export.csv",
        mime="text/csv",
    )


# =============================================================================
# DATA QUALITY WARNINGS & TRANSPARENCY
# =============================================================================

def render_data_quality_panel_extended(
    benchmark_info: Optional[Dict[str, Any]] = None,
    data_freshness: Optional[Dict[str, str]] = None,
    warnings: Optional[List[str]] = None,
    completeness: Optional[Dict[str, float]] = None,
):
    """
    Render comprehensive data quality panel.
    
    Args:
        benchmark_info: Dict with benchmark reliability info
        data_freshness: Dict with last_refresh, next_refresh times
        warnings: List of warning messages
        completeness: Dict of data completeness percentages
    """
    with st.expander("📊 Data Quality & Transparency", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if benchmark_info:
                st.markdown("**Benchmark Reliability**")
                for key, value in benchmark_info.items():
                    st.markdown(f"- {key}: {value}")
            
            if data_freshness:
                st.markdown("**Data Freshness**")
                for key, value in data_freshness.items():
                    st.markdown(f"- {key}: {value}")
        
        with col2:
            if completeness:
                st.markdown("**Data Completeness**")
                for metric, pct in completeness.items():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.progress(pct / 100.0, text=f"{metric}: {pct:.0f}%")
                    with col_b:
                        st.markdown(f"*{pct:.0f}%*")
        
        if warnings:
            st.markdown("---")
            st.markdown("**Data Quality Warnings**")
            for warning in warnings:
                st.warning(warning, icon="⚠️")
        
        st.markdown("---")
        st.caption("[📖 View Glossary & Methods](pages/8_Glossary_Method.py)")


def render_status_badge_row(
    df: pd.DataFrame,
    status_column: str,
    status_colors: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Add status badges to DataFrame for display (using st.write with HTML).
    
    Note: Returns HTML-enhanced column for better visualization.
    
    Args:
        df: DataFrame
        status_column: Column containing status values
        status_colors: Dict of {status: hex_color}
        
    Returns:
        Modified DataFrame with status badges
    """
    if status_colors is None:
        status_colors = {
            "On-Track": "#28a745",
            "At-Risk": "#ffc107",
            "Blocked": "#dc3545",
            "Critical": "#d32f2f",
            "Unknown": "#6c757d",
        }
    
    df = df.copy()
    
    def badge_html(status):
        color = status_colors.get(status, "#6c757d")
        return f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">{status}</span>'
    
    df[status_column] = df[status_column].apply(badge_html)
    return df

# ============================================================================
# PHASE 1B: DRILL-CHAIN UI COMPONENTS
# ============================================================================

def render_breadcrumb_header(
    horizon_weeks: int,
    scope_levels: List[str],
) -> None:
    """
    Render top navigation breadcrumb + horizon selector.
    
    Args:
        horizon_weeks: Current forecast horizon (4, 8, 12, 16)
        scope_levels: List of active scope levels, e.g. ["Company", "Sales", "Fixed Price"]
    """
    col_horizon, col_scope = st.columns([2, 4])
    
    with col_horizon:
        new_horizon = st.radio(
            label="Forecast Horizon",
            options=[4, 8, 12, 16],
            index=[4, 8, 12, 16].index(horizon_weeks),
            horizontal=True,
            key="horizon_selector",
        )
        if new_horizon != horizon_weeks:
            st.session_state['forecast_horizon_weeks'] = new_horizon
            st.rerun()
    
    with col_scope:
        breadcrumb_text = " ▸ ".join(["Company"] + scope_levels)
        st.markdown(f"**Scope:** {breadcrumb_text}")


def render_job_health_card(
    job_row: pd.Series,
) -> None:
    """
    Render job health summary card with human-readable states (no math artifacts).
    
    Args:
        job_row: Single row from job_level dataframe with columns:
                 job_no, status, due_weeks, job_eta_weeks, risk_score, 
                 remaining_hours, expected_hours, actual_hours, job_velocity_hrs_week
    """
    from src.modeling.forecast import translate_job_state
    
    job_id = job_row.get('job_no', 'N/A')
    risk_score = job_row.get('risk_score', np.nan)
    due_weeks = job_row.get('due_weeks', np.nan)
    eta_weeks = job_row.get('job_eta_weeks', np.nan)
    velocity = job_row.get('job_velocity_hrs_week', 0)
    
    status, status_label = translate_job_state(risk_score, due_weeks, eta_weeks, velocity)
    
    remaining = job_row.get('remaining_hours', 0)
    expected = job_row.get('expected_hours', 0)
    actual = job_row.get('actual_hours', 0)
    
    st.subheader(f"Job #{job_id} - {status_label}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if pd.isna(eta_weeks) or np.isinf(eta_weeks):
            eta_display = "No run-rate"
        else:
            eta_display = f"{eta_weeks:.1f}w"
        st.metric("ETA", eta_display)
    
    with col2:
        if pd.isna(due_weeks):
            due_display = "—"
        elif due_weeks < 0:
            due_display = f"🔴 {-due_weeks:.1f}w ago"
        else:
            due_display = f"{due_weeks:.1f}w"
        st.metric("Due", due_display)
    
    with col3:
        if pd.isna(due_weeks) or pd.isna(eta_weeks):
            buffer_display = "—"
        elif np.isinf(eta_weeks):
            buffer_display = "Blocked"
        else:
            buffer = due_weeks - eta_weeks
            buffer_display = f"{buffer:.1f}w"
        st.metric("Buffer", buffer_display)
    
    with col4:
        st.metric("Risk", f"{risk_score:.2f}" if pd.notna(risk_score) else "—")
    
    with col5:
        pct_complete = (actual / expected * 100) if expected > 0 else 0
        st.metric("% Complete", f"{pct_complete:.0f}%")
    
    # Scope breakdown
    st.markdown("---")
    scope_cols = st.columns(3)
    with scope_cols[0]:
        st.write(f"**Benchmark:** {fmt_hours(expected)}")
    with scope_cols[1]:
        st.write(f"**Spent:** {fmt_hours(actual)}")
    with scope_cols[2]:
        st.write(f"**Remaining:** {fmt_hours(remaining)}")


def render_task_status_badge(
    velocity: float,
    remaining_hours: float,
    expected_hours: float,
) -> str:
    """
    Translate task metrics into readable status badge.
    
    Args:
        velocity: Hours per week
        remaining_hours: Hours left to complete
        expected_hours: Benchmark hours for this task
        
    Returns:
        Status string: "Blocked" | "At-Risk" | "On-Track" | "Negligible"
    """
    if remaining_hours < 5:
        return "Negligible"
    
    if velocity == 0:
        return "Blocked"
    
    if remaining_hours > expected_hours:
        return "At-Risk"
    
    return "On-Track"


def render_scope_filtered_table(
    df: pd.DataFrame,
    title: str,
    sortable_columns: Optional[List[str]] = None,
    status_column: Optional[str] = None,
    clickable_row_index: Optional[int] = None,
) -> Optional[int]:
    """
    Render interactive table with sort, filter, and selection.
    
    Args:
        df: DataFrame to display
        title: Table title
        sortable_columns: Column names that should be clickable for sorting
        status_column: Column to apply status-based coloring
        clickable_row_index: If provided, return index of clicked row
        
    Returns:
        Clicked row index (or None if no click)
    """
    st.subheader(title)
    
    if len(df) == 0:
        st.info("No data available for this scope.")
        return None
    
    # Simple display with status coloring
    display_df = df.copy()
    
    if status_column and status_column in display_df.columns:
        display_df = display_df.style.applymap(
            lambda x: 'background-color: #90EE90' if x == 'On-Track' else
                      'background-color: #FFD700' if x == 'At-Risk' else
                      'background-color: #FF6B6B' if x == 'Blocked' else '',
            subset=[status_column]
        )
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    return None
