"""
Layout components: header, breadcrumb, filters.
"""
import streamlit as st
from typing import List, Optional, Callable
import pandas as pd

from src.ui.state import (
    get_state, set_state, get_breadcrumb, get_drill_level,
    drill_to_department, drill_to_category, drill_up, reset_drill
)
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_rate


# =============================================================================
# HEADER AND NAVIGATION
# =============================================================================

def render_header():
    """Render app header with title and time window."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Job Profitability OS")
    
    with col2:
        time_window = get_state("time_window")
        st.caption(f"Time window: {time_window}")


def render_breadcrumb():
    """Render clickable breadcrumb navigation."""
    crumbs = get_breadcrumb()
    level = get_drill_level()
    
    # Build breadcrumb HTML with click handlers
    breadcrumb_parts = []
    
    for i, crumb in enumerate(crumbs):
        if i == len(crumbs) - 1:
            # Current level - not clickable
            breadcrumb_parts.append(f"**{crumb}**")
        else:
            # Previous levels - clickable
            breadcrumb_parts.append(crumb)
    
    breadcrumb_str = " > ".join(breadcrumb_parts)
    
    # Display with buttons for navigation
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        st.markdown(f"**Navigation:** {breadcrumb_str}")
    
    with col2:
        if level != "company":
            if st.button("⬆ Up", key="breadcrumb_up"):
                drill_up()
                st.rerun()
    
    with col3:
        if level != "company":
            if st.button("🏠 Reset", key="breadcrumb_reset"):
                reset_drill()
                st.rerun()


def render_filter_chips():
    """Render active filter chips."""
    filters = []
    
    if get_state("active_jobs_only"):
        filters.append("Active Jobs Only")
    
    if get_state("exclude_leave"):
        filters.append("Excl. Leave")
    
    if not get_state("include_nonbillable"):
        filters.append("Billable Only")
    
    client = get_state("selected_client")
    if client:
        filters.append(f"Client: {client}")
    
    if filters:
        chips = " | ".join([f"`{f}`" for f in filters])
        st.caption(f"Filters: {chips}")


# =============================================================================
# SIDEBAR FILTERS
# =============================================================================

def render_sidebar_filters(df: pd.DataFrame):
    """Render sidebar with global filters."""
    st.sidebar.header("Filters")
    
    # Time window
    time_options = {
        "3m": "Last 3 months",
        "6m": "Last 6 months",
        "12m": "Last 12 months",
        "24m": "Last 24 months",
        "fytd": "FY to date",
        "all": "All time",
    }
    
    current_window = get_state("time_window")
    window_index = list(time_options.keys()).index(current_window) if current_window in time_options else 2
    
    selected_window = st.sidebar.selectbox(
        "Time Window",
        options=list(time_options.keys()),
        format_func=lambda x: time_options[x],
        index=window_index,
        key="filter_time_window"
    )
    set_state("time_window", selected_window)
    
    st.sidebar.divider()
    
    # Toggle filters
    exclude_leave = st.sidebar.checkbox(
        "Exclude Leave",
        value=get_state("exclude_leave"),
        key="filter_exclude_leave"
    )
    set_state("exclude_leave", exclude_leave)
    
    job_state_options = ["All", "Active", "Completed"]
    current_state = get_state("job_state_filter") if get_state("job_state_filter") in job_state_options else "All"
    selected_state = st.sidebar.selectbox(
        "Job State",
        options=job_state_options,
        index=job_state_options.index(current_state),
        key="filter_job_state"
    )
    set_state("job_state_filter", selected_state)
    
    include_nonbillable = st.sidebar.checkbox(
        "Include Non-Billable",
        value=get_state("include_nonbillable"),
        key="filter_include_nonbillable"
    )
    set_state("include_nonbillable", include_nonbillable)
    
    st.sidebar.divider()
    
    # Optional filters (only show if columns exist)
    if "client" in df.columns:
        clients = ["All"] + sorted(df["client"].dropna().unique().tolist())
        selected_client = st.sidebar.selectbox(
            "Client",
            options=clients,
            index=0,
            key="filter_client"
        )
        set_state("selected_client", None if selected_client == "All" else selected_client)
    
    if "job_status" in df.columns:
        statuses = ["All"] + sorted(df["job_status"].dropna().unique().tolist())
        selected_status = st.sidebar.selectbox(
            "Job Status",
            options=statuses,
            index=0,
            key="filter_status"
        )
        set_state("selected_status", None if selected_status == "All" else selected_status)


# =============================================================================
# KPI CARDS
# =============================================================================

def render_kpi_card(label: str, value: str, delta: Optional[str] = None, 
                    delta_color: str = "normal"):
    """Render a single KPI card."""
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color
    )


def render_kpi_strip(metrics: dict):
    """
    Render horizontal strip of KPI cards.
    
    metrics: dict with keys like 'revenue', 'cost', 'margin', etc.
    """
    cols = st.columns(len(metrics))
    
    format_map = {
        "revenue": ("Revenue", fmt_currency),
        "cost": ("Cost", fmt_currency),
        "margin": ("Margin", fmt_currency),
        "margin_pct": ("Margin %", lambda x: fmt_percent(x)),
        "hours": ("Hours", fmt_hours),
        "realised_rate": ("Realised Rate", fmt_rate),
        "utilisation": ("Billable Share", lambda x: fmt_percent(x)),
    }
    
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i]:
            if key in format_map:
                label, formatter = format_map[key]
                formatted_value = formatter(value)
            else:
                label = key.replace("_", " ").title()
                formatted_value = str(value)
            
            st.metric(label=label, value=formatted_value)


# =============================================================================
# DRILL TABLE
# =============================================================================

def render_drill_table(df: pd.DataFrame, 
                       group_col: str,
                       on_row_click: Optional[Callable] = None,
                       key: str = "drill_table"):
    """
    Render drill table with clickable rows.
    
    Args:
        df: DataFrame to display
        group_col: Column name for the grouping dimension
        on_row_click: Callback when row is clicked
        key: Unique key for the table
    """
    # Display columns
    display_cols = [group_col]
    
    # Add metric columns in order
    metric_order = [
        "revenue", "margin_pct", "hours", "hours_variance_pct",
        "unquoted_share", "quote_rate", "realised_rate", "rate_variance",
        "utilisation"
    ]
    
    for col in metric_order:
        if col in df.columns:
            display_cols.append(col)
    
    display_df = df[display_cols].copy()
    
    # Create selection
    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )
    
    # Handle selection
    if selection and selection.selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        selected_value = df.iloc[selected_idx][group_col]
        
        if on_row_click:
            on_row_click(selected_value)


# =============================================================================
# TABS
# =============================================================================

def render_category_tabs():
    """Render Tasks/Staff tabs for category level."""
    subtab = get_state("category_subtab")
    
    tab1, tab2 = st.tabs(["Tasks", "Staff"])
    
    with tab1:
        if subtab != "tasks":
            set_state("category_subtab", "tasks")
        return "tasks"
    
    with tab2:
        if subtab != "staff":
            set_state("category_subtab", "staff")
        return "staff"


# =============================================================================
# SECTION HEADERS
# =============================================================================

def section_header(title: str, description: Optional[str] = None):
    """Render section header with optional description."""
    st.subheader(title)
    if description:
        st.caption(description)


def info_box(title: str, content: str, type: str = "info"):
    """Render info/warning/error box."""
    if type == "info":
        st.info(f"**{title}**: {content}")
    elif type == "warning":
        st.warning(f"**{title}**: {content}")
    elif type == "error":
        st.error(f"**{title}**: {content}")
    elif type == "success":
        st.success(f"**{title}**: {content}")
