"""
Standard table components with drill behaviors.
"""
import streamlit as st
import pandas as pd
from typing import Optional, Callable, List, Dict, Any

from src.ui.formatting import format_metric_df


def drill_table(df: pd.DataFrame,
                group_col: str,
                on_select: Optional[Callable[[str], None]] = None,
                display_cols: Optional[List[str]] = None,
                key: str = "drill_table") -> Optional[str]:
    """
    Render a drill table with clickable rows.
    
    Args:
        df: DataFrame with metrics
        group_col: Column for row labels (clickable)
        on_select: Callback when row is selected
        display_cols: Columns to display (None = auto-detect)
        key: Unique key for the table
    
    Returns:
        Selected value if selection made, else None
    """
    if len(df) == 0:
        st.info("No data to display.")
        return None
    
    # Auto-detect display columns
    if display_cols is None:
        display_cols = [group_col]
        
        # Add metric columns in priority order
        metric_order = [
            "revenue", "margin_pct", "hours", "realised_rate",
            "hours_variance_pct", "unquoted_share", "utilisation",
            "quote_rate", "rate_variance"
        ]
        
        for col in metric_order:
            if col in df.columns:
                display_cols.append(col)
    
    # Prepare display dataframe
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    display_df = format_metric_df(display_df)
    
    # Column name mapping
    col_names = {
        "department_final": "Department",
        "job_category": "Category",
        "category_rev_job": "Category",
        "task_name": "Task",
        "staff_name": "Staff",
        "breakdown": "Breakdown",
        "job_no": "Job",
        "revenue": "Revenue",
        "cost": "Cost",
        "margin": "Margin",
        "margin_pct": "Margin %",
        "hours": "Hours",
        "realised_rate": "Rate",
        "quote_rate": "Quote Rate",
        "rate_variance": "Rate Var",
        "hours_variance_pct": "Hrs Var %",
        "unquoted_share": "Scope Creep",
        "utilisation": "Util %",
        "pct_quote_consumed": "% Consumed",
        "risk_flag": "Risk",
    }
    
    display_df = display_df.rename(columns=col_names)
    
    # Render with selection
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
        
        if on_select:
            on_select(selected_value)
        
        return selected_value
    
    return None


def metric_table(df: pd.DataFrame,
                 group_cols: List[str],
                 metric_cols: Optional[List[str]] = None,
                 sort_by: Optional[str] = None,
                 ascending: bool = False,
                 key: str = "metric_table"):
    """
    Render a read-only metric table.
    """
    if len(df) == 0:
        st.info("No data to display.")
        return
    
    # Select columns
    if metric_cols is None:
        metric_cols = [c for c in df.columns if c not in group_cols]
    
    display_cols = group_cols + metric_cols
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    
    # Sort
    if sort_by and sort_by in display_df.columns:
        display_df = display_df.sort_values(sort_by, ascending=ascending)
    
    # Format
    display_df = format_metric_df(display_df)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True, key=key)


def risk_table(df: pd.DataFrame,
               key: str = "risk_table") -> Optional[str]:
    """
    Render active jobs risk table with color coding.
    """
    if len(df) == 0:
        st.info("No active jobs to display.")
        return None
    
    display_cols = [
        "job_no", "department_final", "job_category",
        "pct_quote_consumed", "scope_creep_pct", "rate_variance",
        "risk_flag"
    ]
    
    # Add optional columns
    if "client" in df.columns:
        display_cols.insert(3, "client")
    if "job_due_date" in df.columns:
        display_cols.insert(4, "job_due_date")
    
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    
    # Format
    display_df = format_metric_df(display_df)
    
    # Column config with color for risk
    column_config = {
        "risk_flag": st.column_config.TextColumn(
            "Risk",
            help="on_track / watch / at_risk"
        ),
    }
    
    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config=column_config,
        key=key,
    )
    
    if selection and selection.selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        return df.iloc[selected_idx]["job_no"]
    
    return None


def staff_capacity_table(df: pd.DataFrame,
                         key: str = "staff_capacity_table"):
    """
    Render staff capacity table.
    """
    if len(df) == 0:
        st.info("No staff data to display.")
        return
    
    display_cols = [
        "staff_name", "department", "period_capacity",
        "billable_capacity", "billable_load", "headroom",
        "trailing_utilisation", "active_job_count"
    ]
    
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    
    # Rename for display
    col_names = {
        "staff_name": "Staff",
        "department": "Department",
        "period_capacity": "Capacity",
        "billable_capacity": "Billable Cap",
        "billable_load": "Load",
        "headroom": "Headroom",
        "trailing_utilisation": "Util %",
        "active_job_count": "Active Jobs",
    }
    display_df = display_df.rename(columns=col_names)
    
    # Sort by headroom descending
    display_df = display_df.sort_values("Headroom", ascending=False)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Capacity": st.column_config.NumberColumn(format="%.0f"),
            "Billable Cap": st.column_config.NumberColumn(format="%.0f"),
            "Load": st.column_config.NumberColumn(format="%.0f"),
            "Headroom": st.column_config.NumberColumn(format="%.0f"),
            "Util %": st.column_config.NumberColumn(format="%.1f%%"),
        },
        key=key,
    )


def task_benchmark_table(df: pd.DataFrame,
                         editable: bool = False,
                         key: str = "task_benchmark_table") -> pd.DataFrame:
    """
    Render task benchmark table, optionally editable.
    """
    if len(df) == 0:
        st.info("No task data to display.")
        return pd.DataFrame()
    
    display_cols = [
        "task_name", "inclusion_rate", "quoted_hours_p50",
        "quoted_hours_p25", "quoted_hours_p75",
        "overrun_risk", "cost_per_hour_median", "quote_rate"
    ]
    
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    
    col_names = {
        "task_name": "Task",
        "inclusion_rate": "Inclusion %",
        "quoted_hours_p50": "Suggested Hrs",
        "quoted_hours_p25": "Low (p25)",
        "quoted_hours_p75": "High (p75)",
        "overrun_risk": "Overrun Risk %",
        "cost_per_hour_median": "Cost/hr",
        "quote_rate": "Quote Rate",
    }
    display_df = display_df.rename(columns=col_names)
    
    if editable:
        return st.data_editor(
            display_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "Inclusion %": st.column_config.NumberColumn(format="%.1f%%", disabled=True),
                "Suggested Hrs": st.column_config.NumberColumn(min_value=0, step=0.5),
                "Low (p25)": st.column_config.NumberColumn(format="%.1f", disabled=True),
                "High (p75)": st.column_config.NumberColumn(format="%.1f", disabled=True),
                "Overrun Risk %": st.column_config.NumberColumn(format="%.0f%%", disabled=True),
                "Cost/hr": st.column_config.NumberColumn(format="$%.0f", disabled=True),
                "Quote Rate": st.column_config.NumberColumn(format="$%.0f/hr", disabled=True),
            },
            key=key,
        )
    else:
        st.dataframe(display_df, use_container_width=True, hide_index=True, key=key)
        return display_df
