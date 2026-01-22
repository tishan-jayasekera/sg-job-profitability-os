"""
Utilisation & Time Use Page

Diagnose utilisation gaps and understand where time is going.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state, get_state, set_state
from src.ui.layout import section_header
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent
from src.ui.charts import scatter_plot, horizontal_bar
from src.data.loader import load_fact_timesheet
from src.data.semantic import utilisation_metrics, exclude_leave, leave_exclusion_mask
from src.data.cohorts import filter_by_time_window
from src.config import config


st.set_page_config(page_title="Utilisation & Time Use", page_icon="⏱️", layout="wide")

init_state()


def calculate_staff_utilisation(df: pd.DataFrame, exclude_leave_rows: bool = True) -> pd.DataFrame:
    """Calculate detailed utilisation metrics per staff."""
    
    if exclude_leave_rows:
        df = exclude_leave(df)
    
    # Staff-level aggregation
    staff_metrics = df.groupby("staff_name").agg(
        department=("department_final", "first"),
        total_hours=("hours_raw", "sum"),
        billable_hours=("hours_raw", lambda x: x[df.loc[x.index, "is_billable"]].sum()),
        util_target=("utilisation_target", "first"),
        fte_scaling=("fte_hours_scaling", "first"),
    ).reset_index()
    
    # Utilisation
    staff_metrics["utilisation"] = np.where(
        staff_metrics["total_hours"] > 0,
        staff_metrics["billable_hours"] / staff_metrics["total_hours"] * 100,
        0
    )
    
    staff_metrics["target_pct"] = staff_metrics["util_target"] * 100
    staff_metrics["util_gap"] = staff_metrics["target_pct"] - staff_metrics["utilisation"]
    
    # Non-billable hours
    staff_metrics["non_billable_hours"] = staff_metrics["total_hours"] - staff_metrics["billable_hours"]
    staff_metrics["non_billable_pct"] = np.where(
        staff_metrics["total_hours"] > 0,
        staff_metrics["non_billable_hours"] / staff_metrics["total_hours"] * 100,
        0
    )
    
    return staff_metrics


def get_breakdown_analysis(df: pd.DataFrame, group_by: str = "breakdown") -> pd.DataFrame:
    """Get time breakdown by a dimension."""
    
    if group_by not in df.columns:
        return pd.DataFrame()
    
    breakdown = df.groupby(group_by).agg(
        hours=("hours_raw", "sum"),
        staff_count=("staff_name", "nunique"),
    ).reset_index()
    
    total_hours = breakdown["hours"].sum()
    breakdown["pct"] = breakdown["hours"] / total_hours * 100 if total_hours > 0 else 0
    
    return breakdown.sort_values("hours", ascending=False)


def get_staff_nonbillable_detail(df: pd.DataFrame, staff_name: str) -> pd.DataFrame:
    """Get non-billable breakdown for a specific staff member."""
    
    df_staff = df[df["staff_name"] == staff_name].copy()
    df_nonbill = df_staff[~df_staff["is_billable"]]
    
    if len(df_nonbill) == 0:
        return pd.DataFrame()
    
    # Group by task or breakdown
    if "breakdown" in df_nonbill.columns and df_nonbill["breakdown"].notna().any():
        group_col = "breakdown"
    else:
        group_col = "task_name"
    
    detail = df_nonbill.groupby(group_col).agg(
        hours=("hours_raw", "sum"),
    ).reset_index()
    
    total = detail["hours"].sum()
    detail["pct"] = detail["hours"] / total * 100 if total > 0 else 0
    
    return detail.sort_values("hours", ascending=False)


def main():
    st.title("Utilisation & Time Use")
    st.caption("Understand where time is going and diagnose utilisation gaps")
    
    # Load data
    df = load_fact_timesheet()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    time_window = st.sidebar.selectbox(
        "Time Window",
        options=["3m", "6m", "12m", "24m"],
        format_func=lambda x: f"Last {x}",
        index=2
    )
    
    df = filter_by_time_window(df, time_window)
    
    departments = ["All"] + sorted(df["department_final"].dropna().unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)
    
    if selected_dept != "All":
        df = df[df["department_final"] == selected_dept]
    
    exclude_leave_toggle = st.sidebar.checkbox("Exclude Leave", value=True)
    
    # =========================================================================
    # SECTION A: OVERALL UTILISATION
    # =========================================================================
    section_header("Overall Utilisation")
    
    util = utilisation_metrics(df, exclude_leave=exclude_leave_toggle)
    
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        st.metric("Billable Hours", fmt_hours(util["billable_hours"].iloc[0]))
    
    with kpi_cols[1]:
        st.metric("Total Hours", fmt_hours(util["total_hours"].iloc[0]))
    
    with kpi_cols[2]:
        st.metric("Utilisation", fmt_percent(util["utilisation"].iloc[0]))
    
    with kpi_cols[3]:
        st.metric("Target", fmt_percent(util["utilisation_target_pct"].iloc[0]))
    
    with kpi_cols[4]:
        gap = util["util_gap"].iloc[0]
        color = "normal" if gap <= 0 else "inverse"
        st.metric("Gap", fmt_percent(gap), delta_color=color)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION B: STAFF SCATTER
    # =========================================================================
    section_header("Staff Utilisation", "X = Utilisation, Y = Non-Billable %")
    
    staff_util = calculate_staff_utilisation(df, exclude_leave_rows=exclude_leave_toggle)
    
    if len(staff_util) > 0:
        # Add department color
        fig = scatter_plot(
            staff_util,
            x="utilisation",
            y="non_billable_pct",
            size="total_hours",
            color="department" if "department" in staff_util.columns else None,
            hover_name="staff_name",
            hover_data=["billable_hours", "non_billable_hours", "util_gap"],
            title="Staff Utilisation vs Non-Billable Share",
            x_title="Billable Utilisation (%)",
            y_title="Non-Billable Share (%)"
        )
        
        # Add target line
        target = staff_util["target_pct"].mean()
        fig.add_vline(x=target, line_dash="dash", line_color="gray", 
                      annotation_text=f"Target: {target:.0f}%")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION C: TIME BREAKDOWN
    # =========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Time Breakdown", "By category")
        
        if exclude_leave_toggle:
            df_analysis = exclude_leave(df)
        else:
            df_analysis = df
        
        breakdown = get_breakdown_analysis(df_analysis, "breakdown")
        
        if len(breakdown) > 0:
            fig = horizontal_bar(
                breakdown,
                x="hours",
                y="breakdown",
                title="Hours by Breakdown Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No breakdown data available")
    
    with col2:
        section_header("By Department")
        
        dept_util = utilisation_metrics(df_analysis, ["department_final"], exclude_leave=False)
        dept_util = dept_util.sort_values("utilisation", ascending=True)
        
        if len(dept_util) > 0:
            fig = horizontal_bar(
                dept_util,
                x="utilisation",
                y="department_final",
                title="Utilisation by Department"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION D: STAFF TABLE
    # =========================================================================
    section_header("Staff Utilisation Table")
    
    # Display table
    display_df = staff_util[[
        "staff_name", "department", "billable_hours", "total_hours",
        "utilisation", "target_pct", "util_gap"
    ]].copy()
    
    display_df = display_df.rename(columns={
        "staff_name": "Staff",
        "department": "Department",
        "billable_hours": "Billable",
        "total_hours": "Total",
        "utilisation": "Util %",
        "target_pct": "Target %",
        "util_gap": "Gap",
    })
    
    # Sort by gap (biggest gap first)
    display_df = display_df.sort_values("Gap", ascending=False)
    
    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Billable": st.column_config.NumberColumn(format="%.0f"),
            "Total": st.column_config.NumberColumn(format="%.0f"),
            "Util %": st.column_config.NumberColumn(format="%.1f%%"),
            "Target %": st.column_config.NumberColumn(format="%.1f%%"),
            "Gap": st.column_config.NumberColumn(format="%.1f%%"),
        },
        key="staff_util_table"
    )
    
    # =========================================================================
    # SECTION E: STAFF DETAIL (when selected)
    # =========================================================================
    if selection and selection.selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        selected_staff = staff_util.iloc[selected_idx]["staff_name"]
        
        st.markdown("---")
        section_header(f"Non-Billable Detail: {selected_staff}")
        
        detail = get_staff_nonbillable_detail(df_analysis, selected_staff)
        
        if len(detail) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = horizontal_bar(
                    detail.head(10),
                    x="hours",
                    y=detail.columns[0],
                    title=f"Non-Billable Hours by {detail.columns[0].replace('_', ' ').title()}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(
                    detail.rename(columns={
                        detail.columns[0]: "Category",
                        "hours": "Hours",
                        "pct": "% of Non-Billable"
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Hours": st.column_config.NumberColumn(format="%.1f"),
                        "% of Non-Billable": st.column_config.NumberColumn(format="%.1f%%"),
                    }
                )
        else:
            st.success(f"{selected_staff} has no non-billable hours in this period.")


if __name__ == "__main__":
    main()
