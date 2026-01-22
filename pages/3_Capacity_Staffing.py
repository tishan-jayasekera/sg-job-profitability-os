"""
Capacity & Staffing Page

Capacity overview, staffing recommendations, and staff scatter analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state, get_state, get_quote_plan
from src.ui.layout import section_header
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent
from src.ui.charts import capacity_bar, scatter_plot
from src.data.loader import load_fact_timesheet
from src.data.semantic import exclude_leave, utilisation_metrics
from src.data.cohorts import filter_by_time_window, get_active_staff, compute_recency_weights
from src.config import config


st.set_page_config(page_title="Capacity & Staffing", page_icon="👥", layout="wide")

init_state()


def calculate_staff_capacity(df: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """Calculate capacity metrics per staff member."""
    
    # Get unique staff with their FTE scaling and utilisation target
    staff_info = df.groupby("staff_name").agg(
        fte_scaling=("fte_hours_scaling", "first"),
        util_target=("utilisation_target", "first"),
        department=("department_final", "first"),
    ).reset_index()
    
    # Calculate capacity
    staff_info["weekly_capacity"] = config.standard_weekly_hours * staff_info["fte_scaling"]
    staff_info["period_capacity"] = staff_info["weekly_capacity"] * weeks
    staff_info["billable_capacity"] = staff_info["period_capacity"] * staff_info["util_target"]
    
    # Calculate recent load (exclude leave)
    df_no_leave = exclude_leave(df)
    recent_cutoff = df["month_key"].max() - pd.DateOffset(weeks=weeks)
    df_recent = df_no_leave[df_no_leave["month_key"] >= recent_cutoff]
    
    # Billable and total load
    billable_load = df_recent[df_recent["is_billable"]].groupby("staff_name")["hours_raw"].sum()
    total_load = df_recent.groupby("staff_name")["hours_raw"].sum()
    
    staff_info = staff_info.merge(
        billable_load.reset_index().rename(columns={"hours_raw": "billable_load"}),
        on="staff_name", how="left"
    )
    staff_info = staff_info.merge(
        total_load.reset_index().rename(columns={"hours_raw": "total_load"}),
        on="staff_name", how="left"
    )
    
    staff_info["billable_load"] = staff_info["billable_load"].fillna(0)
    staff_info["total_load"] = staff_info["total_load"].fillna(0)
    
    # Headroom
    staff_info["headroom"] = staff_info["billable_capacity"] - staff_info["billable_load"]
    
    # Utilisation
    staff_info["utilisation"] = np.where(
        staff_info["period_capacity"] > 0,
        staff_info["billable_load"] / staff_info["period_capacity"] * 100,
        0
    )
    
    # Active jobs
    active_jobs = df_recent.groupby("staff_name")["job_no"].nunique().reset_index()
    active_jobs.columns = ["staff_name", "active_job_count"]
    staff_info = staff_info.merge(active_jobs, on="staff_name", how="left")
    staff_info["active_job_count"] = staff_info["active_job_count"].fillna(0).astype(int)
    
    return staff_info


def get_staff_capability(df: pd.DataFrame, task_name: str = None, 
                         category: str = None) -> pd.DataFrame:
    """Get staff capability scores for a task/category."""
    
    df_weighted = df.copy()
    df_weighted["weight"] = compute_recency_weights(df_weighted)
    df_weighted["hours_weighted"] = df_weighted["hours_raw"] * df_weighted["weight"]
    
    if task_name:
        df_slice = df_weighted[df_weighted["task_name"] == task_name]
    elif category:
        df_slice = df_weighted[df_weighted["job_category"] == category]
    else:
        df_slice = df_weighted
    
    capability = df_slice.groupby("staff_name").agg(
        hours_weighted=("hours_weighted", "sum"),
        hours_total=("hours_raw", "sum"),
        job_count=("job_no", "nunique"),
    ).reset_index()
    
    # Normalize to 0-100 score
    max_hours = capability["hours_weighted"].max()
    if max_hours > 0:
        capability["capability_score"] = capability["hours_weighted"] / max_hours * 100
    else:
        capability["capability_score"] = 0
    
    return capability


def main():
    st.title("Capacity & Staffing")
    
    # Load data
    df = load_fact_timesheet()
    
    # Time window filter
    time_window = st.sidebar.selectbox(
        "Time Window",
        options=["4w", "8w", "12w"],
        format_func=lambda x: f"Last {x}",
        index=0
    )
    weeks = int(time_window.replace("w", ""))
    
    # Filter options
    department = st.sidebar.selectbox(
        "Department",
        options=["All"] + sorted(df["department_final"].dropna().unique().tolist())
    )
    
    if department != "All":
        df = df[df["department_final"] == department]
    
    # =========================================================================
    # SECTION A: CAPACITY OVERVIEW
    # =========================================================================
    section_header("Capacity Overview", f"Based on last {weeks} weeks")
    
    # Calculate capacity
    staff_capacity = calculate_staff_capacity(df, weeks=weeks)
    
    # Totals
    total_supply = staff_capacity["period_capacity"].sum()
    total_billable_cap = staff_capacity["billable_capacity"].sum()
    total_load = staff_capacity["billable_load"].sum()
    total_headroom = staff_capacity["headroom"].sum()
    
    # KPIs
    cap_cols = st.columns(4)
    
    with cap_cols[0]:
        st.metric("Total Supply", fmt_hours(total_supply))
    
    with cap_cols[1]:
        st.metric("Billable Capacity", fmt_hours(total_billable_cap))
    
    with cap_cols[2]:
        st.metric("Current Load", fmt_hours(total_load))
    
    with cap_cols[3]:
        st.metric("Headroom", fmt_hours(total_headroom))
    
    # Capacity bar
    fig = capacity_bar(total_supply, total_load, total_headroom)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION B: QUOTE PLAN STAFFING (if plan exists)
    # =========================================================================
    plan = get_quote_plan()
    
    if plan and plan.tasks:
        section_header("Staffing Recommendations", f"For quote plan: {plan.category}")
        
        # Filter to relevant department/category
        df_slice = df[
            (df["department_final"] == plan.department) &
            (df["job_category"] == plan.category)
        ]
        
        recommendations = []
        
        for task in plan.tasks:
            if task.is_optional:
                continue
            
            # Get capability for this task
            capability = get_staff_capability(df_slice, task_name=task.task_name)
            
            # Merge with capacity
            staff_scored = capability.merge(
                staff_capacity[["staff_name", "headroom", "utilisation"]],
                on="staff_name", how="left"
            )
            
            # Score: balance capability and availability
            staff_scored["availability_score"] = np.clip(staff_scored["headroom"] / task.hours * 50, 0, 100)
            staff_scored["total_score"] = (
                staff_scored["capability_score"] * 0.6 + 
                staff_scored["availability_score"] * 0.4
            )
            
            # Top 3
            top_staff = staff_scored.nlargest(3, "total_score")
            
            for _, s in top_staff.iterrows():
                recommendations.append({
                    "Task": task.task_name,
                    "Hours": task.hours,
                    "Staff": s["staff_name"],
                    "Capability": f"{s['capability_score']:.0f}",
                    "Headroom": fmt_hours(s["headroom"]),
                    "Score": f"{s['total_score']:.0f}",
                })
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
        else:
            st.info("No staffing recommendations available for this plan.")
    else:
        st.info("No quote plan loaded. Build a plan in the Quote Builder to get staffing recommendations.")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION C: STAFF SCATTER
    # =========================================================================
    section_header("Staff Analysis", "Click a point for details")
    
    # Scatter: X = headroom, Y = utilisation
    staff_capacity_display = staff_capacity[staff_capacity["period_capacity"] > 0].copy()
    
    if len(staff_capacity_display) > 0:
        fig = scatter_plot(
            staff_capacity_display,
            x="headroom",
            y="utilisation",
            size="active_job_count",
            hover_name="staff_name",
            hover_data=["department", "billable_load", "billable_capacity"],
            title="Staff Capacity vs Utilisation",
            x_title="Available Headroom (hours)",
            y_title="Utilisation (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Staff table
    section_header("Staff Capacity Table")
    
    display_cols = [
        "staff_name", "department", "period_capacity", "billable_capacity",
        "billable_load", "headroom", "utilisation", "active_job_count"
    ]
    
    display_df = staff_capacity_display[[c for c in display_cols if c in staff_capacity_display.columns]].copy()
    display_df = display_df.rename(columns={
        "staff_name": "Staff",
        "department": "Department",
        "period_capacity": "Capacity",
        "billable_capacity": "Billable Cap",
        "billable_load": "Load",
        "headroom": "Headroom",
        "utilisation": "Util %",
        "active_job_count": "Active Jobs",
    })
    
    st.dataframe(
        display_df.sort_values("Headroom", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Capacity": st.column_config.NumberColumn(format="%.0f"),
            "Billable Cap": st.column_config.NumberColumn(format="%.0f"),
            "Load": st.column_config.NumberColumn(format="%.0f"),
            "Headroom": st.column_config.NumberColumn(format="%.0f"),
            "Util %": st.column_config.NumberColumn(format="%.1f%%"),
        }
    )


if __name__ == "__main__":
    main()
