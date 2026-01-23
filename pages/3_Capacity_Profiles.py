"""
Capacity & Profiles Page

Empirical capacity, load, and capability profiling.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.data.profiles import (
    build_staff_profiles,
    compute_task_expertise,
    compute_category_expertise,
    compute_staff_capacity,
    compute_expected_load,
    compute_headroom,
)
from src.staffing.engine import recommend_staff_for_plan, get_capability_coverage
from src.ui.state import init_state, get_quote_plan
from src.ui.formatting import fmt_hours, fmt_percent
from src.config import config


st.set_page_config(page_title="Capacity & Profiles", page_icon="C", layout="wide")

init_state()


@st.cache_data(show_spinner=False)
def build_profile_data(df: pd.DataFrame, window_weeks: int, training_months: int):
    date_col = "work_date" if "work_date" in df.columns else "month_key"
    df_window = df.copy()
    if date_col in df_window.columns:
        ref = pd.to_datetime(df_window[date_col]).max()
        cutoff = ref - pd.DateOffset(weeks=window_weeks)
        df_window = df_window[pd.to_datetime(df_window[date_col]) >= cutoff]
    
    profiles = build_staff_profiles(df, window_weeks, training_months, df_window=df_window)
    task_expertise = compute_task_expertise(df, training_months, config.RECENCY_HALF_LIFE_MONTHS)
    category_expertise = compute_category_expertise(df, training_months, config.RECENCY_HALF_LIFE_MONTHS)
    capacity = compute_staff_capacity(df_window, window_weeks)
    expected = compute_expected_load(df_window, window_weeks, window_weeks)
    headroom = compute_headroom(capacity, expected)
    return profiles, task_expertise, category_expertise, headroom, capacity, expected, df_window


def _staff_weekly_series(df: pd.DataFrame, staff_name: str) -> pd.DataFrame:
    date_col = "work_date" if "work_date" in df.columns else "month_key"
    df_staff = df[df["staff_name"] == staff_name].copy()
    if len(df_staff) == 0:
        return pd.DataFrame()
    df_staff[date_col] = pd.to_datetime(df_staff[date_col])
    weekly = df_staff.set_index(date_col)["hours_raw"].resample("W").sum().reset_index()
    return weekly.tail(12)


def main():
    st.title("Capacity & Profiles")
    st.caption("Empirical capacity and capability profiling (no targets)")
    
    df = load_fact_timesheet()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    departments = ["All"] + sorted(df["department_final"].dropna().unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)
    
    window = st.sidebar.selectbox("Window", options=[2, 4, 8, 12], index=1, format_func=lambda x: f"{x}w")
    
    with st.sidebar.expander("Eligibility Settings"):
        recency_months = st.slider("Recency months", 1, 12, config.ELIGIBILITY_RECENCY_MONTHS)
        min_hours = st.slider("Min hours", 1, 50, config.ELIGIBILITY_MIN_HOURS)
        min_jobs = st.slider("Min jobs", 1, 10, config.ELIGIBILITY_MIN_JOBS)
    
    if selected_dept != "All":
        df = df[df["department_final"] == selected_dept]
    
    profiles, task_expertise, category_expertise, headroom_df, capacity_df, expected_df, df_window = build_profile_data(
        df, window, config.PROFILE_TRAINING_MONTHS
    )
    
    if len(df_window) == 0:
        st.warning("No activity found in the selected window.")
        return
    
    if len(profiles) == 0:
        st.warning("No active staff found in the selected window.")
        return
    
    # Team capacity KPIs
    st.subheader("Team Capacity")
    kpi_cols = st.columns(5)
    total_staff = profiles["staff_name"].nunique()
    total_staff_all = df["staff_name"].nunique() if "staff_name" in df.columns else total_staff
    total_fte = capacity_df["fte_scaling"].sum() if len(capacity_df) > 0 else 0
    total_capacity_week = total_fte * config.CAPACITY_HOURS_PER_WEEK
    total_capacity = total_capacity_week * window
    total_expected = expected_df["expected_load_hours"].sum() if len(expected_df) > 0 else 0
    total_expected_week = expected_df["avg_weekly_hours"].sum() if len(expected_df) > 0 else 0
    total_headroom = total_capacity - total_expected
    
    with kpi_cols[0]:
        st.metric("Staff Count", f"{total_staff:,}")
    with kpi_cols[1]:
        st.metric("Total FTE", f"{total_fte:.1f}")
    with kpi_cols[2]:
        st.metric("Capacity (Window)", fmt_hours(total_capacity))
    with kpi_cols[3]:
        st.metric("Expected Load", fmt_hours(total_expected))
    with kpi_cols[4]:
        st.metric("Headroom", fmt_hours(total_headroom))
    
    with st.expander("Methodology"):
        st.markdown(
            """
            **Capacity = pure supply.**  
            `Capacity (week) = 38 × fte_hours_scaling`
            `Capacity (window) = Capacity/week × weeks`
            
            **Expected Load = trailing average.**  
            `Expected Load = avg weekly hours (trailing) × weeks`
            
            **Headroom = Capacity - Expected Load**
            
            **Active staff:** only staff with hours in the selected window appear.
            """
        )
        st.caption(f"Active staff: {total_staff} of {total_staff_all} in selected department and window.")
        st.caption(f"Capacity/week: {fmt_hours(total_capacity_week)} | Expected load/week: {fmt_hours(total_expected_week)}")
    
    st.divider()
    
    # Staff profiles table
    st.subheader("Staff Profiles")
    search = st.text_input("Search staff", "")
    display = profiles.copy()
    
    if search:
        display = display[display["staff_name"].str.contains(search, case=False, na=False)]
    
    display["billable_pct"] = display["billable_ratio"] * 100
    def headroom_label(value: float) -> str:
        if pd.isna(value):
            return "-"
        if value < 0:
            return f"NEG {value:.0f}h"
        if value > 20:
            return f"POS {value:.0f}h"
        return f"MID {value:.0f}h"
    
    display["headroom_display"] = display["headroom_hours"].apply(headroom_label)
    
    table_cols = [
        "staff_name",
        "department_final",
        "archetype",
        "avg_hours_per_week",
        "billable_pct",
        "headroom_display",
        "active_jobs_count",
    ]
    
    display_table = display[table_cols].rename(columns={
        "staff_name": "Name",
        "department_final": "Dept",
        "archetype": "Archetype",
        "avg_hours_per_week": "Hrs/Wk",
        "billable_pct": "Bill%",
        "headroom_display": "Headroom",
        "active_jobs_count": "Jobs",
    })
    selection = st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="profiles_table",
    )
    
    selected_staff = None
    if selection and selection.selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        selected_staff = display.iloc[selected_idx]["staff_name"]
    
    if selected_staff:
        with st.expander(f"Staff detail: {selected_staff}", expanded=True):
            weekly = _staff_weekly_series(df, selected_staff)
            if len(weekly) > 0:
                st.line_chart(weekly.set_index(weekly.columns[0])["hours_raw"])
            
            staff_tasks = task_expertise[task_expertise["staff_name"] == selected_staff]
            staff_cats = category_expertise[category_expertise["staff_name"] == selected_staff]
            
            top_tasks = staff_tasks.nlargest(5, "capability_score")[["task_name", "capability_score", "hours_total"]]
            top_cats = staff_cats.nlargest(3, "capability_score")[["category_rev_job", "capability_score", "hours_total"]]
            
            if len(top_tasks) > 0:
                st.markdown("**Top Tasks**")
                st.dataframe(top_tasks, use_container_width=True, hide_index=True)
            
            if len(top_cats) > 0:
                st.markdown("**Top Categories**")
                st.dataframe(top_cats, use_container_width=True, hide_index=True)
            
            if "job_no" in df.columns:
                jobs = df[df["staff_name"] == selected_staff].groupby("job_no")["hours_raw"].sum().reset_index()
                jobs = jobs.rename(columns={"hours_raw": "hours"}).sort_values("hours", ascending=False).head(10)
                st.markdown("**Active Jobs (Top 10 by hours)**")
                st.dataframe(jobs, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Capability coverage
    st.subheader("Capability Coverage")
    coverage = get_capability_coverage(category_expertise)
    if len(coverage) > 0:
        coverage["risk_flag"] = coverage["coverage_risk"].map({
            "Critical": "Critical",
            "Low": "Low",
            "Moderate": "Moderate",
            "Good": "Good",
        })
        st.dataframe(
            coverage.rename(columns={
                "category_rev_job": "Category",
                "staff_count": "Staff w/ Expertise",
                "risk_flag": "Coverage Risk",
            })[["Category", "Staff w/ Expertise", "Coverage Risk"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No category expertise data available.")
    
    st.divider()
    
    # Staffing recommendations
    plan = get_quote_plan()
    if plan and plan.tasks:
        st.subheader("Staffing Recommendations")
        st.caption(f"For: {plan.department} / {plan.category}")
        
        quote_plan = {
            "department": plan.department,
            "category": plan.category,
            "tasks": [
                {"task_name": t.task_name, "hours": t.hours}
                for t in plan.tasks
                if not t.is_optional
            ],
        }
        
        eligibility_config = {
            "recency_months": recency_months,
            "min_hours": min_hours,
            "min_jobs": min_jobs,
        }
        
        recs, warnings = recommend_staff_for_plan(
            quote_plan,
            task_expertise,
            category_expertise,
            headroom_df,
            eligibility_config,
            top_n=3,
        )
        
        if len(recs) > 0:
            summary_rows = []
            for task_name, task_group in recs.groupby("task_name"):
                row = {"Task": task_name, "Hours": task_group["hours"].iloc[0]}
                for rank in [1, 2, 3]:
                    staff = task_group[task_group["rank"] == rank]
                    if len(staff) > 0:
                        row[f"#{rank} Staff"] = staff["staff_name"].iloc[0]
                        row[f"#{rank} Score"] = f"{staff['match_score'].iloc[0]:.0f}"
                    else:
                        row[f"#{rank} Staff"] = "-"
                        row[f"#{rank} Score"] = "-"
                task_warnings = [w.message for w in warnings if w.task == task_name]
                row["Warnings"] = " | ".join(sorted(set(task_warnings))) if task_warnings else ""
                summary_rows.append(row)
            
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No staffing recommendations available.")
    
    st.divider()
    
    # Export
    st.subheader("Export")
    st.download_button(
        "Download Profiles CSV",
        data=profiles.to_csv(index=False),
        file_name="staff_profiles.csv",
        mime="text/csv",
    )
    
    if plan and plan.tasks:
        st.download_button(
            "Download Recommendations CSV",
            data=recs.to_csv(index=False) if len(recs) > 0 else "",
            file_name="staff_recommendations.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
