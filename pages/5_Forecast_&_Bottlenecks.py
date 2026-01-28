"""
5-Level Drill-Chain Forecast & Bottlenecks Dashboard

Level 0: Company Forecast (landing page)
    → click department → Level 1
Level 1: Department Forecast
    → click category → Level 2
Level 2: Category Distribution
    → click job → Level 3
Level 3: Individual Job Detail
    → click task → Level 4
Level 4: Task → FTE Responsibility

Key principle: One drill path, one set of definitions (horizon, velocity window, benchmark basis).
No math artifacts (no inf, no NaN, no divide-by-zero exposed).
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple

from src.config import config
from src.data.loader import load_fact_timesheet
from src.data.cohorts import get_active_jobs
from src.modeling.benchmarks import build_category_benchmarks
from src.modeling.supply import build_velocity_for_active_jobs
from src.modeling.forecast import (
    forecast_remaining_work,
    solve_bottlenecks,
    compute_risk_scores_for_jobs,
    translate_job_state,
    get_company_forecast,
    get_dept_forecast,
    get_category_jobs,
    get_job_tasks,
)
from src.ui.components import (
    render_breadcrumb_header,
    render_job_health_card,
    render_task_status_badge,
    render_scope_filtered_table,
)
from src.ui.charts import (
    apply_layout,
    risk_matrix,
    task_stacked_bar,
    bottleneck_heatmap,
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize drill-chain navigation state."""
    if 'drill_state' not in st.session_state:
        st.session_state['drill_state'] = {
            'level': 0,  # 0-4
            'selected_dept': None,
            'selected_category': None,
            'selected_job_id': None,
            'selected_task_id': None,
            'forecast_horizon_weeks': 12,
            'velocity_lookback_days': 21,
        }


# ============================================================================
# LEVEL 0: COMPANY FORECAST
# ============================================================================

def render_level_0(
    job_level: pd.DataFrame,
    task_level: pd.DataFrame,
) -> None:
    """
    Level 0: Company-level forecast.
    
    Shows: Total demand vs capacity, gap by department, at-risk overview.
    Interaction: Click a department row → navigate to Level 1.
    """
    st.markdown("## Company Forecast")
    
    horizon = st.session_state['drill_state']['forecast_horizon_weeks']
    
    # Company KPI strip
    company_forecast = get_company_forecast(job_level, horizon_weeks=horizon)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Demand",
            f"{company_forecast['total_demand_hours']:,.0f}h",
            delta=f"{company_forecast['gap_hours']:,.0f}h buffer"
        )
    with col2:
        st.metric(
            "Team Capacity",
            f"{company_forecast['total_capacity_hours']:,.0f}h",
            delta=f"Horizon: {horizon}w"
        )
    with col3:
        gap_status = "🟢 Surplus" if company_forecast['gap_hours'] > 0 else "🔴 Oversubscribed"
        st.metric(
            "Gap",
            f"{company_forecast['gap_hours']:,.0f}h",
            delta=gap_status
        )
    with col4:
        st.metric(
            "Gap (FTE)",
            f"{company_forecast['gap_fte']:.1f}",
            delta=f"@ 38h/week"
        )
    
    st.markdown("---")
    
    # Department breakdown
    st.markdown("### Department Breakdown (click to drill)")
    
    dept_df = company_forecast['dept_breakdown'].copy()
    dept_df = dept_df[[
        'department_final',
        'demand_hours',
        'capacity_hours',
        'gap_hours',
        'gap_pct',
        'at_risk_count',
        'avg_risk_score',
    ]].round(2)
    
    # Display with instructions
    st.info("Click on a department name below to drill into that department's forecast.")
    
    for idx, row in dept_df.iterrows():
        dept_name = row['department_final']
        cols = st.columns([2, 1, 1, 1, 1, 1])
        
        with cols[0]:
            if st.button(
                f"🔷 {dept_name}",
                key=f"dept_btn_{dept_name}",
                use_container_width=True,
            ):
                st.session_state['drill_state']['level'] = 1
                st.session_state['drill_state']['selected_dept'] = dept_name
                st.rerun()
        
        with cols[1]:
            st.write(f"{row['demand_hours']:,.0f}h")
        with cols[2]:
            st.write(f"{row['capacity_hours']:,.0f}h")
        with cols[3]:
            st.write(f"{row['gap_hours']:,.0f}h")
        with cols[4]:
            st.write(f"{row['at_risk_count']:.0f}")
        with cols[5]:
            st.write(f"{row['avg_risk_score']:.2f}")
    
    st.markdown("---")
    
    # Risk overview
    st.markdown("### Portfolio Risk Overview")
    
    risk_counts = {
        '🟢 On-Track': (job_level['risk_score'] < 0.2).sum(),
        '🟡 At-Risk': ((job_level['risk_score'] >= 0.2) & (job_level['risk_score'] < 0.7)).sum(),
        '🔴 Critical': (job_level['risk_score'] >= 0.7).sum(),
    }
    
    status_cols = st.columns(3)
    for col, (status, count) in zip(status_cols, risk_counts.items()):
        with col:
            st.info(f"{status}: **{count}** jobs")


# ============================================================================
# LEVEL 1: DEPARTMENT FORECAST
# ============================================================================

def render_level_1(
    job_level: pd.DataFrame,
    task_level: pd.DataFrame,
    dept: str,
) -> None:
    """
    Level 1: Department-level forecast.
    
    Shows: Dept demand vs capacity, breakdown by category, top bottleneck tasks.
    Interaction: Click a category → Level 2; click breadcrumb → Level 0.
    """
    st.markdown(f"## {dept} Forecast")
    
    # Breadcrumb
    col_breadcrumb, col_back = st.columns([4, 1])
    with col_breadcrumb:
        st.markdown(f"**Scope:** Company ▸ {dept}")
    with col_back:
        if st.button("← Back to Company", key="back_to_level0"):
            st.session_state['drill_state']['level'] = 0
            st.session_state['drill_state']['selected_dept'] = None
            st.rerun()
    
    st.markdown("---")
    
    horizon = st.session_state['drill_state']['forecast_horizon_weeks']
    dept_forecast = get_dept_forecast(job_level, dept=dept, horizon_weeks=horizon)
    
    # Department KPI strip
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dept Demand", f"{dept_forecast['dept_demand_hours']:,.0f}h")
    with col2:
        st.metric("Dept Capacity", f"{dept_forecast['dept_capacity_hours']:,.0f}h")
    with col3:
        gap_status = "🟢 Surplus" if dept_forecast['dept_gap_hours'] > 0 else "🔴 Oversubscribed"
        st.metric("Dept Gap", f"{dept_forecast['dept_gap_hours']:,.0f}h", delta=gap_status)
    with col4:
        st.metric("At-Risk Jobs", dept_forecast['at_risk_count'])
    
    st.markdown("---")
    
    # Category breakdown
    st.markdown("### Categories (click to drill)")
    
    cat_df = dept_forecast['category_breakdown'].copy()
    cat_df = cat_df[[
        'category_rev_job',
        'job_count',
        'demand_hours',
        'capacity_hours',
        'gap_hours',
        'at_risk_count',
        'avg_risk_score',
    ]].round(2)
    
    for idx, row in cat_df.iterrows():
        cat_name = row['category_rev_job']
        cols = st.columns([2, 1, 1, 1, 1, 1, 1])
        
        with cols[0]:
            if st.button(
                f"📂 {cat_name}",
                key=f"cat_btn_{cat_name}",
                use_container_width=True,
            ):
                st.session_state['drill_state']['level'] = 2
                st.session_state['drill_state']['selected_category'] = cat_name
                st.rerun()
        
        with cols[1]:
            st.write(f"{row['job_count']:.0f}")
        with cols[2]:
            st.write(f"{row['demand_hours']:,.0f}h")
        with cols[3]:
            st.write(f"{row['capacity_hours']:,.0f}h")
        with cols[4]:
            st.write(f"{row['gap_hours']:,.0f}h")
        with cols[5]:
            st.write(f"{row['at_risk_count']:.0f}")
        with cols[6]:
            st.write(f"{row['avg_risk_score']:.2f}")
    
    st.markdown("---")
    
    # Top bottleneck tasks (dept-scoped)
    st.markdown("### Top Bottleneck Tasks")
    
    dept_tasks = task_level[
        task_level['job_no'].isin(
            job_level[job_level['department_final'] == dept]['job_no'].unique()
        )
    ].copy()
    
    dept_tasks = dept_tasks.sort_values('remaining_task_hours', ascending=False).head(10)
    
    if len(dept_tasks) > 0:
        dept_tasks = dept_tasks.rename(columns={
            "team_velocity_hours_week": "task_velocity_hrs_week",
            "expected_task_hours": "task_hours_p50",
        })
        display_cols = [
            "task_name", "job_no", "remaining_task_hours",
            "task_velocity_hrs_week", "task_hours_p50",
        ]
        display_cols = [c for c in display_cols if c in dept_tasks.columns]
        st.dataframe(dept_tasks[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No bottleneck tasks in this department.")


# ============================================================================
# LEVEL 2: CATEGORY DISTRIBUTION
# ============================================================================

def render_level_2(
    job_level: pd.DataFrame,
    task_level: pd.DataFrame,
    dept: str,
    category: str,
) -> None:
    """
    Level 2: Category-level distribution view.
    
    Shows: Benchmark vs actual, job distribution, worst jobs ranked.
    Interaction: Click a job → Level 3; click breadcrumb → Level 1 or 0.
    """
    st.markdown(f"## {category} (in {dept})")
    
    # Breadcrumb
    col_breadcrumb, col_back = st.columns([4, 1])
    with col_breadcrumb:
        st.markdown(f"**Scope:** Company ▸ {dept} ▸ {category}")
    with col_back:
        if st.button("← Back to Department", key="back_to_level1"):
            st.session_state['drill_state']['level'] = 1
            st.session_state['drill_state']['selected_category'] = None
            st.rerun()
    
    st.markdown("---")
    
    # Get category jobs
    cat_jobs = get_category_jobs(job_level, dept=dept, category=category)
    
    if len(cat_jobs) == 0:
        st.warning(f"No active jobs found in {category}.")
        return
    
    # Category KPI strip
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("# Active Jobs", len(cat_jobs))
    with col2:
        st.metric("Total Remaining", f"{cat_jobs['remaining_hours'].sum():,.0f}h")
    with col3:
        st.metric("Avg Risk Score", f"{cat_jobs['risk_score'].mean():.2f}")
    
    st.markdown("---")
    
    # Worst jobs (ranked by urgency)
    st.markdown("### Jobs (Ranked by Urgency - Click to Detail)")
    
    for idx, job_row in cat_jobs.iterrows():
        job_id = job_row['job_no']
        cols = st.columns([2, 1, 1, 1, 1, 1, 1])
        
        with cols[0]:
            if st.button(
                f"🎯 Job #{job_id}",
                key=f"job_btn_{job_id}",
                use_container_width=True,
            ):
                st.session_state['drill_state']['level'] = 3
                st.session_state['drill_state']['selected_job_id'] = job_id
                st.rerun()
        
        with cols[1]:
            st.write(f"{job_row['remaining_hours']:,.0f}h")
        with cols[2]:
            st.write(f"{job_row['job_velocity_hrs_week']:.1f}h/w")
        with cols[3]:
            if np.isinf(job_row.get('job_eta_weeks', np.inf)):
                st.write("∞")
            else:
                st.write(f"{job_row.get('job_eta_weeks', 0):.1f}w")
        with cols[4]:
            st.write(f"{job_row['due_weeks']:.1f}w")
        with cols[5]:
            status, _ = translate_job_state(
                job_row['risk_score'],
                job_row['due_weeks'],
                job_row.get('job_eta_weeks', np.inf),
                job_row['job_velocity_hrs_week']
            )
            st.write(status)
        with cols[6]:
            st.write(f"{job_row['risk_score']:.2f}")


# ============================================================================
# LEVEL 3: INDIVIDUAL JOB DETAIL
# ============================================================================

def render_level_3(
    job_level: pd.DataFrame,
    task_level: pd.DataFrame,
    job_id: int,
    dept: str,
    category: str,
) -> None:
    """
    Level 3: Individual job deep-dive.
    
    Shows: Job health card, active contributors, task bottleneck matrix.
    Interaction: Click a task → Level 4; click breadcrumb → Level 2, 1, or 0.
    """
    st.markdown(f"## Job #{job_id}")
    
    # Breadcrumb
    col_breadcrumb, col_back = st.columns([4, 1])
    with col_breadcrumb:
        st.markdown(f"**Scope:** Company ▸ {dept} ▸ {category} ▸ #{job_id}")
    with col_back:
        if st.button("← Back to Category", key="back_to_level2"):
            st.session_state['drill_state']['level'] = 2
            st.session_state['drill_state']['selected_job_id'] = None
            st.rerun()
    
    st.markdown("---")
    
    # Get job row
    job_row = job_level[job_level['job_no'] == job_id].iloc[0]
    
    # Health card
    render_job_health_card(job_row)
    
    st.markdown("---")
    
    # Job tasks
    st.markdown("### Bottleneck Tasks (Click to Detail)")
    
    job_tasks = get_job_tasks(task_level, job_id=job_id, min_hours=0)
    job_tasks['show_negligible'] = False  # Toggle later if needed
    
    if len(job_tasks) > 0:
        for idx, task_row in job_tasks.iterrows():
            task_id = task_row.get('task_id', idx)
            task_name = task_row['task_name']
            remaining = task_row['remaining_task_hours']
            velocity = task_row.get('team_velocity_hours_week', task_row.get('task_velocity_hrs_week'))
            expected = task_row.get('expected_task_hours', task_row.get('task_hours_p50'))
            
            status = render_task_status_badge(velocity, remaining, expected)
            
            cols = st.columns([2, 1, 1, 1, 1])
            
            with cols[0]:
                if st.button(
                    f"📋 {task_name}",
                    key=f"task_btn_{task_id}",
                    use_container_width=True,
                ):
                    st.session_state['drill_state']['level'] = 4
                    st.session_state['drill_state']['selected_task_id'] = task_id
                    st.rerun()
            
            with cols[1]:
                st.write(f"{remaining:.0f}h")
            with cols[2]:
                st.write(f"{velocity:.1f}h/w")
            with cols[3]:
                if velocity > 0:
                    st.write(f"{remaining / velocity:.1f}w")
                else:
                    st.write("∞")
            with cols[4]:
                st.write(status)
    else:
        st.info("No tasks tracked for this job yet.")


# ============================================================================
# LEVEL 4: TASK → FTE RESPONSIBILITY
# ============================================================================

def render_level_4(
    job_level: pd.DataFrame,
    task_level: pd.DataFrame,
    job_id: int,
    task_id: int,
    dept: str,
    category: str,
) -> None:
    """
    Level 4: Task-level FTE responsibility and feasibility.
    
    Shows: Task health, active contributors, eligible staff, capacity feasibility, what-if.
    Interaction: Click breadcrumb → Level 3, 2, 1, or 0.
    """
    st.markdown(f"## Task Details (Job #{job_id})")
    
    # Breadcrumb
    col_breadcrumb, col_back = st.columns([4, 1])
    with col_breadcrumb:
        st.markdown(f"**Scope:** Company ▸ {dept} ▸ {category} ▸ #{job_id} ▸ Task")
    with col_back:
        if st.button("← Back to Job", key="back_to_level3"):
            st.session_state['drill_state']['level'] = 3
            st.session_state['drill_state']['selected_task_id'] = None
            st.rerun()
    
    st.markdown("---")
    
    # Get task row
    task_row = task_level[task_level.index == task_id].iloc[0] if task_id in task_level.index else None
    
    if task_row is None:
        st.error(f"Task {task_id} not found.")
        return
    
    task_name = task_row.get('task_name', 'Unknown')
    remaining = task_row['remaining_task_hours']
    velocity = task_row.get('team_velocity_hours_week', task_row.get('task_velocity_hrs_week'))
    expected = task_row.get('expected_task_hours', task_row.get('task_hours_p50'))
    
    # Task summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Task", task_name)
    with col2:
        st.metric("Remaining", f"{remaining:.0f}h")
    with col3:
        st.metric("Velocity", f"{velocity:.1f}h/w")
    with col4:
        if velocity > 0:
            st.metric("Est Complete", f"{remaining / velocity:.1f}w")
        else:
            st.metric("Est Complete", "Blocked")
    
    st.markdown("---")
    
    # What-if scenario
    st.markdown("### What-If Scenario Planning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        added_velocity = st.slider(
            "Add velocity (hrs/week)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            key="added_velocity"
        )
    
    with col2:
        deadline_shift = st.slider(
            "Shift deadline (weeks)",
            min_value=-2,
            max_value=4,
            value=0,
            step=1,
            key="deadline_shift"
        )
    
    # Scenario impact
    new_velocity = velocity + added_velocity
    new_remaining_weeks = remaining / new_velocity if new_velocity > 0 else np.inf
    original_weeks = remaining / velocity if velocity > 0 else np.inf
    
    impact_cols = st.columns(3)
    with impact_cols[0]:
        if np.isinf(original_weeks):
            st.metric("Current", "Blocked")
        else:
            st.metric("Current", f"{original_weeks:.1f}w")
    
    with impact_cols[1]:
        if np.isinf(new_remaining_weeks):
            st.metric("With Changes", "Blocked")
        else:
            st.metric("With Changes", f"{new_remaining_weeks:.1f}w")
    
    with impact_cols[2]:
        if np.isinf(original_weeks) or np.isinf(new_remaining_weeks):
            savings = "Unblocked!"
        else:
            savings = f"{original_weeks - new_remaining_weeks:.1f}w saved"
        st.metric("Improvement", savings)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point: render appropriate level based on session state."""
    st.set_page_config(page_title="Forecast & Bottlenecks", layout="wide")
    st.title("📊 Forecast & Bottlenecks - 5-Level Drill-Chain")
    
    init_session_state()
    
    # Load data
    try:
        df_active = load_fact_timesheet()
        if len(df_active) == 0:
            st.error("No active timesheet data found.")
            return
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Build forecasts
    try:
        benchmarks, task_mix = build_category_benchmarks(df_active)
        
        remaining_by_task = forecast_remaining_work(df_active, benchmarks, task_mix)
        active_jobs = df_active['job_no'].unique().tolist()
        velocity_df = build_velocity_for_active_jobs(df_active, active_jobs, weeks=4)
        job_level = df_active[['job_no', 'department_final', 'category_rev_job']].drop_duplicates()
        task_level, job_level = solve_bottlenecks(remaining_by_task, velocity_df, job_level)
        job_level = compute_risk_scores_for_jobs(job_level)
        
        # Aggregate remaining hours from task_level to job_level for forecasting
        remaining_hours_by_job = task_level.groupby('job_no')['remaining_task_hours'].sum().reset_index()
        remaining_hours_by_job.columns = ['job_no', 'remaining_hours']
        job_level = job_level.merge(remaining_hours_by_job, on='job_no', how='left')
        job_level['remaining_hours'] = job_level['remaining_hours'].fillna(0)
        
    except Exception as e:
        import traceback
        st.error(f"Error computing forecasts: {e}")
        st.error(traceback.format_exc())
        return
    
    # Route to appropriate level
    drill_state = st.session_state['drill_state']
    level = drill_state['level']
    
    try:
        if level == 0:
            render_level_0(job_level, remaining_by_task)
        
        elif level == 1:
            if drill_state['selected_dept']:
                render_level_1(job_level, remaining_by_task, drill_state['selected_dept'])
            else:
                st.warning("No department selected. Returning to company view.")
                st.session_state['drill_state']['level'] = 0
                st.rerun()
        
        elif level == 2:
            if drill_state['selected_dept'] and drill_state['selected_category']:
                render_level_2(
                    job_level,
                    remaining_by_task,
                    drill_state['selected_dept'],
                    drill_state['selected_category'],
                )
            else:
                st.warning("Invalid category selection. Returning.")
                st.session_state['drill_state']['level'] = 1
                st.rerun()
        
        elif level == 3:
            if drill_state['selected_job_id']:
                dept = drill_state.get('selected_dept', 'Unknown')
                cat = drill_state.get('selected_category', 'Unknown')
                render_level_3(
                    job_level,
                    remaining_by_task,
                    drill_state['selected_job_id'],
                    dept,
                    cat,
                )
            else:
                st.warning("Invalid job selection. Returning.")
                st.session_state['drill_state']['level'] = 2
                st.rerun()
        
        elif level == 4:
            if drill_state['selected_job_id'] and drill_state['selected_task_id']:
                dept = drill_state.get('selected_dept', 'Unknown')
                cat = drill_state.get('selected_category', 'Unknown')
                render_level_4(
                    job_level,
                    remaining_by_task,
                    drill_state['selected_job_id'],
                    drill_state['selected_task_id'],
                    dept,
                    cat,
                )
            else:
                st.warning("Invalid task selection. Returning.")
                st.session_state['drill_state']['level'] = 3
                st.rerun()
        
        else:
            st.error(f"Unknown level: {level}")
    
    except Exception as e:
        st.error(f"Rendering error at level {level}: {e}")
        import traceback
        st.write(traceback.format_exc())


if __name__ == "__main__":
    main()
