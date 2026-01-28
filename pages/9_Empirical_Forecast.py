"""
Empirical Lifecycle & Surgical Task Forecaster

Map "remaining DNA" of active jobs against historical task-distribution profiles 
from completed jobs. Forecast demand by skill/task and identify surgical gaps.

5-Level Drill:
- Level 0: Company Pipeline Demand (surgical deficit by task)
- Level 1: Departmental Contribution (dept-level gaps)
- Level 2: Category Benchmarking (historical DNA vs. current jobs)
- Level 3: Job Maturity Card (single job health & next phase warning)
- Level 4: Surgical Intervention (assign FTE to bottleneck task)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state, get_state, set_state
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_count
from src.data.loader import load_fact_timesheet
from src.data.cohorts import get_active_jobs, filter_by_time_window
from src.data.semantic import get_category_col, exclude_leave
from src.modeling.forecast import (
    build_lifecycle_dna,
    compute_job_maturity,
    badge_fte_capabilities,
    compute_surgical_gaps,
)
from src.config import config


st.set_page_config(page_title="Empirical Forecast", page_icon="🔮", layout="wide")
init_state()


# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    h1, h2 { color: #1f4788; font-weight: 700; }
    .metric-card {
        background: linear-gradient(135deg, #f8fafb 0%, #ffffff 100%);
        border: 1px solid #e3e8ef;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .breadcrumb { font-size: 0.9rem; color: #888; margin: 1rem 0; }
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-blocked { background: #ffe5e5; color: #d32f2f; }
    .status-at-risk { background: #fff3e0; color: #f57c00; }
    .status-on-track { background: #e8f5e9; color: #388e3c; }
</style>
""", unsafe_allow_html=True)


def init_drill_state():
    """Initialize session state for drill navigation."""
    defaults = {
        'empirical_drill': {
            'level': 0,  # 0=Company, 1=Dept, 2=Category, 3=Job, 4=Task
            'dept': None,
            'category': None,
            'job_id': None,
            'task_id': None,
        },
        'lookback_jobs': 30,
        'horizon_weeks': 12,
        'recency_months': 6,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_breadcrumb(drill):
    """Render navigation breadcrumb."""
    parts = ["🏢 Company"]
    if drill['level'] >= 1 and drill['dept']:
        parts.append(f"📁 {drill['dept']}")
    if drill['level'] >= 2 and drill['category']:
        parts.append(f"📂 {drill['category']}")
    if drill['level'] >= 3 and drill['job_id']:
        parts.append(f"🎯 Job {drill['job_id']}")
    if drill['level'] >= 4 and drill['task_id']:
        parts.append(f"📋 {drill['task_id']}")
    
    st.markdown(f"<div class='breadcrumb'>{'  ›  '.join(parts)}</div>", unsafe_allow_html=True)


def render_drill_controls(drill):
    """Render drill navigation buttons."""
    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("⬅️ Up", key="drill_up"):
            if drill['level'] > 0:
                drill['level'] -= 1
                if drill['level'] < 4:
                    drill['task_id'] = None
                if drill['level'] < 3:
                    drill['job_id'] = None
                if drill['level'] < 2:
                    drill['category'] = None
                if drill['level'] < 1:
                    drill['dept'] = None
                st.session_state['empirical_drill'] = drill
                st.rerun()
    
    with cols[1]:
        if st.button("🏠 Reset", key="drill_reset"):
            st.session_state['empirical_drill'] = {
                'level': 0,
                'dept': None,
                'category': None,
                'job_id': None,
                'task_id': None,
            }
            st.rerun()


def render_level_0_company(df_all, dna_result):
    """Level 0: Company Pipeline Demand - surgical deficit by task."""
    st.markdown("## 🏢 Company Surgical Pipeline")
    st.caption("Total implied demand vs. roster capability, segmented by task/skill")
    
    # Prepare data
    active_jobs_list = get_active_jobs(df_all)
    if len(active_jobs_list) == 0:
        st.warning("No active jobs found.")
        return
    
    df_active = df_all[df_all['job_no'].isin(active_jobs_list)].copy()
    
    # Aggregate to job level
    job_agg = df_active.groupby(['job_no', 'category_rev_job']).agg(
        actual_hours=('hours_raw', 'sum'),
    ).reset_index()
    
    # Get benchmarks
    benchmarks = dna_result['benchmarks']
    job_agg = job_agg.merge(benchmarks[['category_rev_job', 'median_total_hours']], 
                             on='category_rev_job', how='left')
    job_agg['estimated_total'] = job_agg['median_total_hours'].fillna(job_agg['actual_hours'])
    
    # Compute gaps
    gaps = compute_surgical_gaps(
        job_agg,
        dna_result['dna_profiles'],
        benchmarks,
        st.session_state.get('fte_capabilities', pd.DataFrame()),
    )
    
    gaps_df = gaps['surgical_gaps']
    
    # KPI strip
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Demand", fmt_hours(gaps['total_demand']))
    with col2:
        st.metric("Active Jobs", fmt_count(len(active_jobs_list)))
    with col3:
        st.metric("Roster FTEs", fmt_count(gaps['total_supply_ftes']))
    with col4:
        demand_ftes = gaps['total_demand'] / (config.CAPACITY_HOURS_PER_WEEK * st.session_state['horizon_weeks'])
        st.metric("Implied FTEs", f"{demand_ftes:.1f}")
    
    st.divider()
    
    # Surgical gaps chart
    if len(gaps_df) > 0:
        st.markdown("### 🔴 Surgical Gaps by Task (Demand > Supply)")
        
        gaps_sorted = gaps_df.sort_values('gap_hours', ascending=True).tail(10)  # Top 10 gaps
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=gaps_sorted['task_name'],
            x=gaps_sorted['gap_hours'],
            orientation='h',
            marker_color='#e45756',
            name='Gap (Hours)',
        ))
        fig.update_layout(
            title="Top 10 Task Gaps (Demand > Supply)",
            xaxis_title="Gap Hours",
            yaxis_title="Task",
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table view
        st.markdown("### 📊 All Task Gaps")
        display_df = gaps_df[['task_name', 'total_demand_hours', 'total_supply_hours', 
                               'gap_hours', 'capable_ftes', 'job_count']].copy()
        display_df.columns = ['Task', 'Demand (hrs)', 'Supply (hrs)', 'Gap (hrs)', 'Capable FTEs', 'Jobs']
        display_df = display_df.round(1)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Allow drill to department
        st.markdown("### 🔍 Drill to Department")
        departments = ["All"] + sorted([d for d in df_all['department_final'].unique() if pd.notna(d)])
        selected_dept = st.selectbox("Select Department", departments, key="level0_dept_select")
        
        if selected_dept != "All":
            drill = st.session_state['empirical_drill'].copy()
            drill['level'] = 1
            drill['dept'] = selected_dept
            st.session_state['empirical_drill'] = drill
            
            if st.button("→ Drill to Department", key="drill_to_dept"):
                st.rerun()
    else:
        st.info("No surgical gaps detected. Supply meets demand.")


def render_level_1_department(df_all, dna_result):
    """Level 1: Departmental Contribution - dept-level gaps."""
    drill = st.session_state['empirical_drill']
    dept = drill['dept']
    
    st.markdown(f"## 📁 Department: {dept}")
    st.caption("Task demand and FTE allocation in this department")
    
    df_dept = df_all[df_all['department_final'] == dept]
    active_jobs = get_active_jobs(df_dept)
    
    if len(active_jobs) == 0:
        st.warning(f"No active jobs in {dept}")
        return
    
    df_active = df_dept[df_dept['job_no'].isin(active_jobs)].copy()
    
    # Aggregate by category
    job_agg = df_active.groupby(['job_no', 'category_rev_job']).agg(
        actual_hours=('hours_raw', 'sum'),
    ).reset_index()
    
    benchmarks = dna_result['benchmarks']
    job_agg = job_agg.merge(benchmarks[['category_rev_job', 'median_total_hours']], 
                             on='category_rev_job', how='left')
    
    category_summary = job_agg.groupby('category_rev_job').agg(
        job_count=('job_no', 'nunique'),
        total_actual_hours=('actual_hours', 'sum'),
        median_total_hours=('median_total_hours', 'first'),
    ).reset_index()
    
    category_summary['total_estimated'] = category_summary['median_total_hours'] * category_summary['job_count']
    category_summary['remaining_hours'] = (category_summary['total_estimated'] - 
                                          category_summary['total_actual_hours']).clip(lower=0)
    
    st.dataframe(category_summary, use_container_width=True, hide_index=True)
    
    # Allow drill to category
    st.markdown("### 🔍 Drill to Category")
    categories = ["All"] + sorted([c for c in df_active['category_rev_job'].unique() if pd.notna(c)])
    selected_cat = st.selectbox("Select Category", categories, key="level1_cat_select")
    
    if selected_cat != "All":
        drill['level'] = 2
        drill['category'] = selected_cat
        st.session_state['empirical_drill'] = drill
        
        if st.button("→ Drill to Category", key="drill_to_cat"):
            st.rerun()


def render_level_2_category(df_all, dna_result):
    """Level 2: Category Benchmarking - historical DNA vs. current jobs."""
    drill = st.session_state['empirical_drill']
    dept = drill['dept']
    cat = drill['category']
    
    st.markdown(f"## 📂 Category: {dept} › {cat}")
    st.caption("Historical task distribution profile vs. current active jobs")
    
    df_cat = df_all[(df_all['department_final'] == dept) & (df_all['category_rev_job'] == cat)]
    active_jobs = get_active_jobs(df_cat)
    
    if len(active_jobs) == 0:
        st.warning(f"No active jobs in {cat}")
        return
    
    # Show DNA profile (historical task distribution)
    dna_df = dna_result['dna_profiles']
    cat_dna = dna_df[dna_df['category_rev_job'] == cat]
    
    if len(cat_dna) > 0:
        st.markdown("### 📖 Historical Task Distribution (DNA Profile)")
        
        # Pivot to show quintile x task
        dna_pivot = cat_dna.pivot_table(
            index='task_name',
            columns='quintile',
            values='median_task_pct',
            aggfunc='first',
        )
        st.dataframe(dna_pivot.fillna(0).round(1), use_container_width=True)
    
    # Show current active jobs
    st.markdown("### 🎯 Current Active Jobs")
    
    df_active = df_cat[df_cat['job_no'].isin(active_jobs)].copy()
    
    benchmarks = dna_result['benchmarks']
    bench_row = benchmarks[benchmarks['category_rev_job'] == cat]
    
    if len(bench_row) > 0:
        benchmark_hrs = bench_row['median_total_hours'].iloc[0]
        
        jobs_list = []
        for job_id in active_jobs:
            df_job = df_active[df_active['job_no'] == job_id]
            actual_hrs = df_job['hours_raw'].sum()
            maturity = compute_job_maturity(df_job, benchmark_hrs)
            
            jobs_list.append({
                'job_id': job_id,
                'actual_hours': actual_hrs,
                'maturity_pct': maturity['maturity_pct'],
                'remaining_hours': maturity['remaining_hours'],
            })
        
        jobs_df = pd.DataFrame(jobs_list)
        
        # Allow drill to job
        st.markdown("### 🔍 Drill to Job")
        job_options = jobs_df['job_id'].tolist()
        selected_job = st.selectbox("Select Job", job_options, key="level2_job_select")
        
        if selected_job:
            drill['level'] = 3
            drill['job_id'] = selected_job
            st.session_state['empirical_drill'] = drill
            
            if st.button("→ Drill to Job", key="drill_to_job"):
                st.rerun()


def render_level_3_job(df_all, dna_result):
    """Level 3: Job Maturity Card - single job health & next phase warning."""
    drill = st.session_state['empirical_drill']
    dept = drill['dept']
    cat = drill['category']
    job_id = drill['job_id']
    
    st.markdown(f"## 🎯 Job Health: {job_id}")
    st.caption(f"{dept} › {cat} › Job {job_id}")
    
    df_job = df_all[df_all['job_no'] == job_id]
    
    if len(df_job) == 0:
        st.error(f"Job {job_id} not found")
        return
    
    benchmarks = dna_result['benchmarks']
    bench_row = benchmarks[benchmarks['category_rev_job'] == cat]
    
    if len(bench_row) == 0:
        st.error(f"No benchmark for {cat}")
        return
    
    benchmark_hrs = bench_row['median_total_hours'].iloc[0]
    maturity = compute_job_maturity(df_job, benchmark_hrs)
    
    # Health card
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Maturity", f"{maturity['maturity_pct']:.1f}%")
    with col2:
        st.metric("Actual Hours", fmt_hours(maturity['actual_hours']))
    with col3:
        st.metric("Remaining Hours", fmt_hours(maturity['remaining_hours']))
    with col4:
        status = "🟢 On Track" if maturity['maturity_pct'] < 100 else "🔴 Overrun"
        st.metric("Status", status)
    
    st.divider()
    
    # Task breakdown
    st.markdown("### 📋 Task Breakdown")
    
    task_summary = df_job.groupby('task_name').agg(
        actual_hours=('hours_raw', 'sum'),
    ).reset_index()
    
    st.dataframe(task_summary, use_container_width=True, hide_index=True)
    
    # Next phase warning (using DNA)
    st.markdown("### ⚠️ Next Phase Warning")
    
    dna_df = dna_result['dna_profiles']
    cat_dna = dna_df[dna_df['category_rev_job'] == cat]
    
    if len(cat_dna) > 0 and maturity['maturity_pct'] < 100:
        current_quintile_idx = int(maturity['maturity_pct'] // 20)
        if current_quintile_idx < 4:
            next_quintile = f"Q{current_quintile_idx + 2} ({(current_quintile_idx + 1) * 20 + 1}%-{(current_quintile_idx + 2) * 20}%)"
            next_tasks = cat_dna[cat_dna['quintile'].str.contains(f"Q{current_quintile_idx + 2}")]
            
            if len(next_tasks) > 0:
                st.info(f"**Next phase: {next_quintile}**\n\nHistorically, this phase requires:")
                for _, task_row in next_tasks.iterrows():
                    st.write(f"- {task_row['task_name']}: ~{task_row['median_task_pct']:.0f}% of phase hours")


def render_level_4_intervention(df_all, dna_result):
    """Level 4: Surgical Intervention - assign FTE to bottleneck task."""
    drill = st.session_state['empirical_drill']
    job_id = drill['job_id']
    
    st.markdown(f"## 🔧 Surgical Intervention for Job {job_id}")
    st.caption("Assign FTE capacity to accelerate job completion")
    
    df_job = df_all[df_all['job_no'] == job_id]
    
    # Get bottleneck tasks (highest remaining hours with low velocity)
    task_summary = df_job.groupby('task_name').agg(
        actual_hours=('hours_raw', 'sum'),
        staff_count=('staff_name', 'nunique'),
    ).reset_index()
    
    st.markdown("### 📊 Task Status")
    st.dataframe(task_summary, use_container_width=True, hide_index=True)
    
    # Allow selection of task to intervene
    st.markdown("### 🎯 Select Task to Intervene")
    task_options = task_summary['task_name'].tolist()
    selected_task = st.selectbox("Task", task_options, key="level4_task_select")
    
    if selected_task:
        drill['level'] = 4
        drill['task_id'] = selected_task
        st.session_state['empirical_drill'] = drill
        
        # Show FTE recommendations
        st.markdown(f"### 💼 Available FTEs for {selected_task}")
        
        fte_caps = st.session_state.get('fte_capabilities', pd.DataFrame())
        task_capable = fte_caps[fte_caps['task_name'] == selected_task]
        
        if len(task_capable) > 0:
            st.dataframe(task_capable[['staff_name', 'proficiency_level', 'total_hours']], 
                        use_container_width=True, hide_index=True)
        else:
            st.warning(f"No staff with {selected_task} capability found in last 6 months.")


def main():
    """Main page controller."""
    init_drill_state()
    
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Configuration")
    st.session_state['lookback_jobs'] = st.sidebar.slider(
        "Lookback completed jobs (DNA)", 5, 50, 30, step=5
    )
    st.session_state['horizon_weeks'] = st.sidebar.slider(
        "Forecast horizon (weeks)", 4, 52, 12, step=4
    )
    st.session_state['recency_months'] = st.sidebar.slider(
        "FTE skill recency (months)", 1, 12, 6, step=1
    )
    
    # Load data
    st.markdown("## 🔮 Empirical Lifecycle & Surgical Task Forecaster")
    st.caption("Forecast job completion by mapping 'remaining DNA' of active jobs against historical task profiles")
    
    # Progress indicator
    with st.spinner("Loading data..."):
        df_raw = load_fact_timesheet()
        
        # Filter to completed jobs for DNA building
        df_completed = df_raw[df_raw['job_completed_date'].notna()].copy()
        
        # Build DNA profiles (cache this)
        dna_result = build_lifecycle_dna(df_completed, st.session_state['lookback_jobs'])
        
        # Build FTE capabilities
        df_recent = filter_by_time_window(df_raw, f"{st.session_state['recency_months']}m", date_col='month_key')
        fte_capabilities = badge_fte_capabilities(df_recent, st.session_state['recency_months'])
        st.session_state['fte_capabilities'] = fte_capabilities
        
        # Render drill breadcrumb and controls
        drill = st.session_state['empirical_drill']
        render_breadcrumb(drill)
        render_drill_controls(drill)
        
        st.divider()
        
        # Show errors from DNA building
        if dna_result['errors']:
            with st.expander(f"⚠️ {len(dna_result['errors'])} categories with insufficient data"):
                for err in dna_result['errors']:
                    st.caption(f"- {err}")
        
        # Render level-specific content
        if drill['level'] == 0:
            render_level_0_company(df_raw, dna_result)
        elif drill['level'] == 1:
            render_level_1_department(df_raw, dna_result)
        elif drill['level'] == 2:
            render_level_2_category(df_raw, dna_result)
        elif drill['level'] == 3:
            render_level_3_job(df_raw, dna_result)
        elif drill['level'] == 4:
            render_level_4_intervention(df_raw, dna_result)


if __name__ == "__main__":
    main()
