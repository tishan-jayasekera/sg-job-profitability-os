"""
Remaining work forecast based on empirical benchmarks.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple


def forecast_remaining_work(
    df_active: pd.DataFrame,
    benchmarks: pd.DataFrame,
    task_mix: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each active job, compute remaining task hours based on benchmark shape.
    Returns a task-level remaining dataframe.
    """
    if len(df_active) == 0:
        return pd.DataFrame()

    job_meta = df_active.groupby("job_no").agg(
        department_final=("department_final", "first"),
        category_rev_job=("category_rev_job", "first"),
    ).reset_index()

    actual_task = df_active.groupby(["job_no", "task_name"]).agg(
        actual_task_hours=("hours_raw", "sum"),
    ).reset_index()

    job_actual_total = df_active.groupby("job_no")["hours_raw"].sum().rename("actual_job_hours")
    job_meta = job_meta.merge(job_actual_total.reset_index(), on="job_no", how="left")

    bench = benchmarks.rename(columns={"category_rev_job": "category_rev_job"})
    job_meta = job_meta.merge(
        bench[["department_final", "category_rev_job", "job_total_hours_p50"]],
        on=["department_final", "category_rev_job"],
        how="left",
    )
    job_meta["projected_eac_hours"] = job_meta["job_total_hours_p50"].fillna(job_meta["actual_job_hours"])

    task_mix_use = task_mix.copy()
    remaining_rows = []

    for _, job in job_meta.iterrows():
        job_no = job["job_no"]
        dept = job["department_final"]
        cat = job["category_rev_job"]
        projected_total = job["projected_eac_hours"]

        mix = task_mix_use[
            (task_mix_use["department_final"] == dept) &
            (task_mix_use["category_rev_job"] == cat)
        ]
        actual = actual_task[actual_task["job_no"] == job_no]

        if len(mix) == 0:
            # Fallback: use actual task share so far
            actual_sum = actual["actual_task_hours"].sum()
            if actual_sum > 0:
                mix = actual.copy()
                mix["task_share_pct"] = mix["actual_task_hours"] / actual_sum * 100
                mix = mix[["task_name", "task_share_pct"]]
            else:
                continue

        for _, t in mix.iterrows():
            task_name = t["task_name"]
            expected_task = projected_total * (t["task_share_pct"] / 100.0) if pd.notna(t["task_share_pct"]) else np.nan
            actual_task_hours = actual.loc[actual["task_name"] == task_name, "actual_task_hours"]
            actual_task_hours = actual_task_hours.iloc[0] if len(actual_task_hours) > 0 else 0.0
            remaining = max((expected_task or 0) - actual_task_hours, 0.0) if pd.notna(expected_task) else 0.0
            overrun = actual_task_hours > (expected_task or 0) if pd.notna(expected_task) else False

            remaining_rows.append({
                "job_no": job_no,
                "department_final": dept,
                "category_rev_job": cat,
                "task_name": task_name,
                "projected_eac_hours": projected_total,
                "expected_task_hours": expected_task,
                "actual_task_hours": actual_task_hours,
                "remaining_task_hours": remaining,
                "is_overrun": overrun,
            })

    return pd.DataFrame(remaining_rows)


def solve_bottlenecks(
    remaining_df: pd.DataFrame,
    velocity_df: pd.DataFrame,
    df_jobs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge remaining work with team velocity to compute ETAs and bottlenecks.
    Returns:
        task_level: remaining + velocity + ETA + action flags
        job_level: job ETA and status
    """
    if len(remaining_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    merged = remaining_df.merge(
        velocity_df,
        on=["job_no", "task_name"],
        how="left",
    )
    merged["team_velocity_hours_week"] = merged["team_velocity_hours_week"].fillna(0)
    merged["weeks_to_complete"] = np.where(
        merged["team_velocity_hours_week"] > 0,
        merged["remaining_task_hours"] / merged["team_velocity_hours_week"],
        np.inf,
    )
    merged["is_bottleneck"] = (merged["remaining_task_hours"] > 0) & (merged["team_velocity_hours_week"] == 0)

    job_eta = merged.groupby("job_no")["weeks_to_complete"].max().rename("job_eta_weeks")
    job_level = df_jobs.drop_duplicates(subset=["job_no"]).merge(job_eta.reset_index(), on="job_no", how="left")

    if "job_due_date" in df_jobs.columns:
        due = df_jobs.groupby("job_no")["job_due_date"].first()
        job_level = job_level.merge(due.reset_index(), on="job_no", how="left")
        job_level["job_due_date"] = pd.to_datetime(job_level["job_due_date"], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        job_level["due_weeks"] = (job_level["job_due_date"] - now).dt.days / 7
    else:
        job_level["due_weeks"] = np.nan

    def _status(row: pd.Series) -> str:
        if pd.isna(row["job_eta_weeks"]):
            return "Unknown"
        if np.isinf(row["job_eta_weeks"]):
            return "Blocked"
        if pd.notna(row["due_weeks"]) and row["job_eta_weeks"] > row["due_weeks"]:
            return "At Risk"
        return "On Track"

    job_level["status"] = job_level.apply(_status, axis=1)
    return merged, job_level


def compute_risk_score(due_weeks: float, eta_weeks: float) -> float:
    """
    Compute risk score for a job in range [0, 1.0].
    
    Formula:
        risk_score = 1.0 - max(0, (due_weeks - eta_weeks) / due_weeks)
    
    Interpretation:
        - 0.0: On-track (plenty of buffer)
        - 0.2-0.5: Yellow (moderate caution)
        - 0.5-0.8: High risk (tight timeline)
        - 0.8-1.0: Critical risk (overdue or very tight)
        - NaN: Due date or ETA missing
    
    Args:
        due_weeks: Weeks until due date (now to due_date)
        eta_weeks: Forecasted weeks to completion (ETA)
        
    Returns:
        risk_score: Float in [0, 1.0] or NaN if data missing
    """
    if due_weeks is None or eta_weeks is None:
        return np.nan
    if pd.isna(due_weeks) or pd.isna(eta_weeks):
        return np.nan
    if due_weeks <= 0:
        return 1.0  # Already overdue = critical risk
    if np.isinf(eta_weeks):
        return 1.0  # Infinite ETA (blocked) = critical risk
    
    buffer = due_weeks - eta_weeks
    score = 1.0 - max(0.0, buffer / due_weeks)
    return float(min(max(score, 0.0), 1.0))


def compute_risk_scores_for_jobs(job_level: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized risk scoring for all jobs.
    
    Args:
        job_level: DataFrame with 'due_weeks' and 'job_eta_weeks' columns
        
    Returns:
        Same DataFrame with added 'risk_score' column
    """
    job_level = job_level.copy()
    job_level["risk_score"] = job_level.apply(
        lambda row: compute_risk_score(row.get("due_weeks"), row.get("job_eta_weeks")),
        axis=1
    )
    return job_level

# ============================================================================
# PHASE 1B: 5-LEVEL DRILL-CHAIN SCOPING FUNCTIONS
# ============================================================================

def translate_job_state(
    risk_score: float,
    due_weeks: float,
    eta_weeks: float,
    velocity: float,
) -> Tuple[str, str]:
    """
    Translate raw math (risk_score, due_weeks, eta_weeks) into human-readable states.
    
    Args:
        risk_score: Float [0, 1.0] or NaN
        due_weeks: Weeks until due (can be negative if overdue)
        eta_weeks: Forecasted weeks to completion (can be inf)
        velocity: Hours per week (can be 0)
        
    Returns:
        (status, label) tuple where:
        - status: "On-Track" | "At-Risk" | "Blocked" | "Overdue"
        - label: Human-readable explanation with emoji
    """
    # Handle special cases first
    if velocity == 0 and eta_weeks > 0:
        return ("Blocked", "🔴 No run-rate detected (0 hrs/week)")
    
    if pd.isna(due_weeks) or due_weeks < -52:  # Very far overdue
        return ("Overdue", "🔴 Severely overdue")
    
    if due_weeks < 0:
        return ("Overdue", f"🔴 Overdue by {-due_weeks:.1f} weeks")
    
    # Use risk_score to classify
    if pd.isna(risk_score):
        return ("Unknown", "⚪ Insufficient data")
    
    if risk_score < 0.2:
        return ("On-Track", f"🟢 On track (buffer: {due_weeks:.1f}w)")
    elif risk_score < 0.7:
        return ("At-Risk", f"🟡 At risk (buffer: {due_weeks:.1f}w)")
    else:
        return ("Blocked", f"🔴 Critical (buffer: {due_weeks:.1f}w)")


def get_company_forecast(
    job_level: pd.DataFrame,
    horizon_weeks: int = 12,
) -> dict:
    """
    Company-level summary: demand, capacity, gap by department.
    
    Args:
        job_level: Full job-level forecast dataframe
        horizon_weeks: Forecast horizon in weeks
        
    Returns:
        dict with:
        - 'total_demand_hours': Sum of remaining hours
        - 'total_capacity_hours': Team capacity in hours
        - 'gap_hours': Capacity - Demand
        - 'gap_fte': Gap in FTE equivalent
        - 'dept_breakdown': DataFrame with dept-level metrics sorted by gap
    """
    if len(job_level) == 0:
        return {
            'total_demand_hours': 0,
            'total_capacity_hours': 0,
            'gap_hours': 0,
            'gap_fte': 0,
            'dept_breakdown': pd.DataFrame(),
        }
    
    # Assume from config or calculate
    hours_per_week = 38
    team_ftes = 1  # Placeholder; should come from config
    
    total_demand = job_level['remaining_hours'].sum()
    total_capacity = team_ftes * hours_per_week * horizon_weeks
    gap = total_capacity - total_demand
    gap_fte = gap / (hours_per_week * horizon_weeks) if hours_per_week * horizon_weeks > 0 else 0
    
    # Department breakdown
    dept_breakdown = job_level.groupby('department_final').agg(
        demand_hours=('remaining_hours', 'sum'),
        at_risk_count=('risk_score', lambda x: (x > 0.7).sum()),
        avg_risk_score=('risk_score', 'mean'),
    ).reset_index()
    
    dept_breakdown['capacity_hours'] = team_ftes * hours_per_week * horizon_weeks / len(dept_breakdown)
    dept_breakdown['gap_hours'] = dept_breakdown['capacity_hours'] - dept_breakdown['demand_hours']
    dept_breakdown['gap_pct'] = (dept_breakdown['gap_hours'] / dept_breakdown['demand_hours'] * 100).round(1)
    dept_breakdown = dept_breakdown.sort_values('gap_hours')
    
    return {
        'total_demand_hours': round(total_demand, 1),
        'total_capacity_hours': round(total_capacity, 1),
        'gap_hours': round(gap, 1),
        'gap_fte': round(gap_fte, 2),
        'dept_breakdown': dept_breakdown,
    }


def get_dept_forecast(
    job_level: pd.DataFrame,
    dept: str,
    horizon_weeks: int = 12,
) -> dict:
    """
    Department-level summary: demand, capacity, gap by category.
    
    Args:
        job_level: Full job-level forecast dataframe
        dept: Department name filter
        horizon_weeks: Forecast horizon in weeks
        
    Returns:
        dict with dept-scoped metrics and category breakdown
    """
    dept_jobs = job_level[job_level['department_final'] == dept]
    
    if len(dept_jobs) == 0:
        return {
            'dept_demand_hours': 0,
            'dept_capacity_hours': 0,
            'dept_gap_hours': 0,
            'at_risk_count': 0,
            'category_breakdown': pd.DataFrame(),
        }
    
    hours_per_week = 38
    team_ftes = 1
    
    dept_demand = dept_jobs['remaining_hours'].sum()
    dept_capacity = team_ftes * hours_per_week * horizon_weeks
    dept_gap = dept_capacity - dept_demand
    
    # Category breakdown
    cat_breakdown = dept_jobs.groupby('category_rev_job').agg(
        job_count=('job_no', 'nunique'),
        demand_hours=('remaining_hours', 'sum'),
        avg_risk_score=('risk_score', 'mean'),
        at_risk_count=('risk_score', lambda x: (x > 0.7).sum()),
    ).reset_index()
    
    cat_breakdown['capacity_hours'] = team_ftes * hours_per_week * horizon_weeks / len(cat_breakdown)
    cat_breakdown['gap_hours'] = cat_breakdown['capacity_hours'] - cat_breakdown['demand_hours']
    cat_breakdown = cat_breakdown.sort_values('gap_hours')
    
    return {
        'dept_demand_hours': round(dept_demand, 1),
        'dept_capacity_hours': round(dept_capacity, 1),
        'dept_gap_hours': round(dept_gap, 1),
        'at_risk_count': int((dept_jobs['risk_score'] > 0.7).sum()),
        'category_breakdown': cat_breakdown,
    }


def get_category_jobs(
    job_level: pd.DataFrame,
    dept: str,
    category: str,
) -> pd.DataFrame:
    """
    Category-level job list: all active jobs in this category, ranked by urgency.
    
    Args:
        job_level: Full job-level forecast dataframe
        dept: Department name filter
        category: Category name filter
        
    Returns:
        Filtered and sorted job-level dataframe (worst jobs first)
    """
    cat_jobs = job_level[
        (job_level['department_final'] == dept) &
        (job_level['category_rev_job'] == category)
    ].copy()
    
    if len(cat_jobs) == 0:
        return pd.DataFrame()
    
    # Sort by: overdue first, then high remaining, then low velocity
    cat_jobs['sort_key_overdue'] = (cat_jobs['due_weeks'] < 0).astype(int)
    cat_jobs = cat_jobs.sort_values(
        by=['sort_key_overdue', 'remaining_hours', 'job_velocity_hrs_week'],
        ascending=[False, False, True]
    ).drop('sort_key_overdue', axis=1)
    
    return cat_jobs


def get_job_tasks(
    remaining_by_task: pd.DataFrame,
    job_id: int,
    min_hours: float = 5.0,
) -> pd.DataFrame:
    """
    Job-level task list: all tasks for a job, ranked by urgency (no math artifacts).
    
    Args:
        remaining_by_task: Task-level dataframe from forecast_remaining_work()
        job_id: Job number to filter
        min_hours: Hide tasks with remaining < this threshold
        
    Returns:
        Filtered and sorted task-level dataframe
    """
    job_tasks = remaining_by_task[remaining_by_task['job_no'] == job_id].copy()
    
    if len(job_tasks) == 0:
        return pd.DataFrame()
    
    # Filter out negligible tasks by default (can be shown via toggle)
    job_tasks['is_negligible'] = job_tasks['remaining_task_hours'] < min_hours
    
    # Sort: blocked > at-risk > on-track, then by remaining (desc), then velocity (asc)
    # Status: "Blocked" if velocity=0, "At-Risk" if remaining > est, "On-Track" otherwise
    job_tasks['est_hours'] = job_tasks['task_hours_p50']
    job_tasks['is_blocked'] = job_tasks['task_velocity_hrs_week'] == 0
    job_tasks['is_at_risk'] = (
        (job_tasks['remaining_task_hours'] > job_tasks['est_hours']) &
        (job_tasks['task_velocity_hrs_week'] > 0)
    )
    
    job_tasks['sort_blocked'] = job_tasks['is_blocked'].astype(int)
    job_tasks['sort_at_risk'] = job_tasks['is_at_risk'].astype(int)
    
    job_tasks = job_tasks.sort_values(
        by=['sort_blocked', 'sort_at_risk', 'remaining_task_hours', 'task_velocity_hrs_week'],
        ascending=[False, False, False, True]
    ).drop(['sort_blocked', 'sort_at_risk'], axis=1)
    
    return job_tasks


# ============================================================================
# EMPIRICAL LIFECYCLE FORECASTING: DNA PROFILER, MATURITY, SKILLS, GAPS
# ============================================================================

def build_lifecycle_dna(
    df_completed: pd.DataFrame,
    lookback_jobs: int = 30,
) -> dict:
    """
    Build historical lifecycle DNA profiles from completed jobs.
    
    For each (department, category):
    1. Identify last X completed jobs
    2. Normalize to 0-100% completion timeline using median total hours
    3. Divide into quintiles (0-20%, 21-40%, etc.)
    4. Compute median task mix per quintile
    5. Identify staff roles executing each task
    
    Args:
        df_completed: Filtered to completed jobs only
        lookback_jobs: Number of recent completed jobs per category (default 30)
        
    Returns:
        dict with keys:
        - 'dna_profiles': DataFrame (category_key, quintile, task_name, median_task_pct, staff_roles)
        - 'benchmarks': DataFrame (category_key, median_total_hours, p25_hours, p75_hours)
        - 'errors': List of categories with insufficient data
    """
    if len(df_completed) == 0:
        return {'dna_profiles': pd.DataFrame(), 'benchmarks': pd.DataFrame(), 'errors': []}
    
    df_completed = df_completed[df_completed['job_no'].notna()].copy()
    
    # Filter out records with missing category
    df_completed = df_completed[df_completed['category_rev_job'].notna()].copy()
    
    # Group by category
    categories = df_completed.groupby('category_rev_job')['job_no'].nunique().reset_index()
    categories.columns = ['category_rev_job', 'job_count']
    
    dna_rows = []
    benchmark_rows = []
    errors = []
    
    for _, row in categories.iterrows():
        cat = row['category_rev_job']
        
        # Get last X completed jobs
        cat_jobs = df_completed[df_completed['category_rev_job'] == cat]['job_no'].unique()
        if len(cat_jobs) < 3:  # Require at least 3 jobs for reliable profile
            errors.append(f"{cat} (only {len(cat_jobs)} completed jobs)")
            continue
        
        sample_jobs = cat_jobs[-lookback_jobs:]  # Last X jobs
        df_sample = df_completed[df_completed['job_no'].isin(sample_jobs)]
        
        # Calculate job totals for each job
        job_totals = df_sample.groupby('job_no')['hours_raw'].sum()
        median_total_hours = job_totals.median()
        p25_total_hours = job_totals.quantile(0.25)
        p75_total_hours = job_totals.quantile(0.75)
        
        if median_total_hours <= 0:
            errors.append(f"{cat} (median hours = {median_total_hours})")
            continue
        
        # For each job, compute maturity % and assign to quintile
        df_sample['maturity_pct'] = (df_sample.groupby('job_no')['hours_raw'].cumsum() / 
                                      df_sample['job_no'].map(job_totals)) * 100
        df_sample['quintile'] = pd.cut(
            df_sample['maturity_pct'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Q1 (0-20%)', 'Q2 (21-40%)', 'Q3 (41-60%)', 'Q4 (61-80%)', 'Q5 (81-100%)'],
            include_lowest=True,
        )
        
        # Task mix by quintile
        task_mix = df_sample.groupby(['quintile', 'task_name']).agg(
            hours=('hours_raw', 'sum'),
            staff_roles=('staff_name', lambda x: ', '.join(x.unique())),
        ).reset_index()
        
        total_by_q = task_mix.groupby('quintile')['hours'].sum()
        task_mix['median_task_pct'] = task_mix.apply(
            lambda r: (r['hours'] / total_by_q.get(r['quintile'], 1)) * 100, axis=1
        )
        
        task_mix['category_rev_job'] = cat
        dna_rows.append(task_mix[['category_rev_job', 'quintile', 'task_name', 'median_task_pct', 'staff_roles']])
        
        # Benchmark row
        benchmark_rows.append({
            'category_rev_job': cat,
            'job_count_in_sample': len(sample_jobs),
            'median_total_hours': round(median_total_hours, 1),
            'p25_total_hours': round(p25_total_hours, 1),
            'p75_total_hours': round(p75_total_hours, 1),
        })
    
    dna_df = pd.concat(dna_rows, ignore_index=True) if dna_rows else pd.DataFrame()
    bench_df = pd.DataFrame(benchmark_rows)
    
    return {
        'dna_profiles': dna_df,
        'benchmarks': bench_df,
        'errors': errors,
    }


def compute_job_maturity(
    df_job: pd.DataFrame,
    benchmark_total_hours: float,
) -> dict:
    """
    Compute maturity % for a single active job.
    
    Args:
        df_job: Filtered to single job_no
        benchmark_total_hours: Median total hours from category benchmark (p50)
        
    Returns:
        dict with:
        - 'maturity_pct': Actual hours to date / benchmark (%)
        - 'actual_hours': Total hours to date
        - 'estimated_total': Benchmark hours
        - 'remaining_hours': max(0, estimated - actual)
        - 'confidence_low': maturity using p25 benchmark
        - 'confidence_high': maturity using p75 benchmark
    """
    actual_hours = df_job['hours_raw'].sum()
    
    if benchmark_total_hours <= 0:
        return {
            'maturity_pct': 0,
            'actual_hours': actual_hours,
            'estimated_total': np.nan,
            'remaining_hours': np.nan,
            'confidence_low': np.nan,
            'confidence_high': np.nan,
        }
    
    maturity = (actual_hours / benchmark_total_hours) * 100
    remaining = max(0, benchmark_total_hours - actual_hours)
    
    return {
        'maturity_pct': round(maturity, 1),
        'actual_hours': round(actual_hours, 1),
        'estimated_total': round(benchmark_total_hours, 1),
        'remaining_hours': round(remaining, 1),
        'confidence_low': round((actual_hours / (benchmark_total_hours * 1.5)) * 100, 1),  # P75 scenario
        'confidence_high': round((actual_hours / (benchmark_total_hours * 0.67)) * 100, 1),  # P25 scenario
    }


def badge_fte_capabilities(
    df_recent: pd.DataFrame,
    recency_months: int = 6,
) -> pd.DataFrame:
    """
    Badge current FTEs with skills based on last N months of logged hours.
    
    Args:
        df_recent: Filtered to recent months (use filter_by_time_window(df, '6m'))
        recency_months: Lookback window
        
    Returns:
        DataFrame with columns:
        - 'staff_name'
        - 'task_name' (each row is a staff-task combination)
        - 'total_hours'
        - 'avg_hours_per_month'
        - 'proficiency_level' ('Novice' if <20h, 'Intermediate' if 20-100h, 'Expert' if >100h)
    """
    if len(df_recent) == 0:
        return pd.DataFrame()
    
    # Filter out records with missing staff or task names
    df_recent = df_recent[df_recent['staff_name'].notna() & df_recent['task_name'].notna()].copy()
    
    if len(df_recent) == 0:
        return pd.DataFrame()
    
    staff_task = df_recent.groupby(['staff_name', 'task_name']).agg(
        total_hours=('hours_raw', 'sum'),
    ).reset_index()
    
    staff_task['avg_hours_per_month'] = staff_task['total_hours'] / recency_months
    
    def _proficiency(hours):
        if hours < 20:
            return 'Novice'
        elif hours < 100:
            return 'Intermediate'
        else:
            return 'Expert'
    
    staff_task['proficiency_level'] = staff_task['total_hours'].apply(_proficiency)
    
    return staff_task.sort_values(['staff_name', 'total_hours'], ascending=[True, False])


def compute_surgical_gaps(
    active_jobs: pd.DataFrame,
    dna_profiles: pd.DataFrame,
    benchmarks: pd.DataFrame,
    fte_capabilities: pd.DataFrame,
) -> dict:
    """
    Compute surgical gaps: where is demand > supply by task/skill?
    
    Args:
        active_jobs: Job-level dataframe with job_no, category_rev_job, actual_hours
        dna_profiles: Output from build_lifecycle_dna()['dna_profiles']
        benchmarks: Output from build_lifecycle_dna()['benchmarks']
        fte_capabilities: Output from badge_fte_capabilities()
        
    Returns:
        dict with:
        - 'surgical_gaps': DataFrame (task_name, demand_hours, supply_ftes, gap_hours, gap_rank)
        - 'total_demand': Sum of remaining hours
        - 'total_supply_ftes': Count of unique staff
    """
    if len(active_jobs) == 0 or len(dna_profiles) == 0:
        return {
            'surgical_gaps': pd.DataFrame(),
            'total_demand': 0,
            'total_supply_ftes': 0,
        }
    
    # Filter out jobs with missing required fields
    active_jobs = active_jobs[
        active_jobs['job_no'].notna() & 
        active_jobs['category_rev_job'].notna() & 
        active_jobs['actual_hours'].notna()
    ].copy()
    
    if len(active_jobs) == 0:
        return {
            'surgical_gaps': pd.DataFrame(),
            'total_demand': 0,
            'total_supply_ftes': len(fte_capabilities['staff_name'].unique()) if len(fte_capabilities) > 0 else 0,
        }
    
    # For each active job, compute remaining task demand
    remaining_demand = []
    
    for _, job in active_jobs.iterrows():
        job_no = job['job_no']
        cat = job['category_rev_job']
        actual_hrs = job['actual_hours']
        
        # Get benchmark
        bench_row = benchmarks[benchmarks['category_rev_job'] == cat]
        if len(bench_row) == 0:
            continue
        
        benchmark_hrs = bench_row['median_total_hours'].iloc[0]
        maturity_pct = (actual_hrs / benchmark_hrs) * 100 if benchmark_hrs > 0 else 0
        
        # Get DNA tasks for remaining portion
        dna_cat = dna_profiles[dna_profiles['category_rev_job'] == cat]
        if len(dna_cat) == 0:
            continue
        
        # Determine which quintile range we're in and use remaining portion
        for _, dna_row in dna_cat.iterrows():
            remaining_demand.append({
                'job_no': job_no,
                'task_name': dna_row['task_name'],
                'category_rev_job': cat,
                'maturity_pct': maturity_pct,
                'task_demand_hours': (benchmark_hrs - actual_hrs) * (dna_row['median_task_pct'] / 100),
                'staff_roles_historical': dna_row['staff_roles'],
            })
    
    if not remaining_demand:
        return {
            'surgical_gaps': pd.DataFrame(),
            'total_demand': 0,
            'total_supply_ftes': len(fte_capabilities['staff_name'].unique()),
        }
    
    remaining_df = pd.DataFrame(remaining_demand)
    
    # Aggregate by task
    task_demand = remaining_df.groupby('task_name').agg(
        total_demand_hours=('task_demand_hours', 'sum'),
        job_count=('job_no', 'nunique'),
    ).reset_index()
    
    # Supply by task (from FTE capabilities)
    task_supply = fte_capabilities.groupby('task_name').agg(
        capable_ftes=('staff_name', 'nunique'),
        total_supply_hours=('total_hours', 'sum'),
    ).reset_index()
    
    # Merge and compute gap
    gaps = task_demand.merge(task_supply, on='task_name', how='left')
    gaps['capable_ftes'] = gaps['capable_ftes'].fillna(0)
    gaps['gap_hours'] = gaps['total_demand_hours'] - gaps['total_supply_hours']
    gaps['gap_rank'] = gaps['gap_hours'].rank(ascending=False)
    
    gaps = gaps.sort_values('gap_hours', ascending=False)
    
    return {
        'surgical_gaps': gaps,
        'total_demand': round(task_demand['total_demand_hours'].sum(), 1),
        'total_supply_ftes': len(fte_capabilities['staff_name'].unique()),
    }
