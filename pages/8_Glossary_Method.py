"""
Glossary & Method Page

Stop metric drift with clear definitions and formulas.
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state
from src.ui.layout import section_header


st.set_page_config(page_title="Glossary & Method", page_icon="📖", layout="wide")

init_state()


def main():
    st.title("Glossary & Method")
    st.caption("Definitions, formulas, and methodology documentation")
    
    # =========================================================================
    # CANONICAL HIERARCHY
    # =========================================================================
    section_header("Canonical Drill Hierarchy")
    
    st.markdown("""
    All navigation and analysis follows this hierarchy:
    
    **Company → `department_final` → `category_rev_job` → `staff_name` → `breakdown` → `task_name`**
    
    Everything else (client, business unit, state, role, etc.) is **filter-only**.
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # CORE METRICS
    # =========================================================================
    section_header("Core Metric Definitions")
    
    st.markdown("""
    ### Revenue
    
    | Metric | Formula | Notes |
    |--------|---------|-------|
    | **Revenue** | `Σ rev_alloc` | Always use `rev_alloc`, never sum monthly revenue pools directly |
    | **Realised Rate** | `Σ rev_alloc / Σ hours_raw` | Revenue per hour worked |
    
    ### Costs
    
    | Metric | Formula | Notes |
    |--------|---------|-------|
    | **Cost** | `Σ base_cost` | Direct labour cost |
    | **Cost Rate** | `Σ base_cost / Σ hours_raw` | Cost per hour |
    
    ### Profitability
    
    | Metric | Formula | Notes |
    |--------|---------|-------|
    | **Margin** | `Revenue - Cost` | Gross margin |
    | **Margin %** | `Margin / Revenue × 100` | As percentage of revenue |
    
    ### Hours
    
    | Metric | Formula | Notes |
    |--------|---------|-------|
    | **Hours** | `Σ hours_raw` | Actual hours worked (can be negative for adjustments) |
    | **Billable Hours** | `Σ hours_raw WHERE is_billable = true` | Hours charged to clients |
    | **Non-Billable Hours** | `Σ hours_raw WHERE is_billable = false` | Internal hours |
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # QUOTE METRICS
    # =========================================================================
    section_header("Quote & Delivery Metrics")
    
    st.markdown("""
    ### Quote Aggregation Safety
    
    **Critical:** Quote fields (`quoted_time_total`, `quoted_amount_total`) repeat on every daily 
    fact row for the same job-task. You must **deduplicate at (job_no, task_name) level** before summing.
    
    ```python
    # CORRECT
    quote_total = df.groupby(['job_no', 'task_name'])['quoted_time_total'].first().sum()
    
    # WRONG (inflates by row count!)
    quote_total = df['quoted_time_total'].sum()
    ```
    
    ### Definitions
    
    | Metric | Formula | Notes |
    |--------|---------|-------|
    | **Quoted Hours** | `Σ quoted_time_total` (after job-task dedupe) | Total quoted hours |
    | **Quoted Amount** | `Σ quoted_amount_total` (after job-task dedupe) | Total quoted value |
    | **Quote Rate** | `Quoted Amount / Quoted Hours` | Quoted $/hr |
    | **Hours Variance** | `Actual Hours - Quoted Hours` | Overrun if positive |
    | **Hours Variance %** | `Hours Variance / Quoted Hours × 100` | As percentage |
    
    ### Scope Creep
    
    | Metric | Formula | Notes |
    |--------|---------|-------|
    | **Unquoted Hours** | `Σ hours_raw WHERE quote_match_flag ≠ 'matched'` | Work without quotes |
    | **Scope Creep %** | `Unquoted Hours / Total Hours × 100` | Share of unquoted work |
    
    ### Rate Capture
    
    | Metric | Formula | Notes |
    |--------|---------|-------|
    | **Rate Variance** | `Realised Rate - Quote Rate` | Positive = capturing more |
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # UTILISATION
    # =========================================================================
    section_header("Utilisation Metrics")
    
    st.markdown("""
    ### Leave Exclusion
    
    **Default:** Exclude rows where `task_name` contains 'leave' (case-insensitive) from utilisation calculations.
    
    ### Definitions
    
    | Metric | Formula | Notes |
    |--------|---------|-------|
    | **Utilisation** | `Billable Hours / Total Hours × 100` | Excluding leave |
    | **Target** | `utilisation_target × 100` | Staff-level target |
    | **Util Gap** | `Target - Utilisation` | Positive = underperforming |
    
    ### Rolling Up Targets
    
    When aggregating across staff, use **hours-weighted average** for target:
    
    ```python
    weighted_target = (utilisation_target × hours_raw).sum() / hours_raw.sum()
    ```
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # CAPACITY
    # =========================================================================
    section_header("Capacity Model")
    
    st.markdown("""
    ### Base Assumptions
    
    | Parameter | Default | Notes |
    |-----------|---------|-------|
    | **Standard Weekly Hours** | 38 | Full-time equivalent |
    | **FTE Scaling** | `fte_hours_scaling` | 1.0 = full-time, 0.5 = part-time |
    | **Utilisation Target** | `utilisation_target` | Typically 0.8 (80%) |
    
    ### Formulas
    
    | Metric | Formula |
    |--------|---------|
    | **Weekly Capacity** | `38 × fte_hours_scaling` |
    | **Period Capacity** | `Weekly Capacity × weeks_in_period` |
    | **Billable Capacity** | `Period Capacity × utilisation_target` |
    | **Headroom** | `Billable Capacity - Trailing Billable Load` |
    
    ### Implied FTE Demand
    
    ```
    Implied FTE = Quoted Hours / (weeks × 38 × util_target)
    ```
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # REVENUE ALLOCATION
    # =========================================================================
    section_header("Revenue Allocation Algorithm")
    
    st.markdown("""
    Revenue is allocated from monthly pools to daily timesheet rows based on hours contribution.
    
    ### Process
    
    1. **Get Revenue Pool:** Total revenue for (job_no, month_key)
    2. **Get Denominator Hours:**
       - First try: Sum of billable hours in job-month
       - Fallback: Sum of all hours in job-month
       - If zero: Revenue remains unallocated
    3. **Allocate:** `row_revenue = (row_hours / denom_hours) × revenue_pool`
    
    ### Allocation Modes
    
    | Mode | Description |
    |------|-------------|
    | `billable_hours` | Allocated using billable hours as denominator |
    | `all_hours` | Allocated using all hours (when no billable) |
    | `no_hours` | Cannot allocate (no timesheet hours) |
    
    ### Reconciliation
    
    For each job-month: `Σ rev_alloc ≈ rev_total_job_month` (within $0.01 tolerance)
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # DEPARTMENT DERIVATION
    # =========================================================================
    section_header("Department Derivation")
    
    st.markdown("""
    **Revenue is the source of truth for department**, not timesheet.
    
    ### Precedence
    
    ```
    department_final = COALESCE(
        department_rev_job_month,  -- Revenue dept for this job-month
        department_rev_job,        -- Revenue dept for job (any month)
        department_ts_raw          -- Timesheet dept (fallback if enabled)
    )
    ```
    
    ### Selection When Multiple Departments
    
    When a job-month has multiple departments in revenue:
    1. Group revenue by department
    2. Sum `ABS(amount)` per department
    3. Select department with highest absolute amount
    4. Tie-breakers: highest non-abs sum, then alphabetical
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # ACTIVE DEFINITIONS
    # =========================================================================
    section_header("Active Entity Definitions")
    
    st.markdown("""
    ### Active Jobs
    
    A job is **active** if:
    - NOT completed (`job_completed_date` is null OR `job_status` ≠ 'Completed')
    - AND has timesheet activity in last **21 days** (configurable)
    
    ### Active Staff
    
    A staff member is **active** if:
    - Has logged ≥ 1 hour in last **6 months** (configurable)
    
    ### Recency Weighting
    
    Optional exponential decay for benchmarks:
    
    ```
    weight = 0.5^(months_ago / half_life)
    ```
    
    Default half-life: **6 months**
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # RISK FLAGS
    # =========================================================================
    section_header("Risk Flag Definitions")
    
    st.markdown("""
    ### Job Risk Status
    
    | Status | Condition |
    |--------|-----------|
    | **On Track** | Quote consumed < 80% |
    | **Watch** | 80% ≤ Quote consumed < 100% |
    | **At Risk** | Quote consumed ≥ 100% |
    
    ### Overrun Definitions
    
    | Type | Condition |
    |------|-----------|
    | **Overrun** | Actual hours > Quoted hours |
    | **Severe Overrun** | Actual hours > Quoted hours × 1.2 |
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # TIME WINDOWS
    # =========================================================================
    section_header("Time Window Options")
    
    st.markdown("""
    | Option | Definition |
    |--------|------------|
    | **3m** | Last 3 calendar months |
    | **6m** | Last 6 calendar months |
    | **12m** | Last 12 calendar months |
    | **24m** | Last 24 calendar months |
    | **FYTD** | Australian FY to date (Jul 1 - current) |
    | **All** | All available data |
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # DATA SOURCES
    # =========================================================================
    section_header("Data Sources")
    
    st.markdown("""
    ### Primary Tables
    
    | Table | Grain | Purpose |
    |-------|-------|---------|
    | `fact_timesheet_day_enriched` | staff × job × task × date | Primary fact table |
    | `fact_job_task_month` | job × task × month | Aggregated view |
    | `audit_revenue_reconciliation` | job × month | Reconciliation checks |
    | `audit_unallocated_revenue` | job × month | Unallocated revenue |
    
    ### Precomputed Marts
    
    | Mart | Grain | Purpose |
    |------|-------|---------|
    | `cube_dept_month` | dept × month | Department time series |
    | `cube_dept_category_month` | dept × cat × month | Category trends |
    | `cube_dept_category_task` | dept × cat × task | Task benchmarks |
    | `cube_dept_category_staff` | dept × cat × staff | Staff performance |
    | `active_jobs_snapshot` | job | Active job status |
    | `job_mix_month` | cohort month | Job intake analysis |
    """)


if __name__ == "__main__":
    main()
