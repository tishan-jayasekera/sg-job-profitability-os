# Empirical Lifecycle Forecaster - Build Complete

## 🚀 What Was Built

### New Page: `pages/9_Empirical_Forecast.py` (526 lines)

A 5-level drill-down interface that forecasts job completion by mapping the "remaining DNA" (task distribution) of active jobs against historical profiles from completed jobs.

**Access it from Streamlit sidebar:** Page 9 › Empirical Forecast

### New Functions in `src/modeling/forecast.py`

#### 1. `build_lifecycle_dna(df_completed, lookback_jobs=30)`
**Purpose:** Build historical task distribution profiles ("DNA") from completed jobs

**Returns:**
```python
{
    'dna_profiles': DataFrame with columns:
        - category_rev_job
        - quintile (Q1-Q5, representing 0-100% completion)
        - task_name
        - median_task_pct (% of quintile hours)
        - staff_roles_historical
    
    'benchmarks': DataFrame with:
        - category_rev_job
        - median_total_hours (p50)
        - p25_total_hours (optimistic)
        - p75_total_hours (pessimistic)
    
    'errors': List of categories with insufficient data
}
```

#### 2. `compute_job_maturity(df_job, benchmark_total_hours)`
**Purpose:** Calculate how far through its lifecycle a job is

**Returns:**
```python
{
    'maturity_pct': Job completion % (actual / benchmark * 100)
    'actual_hours': Hours logged to date
    'estimated_total': Benchmark total hours
    'remaining_hours': max(0, estimated - actual)
    'confidence_low': Maturity using P75 benchmark (pessimistic)
    'confidence_high': Maturity using P25 benchmark (optimistic)
}
```

#### 3. `badge_fte_capabilities(df_recent, recency_months=6)`
**Purpose:** Identify which staff members can execute which tasks

**Returns:** DataFrame with columns:
- `staff_name`
- `task_name`
- `total_hours` (in lookback window)
- `avg_hours_per_month`
- `proficiency_level` ('Novice' <20h, 'Intermediate' 20-100h, 'Expert' >100h)

#### 4. `compute_surgical_gaps(active_jobs, dna_profiles, benchmarks, fte_capabilities)`
**Purpose:** Find where demand exceeds supply by task/skill

**Returns:**
```python
{
    'surgical_gaps': DataFrame with columns:
        - task_name
        - total_demand_hours (from active jobs' remaining work)
        - total_supply_hours (hours logged by capable FTEs)
        - gap_hours (demand - supply)
        - capable_ftes (count of staff who can do this task)
        - job_count (jobs requiring this task)
        - gap_rank (1=largest gap)
    
    'total_demand': Sum of all remaining hours across active jobs
    'total_supply_ftes': Count of unique staff on roster
}
```

---

## 📊 5-Level Drill Structure

### Level 0: Company Pipeline Demand
**What:** Bird's-eye view of surgical deficits across entire organization

**Shows:**
- 4 KPI cards: Total demand, Active jobs, Roster FTEs, Implied FTEs needed
- Bar chart of top 10 task gaps (demand - supply)
- Full table of all task gaps
- Drill selector to go to Department level

**Use Case:** Executive dashboard—"Where are we short on skills?"

### Level 1: Departmental Contribution
**What:** Breakdown by department

**Shows:**
- Demand and estimated hours by job category
- Department-level metrics
- Drill selector to go to Category level

**Use Case:** Understand which departments drive overall gaps

### Level 2: Category Benchmarking
**What:** Historical task distribution patterns for this category

**Shows:**
- Quintile × Task matrix (historical DNA profile)
- Current active jobs with maturity % and remaining hours
- Drill selector to go to Job level

**Use Case:** Compare: "Are we following the historical pattern?" or "Is this job abnormal?"

### Level 3: Job Maturity Card
**What:** Health of a single active job

**Shows:**
- Maturity %, actual vs. remaining hours, status badge
- Task breakdown table
- Next phase warning with historically required tasks

**Use Case:** "What should this job do next? What does history say?"

### Level 4: Surgical Intervention
**What:** Assignment recommendations for a bottleneck task

**Shows:**
- Task status across the job
- FTE recommendations (staff badged in this task, their proficiency)
- Available capacity analysis

**Use Case:** "Who can I assign to unblock this task?"

---

## 🎮 Navigation

**Up Button:** Move up one level (resets children)
**Reset Button:** Return to Level 0 (clear all selections)
**Breadcrumb:** Shows current drill path
**Drill Selectors:** Selectbox at each level to choose next drill

---

## ⚙️ Configuration (Sidebar)

- **Lookback completed jobs (DNA):** 5–50 (default 30)
  - Larger = more historical data, better profiles, but older patterns
  
- **Forecast horizon (weeks):** 4–52 (default 12)
  - How far into the future to compute capacity
  
- **FTE skill recency (months):** 1–12 (default 6)
  - Time window to badge staff with skills (recent activity = current capability)

---

## 🔍 How It Works (The Math)

### DNA Building
1. Get all completed jobs per category
2. For each job, calculate **maturity** = (cumulative hours / total job hours) × 100
3. Assign each hour log to a quintile (Q1: 0-20%, Q2: 21-40%, etc.)
4. For each quintile, compute median % of hours per task
5. Result: A "DNA profile" showing historical task mix across lifecycle

### Maturity Calculation
- **Maturity %** = (actual hours to date / category benchmark) × 100
- **Confidence bands:**
  - Pessimistic (P75): Uses higher benchmark → lower maturity %
  - Optimistic (P25): Uses lower benchmark → higher maturity %

### Surgical Gaps
1. For each active job: compute remaining hours = (benchmark − actual)
2. Allocate remaining hours to tasks using DNA profile (e.g., if job is 20% mature, use Q3–Q5 task mix)
3. Aggregate remaining hours by task across all active jobs → **Total Demand**
4. Sum hours logged by staff per task in last 6 months → **Total Supply**
5. Gap = Demand − Supply (positive = understaffed for this task)

---

## ✅ Validation

✅ All existing forecast functions preserved (no breaking changes)  
✅ Syntax validated  
✅ All imports resolved  
✅ Follows existing UI patterns from other pages  
✅ Ready for Streamlit execution  

---

## 🚦 Next Steps

1. **Test in Streamlit:** `streamlit run app.py` → Navigate to Page 9
2. **Verify DNA profiles:** Check if completed jobs have sufficient data per category
3. **Refine styling:** Adjust colors/spacing as needed
4. **Add caching:** Wrap `build_lifecycle_dna()` with `@st.cache_data` if performance needed
5. **Extend what-if scenarios:** Level 4 can add slider to simulate FTE allocation and see impact on job ETA

---

## 📝 Key Differences from Traditional Forecast

| Aspect | Traditional | Empirical Lifecycle |
|--------|-------------|-------------------|
| **Demand Source** | Quoted hours | Remaining DNA (historical task mix) |
| **Supply Definition** | Total roster capacity | Task-specific FTE capability |
| **Gap Analysis** | Total hours | Task-by-task surgical gaps |
| **Maturity Signal** | Hours vs. quote variance | Hours vs. category benchmark |
| **Intervention** | Generic "assign more FTEs" | Named staff with proven skill badges |
