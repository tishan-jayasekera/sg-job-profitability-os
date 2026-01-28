# 🎯 PHASE 1B: DRILL-CHAIN ARCHITECTURE REDESIGN

**Problem**: Phase 1 built tabs but not a coherent **drill path**. Users land in "job deep dive" without context.

**Solution**: 5-level hierarchical navigation with **one set of definitions** (horizon, velocity window, benchmark basis) flowing through all levels.

---

## 🏗️ THE 5-LEVEL DRILL-CHAIN ARCHITECTURE

```
Level 0: Company Forecast
    ↓ click Department
Level 1: Department Forecast
    ↓ click Job Type
Level 2: Job Type Distribution
    ↓ click Job
Level 3: Individual Job Health
    ↓ click Task
Level 4: Task → FTE Responsibility
```

**Key principle**: Every view is filtered to current scope. Same metric definitions, same time window, same benchmark basis across all 5 levels.

---

## 📊 LEVEL 0: COMPANY FORECAST (LANDING PAGE)

**One question**: "Are we in surplus or deficit? Over what horizon? Where is it concentrated?"

### Layout
```
┌─ Forecast Horizon Selector: 4 / 8 / 12 / 16 weeks
│
├─ [KPI Strip]
│  ├─ Company Demand (hours)
│  ├─ Company Capacity (hours)
│  ├─ Gap (hours) + Gap (FTE equivalent)
│  └─ Gap %
│
├─ [Waterfall Chart]
│  └─ Demand breakdown by dept → which depts drive the gap
│
├─ [Department Ranked Table]
│  ├─ Dept Name | Demand | Capacity | Gap | Gap % | At-Risk Jobs
│  ├─ Sorted by Gap DESC (largest gaps first)
│  └─ ✅ Row click → Level 1 (that department)
│
└─ [Data Quality Panel]
   └─ Same as before + horizon used + velocity lookback window
```

### Key outputs
- **Demand formula**: sum(remaining_hours for active jobs in horizon window)
- **Capacity formula**: team FTE × hours/week × weeks in horizon
- **Gap formula**: capacity - demand (negative = oversubscribed)
- **At-risk count**: jobs with risk_score > 0.7 in this dept

### Math exception handling
- If capacity == 0: show "No capacity data" (not NaN)
- If demand == 0: show 0 (explicit)

---

## 📊 LEVEL 1: DEPARTMENT FORECAST

**One question**: "Within this department's gap, which job types / categories are the culprits? What's the delivery reality (planned vs actual)?"

### Layout (Scoped to selected department)
```
┌─ Breadcrumb: Company ▸ [Department Name]
│
├─ [KPI Strip - Dept Scope]
│  ├─ Dept Demand | Capacity | Gap | At-Risk Jobs | Top Bottleneck Task
│
├─ [Job Type Breakdown Table]
│  ├─ Columns: Category | # Active Jobs | Demand | Capacity | Gap | Risk Score (avg)
│  ├─ Sorted by Gap DESC
│  └─ ✅ Row click → Level 2 (that category within dept)
│
├─ [Top Bottleneck Tasks - Dept Scoped]
│  ├─ Task Name | Job | Remaining | Velocity | Est Weeks
│  ├─ Top 5 by (remaining_hours / velocity)
│  └─ ✅ Task click → Level 3 (that job) + Level 4 (that task)
│
└─ [Breadcrumb Interactions]
   └─ Click "Company" to return to Level 0
```

### Key outputs
- Filter data to: `dept == selected_dept AND status == 'active'`
- Category demand: `sum(remaining_hours) by category`
- Category capacity: `team_ftes * hours_per_week * horizon_weeks` (filtered to dept)

---

## 📊 LEVEL 2: JOB TYPE / CATEGORY DISTRIBUTION

**One question**: "Why is this category blowing up capacity? What's the benchmark shape vs. actual delivery pattern? Which jobs are worst?"

### Layout (Scoped to Dept + Category)
```
┌─ Breadcrumb: Company ▸ Department ▸ [Category Name]
│
├─ [KPI Strip - Category Scope]
│  ├─ Category Demand | Capacity | Gap | # Active Jobs | Avg Risk Score
│
├─ [Benchmark vs Actual Comparison]
│  ├─ Stacked bar chart:
│  │  ├─ X-axis: Benchmark (p50 from completed jobs in category)
│  │  ├─ Compare to: Avg actual hours (current active jobs)
│  │  ├─ Plus: Avg remaining hours (what's left)
│  │  └─ Insight: "Category runs [20%] over benchmark on average"
│
├─ [Distribution Scatter]
│  ├─ X-axis: Time Buffer (weeks until due - weeks to complete)
│  ├─ Y-axis: Remaining Hours
│  ├─ Bubble size: Team velocity
│  ├─ Bubble color: Risk score
│  ├─ Each bubble = one job in this category
│  └─ ✅ Bubble click → Level 3 (that job)
│
├─ [Worst Jobs Ranked Table]
│  ├─ Columns: Job # | Remaining | Velocity | Est Weeks | Risk | Status
│  ├─ Sorted by: (overdue_weeks DESC, remaining_hours DESC, velocity ASC)
│  ├─ Status: On-Track | At-Risk | Blocked | Overdue
│  └─ ✅ Row click → Level 3 (that job)
│
└─ [Breadcrumb Interactions]
   └─ Click "Department" to return to Level 1
```

### Key outputs
- Filter data to: `dept == selected_dept AND category == selected_category AND status == 'active'`
- Benchmark shape: median hours from completed jobs in this category
- Actual pattern: distribution of remaining_hours across active jobs

---

## 📊 LEVEL 3: INDIVIDUAL JOB DEEP-DIVE

**One question**: "What's left to do, what's moving, what's stuck, and who's working on it?"

### Layout (Scoped to Job)
```
┌─ Breadcrumb: Company ▸ Department ▸ Category ▸ [Job #]
│
├─ [Job Health Card - HUMAN READABLE]
│  ├─ Status: On-Track | At-Risk | Blocked | Overdue (NOT raw numbers)
│  │
│  ├─ Row 1: ETA
│  │  ├─ If velocity > 0: "Est complete: [DATE] ([WEEKS] weeks from now)"
│  │  └─ If velocity == 0: "⚠️ No run-rate detected" (NOT "Inf")
│  │
│  ├─ Row 2: Due Date
│  │  ├─ If due in future: "Due: [DATE] ([WEEKS] weeks from now)"
│  │  └─ If overdue: "🔴 Overdue: [WEEKS] weeks"
│  │
│  ├─ Row 3: Time Buffer
│  │  ├─ If buffer > 2 weeks: "🟢 Safe: [WEEKS] week cushion"
│  │  ├─ If buffer 0-2 weeks: "🟡 Tight: [WEEKS] week cushion"
│  │  └─ If buffer < 0: "🔴 At risk: [WEEKS] weeks overdue"
│  │
│  ├─ Row 4: Risk Score
│  │  ├─ 0.0-0.2: "🟢 On Track (score: [X])"
│  │  ├─ 0.2-0.7: "🟡 At Risk (score: [X])"
│  │  └─ 0.7-1.0: "🔴 Critical (score: [X])"
│  │
│  ├─ Row 5: Scope
│  │  ├─ Total hours: [X] hrs (benchmark) + [Y] hrs added scope
│  │  ├─ Spent: [Z] hrs ([%] complete)
│  │  └─ Remaining: [W] hrs ([%] to go)
│
├─ [Active Contributors]
│  ├─ Name | Hours Last 21d | % of Job | Trend (📈/→/📉)
│  └─ (Who's actually working on this job right now)
│
├─ [Top 10 Bottleneck Tasks - JOB SCOPED]
│  ├─ Columns: Task | Status | Remaining | Velocity | Est Weeks | Assigned | Priority
│  ├─ Sorted by: 
│  │  ├─ First: Status (Blocked > At-Risk > On-Track)
│  │  ├─ Then: Remaining DESC
│  │  ├─ Then: Velocity ASC
│  ├─ Status colors:
│  │  ├─ 🔴 Blocked: 0 hrs/week AND > 1 hour remaining
│  │  ├─ 🟡 At-Risk: velocity < est_velocity
│  │  ├─ 🟢 On-Track: velocity >= est_velocity
│  │  └─ ⚪ Negligible: < 5 remaining hours (filtered out by default)
│  ├─ Filter button: "Hide negligible tasks" (ON by default)
│  └─ ✅ Task row click → Level 4 (that task)
│
└─ [Breadcrumb Interactions]
   └─ Click "Category" to return to Level 2
```

### Key math exception handling
- **`inf` ETA** → "No run-rate detected" (0 hrs/week velocity)
- **Negative due** → "Overdue by X weeks"
- **Negative time buffer** → Show as overdue status
- **NaN risk score** → "Insufficient data" (missing benchmark or velocity)

---

## 📊 LEVEL 4: TASK → FTE RESPONSIBILITY & FEASIBILITY

**One question**: "Who can do this work, are they available, what's the capacity path to completion?"

### Layout (Scoped to Task)
```
┌─ Breadcrumb: Company ▸ Department ▸ Category ▸ Job ▸ [Task Name]
│
├─ [Task Health Summary]
│  ├─ Task Name | Status | Remaining Hours | Expected Hours | Benchmark Hours
│  ├─ Velocity (last 21d) | Est Completion | Risk
│
├─ [Active Contributors]
│  ├─ Name | Hours Last 21d | Capacity % | 21d Trend
│  └─ (Who is actually working on this task right now)
│
├─ [Eligible Contributors - By Skill Match]
│  ├─ (People who have done similar tasks, sorted by recency)
│  ├─ Name | Last Done | Months Ago | Skill Confidence
│  └─ "Could reallocate these people to unblock this task"
│
├─ [Capacity Feasibility Matrix]
│  ├─ Current state: 
│  │  ├─ Remaining: [X] hours
│  │  ├─ Velocity: [Y] hrs/week
│  │  ├─ Est completion: [DATE]
│  │
│  ├─ What-if scenarios (interactive sliders):
│  │  ├─ Slider 1: Add hours/week (0 to +10 hrs/week)
│  │  │  └─ New ETA: [DATE] (saves [Z] days)
│  │  │
│  │  ├─ Slider 2: Shift deadline (-2 to +4 weeks)
│  │  │  └─ New buffer: [Z] weeks
│  │  │
│  │  └─ Live summary: "If we add 2 FTE: complete in [DATE]"
│
├─ [Recommendation Engine]
│  ├─ "To unblock this task in 1 week, allocate:"
│  │  ├─ [Name A] for [X] hrs/week (available)
│  │  ├─ [Name B] for [Y] hrs/week (available)
│  │  └─ New velocity would be: [Z] hrs/week → complete [DATE]
│
└─ [Breadcrumb Interactions]
   └─ Click "Job" to return to Level 3
```

### Key outputs
- **Active contributors**: Last 21d hours on this exact task
- **Eligible contributors**: Historical skill match + recency
- **Feasibility**: What additional velocity is needed to hit deadline
- **Recommendations**: Which people to pull in, for how long

---

## 🔄 STATE MANAGEMENT (Streamlit Session State)

```python
st.session_state.drill_state = {
    'level': 0,              # 0, 1, 2, 3, or 4
    'selected_dept': None,   # string dept_name
    'selected_category': None,
    'selected_job_id': None,
    'selected_task_id': None,
    'forecast_horizon_weeks': 12,  # Carries through ALL levels
    'velocity_lookback_days': 21,  # Consistent across levels
}
```

**On navigation**:
- Clicking a row at Level N → updates state → re-renders at Level N+1
- Clicking breadcrumb → sets level + clears all deeper selections

---

## 🎨 BREADCRUMB HEADER (Always Visible)

```
┌──────────────────────────────────────────────────────────────┐
│ Forecast Horizon: [4 weeks ▼]  |  Scope: Company ▸ Dept ▸ Category ▸ Job ▸ Task
└──────────────────────────────────────────────────────────────┘
```

**Behavior**:
- Forecast horizon applies to **all 5 levels**
- Each breadcrumb level is clickable → jumps to that level
- Greyed out if not yet selected
- Shows current selections clearly

---

## 📐 DATA PIPELINE (Core Functions)

### New scoping functions

```python
# src/modeling/forecast.py

def get_company_forecast(horizon_weeks: int) -> dict:
    """Company-level: demand, capacity, gap by department."""
    
def get_dept_forecast(dept: str, horizon_weeks: int) -> dict:
    """Dept-level: demand, capacity, gap by category."""
    
def get_category_jobs(dept: str, category: str) -> list:
    """Category-level: list of active jobs, sorted by urgency."""
    
def get_job_tasks(job_id: int) -> DataFrame:
    """Job-level: tasks, with no math artifacts."""
    
def get_task_capacity_path(task_id: int) -> dict:
    """Task-level: eligible staff, feasibility, scenarios."""
```

### Data filtering logic

```python
# ALWAYS filter by:
# 1. Status == 'active'  (defined at Level 0 via date rules)
# 2. Dept (if Level 1+)
# 3. Category (if Level 2+)
# 4. Job ID (if Level 3+)
# 5. Task ID (if Level 4)
# 6. Horizon window (all levels use same horizon_weeks)
# 7. Velocity lookback (all levels use same velocity_lookback_days)
```

---

## 🔧 MATH EXCEPTION HANDLING (Critical)

| Exception | Raw Value | Translation |
|-----------|-----------|-------------|
| ETA is infinity | `inf` | "No run-rate detected" (velocity = 0) |
| ETA is NaN | `nan` | "Insufficient data" (no benchmark or velocity) |
| Due date is past | negative weeks | "Overdue by X weeks" |
| Time buffer is negative | negative | "At risk: X weeks behind" |
| Risk score is NaN | `nan` | "Cannot calculate (missing data)" |
| Capacity is zero | 0 | "No capacity data" (might mean "unlimited" in context) |

**Rule**: **Never show raw math artifacts to user**. Translate to human-readable states.

---

## ✅ ACCEPTANCE CRITERIA (PHASE 1B)

### Navigation & Flow
- [ ] All 5 levels render correctly (no errors)
- [ ] Breadcrumb always shows current scope
- [ ] Clicking any row/bubble advances to next level with correct filters
- [ ] Clicking breadcrumb returns to that level
- [ ] Forecast horizon selector applies to all 5 levels

### Math Artifact Removal
- [ ] No `inf` shown anywhere; replaced with "No run-rate detected"
- [ ] No negative due dates; shown as "Overdue by X weeks"
- [ ] No NaN in risk scores; shown as "Insufficient data" if needed
- [ ] All percentage calculations safe (no divide-by-zero)

### Data Quality & Consistency
- [ ] Scope filtering applied consistently (dept → category → job → task)
- [ ] Same velocity window, horizon, benchmark basis across all levels
- [ ] Sorting logic applied (esp. Level 2 & 3: worst jobs/tasks first)

### UX & Clarity
- [ ] Job health card shows states (On-Track/At-Risk/Blocked), not raw numbers
- [ ] Task status clearly distinguishes "Blocked" vs "At-Risk" vs "On-Track"
- [ ] Active contributors visible at job & task level
- [ ] What-if scenario only at Level 4 (not floating)

### Performance
- [ ] Level 0 loads in <2 seconds (aggregations)
- [ ] Level 1-3 drill-down instant (<300ms, already-filtered data)
- [ ] Level 4 loads in <1 second (filtered task data)

---

## 📁 FILES TO MODIFY

### Core Changes
- **`pages/5_Forecast_&_Bottlenecks.py`** — Complete rebuild as 5-level navigation
- **`src/modeling/forecast.py`** — Add scoping functions (get_company_forecast, get_dept_forecast, etc.)
- **`src/ui/components.py`** — Add job health card (with state translation), task status renderer, breadcrumb header
- **`src/ui/charts.py`** — Ensure all charts respect scope filters

### Supporting
- **`src/config.py`** — Define "active job" criteria (worked in last N days OR due in next M weeks)
- **`src/data/loader.py`** — Ensure date fields are available for filtering

---

## 🎬 EXECUTION ROADMAP

**Phase 1B: Drill-Chain Restructure (1-2 weeks)**

1. **Week 1 - Architecture & Levels 0-2**
   - Set up session state for drill navigation
   - Implement Level 0: company forecast + dept ranking table
   - Implement Level 1: dept forecast + category breakdown
   - Implement Level 2: category distribution + worst jobs table
   - Breadcrumb header (clickable scope)

2. **Week 2 - Levels 3-4 & Exception Handling**
   - Implement Level 3: job health card (with state translation), task table
   - Implement Level 4: task responsibility chain + capacity what-if
   - Math exception handling across all levels
   - Testing & bug fixes

---

## 🎯 SUCCESS METRICS (PHASE 1B)

When this is deployed, users should be able to answer:

> "Our company is oversubscribed by 200 hours. Which department is the problem? Why? Which jobs? Which tasks? Who needs to be reallocated?"

**In < 90 seconds** by drilling: Level 0 → click dept → Level 1 → click category → Level 2 → click job → Level 3 → click task → Level 4 (see recommendation).

**Before Phase 1B**: Users jump around, see `inf`, get confused.  
**After Phase 1B**: One narrative thread, every view answers a question, no math artifacts.

---

## 🚀 READY TO BUILD?

This is **the core fix** for "pile of widgets" → "coherent analysis tool."

When you're ready, I'll implement all 5 levels with:
1. Proper state management (drill tracking)
2. Consistent data filtering (scope respected everywhere)
3. Human-readable states (no `inf`)
4. Breadcrumb navigation (always clear where you are)

Let's go. 🎯
