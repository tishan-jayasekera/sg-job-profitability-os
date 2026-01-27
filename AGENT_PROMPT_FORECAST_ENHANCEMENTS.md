# Agent Prompt: Forecast & Bottlenecks Page Enhancement

## Objective
Build comprehensive improvements to the `pages/5_Forecast_&_Bottlenecks.py` page to transform it from a functional bottleneck-detection tool into a strategic operational planning system with better structure, forecasting rigor, interactivity, and decision-support capabilities.

## Context
Read the detailed analysis in `FORECAST_BOTTLENECKS_DILIGENCE.md` (in repo root). Key findings:
- Current page has 6 disconnected sections; lacks narrative flow and clear hierarchy
- Forecasts are point estimates (p50) with no confidence intervals or uncertainty bands
- Bottlenecks identified but no remediation path or scenario planning
- UI is dense with scattered KPIs; tables are hard to scan
- Data quality assumptions (benchmark reliability, FTE scaling) hidden from users
- No cross-page integration or alerting mechanism

## Phase 1 Deliverables (PRIMARY FOCUS - 40-50 hours)

### Deliverable 1: Restructure Page Architecture (3-Act Narrative)
**Goal**: Transform 6 disconnected sections into coherent Act 1 → Act 2 → Act 3 flow

**Requirements**:
1. **Act 1 - Portfolio Health Dashboard**
   - KPI strip at top: Total active jobs, total remaining hours, portfolio ETA, % at-risk/blocked, capacity runway
   - Risk heat-map: Jobs sorted by (due_date - eta_weeks); color-coded (green/yellow/red); clickable for drill-down
   - Capacity vs. demand forecast chart: Weekly demand line + capacity band; highlight forecast period
   - Top 5 bottlenecks table: Job, task, remaining hours, velocity, skill gap; sortable by impact
   - Clear section purpose statement: "Identify portfolio-level risks and resource constraints"

2. **Act 2 - Job Deep-Dive**
   - Retain chain controls (dept → category → job) but keep them above fold
   - Redesigned job health card: ETA, due date, status (On-Track/At-Risk/Blocked), % complete, velocity trend mini-chart
   - Task shape vs. reality: Stacked bar chart showing (benchmark → actual → remaining) for each task
   - Task bottleneck matrix table:
     - Columns: Task, Remaining Hrs, Velocity (hrs/wk), Est. Weeks, Status (flag), Recommendation
     - Sorting: Default by "Est. Weeks" DESC
     - Color-coding: Green (on-track), yellow (at-risk), red (blocked)
   - Clear section purpose statement: "Decompose job and identify task-level bottlenecks"

3. **Act 3 - Action & Execution**
   - Recommended staffing plan table: Recommended staff by task, current capacity, allocation impact
   - Simple what-if interface: "Add [N] FTE" slider (0-3), "Delay deadline" slider (-2 to +4 weeks); auto-recompute ETA and cost delta
   - Execution summary: "New ETA: [X] weeks (baseline [Y]) | Cost: +$[Z] | Risk: [level]"
   - CTA buttons: "Approve Staffing Plan", "Export Scenario", "Send to Project Manager"
   - Clear section purpose statement: "Plan remediation and execute staffing changes"

**Implementation Notes**:
- Use Streamlit `st.tabs()` for Act 1 | Act 2 | Act 3 navigation (not sequential sections)
- Move "Department Forecast" and "Category Forecast" views to a separate "Comparison" view within Act 1 (collapsible)
- Remove redundant "Section 4 - Job Forecast Summary" KPI strip; consolidate into job health card
- Each Act should fit on one screen (minimize scrolling)

---

### Deliverable 2: Implement Risk Scoring & Prioritization
**Goal**: Replace arbitrary sorting with smart risk-based prioritization

**Requirements**:
1. Compute risk score for each job:
   ```
   risk_score = 1.0 - max(0, (due_weeks - eta_weeks) / due_weeks)
   where:
     - negative (due_weeks - eta_weeks) → risk_score > 1.0 (overdue, capped at 1.0)
     - risk_score ∈ [0, 1.0] where 0 = on-track, 1.0 = severe risk
   ```

2. Create risk matrix visualization (Act 1):
   - X-axis: "Time Buffer (weeks)" = due_weeks - eta_weeks
   - Y-axis: "Remaining Work (hours)"
   - Bubble: Each job; bubble size = team_velocity (larger = faster)
   - Color: Risk score (green: 0-0.2, yellow: 0.2-0.7, red: 0.7-1.0)
   - Interactive: Click bubble to drill to job detail (Act 2)
   - Add horizontal line at x=0 (due date = now) and x=2 (warning threshold)

3. Sort jobs in heat-map and bottlenecks table by risk_score DESC by default

4. Add toggle: "Sort by: Most Urgent | Highest Impact | Quickest to Fix | Highest Cost of Delay"

**Location**: New function `compute_risk_score()` in `src/modeling/forecast.py`; new chart component in `src/ui/charts.py`

---

### Deliverable 3: Add Tabbed Navigation & Clean Up UI
**Goal**: Reduce cognitive load via progressive disclosure and cleaner visual hierarchy

**Requirements**:
1. Replace 6 sequential sections with 3 tabs at top of page:
   - Tab 1 (default): "📊 Portfolio Dashboard" → Act 1 content
   - Tab 2: "🎯 Job Deep-Dive" → Act 2 content
   - Tab 3: "👥 Staffing & Scenarios" → Act 3 content

2. KPI strip redesign:
   - Show only 4-5 most critical KPIs (not scattered across sections)
   - Each KPI: number + status indicator (green/yellow/red) + mini-sparkline (4-week trend)
   - On hover: Show definition & calculation method in tooltip
   - Example: "Portfolio ETA: 12 weeks ⚠️ | (trend: ↑ +2wks vs. last week)"

3. Table improvements:
   - Sort controls: Click column header to sort ascending/descending
   - Filter row: Dropdowns for Status (All / On-Track / At-Risk / Blocked)
   - Color-coded rows: Green (healthy) | Yellow (caution) | Red (critical)
   - Expandable rows: Click "+" to reveal task-level details inline (not modal)
   - CSV export button for each table

4. Mobile responsiveness:
   - On screens <768px: Collapse tabs to dropdown, stack KPI columns, convert tables to cards
   - Test in Streamlit on mobile device

**Location**: Refactor `pages/5_Forecast_&_Bottlenecks.py` main() function; new components in `src/ui/components.py`

---

### Deliverable 4: Data Quality & Transparency Layer
**Goal**: Surface assumptions, benchmark reliability, and data freshness to build user trust

**Requirements**:
1. Add "Data Quality" collapsible section at bottom of each tab:
   - Show: Benchmark source (e.g., "Category 'Implementation': p50=240 hrs from 12 completed projects")
   - Show: FTE scaling (e.g., "Team capacity scaled by 95% per config")
   - Show: Active job definition (e.g., "Active = worked in last 21 days OR due in next 30 days")
   - Show: Data freshness (e.g., "Last refresh: 2 hours ago | Next refresh: in 1 hour")
   - Show: % completeness (e.g., "84% of tasks have velocity data, 76% have due dates")

2. Add warning banners for data quality issues:
   - ⚠️ Low-confidence benchmark: "Forecast for Category [X] based on only [N] projects; recommend manual review"
   - ⚠️ Zero velocity: "Task [Y] has zero team velocity; may indicate new skill, gap, or not started. Verify with PM."
   - ⚠️ Extreme ETA: "Job [Z] forecast shows ETA > 30 weeks; recommend reviewing scope with stakeholder"
   - ⚠️ Task overrun: "Task [A] has exceeded benchmark by 30%; watch for further slippage"

3. Add glossary: Link to "8_Glossary_Method.py" for term definitions

**Location**: New function `render_data_quality_panel()` in `src/ui/components.py`; extend `src/data/loader.py` to track benchmark reliability (count of source projects)

---

### Deliverable 5: Enhanced Visualizations
**Goal**: Replace dense tables with intuitive, interactive charts

**Requirements**:
1. **Risk Matrix** (Act 1):
   - Already described in Deliverable 2
   - Add shaded regions: Green zone (on-track), Yellow zone (at-risk), Red zone (critical)

2. **Task Stacked Bar** (Act 2):
   - For each task: Show benchmark (expected) → actual → remaining as stacked bar
   - Color: Expected (light blue), Actual (darker blue), Remaining (orange)
   - On hover: Show hours and % of total
   - Highlight overrun bars in red

3. **Bottleneck Heatmap** (Act 1 - new):
   - X-axis: Tasks (active across jobs)
   - Y-axis: Jobs
   - Cell color: Status of task in that job (green/yellow/red/gray)
   - Cell size/annotation: Remaining hours or % complete
   - On click: Drill to job detail

4. **Capacity Runway Chart** (Act 1):
   - Existing weekly demand line chart enhanced with:
   - Capacity band (upper bound = current capacity)
   - Forecast period shaded (light gray)
   - Highlight deficit weeks (red bars)
   - Add legend: "Capacity", "Actual Demand", "Forecast Period"

**Location**: Update `src/ui/charts.py` with new chart functions; call from page components

---

## Phase 1 Testing & Validation

### Unit Tests
- [ ] Test `compute_risk_score()` with edge cases (overdue, zero remaining hours)
- [ ] Test `forecast_remaining_work()` with benchmarks of varying quality
- [ ] Test `solve_bottlenecks()` with zero velocity scenarios

### Integration Tests
- [ ] Verify risk heat-map updates when job is selected
- [ ] Verify Act 2 deep-dive reflects same data as Act 1 heat-map
- [ ] Verify export CSV includes all displayed columns

### Usability Tests
- [ ] Have 3-5 power users (PMs, managers) review new structure
- [ ] Time how long to: (a) identify top 3 risks, (b) drill to job detail, (c) understand a bottleneck
- [ ] Collect feedback on terminology, color usage, chart clarity
- [ ] Mobile test: Can tabs and tables be used effectively on iPad/phone?

---

## Acceptance Criteria for Phase 1

- [ ] Page renders without errors on datasets with 10-1000 active jobs
- [ ] Risk heat-map correctly sorts jobs by risk_score
- [ ] All 3 tabs (Portfolio | Job | Staffing) display correctly with representative data
- [ ] Data quality panel accurately reflects benchmark reliability (N completed projects)
- [ ] Warning banners appear correctly for low-confidence forecasts and zero-velocity tasks
- [ ] KPI strip displays 4-5 key metrics with trend sparklines
- [ ] Tables support sorting (click header) and filtering (Status dropdown)
- [ ] CSV export includes all visible rows and columns
- [ ] Mobile responsiveness tested: Layout adapts to <768px screens
- [ ] Usability testing: 80% of testers can identify top 3 portfolio risks in <2 minutes
- [ ] Code passes lint (ruff/pylint); docstrings added to new functions
- [ ] Performance: Page loads in <3 seconds on 1000-job dataset

---

## Success Metrics (Phase 1 completion)
- **Adoption**: Page usage increases from ~2x/week to ~5x/week
- **User satisfaction**: NPS improves from ~20 to ~35 (survey 5-10 users)
- **Decision quality**: First-time managers say forecast helps them prioritize by risk (qualitative feedback)
- **Performance**: Page load time <3 seconds (vs. current ~5 seconds)

---

## Notes on Phase 2 & Beyond

This prompt focuses on **Phase 1 (structure, risk prioritization, transparency)**. Subsequent phases will add:
- **Phase 2**: Confidence intervals (p10/p90 ETAs), velocity trend detection, forecast accuracy tracking, scenario modeling
- **Phase 3**: Interactive what-if scenarios, cross-page linking, advanced staffing recommendations
- **Phase 4**: Automated alerts/notifications, email export, mobile app

---

## Key Files to Modify

**Primary**:
- `pages/5_Forecast_&_Bottlenecks.py` — Complete refactor (main structure, tabs, Act logic)

**Secondary**:
- `src/modeling/forecast.py` — Add `compute_risk_score()` function
- `src/ui/charts.py` — Add risk matrix, bottleneck heatmap, enhanced visualizations
- `src/ui/components.py` — Add `render_data_quality_panel()`, KPI strip, interactive tables

**Tertiary**:
- `src/data/loader.py` — Track benchmark reliability (source project count)
- `src/data/profiles.py` — Validate existing functions work with new page structure

---

## Constraints & Assumptions
- Streamlit version ≥1.28 (supports `st.tabs()`)
- Data already in `fact_timesheet_day_enriched` parquet with required columns
- `src/modeling/forecast.py` functions already exist and work correctly
- No new data pipelines needed (Phase 1 only enhances analysis layer)
- User has Plotly installed (existing dependency)

---

## Handoff Notes
Once this agent completes Phase 1, the improvements should be:
1. Merged to a feature branch (not main) for stakeholder review
2. Tested with 3-5 power users for feedback
3. Refined based on feedback
4. Documented in updated README.md with new page walkthrough
5. Handed off to Phase 2 development
