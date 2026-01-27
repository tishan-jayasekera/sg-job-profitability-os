# Forecast & Bottlenecks Page - Due Diligence Analysis & Improvement Recommendations

## Executive Summary
The **Forecast & Bottlenecks** page is a critical operational tool designed to predict project completion timelines, identify resource constraints, and recommend staffing solutions for active jobs. It combines empirical benchmarks from completed projects with real-time velocity data to provide early warning of capacity crises. However, the current implementation exhibits structural limitations, UX fragmentation, and analytical gaps that reduce its utility for executive decision-making and operational planning.

---

## Current State Analysis

### Architecture Overview
- **Data Flow**: Loads active jobs → applies category benchmarks (p50) → forecasts remaining work → calculates team velocity → computes ETAs & bottleneck status
- **Key Modules**: 
  - `src/modeling/forecast.py` - Remaining work calculation & ETA logic
  - `src/modeling/supply.py` - Team velocity aggregation (assumed)
  - `src/staffing/engine.py` - Staff recommendation engine
  - `src/data/profiles.py` - Expertise & capacity profiling
- **Hierarchy**: Company → Department → Category → Job → Task → Staff
- **UI Pattern**: 6 sequential sections with static chain controls (dropdowns)

### Current Strengths
1. **Hierarchical Navigation** - Logical top-down drill path from company forecast to task-level bottlenecks
2. **Dual Perspective** - Shows both remaining work AND velocity to contextualize ETAs
3. **Benchmark-Driven** - Uses empirical p50 from completed jobs (defensible, repeatable)
4. **Bottleneck Detection** - Clear identification of zero-velocity tasks (blocked work)
5. **Staff Recommendations** - Automated matching of available staff to blocked tasks
6. **Capacity Modeling** - Incorporates actual team FTE and trailing 4-week velocity
7. **Forecast Visualization** - Time-series view with capacity bands
8. **Multi-Level Aggregation** - Department and category views for portfolio insights

### Critical Gaps & Pain Points

#### **1. Structure & Organization Issues**
- **Unclear Section Purpose**: 6 sections lack coherent narrative flow; feels like disconnected reports
- **Redundant Breakdowns**: Dept/Category summaries repeat data without differentiation or comparative value
- **"Section 4 - Job Forecast Summary"** placed after company/dept/category sections (illogical sequencing)
- **Static Chain Controls**: Limited to sequential dept → category → job filtering; no ability to compare jobs side-by-side
- **No Context Switching**: Once job is selected, all downstream viz is locked to that job; can't pivot to similar jobs

#### **2. Functionality & Business Logic Gaps**
- **No Risk Ranking**: Tasks/jobs shown unranked; no priority or urgency sorting
- **Missing Confidence Intervals**: ETA shown as point estimate (p50); no uncertainty bands (p10-p90)
- **Velocity Extrapolation Risk**: Uses 4-week trailing average with no drift detection or outlier handling
- **Benchmark Generalization**: If job category has <N completed projects, benchmark may be unreliable (not flagged)
- **Zero-Velocity Assumption**: Treats zero velocity as binary bottleneck; doesn't distinguish between "no data" vs "structurally blocked"
- **No Remediation Path**: Identified bottlenecks have staff recommendation but no timeline/effort estimate for fixing
- **Capacity Ignored in Forecast**: Forecast period assumes steady demand; doesn't model existing capacity drain
- **Multi-Skill Tasks Opaque**: Task velocity aggregates all staff; doesn't show which skillset is bottleneck
- **No Scenario Planning**: Can't simulate "what-if" staffing changes or rebalancing

#### **3. UI/UX Deficiencies**
- **Information Overload**: 6 sections × multiple viz each = excessive vertical scrolling, no clear CTA
- **Poor Visual Hierarchy**: All sections weighted equally; no distinction between strategic vs. tactical
- **Metrics Scattered**: KPIs for company/dept/category are in tables; no unified dashboard view
- **Color Coding Weak**: Danger/warning states use standard red/orange but not consistently applied
- **Dataframe Tables Bland**: Wide tables with >6 columns; hard to scan and identify actionable items
- **No Drill-Through**: Charts don't link to underlying data or cross-reference other pages
- **Export Capability Missing**: No CSV/Excel downloads for reporting or external analysis
- **Missing Tooltips/Help**: No contextual definitions (e.g., "What is team velocity?", "How is ETA calculated?")

#### **4. Data Quality & Validation Gaps**
- **Benchmark Reliability Unmarked**: No indicator if benchmark is based on 1 job vs. 100 jobs
- **Fallback Logic Silent**: Code reverts to actual task share if mix unavailable; no warning to user
- **Extreme ETA Unhandled**: ETAs of infinity (zero velocity) shown as "—"; no flag for systemic bottleneck
- **Date Assumptions Brittle**: "job_due_date", "job_completed_date" optional; fallback silent if missing
- **FTE Scaling Hidden**: Uses "fte_hours_scaling" config but not surfaced in UI; users unaware of adjustment
- **Active Job Definition Unclear**: Depends on `get_active_jobs()` (likely recency-based) but criteria not shown

#### **5. Analytics & Insights Deficiency**
- **No Root Cause Analysis**: Bottleneck identified but not analyzed for cause (skill gap, availability, priority)
- **No Trend Analysis**: Weekly actuals shown but no comparison to prior forecasts (forecast accuracy tracking)
- **No Cross-Job Insights**: Can't see if bottleneck is job-specific or systemic (e.g., all jobs need X skill)
- **No Portfolio Risk**: No aggregated view of probability of portfolio completion vs. deadline
- **No Capacity Utilization Trend**: Current demand vs. available capacity not tracked over time
- **No Staff Load Distribution**: Can't see if recommendations would overload staff or create new bottlenecks
- **No Cost Impact**: Bottleneck may have financial consequence (overrun, late finish); not quantified
- **No Historical Forecast Accuracy**: No metric showing if past forecasts were accurate (model validation)

---

## Findings Summary

| Dimension | Current State | Issue Severity |
|-----------|---------------|-----------------|
| **Structure** | 6 disconnected sections | HIGH - User confused by flow |
| **Functionality** | Bottleneck detection only | HIGH - No remediation path |
| **Forecasting** | Point estimates (p50) | HIGH - No uncertainty/risk bands |
| **Risk Prioritization** | None | HIGH - Can't prioritize actions |
| **UI Clarity** | Dense tables & scattered KPIs | MEDIUM - Cognitive load high |
| **Data Validation** | Minimal transparency | MEDIUM - Users unaware of data quality |
| **Scenario Planning** | No what-if modeling | MEDIUM - Can't test interventions |
| **Historical Tracking** | No forecast accuracy metrics | LOW - Can improve over time |

---

## Detailed Recommendations

### **RECOMMENDATION 1: Restructure Page for Narrative Clarity**
**Problem**: 6 sections lack coherent story; user must jump between company/dept/category/job views.

**Solution**:
- **Reorganize as 3-Act Structure**:
  1. **Act 1 - Portfolio Health (Company-Level Risk Dashboard)**
     - KPI strip: Total active jobs, total remaining hours, portfolio ETA, % at-risk/blocked
     - Risk heat-map: Jobs sorted by ETA vs. due date (green/yellow/red)
     - Capacity runway: Forecast period capacity vs. demand chart
     - Top 5 Bottlenecks: Table of zero-velocity tasks across all jobs (with job/task/skill)
     - CTA: "Drill into [Job Name]" or "View All Bottlenecks"
  
  2. **Act 2 - Job Decomposition (Dept/Category/Job Deep-Dive)**
     - Retained chain controls (dept → category → job) but add "Compare X Similar Jobs" option
     - Job health card: ETA, due date, status, % complete, velocity trend
     - Task shape vs. reality: Stacked bar or waterfall (expected → actual → remaining)
     - Task bottleneck matrix: Table with task, remaining hrs, velocity, weeks-to-complete, risk flag, recommendation
     - Capacity impact: Show if recommended staffing would over/under-allocate
     - CTA: "Approve Staffing Recommendation" (with downstream workflow integration)
  
  3. **Act 3 - Action & Execution (Staffing & Contingency Planning)**
     - Recommended staffing plan: Interactive table (add/remove staff, see impact on timeline)
     - What-if scenarios: "Delay start by 2 weeks", "Add 1 FTE", "Rebalance tasks"
     - Execution checklist: Countdown to critical path items, dependencies
     - CTA: "Export Plan" or "Send Notification to Project Manager"

- **Benefits**: Clear narrative, progressive disclosure, reduced scrolling, aligned with user mental model (portfolio → job → action)

---

### **RECOMMENDATION 2: Enhance Forecasting with Uncertainty & Confidence Bands**
**Problem**: ETA shown as point estimate (p50); no indication of accuracy or risk range.

**Solution**:
- **Add Confidence Intervals**:
  - Compute ETA at p10, p50, p90 using bootstrap from completed-job variance
  - Display as range: "ETA: 4 weeks (3-6 weeks, 80% confidence)"
  - Visualize as band on timeline (light shading for p10-p90, darker for p25-p75)
  
- **Benchmark Reliability Indicator**:
  - Flag if benchmark is based on <5 completed projects (low confidence)
  - Show "Benchmark: p50 from [N] projects" in tooltip
  - Option to override with custom estimate if benchmark weak
  
- **Forecast Accuracy Tracking**:
  - Log each forecast (job, ETA, date generated)
  - Compare actual completion to prior forecast
  - Display "Forecast Error" metric (actual - predicted) on job card
  - Aggregate accuracy by category to identify systematic bias

- **Velocity Trend & Drift Detection**:
  - Plot 4-week trailing velocity with trend line
  - Flag if velocity declining >10% week-over-week (early warning)
  - Option to select "conservative" (p25) vs. "aggressive" (p75) velocity for re-forecast

- **Implementation**: Add `compute_eia_intervals()` function to `forecast.py`; update job_level output with p10/p90 columns; new chart type in UI layer

- **Benefits**: Users understand forecast uncertainty, can make risk-informed decisions, system improves over time

---

### **RECOMMENDATION 3: Implement Risk-Based Prioritization & Sorting**
**Problem**: Jobs/tasks/bottlenecks shown in arbitrary order; no clear urgency ranking.

**Solution**:
- **Risk Score for Each Job**:
  ```
  risk_score = (1 - (due_weeks - eta_weeks) / due_weeks) * urgency_weight
  where:
    - due_weeks: weeks until due date
    - eta_weeks: forecasted weeks to completion
    - urgency_weight: inversely proportional to due_weeks (jobs due sooner weighted higher)
  ```
  - Result: 0 (on-track) to 1.0 (severe risk)
  
- **Risk Matrix Visualization**:
  - X-axis: "Time Buffer" (due_weeks - eta_weeks); negative = overdue
  - Y-axis: "Remaining Work" (remaining hours); larger = more work
  - Bubble size: Team velocity (larger = faster resolution)
  - Color: Risk score (green/yellow/red)
  - Able to click bubble to drill into job
  
- **Bottleneck Priority Ranking**:
  - Rank tasks by "Impact on Portfolio ETA"
  - Tasks on critical path (impact > N weeks) shown first
  - Secondary sort: "Hours to Unblock" (effort to add capacity)
  
- **Auto-Sort Options**:
  - "Show Most Urgent" (default)
  - "Show Highest-Impact"
  - "Show Quickest-to-Fix"
  - "Show Highest-Cost-of-Delay"

- **Implementation**: Add `compute_risk_score()`, `compute_critical_path()` functions; new risk matrix component in UI

- **Benefits**: Executive can focus on highest-impact items, avoids analysis paralysis, clear action priorities

---

### **RECOMMENDATION 4: Add Remediation Path & Scenario Planning**
**Problem**: Bottleneck identified; staff recommendation given; no path to execution or contingency.

**Solution**:
- **Remediation Wizard**:
  - Step 1: Confirm bottleneck (show task, remaining hrs, current velocity)
  - Step 2: Select remedy (add staff, rebalance tasks, extend deadline, reduce scope)
  - Step 3: Simulate impact (show updated ETA, cost, risks)
  - Step 4: Approve & notify (send alert to PM/manager with action items)
  
- **Scenario Planning**:
  - Build scenarios for job: "Baseline", "Add 1 FTE", "Add 2 FTE", "Delay by 1 week"
  - For each scenario: show new ETA, cost delta, dependency impact
  - Compare scenarios side-by-side
  - Save & export scenarios for stakeholder discussion
  
- **What-If Interface**:
  - Interactive controls: "FTE to Add" slider (0-3), "Task Rebalance" dropdown, "Deadline Shift" slider (-2 to +4 weeks)
  - Real-time re-forecast as controls change
  - Show breakdown: "New ETA: 6 weeks (baseline 8 wks) | Cost: +$50k | Risk: Medium"
  
- **Recommendation Confidence**:
  - Show if staff recommendation would over-allocate (e.g., "Adding Jane would put her at 105% capacity")
  - Flag if staff lacks recent experience in required skill (recency alert)
  - Suggest alternative staff if primary recommendation risky
  
- **Implementation**: New modal component; extend `recommend_staff_for_plan()` to return multiple ranked options; add `simulate_staffing_impact()` function

- **Benefits**: Users can test interventions before commitment, build stakeholder alignment, de-risk decision-making

---

### **RECOMMENDATION 5: Improve UI/UX - Streamline & Add Interactivity**
**Problem**: Dense tables, scattered KPIs, no drill-through or cross-references.

**Solution**:
- **KPI Strip Redesign**:
  - Show top KPIs consistently (company total, selected dept total, selected job total)
  - Each KPI includes +/- sparkline showing trend (last 4 weeks)
  - Color-coded status indicator (green/yellow/red)
  - Tooltip: Definition, calculation method, benchmark
  
- **Tabbed Interface for Navigation**:
  - Tab 1: Portfolio Dashboard (company-level heatmap, top risks)
  - Tab 2: Job Deep-Dive (chain controls, job card, task decomposition)
  - Tab 3: Staffing & Scenarios (recommendations, what-if, approval workflow)
  - Tab 4: Historical Forecast (accuracy tracking, lessons learned)
  
- **Interactive Tables**:
  - Sort by any column (default: risk score DESC)
  - Filter by status (On-Track, At-Risk, Blocked, Completed)
  - Inline edit: Double-click to adjust ETA if user has better info
  - Row color-code: Green/Yellow/Red based on status
  - Expandable rows: Click to see task details inline
  - Export to CSV/Excel button for each table
  
- **Chart Enhancements**:
  - Heatmap for job status (x-axis: job, y-axis: task; color = status) → hover to see details
  - Gantt chart: Job timeline with actual vs. forecast bars, dependencies
  - Risk matrix: Scatter plot (time buffer vs. remaining work) with bubble = job
  - All charts clickable → drill to job/task detail
  
- **Contextual Help**:
  - Info icons next to section titles → popover with definition & formula
  - Glossary link at bottom of page
  - Examples: "What does 'zero velocity' mean?"
  
- **Responsive Layout**:
  - On mobile: Collapse dept/category/job selectors to dropdown (not 3-column layout)
  - Prioritize top KPI strip and risk heatmap
  - Tables become stacked cards on small screens
  
- **Implementation**: Refactor UI using `st.tabs()`, `st.columns()`, custom CSS; add Plotly click handlers; new reusable components in `src/ui/`

- **Benefits**: Cleaner visual hierarchy, faster scanning, lower cognitive load, mobile-friendly, more engaging

---

### **RECOMMENDATION 6: Add Data Quality & Transparency Layer**
**Problem**: Benchmark reliability, fallback logic, FTE scaling all hidden from user.

**Solution**:
- **Data Quality Dashboard**:
  - Section at bottom: "Data Quality & Assumptions"
  - Show:
    - Benchmark source: "Category [X]: based on [N] completed projects (p50: [Y] hrs)"
    - FTE scaling: "Team capacity scaled by [Z]% (config: FTE_SCALING=[Z])"
    - Active job criteria: "Active = worked in last [N] days or due in next [M] weeks"
    - Completeness: "% of tasks with velocity data, % with due dates, % with task mix"
    - Data recency: "Last data refresh: [TIME] | Next refresh: [TIME]"
  
- **Warnings & Caveats**:
  - If benchmark weak (<5 projects): "⚠️ Low-confidence forecast based on [N] projects; recommend manual review"
  - If zero velocity: "⚠️ Zero velocity may indicate: (1) new task type, (2) skill gap, (3) not yet started. Verify."
  - If extreme ETA: "⚠️ ETA > [30] weeks suggests potential data error or scope issue; review with PM"
  - If overrun detected: "⚠️ Task actual hrs exceed benchmark; may indicate underestimation or scope creep"
  
- **Audit Trail**:
  - Log each forecast generated (user, date, selections, output KPIs)
  - Store in lightweight database or CSV
  - Display "Last forecast: [DATE] by [USER]" on page
  - Link to "View Forecast History" (show how forecast has evolved week-over-week)
  
- **Implementation**: Add `DataQualityPanel` component; extend loader to track data freshness; add audit log to config

- **Benefits**: Users understand limitations, can adjust decisions accordingly, auditability for compliance

---

### **RECOMMENDATION 7: Integrate Cross-Page Analytics**
**Problem**: Page is siloed; no connection to profitability, delivery, or staffing pages.

**Solution**:
- **Links to Other Pages**:
  - On job card: Link to "Active Delivery" page for that job (delivery status, burndown)
  - On job card: Link to "Executive Summary" profitability view for that job (margin, rate capture)
  - On bottleneck task: Link to "Capacity Profiles" page for staff recommendations
  - On staffing recommendation: "See full capacity profile for [Staff Name]" link
  
- **Data Hand-Off**:
  - When user clicks "Approve Staffing Recommendation": Pass staffing plan to "Capacity Profiles" page
  - "Active Delivery" page can reference forecast as baseline for variance analysis
  - "Executive Summary" can show forecast impact on margin (if timeline extends)
  
- **Consistent KPI Definition**:
  - Use same definitions across pages (e.g., "active job" same everywhere)
  - Cross-page KPI reconciliation: Portfolio remaining hours on Forecast should match sum on Active Delivery
  
- **Implementation**: Add hyperlinks using `st.write("[Link](:/path/to/page)")`, shared state in `src/ui/state.py`

- **Benefits**: Users see holistic view, reduces context-switching, enables integrated decision-making

---

### **RECOMMENDATION 8: Build Automated Alerting & Notifications**
**Problem**: Critical bottlenecks may go unnoticed; decisions delayed.

**Solution**:
- **Alert Rules**:
  - Job ETA > due date (forecast "At Risk" status)
  - Zero-velocity task blocking >N hours
  - Portfolio capacity deficit in next 2 weeks
  - Forecast accuracy error >20% (model degradation)
  - Task overrun >20% (early warning of further slippage)
  
- **Notification Channels**:
  - In-app: Alert banner at top of Forecast page (dismissible)
  - Email: Daily digest of alerts (configurable by role)
  - Slack: Channel for critical bottlenecks (if Slack integration available)
  
- **Alert Severity Levels**:
  - 🔴 Critical: Job due in <1 week and at risk
  - 🟠 High: Job at risk or portfolio deficit
  - 🟡 Medium: Zero-velocity task or declining velocity trend
  - 🟢 Info: New completed projects, forecast model updated
  
- **Implementation**: Add notification logic to forecast logic; integrate with email/Slack APIs; add alert config to `src/config.py`

- **Benefits**: Proactive problem identification, faster response time, reduced reliance on manual checks

---

## Implementation Roadmap

### **Phase 1 (Week 1-2): Foundation & Quick Wins** ⚡
- [ ] **R1**: Restructure page into 3-act narrative (reorder sections, add section headers with clear purpose)
- [ ] **R5**: Implement tabbed interface (Portfolio Dashboard | Job Deep-Dive | Staffing & Scenarios)
- [ ] **R5**: Enhance tables with sorting, filtering, color-coding
- [ ] **R3**: Add risk score computation & risk matrix visualization
- [ ] **R6**: Add Data Quality panel (benchmark reliability, assumptions transparency)

**Effort**: 40-50 hours | **Impact**: HIGH (structure clarity, risk prioritization, transparency)

---

### **Phase 2 (Week 3-4): Analytics & Forecasting Enhancements** 📊
- [ ] **R2**: Implement confidence intervals (p10/p50/p90) for ETAs
- [ ] **R2**: Add velocity trend & drift detection
- [ ] **R2**: Build forecast accuracy tracking (historical predictions vs. actuals)
- [ ] **R4**: Build remediation wizard (step-by-step bottleneck resolution)
- [ ] **R5**: Implement Gantt chart & heatmap visualizations

**Effort**: 50-60 hours | **Impact**: HIGH (better forecasting, scenario capability, deeper insights)

---

### **Phase 3 (Week 5-6): What-If & Execution** 🎯
- [ ] **R4**: Build scenario planning interface (interactive sliders, side-by-side comparison)
- [ ] **R4**: Enhanced staffing recommendations (confidence, alternatives, impact analysis)
- [ ] **R5**: Context menus & drill-through (charts linkable to detail)
- [ ] **R7**: Cross-page linking (Forecast ↔ Active Delivery ↔ Executive Summary)

**Effort**: 40-50 hours | **Impact**: MEDIUM-HIGH (operational efficiency, decision support)

---

### **Phase 4 (Week 7-8): Automation & Polish** 🚀
- [ ] **R8**: Build alert system (rules, notifications, channels)
- [ ] **R5**: Mobile responsiveness (tabs collapse, cards stack)
- [ ] **R5**: Export functionality (CSV, Excel, PDF reports)
- [ ] **R6**: Add glossary & contextual help (info icons, tooltips)
- [ ] Testing, performance tuning, documentation

**Effort**: 30-40 hours | **Impact**: MEDIUM (automation, accessibility, adoption)

---

## Success Criteria

- **Adoption**: >80% of project managers use page weekly (vs. current <50%)
- **Decision Quality**: Forecast accuracy improves to ±15% (vs. current ±25%)
- **Alert Response Time**: Critical bottlenecks resolved within 2 days of alert (vs. current ad-hoc)
- **Scenario Testing**: >30% of decisions informed by scenario planning (baseline: 0%)
- **User Satisfaction**: NPS >40 on page UX survey (vs. current ~20)
- **Data Freshness**: Forecasts updated daily (vs. manual weekly)

---

## Technical Considerations

### Dependencies to Add/Enhance
- **Scenario Engine**: Extend `forecast.py` with `simulate_staffing_impact()`, `simulate_timeline_shift()` functions
- **Confidence Intervals**: Bootstrap or empirical quantile calculation on historical forecast errors
- **Alert System**: Basic email/Slack integration; consider using APScheduler for scheduled checks
- **Audit Logging**: Lightweight logging to CSV or SQLite for forecast history
- **UI Components**: Tabbed interface, interactive tables, drill-through capabilities (Streamlit 1.28+)

### Performance Considerations
- Cache forecasts for 1 hour (vs. current per-page-load)
- Pre-compute risk scores on data refresh (not on every page load)
- Paginate tables if >100 rows (lazy load details on click)
- Consider Parquet marts for large datasets (if not already in use)

### Testing Strategy
- Unit tests: `test_forecast.py` for ETA calculation, confidence intervals, risk scoring
- Integration tests: Verify cross-page KPI reconciliation
- Usability tests: A/B test new structure with 5-10 power users
- Accuracy tests: Compare forecasts to actual completions (weekly validation)

---

## Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Forecast confidence intervals increase cognitive load | Progressive disclosure: show p50 by default, p10/p90 in tooltip |
| What-if scenarios encourage over-planning & analysis paralysis | Limit to 3-4 predefined scenarios; disable custom scenarios in v1 |
| Alerting fatigue (too many notifications) | Start conservative (only Critical alerts); users can tune sensitivity |
| Performance degradation with large datasets | Profile & cache aggressively; test with 10,000+ active jobs |
| Users ignore data quality warnings | Make warnings prominent (red banner); require acknowledgment for high-risk decisions |

---

## Appendix: Code Architecture Notes

### New Functions to Add
```python
# src/modeling/forecast.py
def compute_eta_intervals(remaining_df, velocity_df, boot_samples=1000) -> DataFrame
  # p10, p50, p90 ETAs with confidence bands
  
def detect_velocity_drift(velocity_series, window=4, threshold=0.1) -> bool
  # Flag if velocity declining >10% week-over-week
  
def compute_risk_score(job_row) -> float
  # Risk = f(time_buffer, remaining_work, velocity_confidence)

# src/modeling/remediation.py (new)
def simulate_staffing_impact(job, recommended_staff) -> dict
  # Returns: new_eta, cost_delta, overallocation_flags, risks
  
def build_scenarios(job, base_scenario) -> List[dict]
  # Generates: baseline, +1FTE, +2FTE, -1week_deadline, etc.

# src/ui/components.py (new/extend)
def render_risk_matrix(job_level: DataFrame) -> go.Figure
  # Scatter: time_buffer vs. remaining_work, bubble=velocity, color=status
  
def render_scenario_comparison(scenarios: List[dict]) -> Streamlit.columns
  # Side-by-side scenario cards

# src/data/audit.py (new)
def log_forecast(user, selections, output_kpis) -> None
  # Audit trail for forecast history
```

### UI Component Refactor
```python
# pages/5_Forecast_&_Bottlenecks.py (refactored)
def main():
    # Initialize tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Dashboard",
        "🎯 Job Deep-Dive",
        "👥 Staffing & Scenarios",
        "📈 Forecast History"
    ])
    
    # Tab 1: Portfolio Dashboard
    with tab1:
        _render_portfolio_dashboard(job_level, task_level)
    
    # Tab 2: Job Deep-Dive
    with tab2:
        _render_job_deep_dive(job_level, task_level, bench_summary)
    
    # Tab 3: Staffing & Scenarios
    with tab3:
        _render_staffing_scenarios(selected_job, selected_task, ...)
    
    # Tab 4: Forecast History
    with tab4:
        _render_forecast_history(...)
```

---

## Conclusion

The Forecast & Bottlenecks page is a foundational tool for operational planning but requires significant structural, analytical, and UX enhancements to reach its potential. The recommended improvements focus on:

1. **Clarity**: Reorganize into coherent narrative (portfolio → job → action)
2. **Rigor**: Add uncertainty quantification, confidence metrics, scenario modeling
3. **Usability**: Streamline navigation, add interactivity, reduce cognitive load
4. **Transparency**: Surface data quality assumptions, alert rules, audit trails
5. **Integration**: Link to other pages, enable cross-functional workflows
6. **Automation**: Build alerts, notifications, approval workflows

**Estimated Total Effort**: 160-200 hours across 4 phases  
**Expected ROI**: 3-5x improvement in forecast accuracy, 2x faster decision-making, 50% reduction in manual bottleneck tracking

**Next Step**: Review recommendations with stakeholders; prioritize Phase 1 for immediate implementation.
