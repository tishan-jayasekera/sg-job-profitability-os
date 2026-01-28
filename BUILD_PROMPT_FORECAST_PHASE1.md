# 🎯 BUILD PROMPT: Forecast & Bottlenecks Phase 1 Enhancements

## TL;DR - What to Build
Restructure the Forecast & Bottlenecks page from 6 disconnected sections into a **3-tab dashboard** with **risk-based prioritization** and **data transparency**. Goal: Transform from bottleneck-detection tool into strategic operational planning system.

---

## 🔧 WHAT TO BUILD (Prioritized)

### Priority 1: Tab-Based Navigation (HIGHEST IMPACT)
Replace 6 sequential sections with 3 tabs:
- **Tab 1: Portfolio Dashboard** — Risk heat-map, KPI strip, top 5 bottlenecks
- **Tab 2: Job Deep-Dive** — Chain controls, job health card, task decomposition, bottleneck matrix
- **Tab 3: Staffing & Scenarios** — Recommended staff, what-if slider, execution summary

**File**: `pages/5_Forecast_&_Bottlenecks.py`  
**Time**: 12-14 hours  
**Acceptance**: Page loads in <3s, all 3 tabs display correctly, no console errors

---

### Priority 2: Risk Heat-Map & Sorting (CRITICAL FOR USABILITY)
Create visual scatter plot showing job risk:
- **X-axis**: Time buffer (weeks until due) - (forecasted weeks to complete)
- **Y-axis**: Remaining work (hours)
- **Bubble size**: Team velocity (hours/week)
- **Color**: Risk score (green < 0.2 | yellow 0.2-0.7 | red 0.7-1.0)
- **Clickable**: Click bubble → drill to job detail
- **Add sort toggles**: "Most Urgent" | "Highest Impact" | "Quickest to Fix"

**New function**: `compute_risk_score()` in `src/modeling/forecast.py`
```python
risk_score = 1.0 - max(0, (due_weeks - eta_weeks) / due_weeks)
# Output: 0 (on-track) to 1.0 (severe risk)
```

**New chart**: `render_risk_matrix()` in `src/ui/charts.py`

**File to add**: Plotly scatter chart in Tab 1  
**Time**: 8-10 hours  
**Acceptance**: Risk matrix renders, risk scores match calculations, clicking bubble works

---

### Priority 3: Data Quality Transparency Panel (HIGH TRUST)
Add collapsible section at bottom of each tab showing:
- Benchmark source: "Category [X]: p50 = [Y] hrs from [N] completed projects"
- FTE scaling: "Team capacity scaled by [Z]% per config"
- Active job definition: "Active = worked in last [N] days OR due in next [M] weeks"
- Data freshness: "Last refresh: [TIME] | Next refresh: [TIME]"
- Completeness: "% tasks with velocity, % with due dates"

**Add warning banners** for:
- ⚠️ Low-confidence benchmark (<5 projects)
- ⚠️ Zero velocity (verify if real bottleneck or no data)
- ⚠️ Extreme ETA (>30 weeks suggests scope issue)
- ⚠️ Task overrun (>20% over benchmark)

**New function**: `render_data_quality_panel()` in `src/ui/components.py`  
**Extend**: `src/data/loader.py` to track benchmark source project count

**Time**: 6-8 hours  
**Acceptance**: Panel displays benchmark reliability, warning banners appear for edge cases, no data quality hiding

---

### Priority 4: UI Improvements (CLEAN & SCANNABLE)
Make tables & KPIs less cluttered:

**KPI Strip Redesign**:
- Show only 4-5 most critical KPIs (not scattered)
- Each KPI: number + status indicator (🟢/🟡/🔴) + mini-sparkline (4-week trend)
- Tooltip on hover: Definition & calculation

**Table Enhancements**:
- **Sort**: Click column header to sort ascending/descending (default: risk_score DESC)
- **Filter**: "Status" dropdown (All | On-Track | At-Risk | Blocked)
- **Color rows**: Green (healthy) | Yellow (caution) | Red (critical)
- **Expand rows**: Click "+" to see task details inline (not modal)
- **Export**: CSV button for each table

**Mobile Responsive**:
- Screens <768px: Collapse tabs to dropdown, stack KPI columns, convert tables to cards

**Time**: 10-12 hours  
**Acceptance**: Tables sortable & filterable, KPI strip shows 4-5 metrics, mobile layout tested on iPad

---

### Priority 5: Enhanced Visualizations (INTUITIVE & INTERACTIVE)
Add/enhance charts:

**Task Stacked Bar** (Tab 2):
- Each task: benchmark (light blue) → actual (dark blue) → remaining (orange)
- Hover: Show hours & % of total
- Highlight overruns in red

**Bottleneck Heatmap** (Tab 1):
- X-axis: Tasks (across active jobs)
- Y-axis: Jobs
- Cell color: Status (green/yellow/red/gray)
- Cell annotation: Remaining hours
- Click → drill to job

**Capacity Runway** (Tab 1 - enhance existing):
- Weekly demand line + capacity band
- Forecast period shaded
- Deficit weeks highlighted red

**Time**: 8-10 hours  
**Acceptance**: All charts render, interactive elements work, no performance lag with 500+ jobs

---

## ✅ ACCEPTANCE CRITERIA (ALL REQUIRED)

### Functionality
- [ ] Page renders without errors on datasets with 10-1000 active jobs
- [ ] All 3 tabs (Portfolio | Job | Staffing) display correctly
- [ ] Risk heat-map sorts jobs by risk_score ascending/descending
- [ ] Risk scores calculated correctly (test with edge cases: overdue, zero remaining hrs)
- [ ] Warning banners appear for low-confidence benchmarks, zero velocity, extreme ETAs
- [ ] Tables support: sort (click header), filter (status dropdown), expand rows
- [ ] CSV export button works and includes all visible rows/columns
- [ ] Clicking risk matrix bubble navigates to job detail

### Performance
- [ ] Page loads in <3 seconds on 1000-job dataset
- [ ] Tab switching is instant (<300ms)
- [ ] Charts render without lag

### UX & Accessibility
- [ ] Mobile responsive: layout adapts to <768px screens
- [ ] Tooltips explain KPIs, metrics, risk score calculation
- [ ] Color-blind friendly: don't rely on color alone (use icons, labels)
- [ ] Data quality panel visible and clearly written
- [ ] Glossary link available for term definitions

### Code Quality
- [ ] New functions have docstrings explaining inputs/outputs
- [ ] Code passes ruff/pylint checks
- [ ] No console errors or warnings
- [ ] Unit tests for `compute_risk_score()` with edge cases

### Testing
- [ ] Usability test with 3-5 power users: Can they identify top 3 portfolio risks in <2 min?
- [ ] Feedback: 80%+ say structure is clearer than original
- [ ] Cross-check: Risk heat-map jobs match bottleneck list jobs

---

## 📁 FILES TO MODIFY

**Primary (Heavy Lift)**:
- `pages/5_Forecast_&_Bottlenecks.py` — Complete refactor (tabs, Act 1/2/3 structure, new components)

**Secondary (Medium Lift)**:
- `src/modeling/forecast.py` — Add `compute_risk_score()` function
- `src/ui/charts.py` — Add `render_risk_matrix()`, task stacked bar, bottleneck heatmap
- `src/ui/components.py` — Add `render_data_quality_panel()`, KPI strip, table wrappers

**Tertiary (Light Touch)**:
- `src/data/loader.py` — Track benchmark reliability (extend load function to include source project count)
- `src/data/profiles.py` — No changes (validate existing functions work with new page)

**New Files** (Optional, if refactoring into separate modules):
- `src/modeling/risk.py` — Could hold risk scoring logic
- `src/ui/table_components.py` — Could hold reusable table wrappers

---

## 🚀 EXECUTION ROADMAP (4-5 Weeks)

**Week 1**: Tabs + Portfolio Dashboard (Risk heat-map, KPI strip, top bottlenecks)  
**Week 2**: Job Deep-Dive (Chain controls, job card, task matrix) + Data Quality Panel  
**Week 3**: Enhanced Visualizations (Stacked bars, heatmaps) + UI Polish (sorting, filtering, color-coding)  
**Week 4**: Mobile Responsiveness + Testing + Documentation  
**Week 5**: Buffer for refinement based on usability feedback

---

## 🎯 SUCCESS DEFINITION

**Phase 1 Success = Page adoption increases from 2x/week to 5x/week + NPS improves from ~20 to ~35**

This means:
- Users find it valuable for prioritizing risks
- Structure makes sense (no "where do I start?" confusion)
- Data quality transparency builds trust
- Risk ranking saves time vs. manual scanning

---

## 📖 REFERENCE DOCUMENTS

If you need deeper context:
- **`AGENT_PROMPT_FORECAST_ENHANCEMENTS.md`** — Full Phase 1 spec with examples
- **`FORECAST_BOTTLENECKS_DILIGENCE.md`** — Complete analysis with Phase 2-4 roadmap
- **`FORECAST_EXECUTIVE_SUMMARY.md`** — High-level findings & why this matters

---

## ❓ QUESTIONS FOR AGENT BEFORE STARTING

1. **Streamlit version**: Confirm ≥1.28 (needed for `st.tabs()`)
2. **Data quality**: Are all benchmarks tracked with source project count? If not, we may need to infer or use a default confidence.
3. **Stakeholder alignment**: Has anyone validated that risk heat-map is the right prioritization mechanism? (Or should we use different axes/colors?)
4. **Timeline**: Are we aiming for Phase 1 launch in 4 weeks? Any hard deadline?
5. **Rollback plan**: If Phase 1 breaks something, do we have the old page saved or can we quickly revert?

---

## 🎬 START HERE

1. Read this prompt (you're here)
2. Read `AGENT_PROMPT_FORECAST_ENHANCEMENTS.md` for detailed requirements
3. Run current page locally: `streamlit run app.py`, navigate to Forecast tab
4. Understand current data flows by reading `src/modeling/forecast.py`
5. Create feature branch: `git checkout -b feature/forecast-phase1-enhancements`
6. Start with Priority 1 (tabs): Refactor page main() into `_render_portfolio_dashboard()`, `_render_job_deepdive()`, `_render_staffing_scenarios()`
7. Add Priority 2 (risk heat-map): Implement `compute_risk_score()`, then `render_risk_matrix()`
8. Iterate & test

---

**Ready to build? Let's go! 🚀**
