# ⚡ QUICK BRIEF: Forecast Page Fix (1-Minute Read)

## Problem
Forecast page has 6 confusing sections, point-estimate forecasts with no confidence, identified bottlenecks but no "what-if" scenarios, and hidden data assumptions.

## Solution (Phase 1 = 40-50 hours)
Build 3 things:

### 1️⃣ Tabs (Restructure for clarity)
```
Tab 1: Portfolio Dashboard
  → Risk heat-map (jobs by time buffer × remaining work)
  → Top 5 bottlenecks table
  → KPI strip (active jobs, total hours, ETA, % at-risk)

Tab 2: Job Deep-Dive  
  → Chain controls (dept → category → job)
  → Job health card (ETA, due, status, % complete)
  → Task bottleneck matrix

Tab 3: Staffing & Scenarios
  → Recommended staff
  → What-if slider (add FTE, extend deadline)
  → Impact summary
```

### 2️⃣ Risk Scoring (Smart prioritization)
```
risk_score = 1.0 - max(0, (due_weeks - eta_weeks) / due_weeks)
Output: 0 (on-track) to 1.0 (critical)

Risk heat-map: X=time_buffer, Y=remaining_hrs, bubble_size=velocity, color=risk
Sort by: "Most Urgent" | "Highest Impact" | "Quickest to Fix"
```

### 3️⃣ Transparency Panel (Build trust)
```
Show for each forecast:
  • Benchmark source: "p50 from [N] completed projects"
  • FTE scaling: "Capacity scaled by [Z]%"
  • Active job criteria: "Worked in last [D] days OR due in [M] weeks"
  • Data freshness: "Last refresh: [TIME]"
  
Warn if:
  ⚠️ Benchmark based on <5 projects
  ⚠️ Zero velocity (real bottleneck or no data?)
  ⚠️ ETA > 30 weeks (scope issue?)
```

---

## Files to Touch
- `pages/5_Forecast_&_Bottlenecks.py` — Refactor main (add tabs, break into 3 acts)
- `src/modeling/forecast.py` — Add `compute_risk_score()` function
- `src/ui/charts.py` — Add `render_risk_matrix()` and enhanced visualizations
- `src/ui/components.py` — Add `render_data_quality_panel()` and table wrappers
- `src/data/loader.py` — Track benchmark reliability (source project count)

---

## Effort & Timeline
- **Effort**: 40-50 hours
- **Timeline**: 4-5 weeks (sprint)
- **Team**: 1-2 developers

---

## Success Criteria
✅ Page loads <3s on 1000-job dataset  
✅ Risk heat-map correctly sorts jobs by risk_score  
✅ All 3 tabs render without errors  
✅ Data quality panel shows benchmark reliability  
✅ Mobile responsive (<768px)  
✅ Usability test: 80% of users can identify top 3 risks in <2 min  

---

## Next Steps
1. Read `BUILD_PROMPT_FORECAST_PHASE1.md` for full spec
2. Hand off to dev team with acceptance criteria
3. Start with Priority 1 (tabs), then Priority 2 (risk heat-map)
4. Build incrementally; test after each priority

---

**Reference Docs** (in repo root):
- `BUILD_PROMPT_FORECAST_PHASE1.md` — Full build prompt (this section's detailed spec)
- `AGENT_PROMPT_FORECAST_ENHANCEMENTS.md` — Phase 1 detailed spec with examples
- `FORECAST_BOTTLENECKS_DILIGENCE.md` — Complete analysis & Phase 2-4 roadmap
- `FORECAST_EXECUTIVE_SUMMARY.md` — Stakeholder summary

**Let's build this! 🚀**
