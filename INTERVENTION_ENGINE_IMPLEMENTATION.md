# 🚀 Operational Intervention Engine - Implementation Guide

## Overview

The "Quadrant Portfolio: Jobs + Profitability Context" section has been refactored into an **Operational Intervention Engine** - a delivery leader cockpit for prioritizing job interventions.

**Goal**: Delivery leaders can answer 5 questions in <2 minutes:
1. What are the top 5 jobs to intervene on?
2. Why are they at risk (reason codes)?
3. What is driving the issue (tasks / staffing)?
4. What action should be taken next?
5. Is this quadrant improving over time?

---

## Architecture

### Files Created/Modified

#### 1. **New: `src/modeling/intervention.py`** (Core business logic)
   - **Functions**:
     - `compute_intervention_risk_score()` - Calculates 0-100 risk score + reason codes
     - `build_intervention_queue()` - Builds ranked worklist per quadrant
     - `compute_quadrant_health_summary()` - KPIs for quadrant health
     - `get_peer_context()` - Percentile positioning vs peers

#### 2. **New: `src/ui/intervention_components.py`** (UI rendering)
   - **6 Component Functions** (in order):
     1. `render_quadrant_health_summary()` - Section 1: KPI cards
     2. `render_intervention_queue()` - Section 2: Ranked worklist
     3. `render_selected_job_brief()` - Section 3: Selected job details
     4. `render_driver_analysis()` - Section 4: Task + Staffing tabs
     5. `render_peer_context()` - Section 5: Percentile sanity check
     6. `render_quadrant_trend()` - Section 6: Time-series trends
     7. `render_methodology_expander()` - Definitions & threshold documentation

#### 3. **Modified: `src/ui/state.py`** (Session state)
   - **New State Keys**:
     - `intervention_quadrant` - Selected quadrant
     - `intervention_selected_job` - Selected job for review
     - `intervention_shortlist_size` - How many jobs to show (default 10)
     - `intervention_issue_filter` - Filter by issue type

#### 4. **Modified: `pages/6_Job_Mix_and_Demand.py`** (Page orchestration)
   - Added imports for new intervention engine
   - **Ready for refactor**: Replace "Quadrant Portfolio" section (lines 974-1165) with new component calls
   - See **Implementation Checklist** below

---

## 6-Section Flow (Non-Negotiable Order)

### Section 1: Quadrant Health Summary
- **What**: KPI cards showing quadrant-level metrics
- **Component**: `render_quadrant_health_summary(quadrant_jobs, quadrant_name)`
- **Metrics**:
  - Jobs in quadrant (count)
  - Quoted revenue exposure (sum)
  - Median Margin % (Actual vs Quote)
  - Median Realized Rate (vs Quote)
  - % Jobs Breaching Guardrails (risk score > 50)
  - Avg Risk Score

**User question answered**: "What's the health of this quadrant?"

---

### Section 2: Intervention Queue ⭐ (Main Anchor)
- **What**: Ranked worklist of active jobs requiring intervention
- **Component**: `render_intervention_queue(quadrant_jobs, max_rows=10, on_job_select=callback)`
- **Columns** (max 8):
  - Risk Score (0-100)
  - Job ID
  - Primary Issue (top 2-3 reason codes)
  - Margin Δ (Actual - Quote, %pp)
  - Hours Δ (Actual - Quote, %)
  - Rate Δ (Realized - Quoted, $/hr)
  - Job Age (days)
  - Owner
- **Interactions**:
  - Sort: By risk score (DESC) - highest risk first
  - Filter: By issue type (e.g., "Low margin %")
  - Slider: Show top N jobs (default 10)
  - Selection: Click row to see job brief

**User question answered**: "What are the top 5 jobs to intervene on?"

---

### Section 3: Selected Job Brief
- **What**: Auto-populates when user selects a job from queue
- **Component**: `render_selected_job_brief(job_row, peer_segment=None)`
- **Content**:
  - KPI cards: Quoted vs Actual (Margin, Hours, Rate, Timeline)
  - Risk Score + Top issue codes
  - Comparison vs peer median runtime
- **Interaction**: Shows why this specific job is flagged

**User question answered**: "Why is this job at risk?"

---

### Section 4: Driver Analysis (Forensic Layer)
- **What**: Two tabs explaining what's driving the issues
- **Component**: `render_driver_analysis(job_row, task_data=None, staffing_data=None)`

#### Tab A: Task Drivers
- Show task share: Job % vs peer median %
- Highlight positive deltas (overruns)
- Interpretation: "Why are certain tasks consuming more time?"

#### Tab B: Staffing Mix Drivers
- Seniority / Role / Geography breakdown
- Compare actual staff mix vs quoted assumptions
- Realized rate vs quoted rate by staff tier
- Recommendation: "What mix change would improve realized rate?"

**User question answered**: "What's driving the issue (tasks vs staffing)?"

---

### Section 5: Peer Context (Sanity Check)
- **What**: Position job vs peers in the segment
- **Component**: `render_peer_context(job_row, peer_segment)`
- **Metrics**:
  - Runtime percentile (vs peers)
  - Margin percentile (vs peers)
  - Realized rate percentile (vs peers)
- **Interpretation**:
  - 🟢 "Within peer range" → Category-wide issue (pricing / scope?)
  - 🔴 "Outlier vs peers" → Job-specific issue (staffing / delivery?)

**User question answered**: "Is this uniquely broken or structurally misquoted?"

---

### Section 6: Quadrant Trend (Strategic)
- **What**: 3 time-series for selected segment (Quadrant)
- **Component**: `render_quadrant_trend(quadrant_data, quadrant_name)`
- **Metrics**:
  - Median Margin % (trend)
  - Median Realized Rate (trend)
  - % Jobs Breaching Guardrails (trend)
- **Interpretation**: Is the quadrant improving, stable, or deteriorating?

**User question answered**: "Is this quadrant improving over time?"

---

## Risk Score Model

### Formula (0-100 scale)

| Factor | Points | Threshold | Description |
|--------|--------|-----------|-------------|
| **Margin Erosion** | 0-30 | < 15% margin | Alert if below threshold; weighted by gap |
| **Revenue Lag** | 0-25 | Revenue/Quote < 0.7 | Slow billing or scope loss |
| **Scope Creep** | 0-25 | Hours Δ > 10% | Overrun vs quoted hours |
| **Rate Leakage** | 0-20 | Realized/Quoted < 0.85 | Staffing mix or scope issue |
| **Runtime Risk** | 0-20 | Runtime > 1.5x peer median | "Zombie job" risk |

**Cap**: Max 100 points

### Reason Codes (Top 2-3 per job)

- "Low margin %"
- "Revenue lagging quote"
- "Hours overrun vs quote"
- "Realized rate below quote"
- "Runtime exceeds peers"

### Risk Interpretation

- 🟢 **Green (< 30)**: On-track, monitor routine
- 🟡 **Yellow (30-60)**: Watch list, intervention planning
- 🔴 **Red (> 60)**: Critical, immediate action needed

---

## Implementation Checklist

### Phase 1: Verify Imports ✅
- [x] `src/modeling/intervention.py` created
- [x] `src/ui/intervention_components.py` created
- [x] `src/ui/state.py` updated with new keys
- [x] `pages/6_Job_Mix_and_Demand.py` imports added
- [x] Syntax validation passed

### Phase 2: Page Orchestration (TO DO)

In `pages/6_Job_Mix_and_Demand.py`, around line 974:

**Replace the old section:**
```python
# OLD: Lines 974-1165 (Quadrant Portfolio section)
st.subheader("Quadrant Portfolio: Jobs + Profitability Context")
# ... old code ...
```

**With new orchestration:**
```python
# NEW: Operational Intervention Engine (6-section cockpit)
st.markdown("## 🎯 Operational Intervention Engine")
st.caption("Delivery leader cockpit for prioritizing active job interventions. Answer 5 key questions in <2 min.")

# Get quadrant name from context (from earlier in function)
quadrant_name = f"{margin_bucket} × {confidence_bucket}"  # or similar

# Prepare data
quadrant_jobs = quadrant_detail.copy()

# SECTION 1: Quadrant Health Summary
render_quadrant_health_summary(quadrant_jobs, quadrant_name)

# SECTION 2: Intervention Queue (main anchor)
selected_job_id = render_intervention_queue(
    quadrant_jobs,
    max_rows=10,
    on_job_select=lambda job_id: st.session_state.update({"intervention_selected_job": job_id}),
)

# Update session state if job selected
if selected_job_id:
    st.session_state["intervention_selected_job"] = selected_job_id

# SECTION 3: Selected Job Brief (conditional)
if st.session_state.get("intervention_selected_job"):
    selected_job = quadrant_jobs[quadrant_jobs["job_no"] == st.session_state["intervention_selected_job"]]
    if len(selected_job) > 0:
        render_selected_job_brief(selected_job.iloc[0], peer_segment=quadrant_jobs)

        # SECTION 4: Driver Analysis
        render_driver_analysis(
            selected_job.iloc[0],
            task_data=None,  # TODO: Pass task-level data if available
            staffing_data=None,  # TODO: Pass staffing breakdown if available
        )

        # SECTION 5: Peer Context
        render_peer_context(selected_job.iloc[0], peer_segment=quadrant_jobs)

# SECTION 6: Quadrant Trend
# TODO: Build quadrant_trend_data with time-series
# render_quadrant_trend(quadrant_trend_data, quadrant_name)

# Methodology
render_methodology_expander()
```

### Phase 3: Data Preparation (TO DO)

Before calling the components, ensure `quadrant_detail` DataFrame has these columns:

**Required Columns**:
```python
- job_no
- client
- is_active
- margin_pct_to_date
- margin_pct_quote
- revenue_to_date
- quoted_amount
- quote_to_revenue  (= revenue_to_date / quoted_amount)
- actual_hours
- quoted_hours
- hours_overrun_pct  (= (actual - quoted) / quoted * 100)
- realised_rate  (= revenue_to_date / actual_hours)
- quote_rate  (= quoted_amount / quoted_hours)
- runtime_days  (= days since job start)
- peer_median_runtime_days  (= median runtime for peers in category)
- delivery_lead
```

**Optional but Recommended**:
```python
- department_final
- category_rev_job  (or job_category)
- task_data  (if available)
- staffing_mix  (if available)
```

Most of these are already computed in the existing code (lines 974-1050). No changes needed to data prep; just verify the column names match.

### Phase 4: Testing Checklist

- [ ] Page loads without import errors
- [ ] Section 1 displays health KPIs correctly
- [ ] Section 2 queue ranks jobs by risk_score DESC
- [ ] User can select job from queue
- [ ] Section 3 job brief populates on selection
- [ ] Section 4 driver analysis tabs render
- [ ] Section 5 peer context computes percentiles
- [ ] <2 min to find top 5 intervention jobs + reason codes
- [ ] Methodology expander is clear and complete

---

## Data Flow Diagram

```
Quadrant Detail Data
    ↓
[compute_intervention_risk_score] → risk_score, reason_codes
    ↓
[build_intervention_queue] → ranked worklist
    ↓
Section 2: render_intervention_queue()
    ↓
User selects job
    ↓
Section 3: render_selected_job_brief(job_row)
    ↓
Section 4: render_driver_analysis(job_row, task_data, staffing_data)
    ↓
Section 5: render_peer_context(job_row, peer_segment)
    ↓
Section 6: render_quadrant_trend(quadrant_data)
```

---

## Success Criteria

✅ **Delivery leader can answer in <2 min**:
1. "What are the top 5 jobs to intervene on?" → Section 2 Queue
2. "Why are they at risk?" → Section 2 reason codes + Section 3 brief
3. "What is driving the issue?" → Section 4 driver analysis
4. "What action should be taken?" → Section 4 recommendation
5. "Is this quadrant improving?" → Section 6 trend

✅ **No data quality hiding**:
- Reason codes are transparent (see "Methodology")
- Peer context shows if issue is unique or systemic
- Data quality panel available (TODO: add to methodology expander)

✅ **Performance**: Page renders in <3 seconds

---

## Next Steps

1. **Run syntax check**: `python3 -m py_compile pages/6_Job_Mix_and_Demand.py` ✅ Done
2. **Replace Quadrant section** with orchestration code (see Phase 2)
3. **Test page** in Streamlit: `streamlit run app.py`
4. **Gather user feedback** from delivery leaders
5. **Iterate** based on feedback (e.g., add task/staffing data if available)

---

## Notes for Future Enhancements

- **Task-level drill**: Add task mix breakdown (job_task_pct vs peer_median)
- **Staffing breakdown**: Add seniority/role/geography mix comparison
- **What-if scenarios**: "If we moved 2 FTEs from task X to Y, margin would improve by Z%"
- **Recommendation engine**: Auto-suggest actions based on driver analysis
- **Benchmark comparison**: Show quote quality (are these jobs priced right category-wide?)
- **Export & sharing**: Download job brief + recommendations as PDF

---

**Created**: 2025-01-28  
**Version**: 1.0  
**Status**: Ready for phase 2 (page orchestration)
