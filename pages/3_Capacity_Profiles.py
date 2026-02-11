"""
Capacity & Profiles Page

Empirical capacity, load, and capability profiling.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.data.semantic import get_category_col
from src.data.cohorts import get_active_jobs
from src.data.profiles import (
    build_staff_profiles,
    compute_task_expertise,
    compute_category_expertise,
    compute_staff_capacity,
    compute_expected_load,
    compute_headroom,
)
from src.staffing.engine import recommend_staff_for_plan, get_capability_coverage
from src.ui.state import init_state, get_quote_plan
from src.ui.formatting import fmt_hours, fmt_percent
from src.config import config


st.set_page_config(page_title="Capacity & Profiles", page_icon="C", layout="wide")

init_state()


@st.cache_data(show_spinner=False)
def build_profile_data(df: pd.DataFrame, window_weeks: int, training_months: int):
    date_col = "work_date" if "work_date" in df.columns else "month_key"
    df_window = df.copy()
    if date_col in df_window.columns:
        ref = pd.to_datetime(df_window[date_col]).max()
        cutoff = ref - pd.DateOffset(weeks=window_weeks)
        df_window = df_window[pd.to_datetime(df_window[date_col]) >= cutoff]
    
    profiles = build_staff_profiles(df, window_weeks, training_months, df_window=df_window)
    task_expertise = compute_task_expertise(df, training_months, config.RECENCY_HALF_LIFE_MONTHS)
    category_expertise = compute_category_expertise(df, training_months, config.RECENCY_HALF_LIFE_MONTHS)
    capacity = compute_staff_capacity(df_window, window_weeks)
    expected = compute_expected_load(df_window, window_weeks, window_weeks)
    headroom = compute_headroom(capacity, expected)
    return profiles, task_expertise, category_expertise, headroom, capacity, expected, df_window


def _staff_weekly_series(df: pd.DataFrame, staff_name: str) -> pd.DataFrame:
    date_col = "work_date" if "work_date" in df.columns else "month_key"
    df_staff = df[df["staff_name"] == staff_name].copy()
    if len(df_staff) == 0:
        return pd.DataFrame()
    df_staff[date_col] = pd.to_datetime(df_staff[date_col])
    weekly = df_staff.set_index(date_col)["hours_raw"].resample("W").sum().reset_index()
    return weekly.tail(12)


@st.fragment
def _render_active_jobs_overlay_fragment(
    df: pd.DataFrame,
    window: int,
    selected_dept: str,
    selected_category: str,
    category_col: str,
    headroom_df: pd.DataFrame,
) -> None:
    st.subheader("Section 2.1 — Active Jobs Demand Overlay")
    active_job_ids = get_active_jobs(df)
    df_active = df[df["job_no"].isin(active_job_ids)] if "job_no" in df.columns else pd.DataFrame()

    if len(df_active) == 0:
        st.info("No active jobs found for the selected filters.")
        return

    df_active = df_active.copy()
    if "fte_hours_scaling" not in df_active.columns:
        df_active["fte_hours_scaling"] = 1.0
    df_active["fte_hours_scaling"] = df_active["fte_hours_scaling"].fillna(1.0)
    window_capacity = config.CAPACITY_HOURS_PER_WEEK * window
    df_active["fte_equiv_window"] = df_active["hours_raw"] / window_capacity if window_capacity > 0 else np.nan
    df_active["fte_equiv_scaled"] = df_active["hours_raw"] / (window_capacity * df_active["fte_hours_scaling"])

    overlay_cols = st.columns(2)
    with overlay_cols[0]:
        st.markdown("**Active Demand (Bottom‑Up by Task)**")
        task_bottom = df_active.groupby("task_name").agg(
            active_hours=("hours_raw", "sum"),
            fte_equiv=("fte_equiv_window", "sum"),
            fte_equiv_scaled=("fte_equiv_scaled", "sum"),
        ).reset_index().sort_values("active_hours", ascending=False)

        total_active_hours = task_bottom["active_hours"].sum()
        total_fte = task_bottom["fte_equiv"].sum()
        st.caption(
            f"Bottom‑up active demand totals: {fmt_hours(total_active_hours)} hours "
            f"(~{total_fte:.2f} FTE‑equiv across {window}w)."
        )
        st.dataframe(
            task_bottom.head(12).rename(columns={
                "task_name": "Task",
                "active_hours": "Active Hours",
                "fte_equiv": "FTE‑equiv (window)",
                "fte_equiv_scaled": "FTE‑equiv (scaled)",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Active Hours": st.column_config.NumberColumn(format="%.1f"),
                "FTE‑equiv (window)": st.column_config.NumberColumn(format="%.2f"),
                "FTE‑equiv (scaled)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.markdown("**Active Job Task Demand vs Historical Benchmarks**")
        task_active = df_active.groupby("task_name")["hours_raw"].sum().sort_values(ascending=False)
        active_total = task_active.sum()
        task_active = task_active.reset_index().rename(columns={"task_name": "Task", "hours_raw": "Active Hours"})
        task_active["Active Share"] = task_active["Active Hours"] / active_total * 100 if active_total > 0 else np.nan

        df_completed = df.copy()
        if "job_completed_date" in df_completed.columns:
            df_completed = df_completed[df_completed["job_completed_date"].notna()]
        elif "job_status" in df_completed.columns:
            df_completed = df_completed[df_completed["job_status"].str.lower().str.contains("completed", na=False)]
        else:
            df_completed = df_completed.iloc[0:0]

        if selected_dept != "All":
            df_completed = df_completed[df_completed["department_final"] == selected_dept]
        if selected_category != "All":
            df_completed = df_completed[df_completed[category_col] == selected_category]

        bench = df_completed.groupby("task_name")["hours_raw"].sum()
        bench_total = bench.sum()
        bench = bench.reset_index().rename(columns={"task_name": "Task", "hours_raw": "Benchmark Hours"})
        bench["Benchmark Share"] = bench["Benchmark Hours"] / bench_total * 100 if bench_total > 0 else np.nan

        task_overlay = task_active.merge(bench[["Task", "Benchmark Share"]], on="Task", how="left")
        task_overlay["Share Delta (pp)"] = task_overlay["Active Share"] - task_overlay["Benchmark Share"]

        st.dataframe(
            task_overlay.head(12),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Active Hours": st.column_config.NumberColumn(format="%.1f"),
                "Active Share": st.column_config.NumberColumn(format="%.1f%%"),
                "Benchmark Share": st.column_config.NumberColumn(format="%.1f%%"),
                "Share Delta (pp)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption("Benchmark share is derived from completed jobs in the same department/category.")

    with overlay_cols[1]:
        st.markdown("**Active Demand (Bottom‑Up by Job)**")
        if "job_no" in df_active.columns:
            job_bottom = df_active.groupby("job_no").agg(
                active_hours=("hours_raw", "sum"),
                fte_equiv=("fte_equiv_window", "sum"),
            ).reset_index().sort_values("active_hours", ascending=False)
            st.dataframe(
                job_bottom.head(12).rename(columns={
                    "job_no": "Job",
                    "active_hours": "Active Hours",
                    "fte_equiv": "FTE‑equiv (window)",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Active Hours": st.column_config.NumberColumn(format="%.1f"),
                    "FTE‑equiv (window)": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        else:
            st.caption("Job numbers are not available in this dataset.")

        st.markdown("**Active Job Staffing Capacity Forecast**")
        active_staff = df_active.groupby("staff_name")["hours_raw"].sum().rename("active_hours").reset_index()
        cols = ["staff_name", "headroom_hours", "capacity_hours_window", "capacity_hours_week"]
        available_cols = [c for c in cols if c in headroom_df.columns]
        staff_capacity = headroom_df[available_cols].copy()
        staff_view = active_staff.merge(staff_capacity, on="staff_name", how="left")
        staff_view["forecast_available_hours"] = staff_view["headroom_hours"].fillna(0)
        staff_view = staff_view.sort_values("active_hours", ascending=False)

        st.dataframe(
            staff_view.head(12).rename(columns={
                "staff_name": "Staff",
                "active_hours": "Active Hours (window)",
                "headroom_hours": "Headroom (window)",
                "capacity_hours_week": "Capacity / week",
                "forecast_available_hours": "Forecast Available",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Active Hours (window)": st.column_config.NumberColumn(format="%.1f"),
                "Headroom (window)": st.column_config.NumberColumn(format="%.1f"),
                "Capacity / week": st.column_config.NumberColumn(format="%.1f"),
                "Forecast Available": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption("Forecast available uses headroom from trailing window.")


@st.fragment
def _render_staff_profiles_fragment(
    profiles: pd.DataFrame,
    category_expertise: pd.DataFrame,
    selected_dept: str,
    selected_category: str,
    df: pd.DataFrame,
    task_expertise: pd.DataFrame,
) -> None:
    st.subheader("Section 4 — Staff Profiles (Action View)")
    filter_cols = st.columns([0.3, 0.3, 0.4])
    with filter_cols[0]:
        search = st.text_input("Search staff", "")
    with filter_cols[1]:
        only_positive_headroom = st.checkbox("Only positive headroom", value=False)
    with filter_cols[2]:
        only_relevant = st.checkbox("Only relevant to selected category", value=False)

    display = profiles.copy()
    display["billable_pct"] = display["billable_ratio"] * 100

    if search:
        display = display[display["staff_name"].str.contains(search, case=False, na=False)]
    if only_positive_headroom:
        display = display[display["headroom_hours"] > 0]
    if selected_dept != "All":
        display = display[display["department_final"] == selected_dept]

    cat_scores = category_expertise.copy()
    if selected_category != "All":
        cat_scores = cat_scores[cat_scores["category_rev_job"] == selected_category]
    cat_scores = cat_scores.groupby("staff_name")["capability_score"].max().reset_index()
    display = display.merge(cat_scores, on="staff_name", how="left")
    if only_relevant:
        display = display[display["capability_score"].fillna(0) > 0]

    display_table = display[[
        "staff_name", "department_final", "archetype",
        "avg_hours_per_week", "billable_pct",
        "headroom_hours", "capability_score", "active_jobs_count",
    ]].rename(columns={
        "staff_name": "Name",
        "department_final": "Dept",
        "archetype": "Archetype",
        "avg_hours_per_week": "Hrs/Wk",
        "billable_pct": "Bill%",
        "headroom_hours": "Headroom (hrs)",
        "capability_score": "Capability (cat)",
        "active_jobs_count": "Jobs",
    })

    selection = st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="profiles_table",
        column_config={
            "Hrs/Wk": st.column_config.NumberColumn(format="%.1f"),
            "Bill%": st.column_config.NumberColumn(format="%.1f%%"),
            "Headroom (hrs)": st.column_config.NumberColumn(format="%.1f"),
            "Capability (cat)": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    selected_staff = None
    if selection and selection.selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        selected_staff = display.iloc[selected_idx]["staff_name"]

    if not selected_staff:
        return

    st.subheader(f"Section 5 — Staff Deep‑Dive: {selected_staff}")
    detail_cols = st.columns(3)
    staff_row = profiles[profiles["staff_name"] == selected_staff].iloc[0]
    with detail_cols[0]:
        st.metric("Avg Hours / Week", fmt_hours(staff_row.get("avg_hours_per_week")))
    with detail_cols[1]:
        st.metric("Billable %", fmt_percent(staff_row.get("billable_ratio") * 100))
    with detail_cols[2]:
        st.metric("Headroom (hrs)", fmt_hours(staff_row.get("headroom_hours")))

    weekly = _staff_weekly_series(df, selected_staff)
    if len(weekly) > 0:
        st.line_chart(weekly.set_index(weekly.columns[0])["hours_raw"])

    staff_tasks = task_expertise[task_expertise["staff_name"] == selected_staff]
    staff_cats = category_expertise[category_expertise["staff_name"] == selected_staff]

    top_tasks = staff_tasks.nlargest(5, "capability_score")[["task_name", "capability_score", "hours_total"]]
    top_cats = staff_cats.nlargest(3, "capability_score")[["category_rev_job", "capability_score", "hours_total"]]

    if len(top_tasks) > 0:
        st.markdown("**Top Tasks**")
        st.dataframe(top_tasks, use_container_width=True, hide_index=True)
    if len(top_cats) > 0:
        st.markdown("**Top Categories**")
        st.dataframe(top_cats, use_container_width=True, hide_index=True)
    if "job_no" in df.columns:
        jobs = df[df["staff_name"] == selected_staff].groupby("job_no")["hours_raw"].sum().reset_index()
        jobs = jobs.rename(columns={"hours_raw": "hours"}).sort_values("hours", ascending=False).head(10)
        st.markdown("**Active Jobs (Top 10 by hours)**")
        st.dataframe(jobs, use_container_width=True, hide_index=True)


def main():
    st.title("Capacity & Profiles")
    st.caption("Empirical capacity, capability, and coverage diagnostics (no targets)")
    
    df = load_fact_timesheet()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    departments = ["All"] + sorted(df["department_final"].dropna().unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)
    
    window = st.sidebar.selectbox("Window", options=[2, 4, 8, 12], index=1, format_func=lambda x: f"{x}w")
    
    with st.sidebar.expander("Eligibility Settings"):
        recency_months = st.slider("Recency months", 1, 12, config.ELIGIBILITY_RECENCY_MONTHS)
        min_hours = st.slider("Min hours", 1, 50, config.ELIGIBILITY_MIN_HOURS)
        min_jobs = st.slider("Min jobs", 1, 10, config.ELIGIBILITY_MIN_JOBS)
    
    if selected_dept != "All":
        df = df[df["department_final"] == selected_dept]

    category_col = get_category_col(df)
    categories = ["All"] + sorted(df[category_col].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Category", categories)
    if selected_category != "All":
        df = df[df[category_col] == selected_category]
    
    profiles, task_expertise, category_expertise, headroom_df, capacity_df, expected_df, df_window = build_profile_data(
        df, window, config.PROFILE_TRAINING_MONTHS
    )
    
    if len(df_window) == 0:
        st.warning("No activity found in the selected window.")
        return
    
    if len(profiles) == 0:
        st.warning("No active staff found in the selected window.")
        return
    
    # =========================
    # SECTION 1 — EXEC SNAPSHOT
    # =========================
    st.subheader("Section 1 — Executive Supply/Demand Snapshot")
    total_staff = profiles["staff_name"].nunique()
    total_staff_all = df["staff_name"].nunique() if "staff_name" in df.columns else total_staff
    total_fte = capacity_df["fte_scaling"].sum() if len(capacity_df) > 0 else 0
    total_capacity_week = total_fte * config.CAPACITY_HOURS_PER_WEEK
    total_capacity = total_capacity_week * window
    total_expected = expected_df["expected_load_hours"].sum() if len(expected_df) > 0 else 0
    total_expected_week = expected_df["avg_weekly_hours"].sum() if len(expected_df) > 0 else 0
    total_headroom = total_capacity - total_expected
    total_actual = df_window["hours_raw"].sum()
    billable_hours = (
        df_window[df_window["is_billable"] == True]["hours_raw"].sum()
        if "is_billable" in df_window.columns
        else np.nan
    )
    utilisation_total = total_actual / total_capacity if total_capacity > 0 else np.nan
    utilisation_billable = billable_hours / total_capacity if total_capacity > 0 else np.nan

    kpi_cols = st.columns(6)
    with kpi_cols[0]:
        st.metric("Active Staff", f"{total_staff:,}")
    with kpi_cols[1]:
        st.metric("Total FTE", f"{total_fte:.1f}")
    with kpi_cols[2]:
        st.metric("Capacity (Window)", fmt_hours(total_capacity))
    with kpi_cols[3]:
        st.metric("Expected Load", fmt_hours(total_expected))
    with kpi_cols[4]:
        st.metric("Headroom", fmt_hours(total_headroom))
    with kpi_cols[5]:
        st.metric("Utilisation (Total)", fmt_percent(utilisation_total * 100))

    st.caption(
        f"Active staff: {total_staff} of {total_staff_all}. "
        f"Capacity/week: {fmt_hours(total_capacity_week)} | Expected load/week: {fmt_hours(total_expected_week)}."
    )

    with st.expander("Methodology"):
        st.markdown(
            """
            **Capacity = pure supply.**  
            `Capacity (week) = 38 × fte_hours_scaling`  
            `Capacity (window) = Capacity/week × weeks`  
            
            **Expected Load = trailing average.**  
            `Expected Load = avg weekly hours (trailing) × weeks`  
            
            **Headroom = Capacity - Expected Load**  
            **Utilisation (Total)** = Actual Hours ÷ Capacity  
            **Billable Utilisation** = Billable Hours ÷ Capacity  
            """
        )

    st.divider()

    # =========================
    # SECTION 2 — CAPACITY VS DELIVERY
    # =========================
    st.subheader("Section 2 — Capacity vs Delivery")
    trend_cols = st.columns([0.6, 0.4])
    with trend_cols[0]:
        date_col = "work_date" if "work_date" in df_window.columns else "month_key"
        series = df_window.copy()
        series[date_col] = pd.to_datetime(series[date_col])
        weekly = series.set_index(date_col)["hours_raw"].resample("W").sum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly.index, y=weekly.values, mode="lines+markers", name="Actual"))
        if total_capacity_week > 0:
            fig.add_hline(y=total_capacity_week, line_dash="dash", annotation_text="Capacity / week")
        if total_expected_week > 0:
            fig.add_hline(y=total_expected_week, line_dash="dot", annotation_text="Expected / week")
        fig.update_layout(
            height=300,
            margin=dict(t=30, l=10, r=10, b=10),
            xaxis_title="Week",
            yaxis_title="Hours",
        )
        st.plotly_chart(fig, use_container_width=True)
    with trend_cols[1]:
        st.markdown("**Actual Delivery by Category**")
        category_hours = df_window.groupby(category_col)["hours_raw"].sum().sort_values(ascending=False).head(8)
        st.bar_chart(category_hours)
        st.caption("Top categories by actual delivery hours in the selected window.")

    st.divider()

    # =========================
    # SECTION 2.1 — ACTIVE JOBS DEMAND OVERLAY
    # =========================
    _render_active_jobs_overlay_fragment(
        df=df,
        window=window,
        selected_dept=selected_dept,
        selected_category=selected_category,
        category_col=category_col,
        headroom_df=headroom_df,
    )

    # =========================
    # SECTION 3 — COVERAGE & CAPABILITY RISK
    # =========================
    st.subheader("Section 3 — Coverage & Capability Risk")
    coverage = get_capability_coverage(category_expertise)
    if len(coverage) > 0:
        demand = df_window.groupby(category_col)["hours_raw"].sum().rename("demand_hours")
        coverage = coverage.merge(demand.reset_index().rename(columns={category_col: "category_rev_job"}), on="category_rev_job", how="left")

        headroom_lookup = headroom_df[["staff_name", "headroom_hours"]].copy()
        cat_staff = category_expertise[["staff_name", "category_rev_job", "capability_score"]]
        cat_staff = cat_staff[cat_staff["capability_score"] > 0]
        cat_headroom = cat_staff.merge(headroom_lookup, on="staff_name", how="left")
        cat_headroom = cat_headroom.groupby("category_rev_job")["headroom_hours"].sum().reset_index()
        coverage = coverage.merge(cat_headroom, on="category_rev_job", how="left")
        coverage["demand_gap"] = coverage["demand_hours"] - coverage["headroom_hours"]
        coverage["demand_gap"] = coverage["demand_gap"].fillna(0)

        coverage["Coverage Risk"] = coverage["coverage_risk"]
        coverage = coverage.rename(columns={
            "category_rev_job": "Category",
            "staff_count": "Staff w/ Expertise",
            "demand_hours": "Demand Hours",
            "headroom_hours": "Available Headroom",
            "demand_gap": "Gap (Demand - Headroom)",
        })
        st.dataframe(
            coverage[["Category", "Staff w/ Expertise", "Coverage Risk", "Demand Hours", "Available Headroom", "Gap (Demand - Headroom)"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Demand Hours": st.column_config.NumberColumn(format="%.1f"),
                "Available Headroom": st.column_config.NumberColumn(format="%.1f"),
                "Gap (Demand - Headroom)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
    else:
        st.info("No category expertise data available.")

    st.divider()

    # =========================
    # SECTION 4 — STAFF PROFILES (ACTION VIEW)
    # =========================
    _render_staff_profiles_fragment(
        profiles=profiles,
        category_expertise=category_expertise,
        selected_dept=selected_dept,
        selected_category=selected_category,
        df=df,
        task_expertise=task_expertise,
    )
    
    st.divider()
    
    # =========================
    # SECTION 6 — STAFFING RECOMMENDATIONS
    # =========================
    plan = get_quote_plan()
    if plan and plan.tasks:
        st.subheader("Section 6 — Staffing Recommendations")
        st.caption(f"For: {plan.department} / {plan.category}")
        
        quote_plan = {
            "department": plan.department,
            "category": plan.category,
            "tasks": [
                {"task_name": t.task_name, "hours": t.hours}
                for t in plan.tasks
                if not t.is_optional
            ],
        }
        
        eligibility_config = {
            "recency_months": recency_months,
            "min_hours": min_hours,
            "min_jobs": min_jobs,
        }
        
        recs, warnings = recommend_staff_for_plan(
            quote_plan,
            task_expertise,
            category_expertise,
            headroom_df,
            eligibility_config,
            top_n=3,
        )
        
        if len(recs) > 0:
            summary_rows = []
            for task_name, task_group in recs.groupby("task_name"):
                row = {"Task": task_name, "Hours": task_group["hours"].iloc[0]}
                for rank in [1, 2, 3]:
                    staff = task_group[task_group["rank"] == rank]
                    if len(staff) > 0:
                        row[f"#{rank} Staff"] = staff["staff_name"].iloc[0]
                        row[f"#{rank} Score"] = f"{staff['match_score'].iloc[0]:.0f}"
                    else:
                        row[f"#{rank} Staff"] = "-"
                        row[f"#{rank} Score"] = "-"
                task_warnings = [w.message for w in warnings if w.task == task_name]
                row["Warnings"] = " | ".join(sorted(set(task_warnings))) if task_warnings else ""
                summary_rows.append(row)
            
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No staffing recommendations available.")
    
    st.divider()
    
    # Export
    st.subheader("Export")
    st.download_button(
        "Download Profiles CSV",
        data=profiles.to_csv(index=False),
        file_name="staff_profiles.csv",
        mime="text/csv",
    )
    
    if plan and plan.tasks:
        st.download_button(
            "Download Recommendations CSV",
            data=recs.to_csv(index=False) if len(recs) > 0 else "",
            file_name="staff_recommendations.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
