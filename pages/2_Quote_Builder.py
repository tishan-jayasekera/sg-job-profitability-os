"""
Quote Builder Page

Build quote templates from historical benchmarks with task recommendations,
economics preview, and export capabilities.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import (
    init_state, get_state, set_state,
    QuotePlan, QuotePlanTask, get_quote_plan, set_quote_plan
)
from src.ui.layout import section_header
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_rate
from src.data.loader import load_fact_timesheet, load_mart
from src.data.semantic import safe_quote_job_task
from src.data.cohorts import (
    filter_by_time_window, filter_active_staff,
    compute_recency_weights, get_benchmark_metadata
)
from src.config import config


st.set_page_config(page_title="Quote Builder", page_icon="📝", layout="wide")

init_state()


def get_task_benchmarks(df: pd.DataFrame, department: str, category: str,
                        recency_weighted: bool = False) -> pd.DataFrame:
    """
    Get task benchmarks for a department/category combination.
    """
    # Filter to department and category
    mask = (df["department_final"] == department) & (df["job_category"] == category)
    df_slice = df[mask].copy()
    
    if len(df_slice) == 0:
        return pd.DataFrame()
    
    # Get job-task level quote data
    job_task = safe_quote_job_task(df_slice)
    
    if len(job_task) == 0:
        return pd.DataFrame()
    
    # Merge actual hours
    actuals = df_slice.groupby(["job_no", "task_name"]).agg(
        actual_hours=("hours_raw", "sum"),
        actual_cost=("base_cost", "sum")
    ).reset_index()
    
    job_task = job_task.merge(actuals, on=["job_no", "task_name"], how="left")
    
    # Apply recency weighting if enabled
    if recency_weighted:
        # Get month for each job-task (use first activity)
        first_month = df_slice.groupby(["job_no", "task_name"])["month_key"].min().reset_index()
        first_month.columns = ["job_no", "task_name", "first_month"]
        job_task = job_task.merge(first_month, on=["job_no", "task_name"], how="left")
        
        job_task["weight"] = compute_recency_weights(
            job_task, date_col="first_month",
            half_life_months=config.recency_half_life_months
        )
    else:
        job_task["weight"] = 1.0
    
    # Aggregate to task level with benchmarks
    task_stats = []
    
    for task in job_task["task_name"].unique():
        task_data = job_task[job_task["task_name"] == task]
        
        # Compute metrics
        n_jobs = task_data["job_no"].nunique()
        
        # Weighted percentiles for quoted hours
        quoted = task_data["quoted_time_total"].dropna()
        weights = task_data.loc[quoted.index, "weight"]
        
        if len(quoted) > 0:
            # Simple weighted stats
            quoted_p25 = quoted.quantile(0.25)
            quoted_p50 = quoted.quantile(0.50)
            quoted_p75 = quoted.quantile(0.75)
        else:
            quoted_p25 = quoted_p50 = quoted_p75 = 0
        
        # Actual hours stats
        actual = task_data["actual_hours"].dropna()
        if len(actual) > 0:
            actual_p50 = actual.quantile(0.50)
        else:
            actual_p50 = 0
        
        # Overrun probability
        if len(task_data) > 0 and "quoted_time_total" in task_data.columns:
            task_data_valid = task_data.dropna(subset=["quoted_time_total", "actual_hours"])
            if len(task_data_valid) > 0:
                overrun_rate = (task_data_valid["actual_hours"] > task_data_valid["quoted_time_total"] * 1.2).mean()
            else:
                overrun_rate = 0
        else:
            overrun_rate = 0
        
        # Cost per hour (median)
        if "actual_cost" in task_data.columns and "actual_hours" in task_data.columns:
            task_data["cost_per_hour"] = np.where(
                task_data["actual_hours"] > 0,
                task_data["actual_cost"] / task_data["actual_hours"],
                np.nan
            )
            cost_per_hour = task_data["cost_per_hour"].median()
        else:
            cost_per_hour = 0
        
        # Quote rate (median)
        if "quote_rate" in task_data.columns:
            quote_rate = task_data["quote_rate"].median()
        else:
            quote_rate = 0
        
        # Inclusion rate (what % of jobs in this slice have this task)
        total_jobs = df_slice["job_no"].nunique()
        inclusion_rate = n_jobs / total_jobs * 100 if total_jobs > 0 else 0
        
        task_stats.append({
            "task_name": task,
            "n_jobs": n_jobs,
            "inclusion_rate": inclusion_rate,
            "quoted_hours_p25": quoted_p25,
            "quoted_hours_p50": quoted_p50,
            "quoted_hours_p75": quoted_p75,
            "actual_hours_p50": actual_p50,
            "overrun_risk": overrun_rate * 100,
            "cost_per_hour": cost_per_hour,
            "quote_rate": quote_rate,
        })
    
    result = pd.DataFrame(task_stats)
    result = result.sort_values("inclusion_rate", ascending=False)
    
    return result


def main():
    st.title("Quote Builder")
    st.caption("Build quote templates from historical benchmarks")
    
    # Load data
    df = load_fact_timesheet()
    
    # =========================================================================
    # INPUT CONTROLS (LEFT RAIL)
    # =========================================================================
    col_inputs, col_main = st.columns([1, 3])
    
    with col_inputs:
        section_header("Configuration")
        
        # Department selection
        departments = sorted(df["department_final"].dropna().unique().tolist())
        selected_dept = st.selectbox(
            "Department",
            options=departments,
            key="quote_dept"
        )
        
        # Category selection (filtered by department)
        if selected_dept:
            categories = sorted(
                df[df["department_final"] == selected_dept]["job_category"]
                .dropna().unique().tolist()
            )
        else:
            categories = []
        
        selected_cat = st.selectbox(
            "Category",
            options=categories,
            key="quote_cat"
        )
        
        st.divider()
        
        # Benchmark window
        window_options = {
            "3m": "Last 3 months",
            "6m": "Last 6 months",
            "12m": "Last 12 months",
            "24m": "Last 24 months",
            "all": "All time",
        }
        
        benchmark_window = st.selectbox(
            "Benchmark Window",
            options=list(window_options.keys()),
            format_func=lambda x: window_options[x],
            index=2,  # Default to 12m
            key="quote_window"
        )
        
        # Recency weighting
        recency_weighted = st.checkbox(
            "Apply Recency Weighting",
            value=get_state("recency_weighted"),
            help="Weight recent jobs more heavily using exponential decay"
        )
        set_state("recency_weighted", recency_weighted)
        
        # Active staff only
        active_staff_only = st.checkbox(
            "Active Staff Only",
            value=get_state("active_staff_only"),
            help="Only include staff with recent activity"
        )
        set_state("active_staff_only", active_staff_only)
    
    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    with col_main:
        if not selected_dept or not selected_cat:
            st.info("Select a department and category to build a quote template.")
            return
        
        # Filter data
        df_filtered = filter_by_time_window(df, benchmark_window)
        
        if active_staff_only:
            df_filtered = filter_active_staff(df_filtered)
        
        # Get benchmark metadata
        mask = (df_filtered["department_final"] == selected_dept) & \
               (df_filtered["job_category"] == selected_cat)
        df_slice = df_filtered[mask]
        
        meta = get_benchmark_metadata(df_slice, recency_weighted=recency_weighted)
        
        # Display metadata
        st.markdown(f"**Benchmark:** {meta['n_jobs']} jobs, {meta['n_staff']} staff")
        if meta["date_min"] and meta["date_max"]:
            st.caption(f"Date range: {meta['date_min'].strftime('%b %Y')} - {meta['date_max'].strftime('%b %Y')}")
        
        st.markdown("---")
        
        # Get task benchmarks
        benchmarks = get_task_benchmarks(
            df_filtered, selected_dept, selected_cat,
            recency_weighted=recency_weighted
        )
        
        if len(benchmarks) == 0:
            st.warning("No task data available for this selection.")
            return
        
        # =====================================================================
        # TASK TEMPLATE TABLE
        # =====================================================================
        section_header("Task Template", "Select tasks and adjust hours for your quote")
        
        # Initialize plan in session state if not exists
        plan_key = f"quote_plan_{selected_dept}_{selected_cat}"
        
        if plan_key not in st.session_state:
            # Default: include tasks with >50% inclusion rate
            default_tasks = benchmarks[benchmarks["inclusion_rate"] >= 50]["task_name"].tolist()
            st.session_state[plan_key] = {
                task: benchmarks[benchmarks["task_name"] == task]["quoted_hours_p50"].iloc[0]
                for task in default_tasks
            }
        
        # Display editable table
        edited_df = st.data_editor(
            benchmarks[[
                "task_name", "inclusion_rate", "quoted_hours_p50",
                "quoted_hours_p25", "quoted_hours_p75",
                "overrun_risk", "cost_per_hour", "quote_rate"
            ]].rename(columns={
                "task_name": "Task",
                "inclusion_rate": "Inclusion %",
                "quoted_hours_p50": "Suggested Hours",
                "quoted_hours_p25": "Low (p25)",
                "quoted_hours_p75": "High (p75)",
                "overrun_risk": "Overrun Risk %",
                "cost_per_hour": "Cost/hr",
                "quote_rate": "Quote Rate",
            }),
            column_config={
                "Task": st.column_config.TextColumn("Task", disabled=True),
                "Inclusion %": st.column_config.NumberColumn("Inclusion %", format="%.1f%%", disabled=True),
                "Suggested Hours": st.column_config.NumberColumn("Hours", min_value=0, step=0.5),
                "Low (p25)": st.column_config.NumberColumn("Low", format="%.1f", disabled=True),
                "High (p75)": st.column_config.NumberColumn("High", format="%.1f", disabled=True),
                "Overrun Risk %": st.column_config.NumberColumn("Risk %", format="%.0f%%", disabled=True),
                "Cost/hr": st.column_config.NumberColumn("Cost/hr", format="$%.0f", disabled=True),
                "Quote Rate": st.column_config.NumberColumn("Rate", format="$%.0f/hr", disabled=True),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="task_editor"
        )
        
        st.markdown("---")
        
        # =====================================================================
        # ECONOMICS PREVIEW
        # =====================================================================
        section_header("Quote Economics")
        
        # Calculate totals from edited data
        total_hours = edited_df["Suggested Hours"].sum()
        
        # Merge back cost/rate info
        edited_df = edited_df.merge(
            benchmarks[["task_name", "cost_per_hour", "quote_rate"]].rename(
                columns={"task_name": "Task"}
            ),
            on="Task",
            how="left",
            suffixes=("", "_bench")
        )
        
        # Use benchmark values for cost and rate
        edited_df["task_cost"] = edited_df["Suggested Hours"] * edited_df["cost_per_hour_bench"].fillna(
            benchmarks["cost_per_hour"].median()
        )
        edited_df["task_value"] = edited_df["Suggested Hours"] * edited_df["quote_rate_bench"].fillna(
            benchmarks["quote_rate"].median()
        )
        
        total_cost = edited_df["task_cost"].sum()
        total_value = edited_df["task_value"].sum()
        total_margin = total_value - total_cost
        margin_pct = total_margin / total_value * 100 if total_value > 0 else 0
        
        # Display
        econ_cols = st.columns(5)
        
        with econ_cols[0]:
            st.metric("Total Hours", fmt_hours(total_hours))
        
        with econ_cols[1]:
            st.metric("Est. Cost", fmt_currency(total_cost))
        
        with econ_cols[2]:
            st.metric("Est. Value", fmt_currency(total_value))
        
        with econ_cols[3]:
            st.metric("Est. Margin", fmt_currency(total_margin))
        
        with econ_cols[4]:
            st.metric("Margin %", fmt_percent(margin_pct))
        
        st.markdown("---")
        
        # =====================================================================
        # ACTIONS
        # =====================================================================
        action_cols = st.columns(4)
        
        with action_cols[0]:
            if st.button("Save Quote Plan", type="primary"):
                # Build plan object
                tasks = []
                for _, row in edited_df.iterrows():
                    if row["Suggested Hours"] > 0:
                        tasks.append(QuotePlanTask(
                            task_name=row["Task"],
                            hours=row["Suggested Hours"],
                            cost_per_hour=row.get("cost_per_hour_bench", 0) or 0,
                            quote_rate=row.get("quote_rate_bench", 0) or 0,
                        ))
                
                plan = QuotePlan(
                    department=selected_dept,
                    category=selected_cat,
                    tasks=tasks,
                    benchmark_window=benchmark_window,
                    recency_weighted=recency_weighted,
                    created_at=datetime.now().isoformat(),
                )
                
                set_quote_plan(plan)
                st.success("Quote plan saved!")
        
        with action_cols[1]:
            if st.button("Send to Capacity Planner"):
                # Save and redirect
                tasks = []
                for _, row in edited_df.iterrows():
                    if row["Suggested Hours"] > 0:
                        tasks.append(QuotePlanTask(
                            task_name=row["Task"],
                            hours=row["Suggested Hours"],
                            cost_per_hour=row.get("cost_per_hour_bench", 0) or 0,
                            quote_rate=row.get("quote_rate_bench", 0) or 0,
                        ))
                
                plan = QuotePlan(
                    department=selected_dept,
                    category=selected_cat,
                    tasks=tasks,
                    benchmark_window=benchmark_window,
                    recency_weighted=recency_weighted,
                    created_at=datetime.now().isoformat(),
                )
                
                set_quote_plan(plan)
                st.switch_page("pages/3_Capacity_Staffing.py")
        
        with action_cols[2]:
            # Export as CSV
            export_df = edited_df[["Task", "Suggested Hours"]].copy()
            export_df["Department"] = selected_dept
            export_df["Category"] = selected_cat
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                "Export CSV",
                data=csv,
                file_name=f"quote_plan_{selected_dept}_{selected_cat}.csv",
                mime="text/csv"
            )
        
        with action_cols[3]:
            if st.button("Clear Plan"):
                st.session_state.pop(plan_key, None)
                st.rerun()


if __name__ == "__main__":
    main()
