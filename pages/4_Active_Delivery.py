"""
Active Delivery Page

Delivery control tower with risk flags, attribution, and interventions.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state, get_state
from src.ui.layout import section_header
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_rate
from src.ui.charts import horizontal_bar, grouped_bar
from src.data.loader import load_fact_timesheet
from src.data.semantic import safe_quote_job_task, profitability_rollup, get_category_col
from src.data.cohorts import get_active_jobs, filter_active_jobs
from src.metrics.rate_capture import compute_rate_variance_diagnosis, compute_rate_metrics
from src.config import config


st.set_page_config(page_title="Active Delivery", page_icon="🎯", layout="wide")

init_state()


def build_active_jobs_view(df: pd.DataFrame) -> pd.DataFrame:
    """Build active jobs with risk metrics."""
    
    # Get active job list
    active_job_nos = get_active_jobs(df)
    df_active = df[df["job_no"].isin(active_job_nos)].copy()
    
    if len(df_active) == 0:
        return pd.DataFrame()
    
    # Job-level aggregation
    category_col = get_category_col(df_active)
    job_agg = df_active.groupby("job_no").agg(
        department_final=("department_final", "first"),
        job_category=(category_col, "first"),
        actual_hours=("hours_raw", "sum"),
        actual_cost=("base_cost", "sum"),
        actual_revenue=("rev_alloc", "sum"),
    ).reset_index()
    
    # Add optional fields
    if "client" in df_active.columns:
        clients = df_active.groupby("job_no")["client"].first()
        job_agg = job_agg.merge(clients.reset_index(), on="job_no", how="left")
    
    if "job_status" in df_active.columns:
        statuses = df_active.groupby("job_no")["job_status"].first()
        job_agg = job_agg.merge(statuses.reset_index(), on="job_no", how="left")
    
    if "job_due_date" in df_active.columns:
        due_dates = df_active.groupby("job_no")["job_due_date"].first()
        job_agg = job_agg.merge(due_dates.reset_index(), on="job_no", how="left")
    
    # Safe quote totals
    job_task = safe_quote_job_task(df_active)
    if len(job_task) > 0:
        job_quotes = job_task.groupby("job_no").agg(
            quoted_hours=("quoted_time_total", "sum"),
            quoted_amount=("quoted_amount_total", "sum"),
        ).reset_index()
        job_agg = job_agg.merge(job_quotes, on="job_no", how="left")
    else:
        job_agg["quoted_hours"] = np.nan
        job_agg["quoted_amount"] = np.nan
    
    # Compute metrics
    job_agg["pct_consumed"] = np.where(
        job_agg["quoted_hours"] > 0,
        job_agg["actual_hours"] / job_agg["quoted_hours"] * 100,
        np.nan
    )
    
    job_agg["hours_variance"] = job_agg["actual_hours"] - job_agg["quoted_hours"]
    job_agg["hours_variance_pct"] = np.where(
        job_agg["quoted_hours"] > 0,
        job_agg["hours_variance"] / job_agg["quoted_hours"] * 100,
        np.nan
    )
    
    # Scope creep per job
    if "quote_match_flag" in df_active.columns:
        scope_by_job = df_active.groupby("job_no").apply(
            lambda x: x[x["quote_match_flag"] != "matched"]["hours_raw"].sum() / x["hours_raw"].sum() * 100
            if x["hours_raw"].sum() > 0 else 0
        ).reset_index()
        scope_by_job.columns = ["job_no", "scope_creep_pct"]
        job_agg = job_agg.merge(scope_by_job, on="job_no", how="left")
    else:
        job_agg["scope_creep_pct"] = 0
    
    # Rate metrics
    job_agg["realised_rate"] = np.where(
        job_agg["actual_hours"] > 0,
        job_agg["actual_revenue"] / job_agg["actual_hours"],
        np.nan
    )
    
    job_agg["quote_rate"] = np.where(
        job_agg["quoted_hours"] > 0,
        job_agg["quoted_amount"] / job_agg["quoted_hours"],
        np.nan
    )
    
    job_agg["rate_variance"] = job_agg["realised_rate"] - job_agg["quote_rate"]
    
    # Risk flag
    job_agg["risk_flag"] = "on_track"
    job_agg.loc[job_agg["pct_consumed"] > 80, "risk_flag"] = "watch"
    job_agg.loc[job_agg["pct_consumed"] > 100, "risk_flag"] = "at_risk"
    
    # Sort by risk
    risk_order = {"at_risk": 0, "watch": 1, "on_track": 2}
    job_agg["risk_order"] = job_agg["risk_flag"].map(risk_order)
    job_agg = job_agg.sort_values(["risk_order", "pct_consumed"], ascending=[True, False])
    
    return job_agg


def get_job_task_breakdown(df: pd.DataFrame, job_no: str) -> pd.DataFrame:
    """Get task breakdown for a specific job."""
    
    df_job = df[df["job_no"] == job_no].copy()
    
    # Task aggregation
    task_agg = df_job.groupby("task_name").agg(
        actual_hours=("hours_raw", "sum"),
        actual_cost=("base_cost", "sum"),
    ).reset_index()
    
    # Quote data
    job_task = safe_quote_job_task(df_job)
    if len(job_task) > 0:
        task_agg = task_agg.merge(
            job_task[["task_name", "quoted_time_total"]].rename(
                columns={"quoted_time_total": "quoted_hours"}
            ),
            on="task_name",
            how="left"
        )
    else:
        task_agg["quoted_hours"] = np.nan
    
    task_agg["variance"] = task_agg["actual_hours"] - task_agg["quoted_hours"]
    task_agg["variance_pct"] = np.where(
        task_agg["quoted_hours"] > 0,
        task_agg["variance"] / task_agg["quoted_hours"] * 100,
        np.nan
    )
    
    return task_agg.sort_values("variance", ascending=False)


def get_job_staff_breakdown(df: pd.DataFrame, job_no: str) -> pd.DataFrame:
    """Get staff breakdown for a specific job."""
    
    df_job = df[df["job_no"] == job_no].copy()
    
    staff_agg = df_job.groupby("staff_name").agg(
        hours=("hours_raw", "sum"),
        cost=("base_cost", "sum"),
        tasks=("task_name", "nunique"),
    ).reset_index()
    
    total_hours = staff_agg["hours"].sum()
    staff_agg["pct_of_job"] = staff_agg["hours"] / total_hours * 100 if total_hours > 0 else 0
    
    return staff_agg.sort_values("hours", ascending=False)


def main():
    st.title("Active Delivery")
    st.caption("Delivery control tower with risk monitoring")
    
    # Load data
    df = load_fact_timesheet()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    departments = ["All"] + sorted(df["department_final"].dropna().unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)
    
    if selected_dept != "All":
        df = df[df["department_final"] == selected_dept]
    
    risk_filter = st.sidebar.multiselect(
        "Risk Status",
        options=["at_risk", "watch", "on_track"],
        default=["at_risk", "watch"]
    )
    
    # Build active jobs view
    jobs_df = build_active_jobs_view(df)
    
    if len(jobs_df) == 0:
        st.warning("No active jobs found.")
        return
    
    # Apply risk filter
    if risk_filter:
        jobs_df = jobs_df[jobs_df["risk_flag"].isin(risk_filter)]
    
    # =========================================================================
    # SECTION A: SUMMARY KPIS
    # =========================================================================
    section_header("Active Jobs Summary")
    
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        st.metric("Active Jobs", len(jobs_df))
    
    with kpi_cols[1]:
        at_risk = len(jobs_df[jobs_df["risk_flag"] == "at_risk"])
        st.metric("At Risk", at_risk)
    
    with kpi_cols[2]:
        watch = len(jobs_df[jobs_df["risk_flag"] == "watch"])
        st.metric("Watch", watch)
    
    with kpi_cols[3]:
        total_hours_var = jobs_df["hours_variance"].sum()
        st.metric("Total Hours Variance", fmt_hours(total_hours_var))
    
    with kpi_cols[4]:
        avg_scope_creep = jobs_df["scope_creep_pct"].mean()
        st.metric("Avg Scope Creep", fmt_percent(avg_scope_creep))
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION B: JOBS TABLE
    # =========================================================================
    section_header("Active Jobs", "Click a row to see details")
    
    # Prepare display columns
    display_cols = ["job_no", "department_final", "job_category"]
    
    if "client" in jobs_df.columns:
        display_cols.append("client")
    
    display_cols.extend([
        "quoted_hours", "actual_hours", "pct_consumed",
        "scope_creep_pct", "risk_flag"
    ])
    
    if "job_due_date" in jobs_df.columns:
        display_cols.insert(-1, "job_due_date")
    
    display_df = jobs_df[[c for c in display_cols if c in jobs_df.columns]].copy()
    
    # Rename for display
    col_rename = {
        "job_no": "Job",
        "department_final": "Department",
        "job_category": "Category",
        "client": "Client",
        "quoted_hours": "Quoted",
        "actual_hours": "Actual",
        "pct_consumed": "% Consumed",
        "scope_creep_pct": "Scope Creep",
        "job_due_date": "Due Date",
        "risk_flag": "Status",
    }
    display_df = display_df.rename(columns=col_rename)
    
    # Selection
    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Quoted": st.column_config.NumberColumn(format="%.0f"),
            "Actual": st.column_config.NumberColumn(format="%.0f"),
            "% Consumed": st.column_config.NumberColumn(format="%.1f%%"),
            "Scope Creep": st.column_config.NumberColumn(format="%.1f%%"),
            "Due Date": st.column_config.DateColumn(),
            "Status": st.column_config.TextColumn(),
        },
        key="jobs_table"
    )
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION C: RATE VARIANCE (TOP-DOWN)
    # =========================================================================
    section_header("Rate Variance - Top-Down View", "Department -> Category -> Job -> Task -> Staff")
    
    category_col = get_category_col(df)
    keys = ["department_final", category_col, "job_no", "task_name", "staff_name"]
    rate_tree = compute_rate_metrics(df, keys)
    if len(rate_tree) > 0:
        rate_tree["rate_gap"] = rate_tree["rate_variance"] * rate_tree["hours"]
        rate_tree["rate_gap_abs"] = rate_tree["rate_gap"].abs()
        rate_tree["node_weight"] = rate_tree["rate_gap_abs"].replace(0, np.nan)
        rate_tree = rate_tree.dropna(subset=["node_weight"])
        rate_tree = rate_tree.rename(columns={category_col: "category"})
        
        total_gap = rate_tree["rate_gap"].sum()
        total_hours = rate_tree["hours"].sum()
        avg_variance = rate_tree["rate_variance"].mean()
        
        kpi_cols = st.columns(3)
        with kpi_cols[0]:
            st.metric("Total Rate Gap", fmt_currency(total_gap))
        with kpi_cols[1]:
            st.metric("Total Hours", fmt_hours(total_hours))
        with kpi_cols[2]:
            st.metric("Avg Rate Var", fmt_rate(avg_variance))
        
        tab1, tab2, tab3 = st.tabs(["Visual Map", "Driver Tables", "Waterfalls"])
        
        with tab1:
            fig = px.treemap(
                rate_tree,
                path=["department_final", "category", "job_no", "task_name", "staff_name"],
                values="node_weight",
                color="rate_variance",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                custom_data=["department_final", "category", "job_no", "task_name", "staff_name"],
            )
            fig.update_layout(margin=dict(t=30, l=10, r=10, b=10))
            selection = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
            )
            
            selected_path = None
            if selection and selection.selection and selection.selection.points:
                point = selection.selection.points[0]
                if "customdata" in point:
                    selected_path = point["customdata"]
                elif "label" in point:
                    selected_path = [point["label"]]
            
            st.markdown("**Selected Path**")
            if selected_path:
                st.code(" > ".join([p for p in selected_path if p]))
            else:
                st.caption("Click a node in the treemap to drill into details.")
            
            st.markdown("**Detail Panel (Job / Task / Staff)**")
            detail = rate_tree.copy()
            if selected_path:
                dept, cat, job, task, staff = selected_path
                if dept:
                    detail = detail[detail["department_final"] == dept]
                if cat:
                    detail = detail[detail["category"] == cat]
                if job:
                    detail = detail[detail["job_no"] == job]
                if task:
                    detail = detail[detail["task_name"] == task]
                if staff:
                    detail = detail[detail["staff_name"] == staff]
            detail = detail.sort_values("rate_gap").head(20)
            st.dataframe(
                detail[[
                    "department_final", "category", "job_no", "task_name", "staff_name",
                    "quote_rate", "realised_rate", "rate_variance", "rate_gap", "hours",
                ]],
                use_container_width=True,
                hide_index=True,
            )
        
        with tab2:
            def top_by(level_keys, label):
                summary = rate_tree.groupby(level_keys).agg(
                    rate_gap=("rate_gap", "sum"),
                    hours=("hours", "sum"),
                ).reset_index()
                summary["rate_variance"] = summary["rate_gap"] / summary["hours"].replace(0, np.nan)
                summary = summary.nsmallest(10, "rate_gap")
                st.markdown(f"**{label}**")
                st.dataframe(summary, use_container_width=True, hide_index=True)
            
            top_by(["department_final"], "Worst Departments")
            top_by(["department_final", "category"], "Worst Categories")
            top_by(["department_final", "category", "job_no"], "Worst Jobs")
            top_by(["department_final", "category", "job_no", "task_name"], "Worst Tasks")
            top_by(["department_final", "category", "job_no", "task_name", "staff_name"], "Worst Staff")
        
        with tab3:
            def waterfall_for(group_cols, label):
                summary = rate_tree.groupby(group_cols).agg(
                    quoted_amount=("quoted_amount", "sum"),
                    rate_gap=("rate_gap", "sum"),
                ).reset_index()
                summary["actual_revenue"] = summary["quoted_amount"] + summary["rate_gap"]
                summary = summary.nsmallest(3, "rate_gap")
                
                for _, row in summary.iterrows():
                    title = f"{label}: {row[group_cols[-1]]}"
                    fig = go.Figure(go.Waterfall(
                        x=["Quoted Revenue", "Rate Gap", "Actual Revenue"],
                        y=[row["quoted_amount"], row["rate_gap"], row["actual_revenue"]],
                        measure=["absolute", "relative", "total"],
                    ))
                    fig.update_layout(title=title, height=260, margin=dict(t=40, l=10, r=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            
            waterfall_for(["department_final"], "Department")
            waterfall_for(["department_final", "category"], "Category")
    else:
        st.info("No rate variance data available.")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION D: RATE VARIANCE DIAGNOSIS
    # =========================================================================
    section_header("Rate Variance Diagnosis", "Identify root causes and actions")
    
    diagnosis = compute_rate_variance_diagnosis(df)
    if len(diagnosis) > 0:
        job_attrs = df.groupby("job_no").agg(
            department_final=("department_final", "first"),
            category=(get_category_col(df), "first"),
        ).reset_index()
        diagnosis = diagnosis.merge(job_attrs, on="job_no", how="left")
        diagnosis["rate_gap"] = diagnosis["rate_variance"] * diagnosis["hours"]
        diagnosis = diagnosis.sort_values("rate_gap").head(15)
        action_map = {
            "No quote": "Backfill quote or flag as non-quoted work",
            "Low volume": "Review with next period data",
            "Scope creep": "Re-scope, change request, or re-quote",
            "Overrun hours": "Adjust delivery plan or scope",
            "Revenue shortfall": "Check invoicing/discounts",
            "Non-billable mix": "Reduce internal/overhead allocation",
            "Rate leakage": "Review rate card vs actuals",
            "Above quote": "Validate rate capture and margin",
        }
        diagnosis["action"] = diagnosis["driver"].map(action_map).fillna("Review variance")
        
        display_cols = [
            "job_no", "department_final", "category",
            "quote_rate", "realised_rate", "rate_variance",
            "unquoted_share", "hours_variance_pct", "billable_ratio",
            "driver", "action",
        ]
        show = diagnosis[[c for c in display_cols if c in diagnosis.columns]].copy()
        show = show.rename(columns={
            "job_no": "Job",
            "department_final": "Department",
            "category": "Category",
            "quote_rate": "Quote Rate",
            "realised_rate": "Realised Rate",
            "rate_variance": "Rate Var",
            "unquoted_share": "Unquoted %",
            "hours_variance_pct": "Hours Var %",
            "billable_ratio": "Billable Ratio",
            "driver": "Driver",
            "action": "Action",
        })
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No rate variance data available.")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION E: JOB DETAILS (when selected)
    # =========================================================================
    if selection and selection.selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        selected_job = jobs_df.iloc[selected_idx]["job_no"]
        
        section_header(f"Job Details: {selected_job}")
        
        # Job summary
        job_row = jobs_df[jobs_df["job_no"] == selected_job].iloc[0]
        
        detail_cols = st.columns(4)
        
        with detail_cols[0]:
            st.metric("Quoted Hours", fmt_hours(job_row["quoted_hours"]))
        
        with detail_cols[1]:
            st.metric("Actual Hours", fmt_hours(job_row["actual_hours"]))
        
        with detail_cols[2]:
            st.metric("% Consumed", fmt_percent(job_row["pct_consumed"]))
        
        with detail_cols[3]:
            st.metric("Status", job_row["risk_flag"].replace("_", " ").title())
        
        # Tabs for breakdown
        tab1, tab2 = st.tabs(["Task Breakdown", "Staff Attribution"])
        
        with tab1:
            task_df = get_job_task_breakdown(df, selected_job)
            
            if len(task_df) > 0:
                # Bar chart
                task_chart_df = task_df.head(10).copy()
                task_chart_df = task_chart_df.melt(
                    id_vars=["task_name"],
                    value_vars=["quoted_hours", "actual_hours"],
                    var_name="Type",
                    value_name="Hours"
                )
                task_chart_df["Type"] = task_chart_df["Type"].replace({
                    "quoted_hours": "Quoted",
                    "actual_hours": "Actual"
                })
                
                fig = grouped_bar(
                    task_chart_df,
                    x="task_name",
                    y=["Hours"],
                    title="Quoted vs Actual Hours by Task"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(
                    task_df[["task_name", "quoted_hours", "actual_hours", "variance", "variance_pct"]].rename(
                        columns={
                            "task_name": "Task",
                            "quoted_hours": "Quoted",
                            "actual_hours": "Actual",
                            "variance": "Variance",
                            "variance_pct": "Var %"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Quoted": st.column_config.NumberColumn(format="%.1f"),
                        "Actual": st.column_config.NumberColumn(format="%.1f"),
                        "Variance": st.column_config.NumberColumn(format="%.1f"),
                        "Var %": st.column_config.NumberColumn(format="%.1f%%"),
                    }
                )
        
        with tab2:
            staff_df = get_job_staff_breakdown(df, selected_job)
            
            if len(staff_df) > 0:
                # Bar chart
                fig = horizontal_bar(
                    staff_df.head(10),
                    x="hours",
                    y="staff_name",
                    title="Hours by Staff Member"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(
                    staff_df[["staff_name", "hours", "pct_of_job", "tasks"]].rename(
                        columns={
                            "staff_name": "Staff",
                            "hours": "Hours",
                            "pct_of_job": "% of Job",
                            "tasks": "Tasks"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Hours": st.column_config.NumberColumn(format="%.1f"),
                        "% of Job": st.column_config.NumberColumn(format="%.1f%%"),
                    }
                )
    else:
        st.info("Select a job from the table above to see detailed breakdown.")


if __name__ == "__main__":
    main()
