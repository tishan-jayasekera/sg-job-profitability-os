"""
Active Campaign Load Snapshot

Company -> Department -> Category -> Job -> Task -> FTE
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.cohorts import get_active_jobs
from src.data.loader import load_fact_timesheet
from src.data.semantic import exclude_leave, get_category_col
from src.ui.formatting import build_job_name_lookup, format_job_label, fmt_hours, fmt_count
from src.ui.state import init_state


st.set_page_config(page_title="Campaign Load Snapshot", page_icon="C", layout="wide")

init_state()

MONTHLY_CAPACITY_HOURS = config.CAPACITY_HOURS_PER_WEEK * 4.33


def _month_label(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return "-"
    return pd.to_datetime(value).strftime("%b %Y")


def _get_months(df: pd.DataFrame) -> list[pd.Timestamp]:
    if "month_key" not in df.columns:
        return []
    months = pd.to_datetime(df["month_key"].dropna().unique())
    months = sorted(months)
    return months


def _compute_hours_pm(
    df: pd.DataFrame,
    group_keys: list[str],
    basis: str,
    window_months: int,
    snapshot_month: pd.Timestamp | None,
) -> pd.DataFrame:
    if "month_key" not in df.columns or "hours_raw" not in df.columns:
        return pd.DataFrame(columns=group_keys + ["hours_pm", "months_in_sample"])

    monthly = df.groupby(group_keys + ["month_key"], dropna=False)["hours_raw"].sum().reset_index()
    if len(monthly) == 0:
        return pd.DataFrame(columns=group_keys + ["hours_pm", "months_in_sample"])

    months = sorted(pd.to_datetime(monthly["month_key"].dropna().unique()))
    if len(months) == 0:
        return pd.DataFrame(columns=group_keys + ["hours_pm", "months_in_sample"])

    if not group_keys:
        monthly_total = monthly.groupby("month_key")["hours_raw"].sum()
        if basis == "snapshot":
            month = snapshot_month if snapshot_month is not None else months[-1]
            hours_pm = float(monthly_total.loc[month]) if month in monthly_total.index else 0.0
            return pd.DataFrame([{"hours_pm": hours_pm, "months_in_sample": 1}])
        months_window = months[-window_months:]
        monthly_total = monthly_total.loc[monthly_total.index.isin(months_window)]
        hours_pm = float(monthly_total.mean()) if len(monthly_total) > 0 else 0.0
        return pd.DataFrame([{"hours_pm": hours_pm, "months_in_sample": len(monthly_total)}])

    if basis == "snapshot":
        month = snapshot_month if snapshot_month is not None else months[-1]
        monthly = monthly[monthly["month_key"] == month]
        result = monthly.groupby(group_keys, dropna=False)["hours_raw"].sum().reset_index()
        result["months_in_sample"] = 1
    else:
        months_window = months[-window_months:]
        monthly = monthly[monthly["month_key"].isin(months_window)]
        result = monthly.groupby(group_keys, dropna=False)["hours_raw"].mean().reset_index()
        months_count = monthly.groupby(group_keys, dropna=False)["month_key"].nunique().reset_index()
        months_count = months_count.rename(columns={"month_key": "months_in_sample"})
        result = result.merge(months_count, on=group_keys, how="left")

    result = result.rename(columns={"hours_raw": "hours_pm"})
    return result


def _add_rollup_counts(df: pd.DataFrame, rollup: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    if len(rollup) == 0:
        return rollup

    result = rollup.copy()

    if not group_keys:
        active_jobs = df["job_no"].nunique() if "job_no" in df.columns else 0
        staff_count = df["staff_name"].nunique() if "staff_name" in df.columns else 0
        result["active_jobs"] = active_jobs
        result["staff_count"] = staff_count
    else:
        if "job_no" in df.columns:
            job_counts = df.groupby(group_keys)["job_no"].nunique().reset_index()
            job_counts = job_counts.rename(columns={"job_no": "active_jobs"})
            result = result.merge(job_counts, on=group_keys, how="left")
        if "staff_name" in df.columns:
            staff_counts = df.groupby(group_keys)["staff_name"].nunique().reset_index()
            staff_counts = staff_counts.rename(columns={"staff_name": "staff_count"})
            result = result.merge(staff_counts, on=group_keys, how="left")

    if "active_jobs" not in result.columns:
        result["active_jobs"] = 0
    if "staff_count" not in result.columns:
        result["staff_count"] = 0

    result["active_jobs"] = result["active_jobs"].fillna(0).astype(int)
    result["staff_count"] = result["staff_count"].fillna(0).astype(int)
    result["avg_hours_per_job"] = np.where(
        result["active_jobs"] > 0,
        result["hours_pm"] / result["active_jobs"],
        np.nan,
    )
    result["implied_fte"] = result["hours_pm"] / MONTHLY_CAPACITY_HOURS
    return result


def _staff_load_table(
    df_scope: pd.DataFrame,
    basis: str,
    window_months: int,
    snapshot_month: pd.Timestamp | None,
) -> pd.DataFrame:
    if "staff_name" not in df_scope.columns:
        return pd.DataFrame()

    staff_hours = _compute_hours_pm(df_scope, ["staff_name"], basis, window_months, snapshot_month)
    if len(staff_hours) == 0:
        return staff_hours

    if "fte_hours_scaling" in df_scope.columns:
        fte_scaling = df_scope.groupby("staff_name")["fte_hours_scaling"].agg(
            lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else config.DEFAULT_FTE_SCALING
        ).reset_index()
        fte_scaling = fte_scaling.rename(columns={"fte_hours_scaling": "fte_scaling"})
    else:
        fte_scaling = pd.DataFrame({
            "staff_name": staff_hours["staff_name"],
            "fte_scaling": config.DEFAULT_FTE_SCALING,
        })

    job_counts = (
        df_scope.groupby("staff_name")["job_no"].nunique().reset_index()
        if "job_no" in df_scope.columns
        else pd.DataFrame({"staff_name": staff_hours["staff_name"], "job_no": 0})
    )
    job_counts = job_counts.rename(columns={"job_no": "active_jobs"})

    result = staff_hours.merge(fte_scaling, on="staff_name", how="left")
    result = result.merge(job_counts, on="staff_name", how="left")

    result["fte_scaling"] = result["fte_scaling"].fillna(config.DEFAULT_FTE_SCALING)
    result["capacity_pm"] = result["fte_scaling"] * MONTHLY_CAPACITY_HOURS
    result["utilisation_pct"] = np.where(
        result["capacity_pm"] > 0,
        result["hours_pm"] / result["capacity_pm"] * 100,
        np.nan,
    )
    return result


def main() -> None:
    st.title("Campaign Load Snapshot")
    st.caption("Company -> Department -> Category -> Job -> Task -> FTE")

    df_raw = load_fact_timesheet()
    category_col = get_category_col(df_raw)

    if "job_no" not in df_raw.columns or "month_key" not in df_raw.columns:
        st.error("Missing required columns: job_no and month_key are required for this snapshot.")
        return

    st.sidebar.header("Snapshot Controls")

    recency_days = st.sidebar.slider(
        "Active job recency (days)",
        min_value=7,
        max_value=120,
        value=config.active_job_recency_days,
        step=1,
    )
    active_only = st.sidebar.checkbox("Active jobs only", value=True)
    exclude_leave_toggle = st.sidebar.checkbox("Exclude leave tasks", value=True)
    include_nonbillable = st.sidebar.checkbox("Include non-billable", value=True)

    basis_option = st.sidebar.selectbox(
        "Monthly basis",
        options=["Latest month", "Trailing 3-mo avg", "Trailing 6-mo avg", "Trailing 12-mo avg"],
        index=0,
    )

    basis = "snapshot" if basis_option == "Latest month" else "avg"
    window_months = 1
    if basis_option == "Trailing 3-mo avg":
        window_months = 3
    elif basis_option == "Trailing 6-mo avg":
        window_months = 6
    elif basis_option == "Trailing 12-mo avg":
        window_months = 12

    df_activity = exclude_leave(df_raw) if exclude_leave_toggle else df_raw.copy()
    active_jobs = get_active_jobs(df_activity, recency_days=recency_days)

    df_scope_base = df_raw.copy()
    if exclude_leave_toggle:
        df_scope_base = exclude_leave(df_scope_base)
    if active_only:
        df_scope_base = df_scope_base[df_scope_base["job_no"].isin(active_jobs)]
    if not include_nonbillable and "is_billable" in df_scope_base.columns:
        df_scope_base = df_scope_base[df_scope_base["is_billable"] == True]

    months_available = _get_months(df_scope_base)
    snapshot_month = None
    if basis == "snapshot" and len(months_available) > 0:
        snapshot_month = st.sidebar.selectbox(
            "Snapshot month",
            options=months_available,
            index=len(months_available) - 1,
            format_func=_month_label,
        )

    if len(df_scope_base) == 0:
        st.info("No activity found for the selected filters.")
        return

    # =========================
    # DRILL CHAIN
    # =========================
    st.subheader("Drill Chain")
    dept_options = ["All"] + sorted(df_scope_base["department_final"].dropna().unique().tolist())
    default_dept = "All"
    for dept in dept_options:
        if isinstance(dept, str) and "advert" in dept.lower():
            default_dept = dept
            break
    dept_index = dept_options.index(default_dept) if default_dept in dept_options else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_dept = st.selectbox("Department", dept_options, index=dept_index)

    df_dept = df_scope_base.copy()
    if selected_dept != "All":
        df_dept = df_dept[df_dept["department_final"] == selected_dept]

    with col2:
        if selected_dept == "All":
            selected_category = "All"
            st.selectbox("Category", ["All"], disabled=True)
        else:
            cat_options = ["All"] + sorted(df_dept[category_col].dropna().unique().tolist())
            selected_category = st.selectbox("Category", cat_options)

    df_cat = df_dept.copy()
    if selected_category != "All":
        df_cat = df_cat[df_cat[category_col] == selected_category]

    job_name_lookup = build_job_name_lookup(df_scope_base)
    with col3:
        if selected_category == "All":
            selected_job = "All"
            st.selectbox("Job", ["All"], disabled=True)
        else:
            job_options = ["All"] + sorted(df_cat["job_no"].dropna().unique().tolist())
            selected_job = st.selectbox(
                "Job",
                job_options,
                format_func=lambda x: format_job_label(x, job_name_lookup) if x != "All" else "All",
            )

    df_job = df_cat.copy()
    if selected_job != "All":
        df_job = df_job[df_job["job_no"] == selected_job]

    with col4:
        if selected_job == "All":
            selected_task = "All"
            st.selectbox("Task", ["All"], disabled=True)
        else:
            task_options = ["All"] + sorted(df_job["task_name"].dropna().unique().tolist())
            selected_task = st.selectbox("Task", task_options)

    df_task = df_job.copy()
    if selected_task != "All":
        df_task = df_task[df_task["task_name"] == selected_task]

    # =========================
    # SNAPSHOT HEADER
    # =========================
    if basis == "snapshot":
        snapshot_label = _month_label(snapshot_month) if snapshot_month else "Latest month"
        st.caption(f"Basis: Snapshot month ({snapshot_label}).")
    else:
        months_window = _get_months(df_scope_base)[-window_months:] if len(months_available) > 0 else []
        if months_window:
            st.caption(
                f"Basis: Trailing average ({_month_label(months_window[0])} - {_month_label(months_window[-1])})."
            )
        else:
            st.caption("Basis: Trailing average.")

    # =========================
    # KPI STRIP (CURRENT SCOPE)
    # =========================
    scope_df = df_scope_base.copy()
    scope_label = "Company"
    if selected_dept != "All":
        scope_df = scope_df[scope_df["department_final"] == selected_dept]
        scope_label = selected_dept
    if selected_category != "All":
        scope_df = scope_df[scope_df[category_col] == selected_category]
        scope_label = f"{scope_label} -> {selected_category}"
    if selected_job != "All":
        scope_df = scope_df[scope_df["job_no"] == selected_job]
        scope_label = f"{scope_label} -> {format_job_label(selected_job, job_name_lookup)}"
    if selected_task != "All":
        scope_df = scope_df[scope_df["task_name"] == selected_task]
        scope_label = f"{scope_label} -> {selected_task}"

    scope_hours = _compute_hours_pm(scope_df, [], basis, window_months, snapshot_month)
    total_hours_pm = float(scope_hours["hours_pm"].iloc[0]) if len(scope_hours) > 0 else 0.0
    active_jobs_count = scope_df["job_no"].nunique() if "job_no" in scope_df.columns else 0
    staff_count = scope_df["staff_name"].nunique() if "staff_name" in scope_df.columns else 0
    avg_hours_job = total_hours_pm / active_jobs_count if active_jobs_count > 0 else np.nan
    implied_fte = total_hours_pm / MONTHLY_CAPACITY_HOURS if MONTHLY_CAPACITY_HOURS > 0 else np.nan

    st.markdown(f"**Current Scope:** {scope_label}")
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.metric("Hours p/m", fmt_hours(total_hours_pm))
    with kpi_cols[1]:
        st.metric("Active Jobs", fmt_count(active_jobs_count))
    with kpi_cols[2]:
        st.metric("Avg Hours/Job", fmt_hours(avg_hours_job))
    with kpi_cols[3]:
        st.metric("Active Staff", fmt_count(staff_count))
    with kpi_cols[4]:
        st.metric("Implied FTE", f"{implied_fte:.2f}" if not pd.isna(implied_fte) else "-")

    st.divider()

    # =========================
    # LEVEL 1 - DEPARTMENT
    # =========================
    st.subheader("Level 1 - Department Snapshot")
    dept_rollup = _compute_hours_pm(df_scope_base, ["department_final"], basis, window_months, snapshot_month)
    dept_rollup = _add_rollup_counts(df_scope_base, dept_rollup, ["department_final"])
    if len(dept_rollup) > 0:
        dept_display = dept_rollup.copy()
        dept_display = dept_display.sort_values("hours_pm", ascending=False)
        dept_display = dept_display.rename(columns={
            "department_final": "Department",
            "hours_pm": "Hours p/m",
            "implied_fte": "Implied FTE",
            "active_jobs": "Active Jobs",
            "avg_hours_per_job": "Avg Hours/Job",
            "staff_count": "Active Staff",
        })
        dept_display["Hours p/m"] = dept_display["Hours p/m"].round(1)
        dept_display["Implied FTE"] = dept_display["Implied FTE"].round(2)
        dept_display["Avg Hours/Job"] = dept_display["Avg Hours/Job"].round(1)
        st.dataframe(dept_display, use_container_width=True, hide_index=True)
    else:
        st.info("No department activity in the selected window.")

    # =========================
    # LEVEL 2 - CATEGORY
    # =========================
    st.subheader("Level 2 - Category Snapshot")
    if selected_dept == "All":
        st.info("Select a department to view category-level load.")
    else:
        cat_rollup = _compute_hours_pm(df_dept, [category_col], basis, window_months, snapshot_month)
        cat_rollup = _add_rollup_counts(df_dept, cat_rollup, [category_col])
        if len(cat_rollup) > 0:
            cat_display = cat_rollup.copy()
            cat_display = cat_display.sort_values("hours_pm", ascending=False)
            cat_display = cat_display.rename(columns={
                category_col: "Category",
                "hours_pm": "Hours p/m",
                "implied_fte": "Implied FTE",
                "active_jobs": "Active Jobs",
                "avg_hours_per_job": "Avg Hours/Job",
                "staff_count": "Active Staff",
            })
            cat_display["Hours p/m"] = cat_display["Hours p/m"].round(1)
            cat_display["Implied FTE"] = cat_display["Implied FTE"].round(2)
            cat_display["Avg Hours/Job"] = cat_display["Avg Hours/Job"].round(1)
            st.dataframe(cat_display, use_container_width=True, hide_index=True)
        else:
            st.info("No category activity for the selected department.")

    # =========================
    # LEVEL 3 - JOB
    # =========================
    st.subheader("Level 3 - Job Snapshot")
    if selected_category == "All":
        st.info("Select a category to view job-level load.")
    else:
        job_rollup = _compute_hours_pm(df_cat, ["job_no"], basis, window_months, snapshot_month)
        if len(job_rollup) > 0:
            staff_counts = df_cat.groupby("job_no")["staff_name"].nunique().reset_index()
            staff_counts = staff_counts.rename(columns={"staff_name": "staff_count"})
            task_counts = df_cat.groupby("job_no")["task_name"].nunique().reset_index()
            task_counts = task_counts.rename(columns={"task_name": "task_count"})
            job_rollup = job_rollup.merge(staff_counts, on="job_no", how="left")
            job_rollup = job_rollup.merge(task_counts, on="job_no", how="left")
            job_rollup["implied_fte"] = job_rollup["hours_pm"] / MONTHLY_CAPACITY_HOURS

            job_display = job_rollup.copy()
            job_display["Job"] = job_display["job_no"].apply(
                lambda x: format_job_label(x, job_name_lookup)
            )
            job_display = job_display.rename(columns={
                "hours_pm": "Hours p/m",
                "implied_fte": "Implied FTE",
                "task_count": "Tasks",
                "staff_count": "Staff",
            })
            job_display = job_display.sort_values("Hours p/m", ascending=False)
            job_display["Hours p/m"] = job_display["Hours p/m"].round(1)
            job_display["Implied FTE"] = job_display["Implied FTE"].round(2)
            st.dataframe(
                job_display[["Job", "Hours p/m", "Implied FTE", "Tasks", "Staff"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No job activity for the selected category.")

    # =========================
    # LEVEL 4 - TASK
    # =========================
    st.subheader("Level 4 - Task Snapshot")
    if selected_job == "All":
        st.info("Select a job to view task-level load.")
    else:
        task_rollup = _compute_hours_pm(df_job, ["task_name"], basis, window_months, snapshot_month)
        if len(task_rollup) > 0:
            staff_counts = df_job.groupby("task_name")["staff_name"].nunique().reset_index()
            staff_counts = staff_counts.rename(columns={"staff_name": "staff_count"})
            task_rollup = task_rollup.merge(staff_counts, on="task_name", how="left")
            task_rollup["implied_fte"] = task_rollup["hours_pm"] / MONTHLY_CAPACITY_HOURS

            task_display = task_rollup.copy()
            task_display = task_display.rename(columns={
                "task_name": "Task",
                "hours_pm": "Hours p/m",
                "implied_fte": "Implied FTE",
                "staff_count": "Staff",
            })
            task_display = task_display.sort_values("Hours p/m", ascending=False)
            task_display["Hours p/m"] = task_display["Hours p/m"].round(1)
            task_display["Implied FTE"] = task_display["Implied FTE"].round(2)
            st.dataframe(task_display, use_container_width=True, hide_index=True)
        else:
            st.info("No task activity for the selected job.")

    # =========================
    # LEVEL 5 - STAFF / FTE
    # =========================
    st.subheader("Level 5 - Staff Load (FTE)")
    staff_scope = df_task.copy()
    if selected_task == "All":
        staff_scope = df_job.copy()
    if selected_job == "All":
        staff_scope = df_cat.copy()
    if selected_category == "All":
        staff_scope = df_dept.copy()
    if selected_dept == "All":
        staff_scope = df_scope_base.copy()

    staff_load = _staff_load_table(staff_scope, basis, window_months, snapshot_month)
    if len(staff_load) > 0:
        staff_display = staff_load.copy()
        staff_display = staff_display.rename(columns={
            "staff_name": "Staff",
            "hours_pm": "Hours p/m",
            "capacity_pm": "Capacity p/m",
            "utilisation_pct": "Utilisation %",
            "active_jobs": "Active Jobs",
            "fte_scaling": "FTE Scaling",
        })
        staff_display = staff_display.sort_values("Hours p/m", ascending=False)
        staff_display["Hours p/m"] = staff_display["Hours p/m"].round(1)
        staff_display["Capacity p/m"] = staff_display["Capacity p/m"].round(1)
        staff_display["Utilisation %"] = staff_display["Utilisation %"].round(1)
        staff_display["FTE Scaling"] = staff_display["FTE Scaling"].round(2)
        st.dataframe(
            staff_display[
                ["Staff", "Hours p/m", "Capacity p/m", "Utilisation %", "FTE Scaling", "Active Jobs"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No staff load available for the selected scope.")


if __name__ == "__main__":
    main()
