"""
Rate Capture Control Tower

Drill chain: Company -> Department -> Category -> Job -> Task/Staff levers.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cohorts import filter_by_time_window
from src.data.loader import load_fact_timesheet
from src.data.semantic import (
    exclude_leave,
    filter_jobs_by_state,
    get_category_col,
    safe_quote_job_task,
)
from src.ui import charts
from src.ui.formatting import (
    build_job_name_lookup,
    fmt_currency,
    fmt_hours,
    fmt_percent,
    fmt_rate,
)
from src.ui.layout import section_header
from src.ui.state import init_state


st.set_page_config(page_title="Rate Capture Control Tower", page_icon="💸", layout="wide")


RC_DEFAULTS = {
    "rc_level": "company",
    "rc_selected_department": None,
    "rc_selected_category": None,
    "rc_selected_job": None,
    "rc_selected_lever": None,
    "rc_selected_task": None,
}


def _init_rc_state():
    for key, default in RC_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _reset_to_company():
    st.session_state["rc_level"] = "company"
    st.session_state["rc_selected_department"] = None
    st.session_state["rc_selected_category"] = None
    st.session_state["rc_selected_job"] = None
    st.session_state["rc_selected_lever"] = None
    st.session_state["rc_selected_task"] = None


def _select_department(department: str):
    st.session_state["rc_level"] = "department"
    st.session_state["rc_selected_department"] = department
    st.session_state["rc_selected_category"] = None
    st.session_state["rc_selected_job"] = None
    st.session_state["rc_selected_lever"] = None
    st.session_state["rc_selected_task"] = None


def _select_category(category: str):
    st.session_state["rc_level"] = "category"
    st.session_state["rc_selected_category"] = category
    st.session_state["rc_selected_job"] = None
    st.session_state["rc_selected_lever"] = None
    st.session_state["rc_selected_task"] = None


def _select_job(job_no: str):
    st.session_state["rc_level"] = "job"
    st.session_state["rc_selected_job"] = job_no
    st.session_state["rc_selected_lever"] = None
    st.session_state["rc_selected_task"] = None


def _select_lever(lever: str):
    st.session_state["rc_level"] = "detail"
    st.session_state["rc_selected_lever"] = lever


def _select_task(task_name: str):
    st.session_state["rc_level"] = "detail"
    st.session_state["rc_selected_task"] = task_name
    if st.session_state.get("rc_selected_lever") is None:
        st.session_state["rc_selected_lever"] = "Task Detail"


def _drill_up():
    level = st.session_state.get("rc_level", "company")
    if level == "detail":
        st.session_state["rc_level"] = "job"
        st.session_state["rc_selected_lever"] = None
        st.session_state["rc_selected_task"] = None
    elif level == "job":
        st.session_state["rc_level"] = "category"
        st.session_state["rc_selected_job"] = None
        st.session_state["rc_selected_lever"] = None
        st.session_state["rc_selected_task"] = None
    elif level == "category":
        st.session_state["rc_level"] = "department"
        st.session_state["rc_selected_category"] = None
        st.session_state["rc_selected_job"] = None
        st.session_state["rc_selected_lever"] = None
        st.session_state["rc_selected_task"] = None
    elif level == "department":
        _reset_to_company()


def _fmt_rate_delta(value: float) -> str:
    if pd.isna(value):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}${value:,.0f}/hr"


def _build_job_label_series(job_no_series: pd.Series, job_name_lookup: dict) -> pd.Series:
    job_no_str = job_no_series.astype(str)
    if not job_name_lookup:
        return job_no_str
    name_series = job_no_str.map(job_name_lookup).fillna("")
    labelled = np.where(name_series.str.strip() != "", job_no_str + " — " + name_series, job_no_str)
    return pd.Series(labelled, index=job_no_series.index)


def _fill_missing_from_job(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns or "job_no" not in df.columns:
        return df
    df = df.copy()
    job_map = df.groupby("job_no")[col].agg(lambda s: s.dropna().iloc[0] if s.dropna().any() else np.nan)
    df[col] = df[col].fillna(df["job_no"].map(job_map))
    return df


def _rate_frame(actuals: pd.DataFrame, quote: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    if group_keys:
        result = actuals.merge(quote, on=group_keys, how="left")
    else:
        result = actuals.copy()
        if len(quote) > 0:
            result["quoted_hours"] = quote["quoted_hours"].iloc[0]
            result["quoted_amount"] = quote["quoted_amount"].iloc[0]
        else:
            result["quoted_hours"] = 0
            result["quoted_amount"] = 0
    result["quote_rate"] = np.where(
        result["quoted_hours"] > 0,
        result["quoted_amount"] / result["quoted_hours"],
        np.nan,
    )
    result["realised_rate"] = np.where(
        result["hours"] > 0,
        result["revenue"] / result["hours"],
        np.nan,
    )
    result["rate_variance"] = result["realised_rate"] - result["quote_rate"]
    result["rate_capture_pct"] = np.where(
        result["quote_rate"] > 0,
        result["realised_rate"] / result["quote_rate"] * 100,
        np.nan,
    )
    return result


def _build_rate_tables(df: pd.DataFrame, category_col: str) -> dict:
    base_cols = [
        "job_no",
        "task_name",
        "department_final",
        category_col,
        "hours_raw",
        "rev_alloc",
        "base_cost",
        "quoted_time_total",
        "quoted_amount_total",
        "quote_match_flag",
    ]
    if "job_name" in df.columns:
        base_cols.append("job_name")
    base_cols = [c for c in base_cols if c in df.columns]
    df_base = df[base_cols].copy()

    job_task = safe_quote_job_task(df_base)
    if len(job_task) == 0:
        return {
            "company": pd.DataFrame(),
            "dept": pd.DataFrame(),
            "category": pd.DataFrame(),
            "job": pd.DataFrame(),
        }

    mapping_source = df_base[["job_no", "task_name", "department_final", category_col, "hours_raw"]].copy()
    mapping_source["hours_raw"] = mapping_source["hours_raw"].fillna(0)
    mapping = mapping_source.groupby(
        ["job_no", "task_name", "department_final", category_col],
        dropna=False,
    )["hours_raw"].sum().reset_index()
    mapping = mapping.sort_values("hours_raw", ascending=False).drop_duplicates(["job_no", "task_name"])
    job_task = job_task.merge(
        mapping[["job_no", "task_name", "department_final", category_col]],
        on=["job_no", "task_name"],
        how="left",
    )

    actuals_company = pd.DataFrame([{
        "hours": df_base["hours_raw"].sum(),
        "revenue": df_base["rev_alloc"].sum(),
    }])
    quote_company = pd.DataFrame([{
        "quoted_hours": job_task["quoted_time_total"].sum(),
        "quoted_amount": job_task["quoted_amount_total"].sum(),
    }])
    company = _rate_frame(actuals_company, quote_company, [])

    actuals_dept = df_base.groupby("department_final").agg(
        hours=("hours_raw", "sum"),
        revenue=("rev_alloc", "sum"),
    ).reset_index()
    quote_dept = job_task.groupby("department_final").agg(
        quoted_hours=("quoted_time_total", "sum"),
        quoted_amount=("quoted_amount_total", "sum"),
    ).reset_index()
    dept = _rate_frame(actuals_dept, quote_dept, ["department_final"])

    actuals_cat = df_base.groupby(["department_final", category_col]).agg(
        hours=("hours_raw", "sum"),
        revenue=("rev_alloc", "sum"),
    ).reset_index()
    quote_cat = job_task.groupby(["department_final", category_col]).agg(
        quoted_hours=("quoted_time_total", "sum"),
        quoted_amount=("quoted_amount_total", "sum"),
    ).reset_index()
    category = _rate_frame(actuals_cat, quote_cat, ["department_final", category_col])

    job_actuals = df_base.groupby("job_no").agg(
        hours=("hours_raw", "sum"),
        revenue=("rev_alloc", "sum"),
    ).reset_index()
    job_quote = job_task.groupby("job_no").agg(
        quoted_hours=("quoted_time_total", "sum"),
        quoted_amount=("quoted_amount_total", "sum"),
    ).reset_index()
    job_dim = df_base.groupby(["job_no", "department_final", category_col], dropna=False)["hours_raw"].sum().reset_index()
    job_dim = job_dim.sort_values("hours_raw", ascending=False).drop_duplicates(["job_no"])
    if "job_name" in df_base.columns:
        job_name = df_base.groupby("job_no")["job_name"].first().reset_index()
        job_dim = job_dim.merge(job_name, on="job_no", how="left")
    job = job_actuals.merge(job_quote, on="job_no", how="left").merge(job_dim, on="job_no", how="left")
    job["quote_rate"] = np.where(
        job["quoted_hours"] > 0,
        job["quoted_amount"] / job["quoted_hours"],
        np.nan,
    )
    job["realised_rate"] = np.where(
        job["hours"] > 0,
        job["revenue"] / job["hours"],
        np.nan,
    )
    job["rate_variance"] = job["realised_rate"] - job["quote_rate"]
    job["rate_capture_pct"] = np.where(
        job["quote_rate"] > 0,
        job["realised_rate"] / job["quote_rate"] * 100,
        np.nan,
    )
    job["job_no"] = job["job_no"].astype(str)

    return {
        "company": company,
        "dept": dept,
        "category": category,
        "job": job,
    }


@st.cache_data(show_spinner=False)
def _get_cached_tables(df: pd.DataFrame, category_col: str) -> dict:
    return _build_rate_tables(df, category_col)


def _rate_table_column_config() -> dict:
    return {
        "Quote Rate": st.column_config.NumberColumn(format="$%.0f"),
        "Realised Rate": st.column_config.NumberColumn(format="$%.0f"),
        "Rate Variance": st.column_config.NumberColumn(format="$%.0f"),
        "Rate Capture %": st.column_config.NumberColumn(format="%.1f%%"),
        "Total Leakage": st.column_config.NumberColumn(format="$%.0f"),
        "Hours": st.column_config.NumberColumn(format="%.1f"),
        "Revenue": st.column_config.NumberColumn(format="$%.0f"),
    }


def _safe_rate_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    display = df.copy()
    if label_col in display.columns:
        display[label_col] = display[label_col].fillna("—")
    if "quote_rate" in display.columns:
        display["quote_rate"] = display["quote_rate"].apply(fmt_rate)
    if "realised_rate" in display.columns:
        display["realised_rate"] = display["realised_rate"].apply(fmt_rate)
    if "rate_variance" in display.columns:
        display["rate_variance"] = display["rate_variance"].apply(_fmt_rate_delta)
    if "rate_capture_pct" in display.columns:
        display["rate_capture_pct"] = display["rate_capture_pct"].apply(fmt_percent)
    if "total_leakage" in display.columns:
        display["total_leakage"] = display["total_leakage"].apply(fmt_currency)
    if "hours" in display.columns:
        display["hours"] = display["hours"].apply(fmt_hours)
    if "revenue" in display.columns:
        display["revenue"] = display["revenue"].apply(fmt_currency)
    if "billable_share_pct" in display.columns:
        display["billable_share_pct"] = display["billable_share_pct"].apply(fmt_percent)
    return display


def _compute_scope_creep(df_job: pd.DataFrame) -> tuple[float, float, float]:
    if "quote_match_flag" not in df_job.columns or "hours_raw" not in df_job.columns:
        return np.nan, 0.0, 0.0
    unquoted_hours = df_job.loc[df_job["quote_match_flag"] != "matched", "hours_raw"].sum()
    total_hours = df_job["hours_raw"].sum()
    pct = (unquoted_hours / total_hours * 100) if total_hours > 0 else np.nan
    return pct, unquoted_hours, total_hours


def _compute_hours_overrun(df_job: pd.DataFrame, job_task: pd.DataFrame | None = None) -> tuple[float, float, float]:
    if "task_name" not in df_job.columns:
        return np.nan, 0.0, 0.0
    if job_task is None:
        job_task = safe_quote_job_task(df_job)
    if len(job_task) == 0 or "quoted_time_total" not in job_task.columns:
        return np.nan, 0.0, 0.0
    matched = job_task
    if "quote_match_flag" in job_task.columns:
        matched = job_task[job_task["quote_match_flag"] == "matched"]
    if len(matched) == 0:
        return np.nan, 0.0, 0.0
    matched_hours = matched["quoted_time_total"].sum()
    actual_matched = df_job.merge(
        matched[["job_no", "task_name"]],
        on=["job_no", "task_name"],
        how="inner",
    )
    actual_hours = actual_matched["hours_raw"].sum()
    pct = (actual_hours - matched_hours) / matched_hours * 100 if matched_hours > 0 else np.nan
    return pct, actual_hours, matched_hours


def _compute_mix_variance(df_job: pd.DataFrame, baseline_cost_rate: float) -> tuple[float, float, float]:
    if "hours_raw" not in df_job.columns or "base_cost" not in df_job.columns:
        return np.nan, np.nan, np.nan
    job_hours = df_job["hours_raw"].sum()
    job_cost = df_job["base_cost"].sum()
    job_cost_rate = job_cost / job_hours if job_hours > 0 else np.nan
    if pd.isna(job_cost_rate) or pd.isna(baseline_cost_rate) or baseline_cost_rate == 0:
        return job_cost_rate, np.nan, np.nan
    delta = job_cost_rate - baseline_cost_rate
    delta_pct = delta / baseline_cost_rate * 100
    return job_cost_rate, delta, delta_pct


def _build_task_rate_table(df_job: pd.DataFrame, job_task: pd.DataFrame | None = None) -> pd.DataFrame:
    if "task_name" not in df_job.columns:
        return pd.DataFrame()
    actuals = df_job.groupby("task_name").agg(
        hours=("hours_raw", "sum"),
        revenue=("rev_alloc", "sum"),
        cost=("base_cost", "sum"),
    ).reset_index()

    if job_task is None:
        job_task = safe_quote_job_task(df_job)
    if len(job_task) > 0:
        quotes = job_task.groupby("task_name").agg(
            quoted_hours=("quoted_time_total", "sum"),
            quoted_amount=("quoted_amount_total", "sum"),
            quote_rate=("quote_rate", "mean"),
            quote_match_flag=("quote_match_flag", "first") if "quote_match_flag" in job_task.columns else ("task_name", "first"),
        ).reset_index()
    else:
        quotes = pd.DataFrame({
            "task_name": [],
            "quoted_hours": [],
            "quoted_amount": [],
            "quote_rate": [],
            "quote_match_flag": [],
        })

    tasks = actuals.merge(quotes, on="task_name", how="left")
    tasks["realised_rate"] = np.where(tasks["hours"] > 0, tasks["revenue"] / tasks["hours"], np.nan)
    tasks["rate_variance"] = tasks["realised_rate"] - tasks["quote_rate"]
    tasks["total_leakage"] = np.where(tasks["rate_variance"] < 0, -tasks["rate_variance"] * tasks["hours"], 0)
    return tasks.sort_values("rate_variance")


def _build_staff_table(df_job: pd.DataFrame) -> pd.DataFrame:
    if "staff_name" not in df_job.columns:
        return pd.DataFrame()
    df_job = df_job.copy()
    if "is_billable" not in df_job.columns:
        df_job["is_billable"] = True
    df_job["billable_hours"] = np.where(df_job["is_billable"], df_job["hours_raw"], 0)
    staff = df_job.groupby("staff_name").agg(
        hours=("hours_raw", "sum"),
        billable_hours=("billable_hours", "sum"),
        revenue=("rev_alloc", "sum"),
        cost=("base_cost", "sum"),
    ).reset_index()
    staff["billable_share_pct"] = np.where(staff["hours"] > 0, staff["billable_hours"] / staff["hours"] * 100, np.nan)
    staff["cost_rate"] = np.where(staff["hours"] > 0, staff["cost"] / staff["hours"], np.nan)
    return staff.sort_values("billable_share_pct")


def _build_job_drill(df_job: pd.DataFrame, baseline_cost_rate: float) -> dict:
    job_task = safe_quote_job_task(df_job)
    scope_pct, unquoted_hours, total_hours = _compute_scope_creep(df_job)
    overrun_pct, actual_matched_hours, quoted_matched_hours = _compute_hours_overrun(df_job, job_task)
    job_cost_rate, mix_delta, mix_delta_pct = _compute_mix_variance(df_job, baseline_cost_rate)
    task_rates = _build_task_rate_table(df_job, job_task)
    staff_rates = _build_staff_table(df_job)
    return {
        "scope_pct": scope_pct,
        "unquoted_hours": unquoted_hours,
        "total_hours": total_hours,
        "overrun_pct": overrun_pct,
        "actual_matched_hours": actual_matched_hours,
        "quoted_matched_hours": quoted_matched_hours,
        "job_cost_rate": job_cost_rate,
        "mix_delta": mix_delta,
        "mix_delta_pct": mix_delta_pct,
        "task_rates": task_rates,
        "staff_rates": staff_rates,
    }


def _get_job_drill_cache(signature: tuple, job_no: str, df_job: pd.DataFrame, baseline_cost_rate: float) -> dict:
    cache = st.session_state.get("rc_job_cache")
    if cache and cache.get("signature") == signature and cache.get("job_no") == job_no:
        return cache["data"]
    data = _build_job_drill(df_job, baseline_cost_rate)
    st.session_state["rc_job_cache"] = {"signature": signature, "job_no": job_no, "data": data}
    return data


def _render_breadcrumb(job_label: str | None):
    parts = ["Company"]
    if st.session_state.get("rc_selected_department"):
        parts.append(st.session_state["rc_selected_department"])
    if st.session_state.get("rc_selected_category"):
        parts.append(st.session_state["rc_selected_category"])
    if job_label:
        parts.append(job_label)
    if st.session_state.get("rc_selected_lever"):
        parts.append(st.session_state["rc_selected_lever"])
    breadcrumb = " > ".join(parts)

    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        st.caption(f"Navigation: {breadcrumb}")
    with col2:
        if st.session_state.get("rc_level", "company") != "company":
            if st.button("⬆ Up", key="rc_up"):
                _drill_up()
                st.rerun()
    with col3:
        if st.session_state.get("rc_level", "company") != "company":
            if st.button("🏠 Reset", key="rc_reset"):
                _reset_to_company()
                st.rerun()


@st.fragment
def _render_rate_capture_drill(
    df_valid: pd.DataFrame,
    rate_tables: dict,
    category_col: str,
    available_departments: list[str],
    min_job_hours: int,
    job_name_lookup: dict,
) -> None:
    section_header("Department Detail", "Select a department to view category performance.")

    dept_rates = rate_tables["dept"].copy()
    dept_rates = dept_rates.dropna(subset=["rate_variance"], how="all")
    dept_rates["total_leakage"] = np.where(
        dept_rates["rate_variance"] < 0,
        -dept_rates["rate_variance"] * dept_rates["hours"],
        0,
    )
    dept_rates = dept_rates.sort_values("rate_variance").reset_index(drop=True)

    dept_options = ["All"] + sorted(available_departments)
    selected_department = st.selectbox(
        "Department",
        options=dept_options,
        index=0,
        key="rc_department_select",
    )
    if selected_department == "All":
        selected_department = None

    if not selected_department:
        return

    df_dept = df_valid[df_valid["department_final"] == selected_department]
    dept_row = dept_rates[dept_rates["department_final"] == selected_department].head(1)
    if len(dept_row) > 0:
        dept_row = dept_row.iloc[0]
        dept_quote_rate = dept_row.get("quote_rate", np.nan)
        dept_realised_rate = dept_row.get("realised_rate", np.nan)
        dept_variance = dept_row.get("rate_variance", np.nan)
    else:
        dept_quote_rate = dept_realised_rate = dept_variance = np.nan

    billable_share = np.nan
    if "is_billable" in df_dept.columns:
        total_hours = df_dept["hours_raw"].sum()
        billable_hours = df_dept.loc[df_dept["is_billable"] == True, "hours_raw"].sum()
        billable_share = (billable_hours / total_hours * 100) if total_hours > 0 else np.nan

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Dept Quote Rate", fmt_rate(dept_quote_rate))
    with kpi_cols[1]:
        st.metric("Dept Realised Rate", fmt_rate(dept_realised_rate))
    with kpi_cols[2]:
        st.metric("Rate Variance", _fmt_rate_delta(dept_variance))
    with kpi_cols[3]:
        st.metric("Billable Share", fmt_percent(billable_share))

    if category_col not in df_dept.columns:
        st.warning("Category field missing from dataset.")
    else:
        cat_rates = rate_tables["category"].copy()
        cat_rates = cat_rates[cat_rates["department_final"] == selected_department]
        cat_rates["total_leakage"] = np.where(
            cat_rates["rate_variance"] < 0,
            -cat_rates["rate_variance"] * cat_rates["hours"],
            0,
        )
        cat_rates = cat_rates.sort_values("rate_variance").reset_index(drop=True)

        display_cols = [
            category_col,
            "quote_rate",
            "realised_rate",
            "rate_variance",
            "rate_capture_pct",
            "total_leakage",
            "hours",
            "revenue",
        ]
        cat_display = cat_rates[display_cols].copy()
        cat_display = cat_display.rename(columns={
            category_col: "Category",
            "quote_rate": "Quote Rate",
            "realised_rate": "Realised Rate",
            "rate_variance": "Rate Variance",
            "rate_capture_pct": "Rate Capture %",
            "total_leakage": "Total Leakage",
            "hours": "Hours",
            "revenue": "Revenue",
        })

        section_header("Categories by Rate Variance", "Department slice (no drill).")
        st.dataframe(
            cat_display,
            use_container_width=True,
            hide_index=True,
            column_config=_rate_table_column_config(),
        )

    show_job_detail = st.checkbox("Show job detail (slower)", value=False, key="rc_show_job_detail")
    if not show_job_detail:
        return

    job_rates = rate_tables["job"].copy()
    job_rates = job_rates[job_rates["department_final"] == selected_department]
    job_rates = job_rates[job_rates["hours"] >= min_job_hours]
    job_rates["total_leakage"] = np.where(
        job_rates["rate_variance"] < 0,
        -job_rates["rate_variance"] * job_rates["hours"],
        0,
    )
    job_rates = job_rates.sort_values("rate_variance").reset_index(drop=True)

    if len(job_rates) == 0:
        st.info("No jobs meet the current filters.")
        return

    show_all_jobs = st.checkbox(
        "Show all jobs (slow)",
        value=False,
        key="rc_jobs_all",
    )
    job_limit = 200
    if not show_all_jobs:
        job_limit = st.slider(
            "Max jobs",
            min_value=25,
            max_value=500,
            value=200,
            step=25,
            key="rc_jobs_limit",
        )

    display_rates = job_rates if show_all_jobs else job_rates.head(job_limit)
    display_rates = display_rates.copy()
    display_rates["job_label"] = _build_job_label_series(display_rates["job_no"], job_name_lookup)
    display_cols = [
        "job_label",
        "quote_rate",
        "realised_rate",
        "rate_variance",
        "rate_capture_pct",
        "total_leakage",
        "hours",
        "revenue",
    ]
    job_display = display_rates[display_cols].rename(columns={
        "job_label": "Job",
        "quote_rate": "Quote Rate",
        "realised_rate": "Realised Rate",
        "rate_variance": "Rate Variance",
        "rate_capture_pct": "Rate Capture %",
        "total_leakage": "Total Leakage",
        "hours": "Hours",
        "revenue": "Revenue",
    })

    section_header("Jobs by Rate Variance", "Department slice (optional detail).")
    st.dataframe(
        job_display,
        use_container_width=True,
        hide_index=True,
        column_config=_rate_table_column_config(),
    )


init_state()
_init_rc_state()

st.title("Rate Capture Control Tower")
st.caption("What rate did we quote vs. what we actually earned, and which lever caused the leakage?")

# Sidebar controls
st.sidebar.header("Filters")

time_window = st.sidebar.selectbox(
    "Time Window",
    options=["6m", "12m", "24m", "all"],
    format_func=lambda x: f"Last {x}" if x != "all" else "All time",
    key="rc_time_window",
)

job_state = st.sidebar.selectbox(
    "Job State",
    options=["All", "Active", "Completed"],
    key="rc_job_state",
)

exclude_leave_toggle = st.sidebar.checkbox("Exclude Leave", value=True, key="rc_exclude_leave")
include_nonbillable = st.sidebar.checkbox("Include Non-Billable", value=True, key="rc_include_nonbillable")
min_job_hours = st.sidebar.slider("Min Job Hours", min_value=0, max_value=100, value=5, step=1, key="rc_min_job_hours")

# Load data
with st.spinner("Loading data..."):
    df_all = load_fact_timesheet()

if len(df_all) == 0:
    st.warning("No data available.")
    st.stop()

date_col = "month_key" if "month_key" in df_all.columns else "work_date"
if date_col in df_all.columns:
    df_all[date_col] = pd.to_datetime(df_all[date_col], errors="coerce")

# Apply filters
if exclude_leave_toggle:
    df_filtered = exclude_leave(df_all)
else:
    df_filtered = df_all

if not include_nonbillable and "is_billable" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["is_billable"] == True]

if date_col in df_filtered.columns:
    df_filtered = filter_by_time_window(df_filtered, time_window, date_col=date_col)

if job_state != "All":
    df_filtered = filter_jobs_by_state(df_filtered, job_state)

if len(df_filtered) == 0:
    st.warning("No data after filters. Adjust filters to continue.")
    st.stop()

category_col = get_category_col(df_filtered)
if "department_final" in df_filtered.columns:
    df_filtered = _fill_missing_from_job(df_filtered, "department_final")
if category_col in df_filtered.columns:
    df_filtered = _fill_missing_from_job(df_filtered, category_col)

valid_mask = pd.Series(True, index=df_filtered.index)
if "department_final" in df_filtered.columns:
    valid_mask = valid_mask & df_filtered["department_final"].notna()
if category_col in df_filtered.columns:
    valid_mask = valid_mask & df_filtered[category_col].notna()

excluded_rows = int((~valid_mask).sum())
df_valid = df_filtered.loc[valid_mask]
if excluded_rows > 0:
    excluded_hours = df_filtered.loc[~valid_mask, "hours_raw"].sum() if "hours_raw" in df_filtered.columns else 0
    st.caption(
        f"Excluded {excluded_rows:,} rows ({excluded_hours:,.1f} hours) with missing department/category to keep rollups reconciled."
    )

job_name_lookup = build_job_name_lookup(df_valid)

rate_tables = _get_cached_tables(df_valid, category_col)

# Validate selections against filtered data
available_departments = df_valid["department_final"].dropna().unique().tolist()
if st.session_state["rc_selected_department"] not in available_departments:
    if st.session_state["rc_selected_department"] is not None:
        _reset_to_company()

selected_department = None
selected_category = None
selected_job = None

# ============================================================================== 
# LEVEL 0 — COMPANY RATE CAPTURE
# ============================================================================== 
section_header("Level 0 — Company Rate Capture", "Are we capturing the value we quoted across the business?")

company_rates = rate_tables["company"]
if len(company_rates) > 0:
    company_row = company_rates.iloc[0]
    quote_rate = company_row.get("quote_rate", np.nan)
    realised_rate = company_row.get("realised_rate", np.nan)
    rate_variance = company_row.get("rate_variance", np.nan)
    rate_capture_pct = company_row.get("rate_capture_pct", np.nan)
    hours = company_row.get("hours", 0)
    leakage = (-rate_variance * hours) if pd.notna(rate_variance) and rate_variance < 0 else 0
else:
    quote_rate = realised_rate = rate_variance = rate_capture_pct = np.nan
    leakage = 0

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Portfolio Quote Rate", fmt_rate(quote_rate))
with kpi_cols[1]:
    st.metric("Portfolio Realised Rate", fmt_rate(realised_rate))
with kpi_cols[2]:
    st.metric("Rate Variance", _fmt_rate_delta(rate_variance))
with kpi_cols[3]:
    st.metric("Total Leakage ($)", fmt_currency(leakage))

waterfall_df = pd.DataFrame([
    {"label": "Quoted Rate", "value": quote_rate, "measure": "absolute"},
    {"label": "Variance", "value": rate_variance, "measure": "relative"},
    {"label": "Realised Rate", "value": realised_rate, "measure": "total"},
])

if waterfall_df["value"].notna().any():
    fig = charts.waterfall_chart(
        waterfall_df,
        labels="label",
        values="value",
        title="Rate Capture Waterfall ($/hr)",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data to render the rate waterfall.")

section_header("Departments by Rate Variance", "Portfolio view (no drill).")

if "department_final" in df_valid.columns:
    dept_rates = rate_tables["dept"].copy()
    dept_rates = dept_rates.dropna(subset=["rate_variance"], how="all")
    dept_rates["total_leakage"] = np.where(
        dept_rates["rate_variance"] < 0,
        -dept_rates["rate_variance"] * dept_rates["hours"],
        0,
    )
    dept_rates = dept_rates.sort_values("rate_variance").reset_index(drop=True)

    display_cols = [
        "department_final",
        "quote_rate",
        "realised_rate",
        "rate_variance",
        "rate_capture_pct",
        "total_leakage",
        "hours",
        "revenue",
    ]
    dept_display = dept_rates[display_cols].copy()
    dept_display = dept_display.rename(columns={
        "department_final": "Department",
        "quote_rate": "Quote Rate",
        "realised_rate": "Realised Rate",
        "rate_variance": "Rate Variance",
        "rate_capture_pct": "Rate Capture %",
        "total_leakage": "Total Leakage",
        "hours": "Hours",
        "revenue": "Revenue",
    })

    st.dataframe(
        dept_display,
        use_container_width=True,
        hide_index=True,
        column_config=_rate_table_column_config(),
    )
else:
    st.warning("Department field missing from dataset.")

_render_rate_capture_drill(
    df_valid=df_valid,
    rate_tables=rate_tables,
    category_col=category_col,
    available_departments=available_departments,
    min_job_hours=min_job_hours,
    job_name_lookup=job_name_lookup,
)

st.caption("Quote fields are deduped at the job-task level to prevent value inflation. Realised rates are hours-weighted.")
