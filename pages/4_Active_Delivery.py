"""
Active Delivery Page

Delivery control tower with risk flags, attribution, and interventions.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state
from src.ui.layout import section_header
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_rate
from src.data.loader import load_fact_timesheet
from src.data.semantic import safe_quote_job_task, get_category_col
from src.data.cohorts import get_active_jobs


st.set_page_config(page_title="Active Delivery", page_icon="🎯", layout="wide")

init_state()

def _pill(text: str, color: str) -> str:
    return f"<span class='sg-pill' style='background:{color};'>{text}</span>"


def _kpi_card(title: str, value: str, subtitle: str = "") -> str:
    subtitle_html = f"<div class='sg-kpi-sub'>{subtitle}</div>" if subtitle else ""
    return (
        "<div class='sg-card'>"
        f"<div class='sg-kpi-title'>{title}</div>"
        f"<div class='sg-kpi-value'>{value}</div>"
        f"{subtitle_html}"
        "</div>"
    )


def _resolve_selected_job(jobs_df: pd.DataFrame, selected_choice: str) -> str:
    if selected_choice and selected_choice != "Auto":
        st.session_state["ad_selected_job"] = selected_choice
        return selected_choice
    if st.session_state.get("ad_selected_job") in jobs_df["job_no"].values:
        return st.session_state["ad_selected_job"]
    return jobs_df.iloc[0]["job_no"]

def _get_date_col(df: pd.DataFrame) -> str:
    return "work_date" if "work_date" in df.columns else "month_key"


def _compute_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dept+category benchmarks from completed jobs."""
    if "job_no" not in df.columns:
        return pd.DataFrame()
    category_col = get_category_col(df)

    job_completion = df.groupby("job_no").agg(
        completed_date=("job_completed_date", "first") if "job_completed_date" in df.columns else ("job_no", "first"),
        job_status=("job_status", "first") if "job_status" in df.columns else ("job_no", "first"),
        department_final=("department_final", "first"),
        job_category=(category_col, "first"),
    ).reset_index()
    if "job_completed_date" in job_completion.columns:
        job_completion["is_completed"] = job_completion["completed_date"].notna()
    elif "job_status" in job_completion.columns:
        job_completion["is_completed"] = job_completion["job_status"].str.lower().str.contains("completed", na=False)
    else:
        job_completion["is_completed"] = False

    completed_jobs = set(job_completion[job_completion["is_completed"] == True]["job_no"].tolist())
    df_completed = df[df["job_no"].isin(completed_jobs)].copy()
    if len(df_completed) == 0:
        return pd.DataFrame()

    date_col = _get_date_col(df_completed)
    df_completed[date_col] = pd.to_datetime(df_completed[date_col], errors="coerce")

    runtime = df_completed.groupby("job_no").agg(
        start_date=(date_col, "min"),
        end_date=(date_col, "max"),
    ).reset_index()
    runtime["runtime_days"] = (runtime["end_date"] - runtime["start_date"]).dt.days + 1
    runtime = runtime.merge(
        job_completion[["job_no", "department_final", "job_category"]],
        on="job_no",
        how="left",
    )

    profit = df_completed.groupby("job_no").agg(
        revenue=("rev_alloc", "sum") if "rev_alloc" in df_completed.columns else ("job_no", "count"),
        cost=("base_cost", "sum") if "base_cost" in df_completed.columns else ("job_no", "count"),
        hours=("hours_raw", "sum"),
    ).reset_index()
    if "is_billable" in df_completed.columns:
        billable = df_completed[df_completed["is_billable"] == True].groupby("job_no")["hours_raw"].sum()
        profit = profit.merge(billable.rename("billable_hours"), on="job_no", how="left")
        profit["billable_hours"] = profit["billable_hours"].fillna(0)
    else:
        profit["billable_hours"] = 0.0
    profit["margin_pct"] = np.where(
        profit["revenue"] > 0,
        (profit["revenue"] - profit["cost"]) / profit["revenue"] * 100,
        np.nan,
    )
    profit["billable_share"] = np.where(
        profit["hours"] > 0,
        profit["billable_hours"] / profit["hours"],
        np.nan,
    )
    profit = profit.merge(
        job_completion[["job_no", "department_final", "job_category"]],
        on="job_no",
        how="left",
    )

    bench = profit.merge(runtime[["job_no", "runtime_days"]], on="job_no", how="left")
    bench = bench.groupby(["department_final", "job_category"]).agg(
        median_runtime_days=("runtime_days", "median"),
        median_margin_pct=("margin_pct", "median"),
        median_billable_share=("billable_share", "median"),
    ).reset_index()
    return bench


def _forecast_job(df_job: pd.DataFrame, quoted_hours: float, quoted_amount: float) -> dict:
    """Forecast remaining hours, ETA, and margin for a job."""
    if len(df_job) == 0:
        return {}
    date_col = _get_date_col(df_job)
    df_job = df_job.copy()
    df_job[date_col] = pd.to_datetime(df_job[date_col], errors="coerce")
    last_date = df_job[date_col].max()

    # Burn rate over last 28 days (fallback to overall)
    cutoff = last_date - timedelta(days=28)
    recent = df_job[df_job[date_col] >= cutoff]
    if len(recent) == 0:
        recent = df_job
    days = max((recent[date_col].max() - recent[date_col].min()).days, 1)
    burn_rate_per_day = recent["hours_raw"].sum() / days if "hours_raw" in recent.columns else np.nan

    actual_hours = df_job["hours_raw"].sum() if "hours_raw" in df_job.columns else 0
    remaining_hours = max((quoted_hours or 0) - actual_hours, 0)
    eta_days = (remaining_hours / burn_rate_per_day) if burn_rate_per_day and burn_rate_per_day > 0 else np.nan
    eta_date = last_date + timedelta(days=float(eta_days)) if pd.notna(eta_days) else pd.NaT

    revenue_to_date = df_job["rev_alloc"].sum() if "rev_alloc" in df_job.columns else np.nan
    cost_to_date = df_job["base_cost"].sum() if "base_cost" in df_job.columns else np.nan
    avg_cost_per_hour = (cost_to_date / actual_hours) if actual_hours > 0 else np.nan
    forecast_cost = cost_to_date + (remaining_hours * avg_cost_per_hour) if pd.notna(avg_cost_per_hour) else np.nan
    forecast_revenue = quoted_amount if pd.notna(quoted_amount) and quoted_amount > 0 else revenue_to_date
    forecast_margin_pct = (
        (forecast_revenue - forecast_cost) / forecast_revenue * 100
        if pd.notna(forecast_revenue) and forecast_revenue > 0 and pd.notna(forecast_cost)
        else np.nan
    )

    return {
        "remaining_hours": remaining_hours,
        "eta_days": eta_days,
        "eta_date": eta_date,
        "forecast_margin_pct": forecast_margin_pct,
        "forecast_revenue": forecast_revenue,
    }


def _risk_score(row: pd.Series) -> tuple:
    """Compute risk score, band, and primary driver."""
    # Normalize inputs
    def _norm(val, max_val):
        if pd.isna(val):
            return 0
        return min(max(val / max_val, 0), 1)

    pct_consumed = _norm(row.get("pct_consumed", 0), 120)
    scope = _norm(row.get("scope_creep_pct", 0), 30)
    rate_var = _norm(abs(row.get("rate_variance", 0)), 50)
    runtime_delta = _norm(max(row.get("runtime_delta_days", 0), 0), 30)
    due_prox = _norm(max(row.get("days_to_due", 0), 0), 30) if pd.notna(row.get("days_to_due")) else 0

    score = (
        0.30 * pct_consumed
        + 0.20 * scope
        + 0.20 * rate_var
        + 0.20 * runtime_delta
        + 0.10 * (1 - due_prox)
    ) * 100

    if score >= 70:
        band = "Red"
    elif score >= 40:
        band = "Amber"
    else:
        band = "Green"

    drivers = {
        "Scope creep": scope,
        "Rate leakage": rate_var,
        "Schedule drift": runtime_delta,
        "Quote burn": pct_consumed,
    }
    primary = max(drivers, key=drivers.get)
    return score, band, primary


def _recommend_action(row: pd.Series) -> str:
    actions = []
    if pd.notna(row.get("scope_creep_pct")) and row["scope_creep_pct"] > 10:
        actions.append("Scope reset + reforecast")
    if pd.notna(row.get("rate_variance")) and row["rate_variance"] < -10:
        actions.append("Staffing mix review")
    if pd.notna(row.get("runtime_delta_days")) and row["runtime_delta_days"] > 7:
        actions.append("Timeline escalation")
    if pd.notna(row.get("pct_consumed")) and row["pct_consumed"] > 90:
        actions.append("Quote burn alert")
    return " | ".join(dict.fromkeys(actions)) if actions else "Monitor"


def build_active_jobs_view(df: pd.DataFrame) -> pd.DataFrame:
    """Build active jobs with risk metrics and forecasts."""
    
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

    # Benchmarks for runtime + margin
    bench = _compute_benchmarks(df)
    if len(bench) > 0:
        job_agg = job_agg.merge(
            bench.rename(columns={"job_category": "job_category"}),
            on=["department_final", "job_category"],
            how="left",
        )
    else:
        job_agg["median_runtime_days"] = np.nan
        job_agg["median_margin_pct"] = np.nan
        job_agg["median_billable_share"] = np.nan

    # Runtime delta vs benchmark
    date_col = _get_date_col(df_active)
    df_dates = df_active.copy()
    df_dates[date_col] = pd.to_datetime(df_dates[date_col], errors="coerce")
    runtime = df_dates.groupby("job_no").agg(
        start_date=(date_col, "min"),
        end_date=(date_col, "max"),
    ).reset_index()
    runtime["runtime_days"] = (runtime["end_date"] - runtime["start_date"]).dt.days + 1
    job_agg = job_agg.merge(runtime[["job_no", "runtime_days"]], on="job_no", how="left")
    job_agg["runtime_delta_days"] = job_agg["runtime_days"] - job_agg["median_runtime_days"]

    # Due date proximity
    if "job_due_date" in job_agg.columns:
        due_dates = pd.to_datetime(job_agg["job_due_date"], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        job_agg["days_to_due"] = (due_dates - now).dt.days
    else:
        job_agg["days_to_due"] = np.nan

    # Forecasts per job
    forecasts = []
    for job_no in job_agg["job_no"].unique():
        df_job = df_active[df_active["job_no"] == job_no]
        quoted_hours = job_agg.loc[job_agg["job_no"] == job_no, "quoted_hours"].iloc[0]
        quoted_amount = job_agg.loc[job_agg["job_no"] == job_no, "quoted_amount"].iloc[0]
        fc = _forecast_job(df_job, quoted_hours, quoted_amount)
        forecasts.append({
            "job_no": job_no,
            "remaining_hours": fc.get("remaining_hours"),
            "eta_days": fc.get("eta_days"),
            "eta_date": fc.get("eta_date"),
            "forecast_margin_pct": fc.get("forecast_margin_pct"),
            "forecast_revenue": fc.get("forecast_revenue"),
        })
    job_agg = job_agg.merge(pd.DataFrame(forecasts), on="job_no", how="left")

    # Risk score and actions
    scores = job_agg.apply(_risk_score, axis=1, result_type="expand")
    job_agg["risk_score"] = scores[0]
    job_agg["risk_band"] = scores[1]
    job_agg["primary_driver"] = scores[2]
    job_agg["recommended_action"] = job_agg.apply(_recommend_action, axis=1)

    job_agg = job_agg.sort_values(["risk_score", "pct_consumed"], ascending=[False, False])
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
    st.markdown(
        """
        <style>
        .sg-band {background:#f7f7f2;border:1px solid #eceae3;border-radius:18px;padding:22px 24px;margin:10px 0 22px;}
        .sg-band h3 {margin:0 0 6px 0;}
        .sg-section-title {font-size:1.2rem;font-weight:700;margin:0;}
        .sg-section-sub {color:#6b6b6b;font-size:0.95rem;margin:4px 0 12px 0;}
        .sg-subtle {color:#6b6b6b;font-size:0.95rem;}
        .sg-divider {height:1px;background:#eceae3;margin:16px 0;}
        .sg-card {background:#ffffff;border:1px solid #eceae3;border-radius:14px;padding:14px 16px;margin:6px 0;}
        .sg-kpi-title {font-size:0.82rem;color:#6b6b6b;text-transform:uppercase;letter-spacing:0.06em;}
        .sg-kpi-value {font-size:1.6rem;font-weight:700;margin-top:4px;}
        .sg-kpi-sub {font-size:0.85rem;color:#7a7a7a;margin-top:4px;}
        .sg-pill {padding:4px 8px;border-radius:999px;color:#1f1f1f;font-size:0.78rem;font-weight:600;margin-right:6px;display:inline-block;}
        .sg-callout {background:#fff7e6;border:1px solid #f6e0b3;border-radius:12px;padding:10px 12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Active Delivery Control Tower")
    st.markdown(
        '<div class="sg-subtle">Executive view of delivery risk, forecast trajectory, and operational interventions.</div>',
        unsafe_allow_html=True,
    )

    df = load_fact_timesheet()

    # ======================================================================
    # CHAIN CONTROLS
    # ======================================================================
    departments = ["All"] + sorted(df["department_final"].dropna().unique().tolist())
    dept_col, cat_col, job_col, band_col = st.columns([0.22, 0.22, 0.36, 0.20])
    with dept_col:
        selected_dept = st.selectbox("Department", departments, key="ad_dept")

    df_dept = df if selected_dept == "All" else df[df["department_final"] == selected_dept]
    category_col = get_category_col(df_dept)
    categories = ["All"] + sorted(df_dept[category_col].dropna().unique().tolist())
    with cat_col:
        selected_category = st.selectbox("Category", categories, key="ad_category")

    df_scope = df_dept if selected_category == "All" else df_dept[df_dept[category_col] == selected_category]
    jobs_df = build_active_jobs_view(df_scope)
    if len(jobs_df) == 0:
        st.warning("No active jobs found for the selected filters.")
        return

    job_options = ["Auto"] + sorted(jobs_df["job_no"].dropna().unique().tolist())
    with job_col:
        selected_job_choice = st.selectbox("Job", job_options, key="ad_job")

    with band_col:
        band_filter = st.multiselect(
            "Risk Band",
            options=["Red", "Amber", "Green"],
            default=["Red", "Amber"],
            key="ad_band",
        )

    if band_filter:
        jobs_df = jobs_df[jobs_df["risk_band"].isin(band_filter)]

    if len(jobs_df) == 0:
        st.info("No jobs match the selected risk bands.")
        return

    breadcrumb = "Company"
    if selected_dept != "All":
        breadcrumb += f"  ›  {selected_dept}"
    if selected_category != "All":
        breadcrumb += f"  ›  {selected_category}"
    if selected_job_choice != "Auto":
        breadcrumb += f"  ›  {selected_job_choice}"
    st.markdown(
        f"{_pill('Chain', '#e6f0ff')}<span class='sg-subtle'>{breadcrumb}</span>",
        unsafe_allow_html=True,
    )

    # ======================================================================
    # EXEC SUMMARY BAND
    # ======================================================================
    jobs_at_risk = jobs_df[jobs_df["risk_score"] >= 70]
    eac_hours = jobs_df["actual_hours"] + jobs_df["remaining_hours"].fillna(0)
    overrun_hours = (eac_hours - jobs_df["quoted_hours"]).fillna(0)
    forecast_margin = jobs_df["forecast_margin_pct"] / 100
    benchmark_margin = jobs_df["median_margin_pct"] / 100
    forecast_revenue = jobs_df["forecast_revenue"]
    margin_shortfall = (benchmark_margin - forecast_margin) * forecast_revenue
    margin_shortfall = margin_shortfall.clip(lower=0).fillna(0)
    margin_at_risk = margin_shortfall[jobs_df["risk_score"] >= 70].sum()

    st.markdown('<div class="sg-band">', unsafe_allow_html=True)
    st.markdown("### Executive Snapshot")
    st.markdown('<div class="sg-subtle">Immediate health of active delivery and exposure.</div>', unsafe_allow_html=True)
    kpi_cols = st.columns(6)
    kpi_cards = [
        _kpi_card("Active Jobs", f"{len(jobs_df)}"),
        _kpi_card("Jobs at Risk", f"{len(jobs_at_risk)}", "Risk score ≥ 70"),
        _kpi_card("Margin at Risk", fmt_currency(margin_at_risk), "Shortfall vs benchmark"),
        _kpi_card("Forecast Overrun Hrs", fmt_hours(overrun_hours[overrun_hours > 0].sum()), "EAC − Quoted"),
        _kpi_card("Avg Risk Score", f"{jobs_df['risk_score'].mean():.1f}", "0–100 scale"),
        _kpi_card("Scope Creep %", fmt_percent(jobs_df["scope_creep_pct"].mean()), "Unquoted share"),
    ]
    for col, card in zip(kpi_cols, kpi_cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================================
    # SECTION 1 — RISK QUEUE
    # ======================================================================
    st.markdown('<div class="sg-band">', unsafe_allow_html=True)
    st.markdown("<div class='sg-section-title'>Section 1 — Risk Queue</div>", unsafe_allow_html=True)
    st.markdown("<div class='sg-section-sub'>Prioritised list of jobs with the highest intervention value.</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sg-card'><div class='sg-kpi-title'>So what</div>"
        "<div class='sg-kpi-sub'>Start with Red bands and the top two drivers. "
        "These are the highest‑impact intervention candidates.</div></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.72, 0.28])
    with left:
        display_cols = [
            "job_no", "client", "department_final", "job_category",
            "risk_score", "risk_band", "primary_driver", "recommended_action",
            "eta_date", "forecast_margin_pct",
        ]
        show = jobs_df[[c for c in display_cols if c in jobs_df.columns]].copy()
        show = show.rename(columns={
            "job_no": "Job",
            "client": "Client",
            "department_final": "Department",
            "job_category": "Category",
            "risk_score": "Risk Score",
            "risk_band": "Risk Band",
            "primary_driver": "Primary Driver",
            "recommended_action": "Recommended Action",
            "eta_date": "ETA Completion",
            "forecast_margin_pct": "Forecast Margin %",
        })
        job_selection = st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Risk Score": st.column_config.NumberColumn(format="%.0f"),
                "Forecast Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                "ETA Completion": st.column_config.DateColumn(),
            },
            key="risk_queue",
        )
        if job_selection and job_selection.selection and job_selection.selection.rows:
            selected_idx = job_selection.selection.rows[0]
            st.session_state["ad_selected_job"] = jobs_df.iloc[selected_idx]["job_no"]

    with right:
        scope_jobs = jobs_df[jobs_df["scope_creep_pct"] > 10]
        rate_jobs = jobs_df[jobs_df["rate_variance"] < 0]
        drift_jobs = jobs_df[jobs_df["runtime_delta_days"] > 0]
        st.markdown("**Risk Driver Signals**")
        st.markdown(_kpi_card("Scope Creep", f"{len(scope_jobs)} jobs", fmt_percent(scope_jobs["scope_creep_pct"].mean())), unsafe_allow_html=True)
        st.markdown(_kpi_card("Rate Leakage", fmt_rate(rate_jobs["rate_variance"].mean()), "Avg variance"), unsafe_allow_html=True)
        drift_mean = drift_jobs["runtime_delta_days"].mean()
        st.markdown(_kpi_card("Schedule Drift", f"{drift_mean:.1f} days" if pd.notna(drift_mean) else "—", "Avg vs benchmark"), unsafe_allow_html=True)
        st.markdown('<div class="sg-callout">Drivers are computed from quote burn, scope creep, rate variance, and runtime drift.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Resolve selected job for downstream sections
    selected_job = _resolve_selected_job(jobs_df, selected_job_choice)
    job_row = jobs_df[jobs_df["job_no"] == selected_job].iloc[0]
    df_job = df_scope[df_scope["job_no"] == selected_job].copy()

    # ======================================================================
    # SECTION 2 — FORECAST & INTERVENTION
    # ======================================================================
    st.markdown('<div class="sg-band">', unsafe_allow_html=True)
    st.markdown("<div class='sg-section-title'>Section 2 — Forecast & Intervention</div>", unsafe_allow_html=True)
    st.markdown("<div class='sg-section-sub'>Mechanics are explicit. Outputs tie to intervention levers.</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sg-card'><div class='sg-kpi-title'>So what</div>"
        "<div class='sg-kpi-sub'>Forecast tells you where the job lands vs quote; "
        "interventions explain how to correct course.</div></div>",
        unsafe_allow_html=True,
    )

    forecast_col, action_col = st.columns(2)
    with forecast_col:
        st.markdown(f"**Forecast Dashboard — {selected_job}**")
        st.markdown(
            "<div class='sg-card'>"
            "<div class='sg-kpi-title'>Forecast Mechanics (Step-by-step)</div>"
            "<div class='sg-kpi-sub'>"
            "1) Burn rate = last 28 days (fallback: full history). "
            "2) Remaining Hours = max(Quoted − Actual, 0). "
            "3) EAC Hours = Actual + Remaining. "
            "4) EAC Margin % uses avg cost per hour and quoted revenue if available."
            "</div></div>",
            unsafe_allow_html=True,
        )
        date_col = _get_date_col(df_job)
        df_job[date_col] = pd.to_datetime(df_job[date_col], errors="coerce")
        timeline = df_job.groupby(date_col)["hours_raw"].sum().sort_index()
        cum_actual = timeline.cumsum()
        quoted_hours = job_row.get("quoted_hours", np.nan)
        remaining = job_row.get("remaining_hours")
        eac = job_row["actual_hours"] + (remaining if pd.notna(remaining) else 0)

        st.markdown("<div class='sg-card'>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum_actual.index,
            y=cum_actual.values,
            mode="lines+markers",
            name="Actual (cumulative)",
        ))
        if pd.notna(quoted_hours):
            fig.add_hline(y=quoted_hours, line_dash="dash", annotation_text="Quoted")
        if pd.notna(eac):
            fig.add_hline(y=eac, line_dash="dot", annotation_text="Forecast EAC")
        fig.update_layout(
            height=280,
            margin=dict(t=30, l=10, r=10, b=10),
            xaxis_title="Date",
            yaxis_title="Cumulative Hours",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"forecast_burndown_{selected_job}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sg-card'>", unsafe_allow_html=True)
        kpi_cols = st.columns(3)
        with kpi_cols[0]:
            st.markdown(_kpi_card("EAC Hours", fmt_hours(eac), "Forecast total"), unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(_kpi_card("EAC Margin %", fmt_percent(job_row.get("forecast_margin_pct")), "Forecast margin"), unsafe_allow_html=True)
        eta_days = job_row.get("eta_days")
        with kpi_cols[2]:
            eta_label = f"{eta_days:.0f}" if pd.notna(eta_days) else "—"
            st.markdown(_kpi_card("ETA Completion (days)", eta_label, "From burn rate"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Inputs panel for traceability
        actual_hours = job_row.get("actual_hours", np.nan)
        quoted_hours = job_row.get("quoted_hours", np.nan)
        burn_rate = np.nan
        if len(timeline) > 1:
            days_span = max((timeline.index.max() - timeline.index.min()).days, 1)
            burn_rate = timeline.sum() / days_span
        remaining = job_row.get("remaining_hours", np.nan)
        st.markdown(
            "<div class='sg-card'>"
            "<div class='sg-kpi-title'>Forecast Inputs</div>"
            "<div class='sg-kpi-sub'>"
            f"Actual Hours: {fmt_hours(actual_hours)} · "
            f"Quoted Hours: {fmt_hours(quoted_hours)} · "
            f"Burn Rate: {fmt_hours(burn_rate)}/day · "
            f"Remaining: {fmt_hours(remaining)}"
            "</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Forecast Drivers (Task Level)**")
        task_breakdown = get_job_task_breakdown(df_scope, selected_job)
        if len(task_breakdown) > 0:
            task_breakdown["overrun_hours"] = task_breakdown["variance"].fillna(0).clip(lower=0)
            total_overrun = task_breakdown["overrun_hours"].sum()
            driver_table = task_breakdown.sort_values("overrun_hours", ascending=False).head(5)
            driver_table["overrun_share"] = np.where(
                total_overrun > 0,
                driver_table["overrun_hours"] / total_overrun * 100,
                np.nan,
            )
            st.dataframe(
                driver_table[["task_name", "actual_hours", "quoted_hours", "overrun_hours", "overrun_share"]].rename(
                    columns={
                        "task_name": "Task",
                        "actual_hours": "Actual",
                        "quoted_hours": "Quoted",
                        "overrun_hours": "Overrun",
                        "overrun_share": "Overrun Share",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Actual": st.column_config.NumberColumn(format="%.1f"),
                    "Quoted": st.column_config.NumberColumn(format="%.1f"),
                    "Overrun": st.column_config.NumberColumn(format="%.1f"),
                    "Overrun Share": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
        else:
            st.caption("No task-level quote data available for driver analysis.")

    with action_col:
        st.markdown("**Intervention Playbook**")
        st.markdown(
            "<div class='sg-card'>"
            "<div class='sg-kpi-title'>Primary Driver</div>"
            f"<div class='sg-kpi-value'>{job_row.get('primary_driver','—')}</div>"
            "<div class='sg-kpi-sub'>Recommendation follows the dominant driver.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("Action summary")
        st.info(job_row.get("recommended_action", "Monitor"))
        st.markdown("**Priority interventions**")
        action_list = [a.strip() for a in str(job_row.get("recommended_action", "")).split("|") if a.strip()]
        if action_list:
            for action in action_list:
                st.write(f"- {action}")
        else:
            st.write("- Monitor")

        st.markdown("**Staff Mix Variance (Billable Share)**")
        if "is_billable" in df_job.columns:
            job_billable = df_job[df_job["is_billable"] == True]["hours_raw"].sum()
            job_total = df_job["hours_raw"].sum()
            job_billable_share = job_billable / job_total if job_total > 0 else np.nan
            bench_share = job_row.get("median_billable_share")
            variance = (job_billable_share - bench_share) * 100 if pd.notna(bench_share) else np.nan
            st.markdown(
                _kpi_card(
                    "Job billable share",
                    fmt_percent(job_billable_share * 100 if pd.notna(job_billable_share) else np.nan),
                    f"Benchmark: {fmt_percent(bench_share * 100) if pd.notna(bench_share) else '—'} · Δ {variance:.1f} pp" if pd.notna(variance) else "Benchmark unavailable",
                ),
                unsafe_allow_html=True,
            )
            staff_mix = df_job.groupby("staff_name").agg(
                hours=("hours_raw", "sum"),
                billable_hours=("hours_raw", lambda x: x[df_job.loc[x.index, "is_billable"]].sum()),
            ).reset_index()
            staff_mix["billable_share"] = np.where(
                staff_mix["hours"] > 0,
                staff_mix["billable_hours"] / staff_mix["hours"] * 100,
                np.nan,
            )
            top_staff = staff_mix.sort_values("hours", ascending=False).head(8)
            st.dataframe(
                top_staff[["staff_name", "hours", "billable_share"]].rename(
                    columns={"staff_name": "Staff", "hours": "Hours", "billable_share": "Billable %"}
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Hours": st.column_config.NumberColumn(format="%.1f"),
                    "Billable %": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
        else:
            st.caption("Billable share not available in the dataset.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================================
    # SECTION 3 — JOB DEEP-DIVE
    # ======================================================================
    st.markdown('<div class="sg-band">', unsafe_allow_html=True)
    st.markdown("<div class='sg-section-title'>Section 3 — Job Deep‑Dive</div>", unsafe_allow_html=True)
    st.markdown("<div class='sg-section-sub'>Traceability from job health → tasks → staff.</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sg-card'><div class='sg-kpi-title'>So what</div>"
        "<div class='sg-kpi-sub'>Use this to pinpoint what’s driving margin erosion and "
        "where to intervene (task or staff).</div></div>",
        unsafe_allow_html=True,
    )

    tab_a, tab_b, tab_c = st.tabs(["Delivery Health", "Profitability", "Drivers"])

    with tab_a:
        st.markdown("**Quote vs Actual vs Forecast**")
        st.markdown(
            "<div class='sg-card'>"
            "<div class='sg-kpi-title'>Traceability</div>"
            "<div class='sg-kpi-sub'>Cumulative hours vs quote and forecast EAC. "
            "This is the single source of truth for delivery drift.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"delivery_health_burndown_{selected_job}")
        st.markdown("**Runtime vs Benchmark**")
        st.markdown(
            "<div class='sg-card'>"
            "<div class='sg-kpi-title'>Benchmark Logic</div>"
            "<div class='sg-kpi-sub'>Median runtime for completed jobs within the same department and category.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        runtime_cols = st.columns(3)
        with runtime_cols[0]:
            st.metric("Runtime (days)", f"{job_row.get('runtime_days', np.nan):.0f}")
        with runtime_cols[1]:
            st.metric("Benchmark (median)", f"{job_row.get('median_runtime_days', np.nan):.0f}")
        with runtime_cols[2]:
            st.metric("Delta (days)", f"{job_row.get('runtime_delta_days', np.nan):.0f}")

    with tab_b:
        st.markdown("**Margin Trend**")
        st.markdown(
            "<div class='sg-card'>"
            "<div class='sg-kpi-title'>Margin Logic</div>"
            "<div class='sg-kpi-sub'>(Revenue − Cost) ÷ Revenue over time.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if "rev_alloc" in df_job.columns and "base_cost" in df_job.columns:
            margin_ts = df_job.groupby(date_col).agg(
                revenue=("rev_alloc", "sum"),
                cost=("base_cost", "sum"),
            ).reset_index()
            margin_ts["margin_pct"] = np.where(
                margin_ts["revenue"] > 0,
                (margin_ts["revenue"] - margin_ts["cost"]) / margin_ts["revenue"] * 100,
                np.nan,
            )
            fig_margin = px.line(margin_ts, x=date_col, y="margin_pct", markers=True)
            fig_margin.update_layout(
                height=280,
                margin=dict(t=30, l=10, r=10, b=10),
                xaxis_title="Date",
                yaxis_title="Margin %",
            )
            st.plotly_chart(fig_margin, use_container_width=True, key=f"margin_trend_{selected_job}")
        st.markdown("**Rate Capture**")
        st.markdown(
            "<div class='sg-card'>"
            "<div class='sg-kpi-title'>Rate Logic</div>"
            "<div class='sg-kpi-sub'>Realised Rate = Actual Revenue ÷ Actual Hours. "
            "Quote Rate = Quoted Amount ÷ Quoted Hours.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        rate_cols = st.columns(2)
        with rate_cols[0]:
            st.metric("Realised Rate", fmt_rate(job_row.get("realised_rate")))
        with rate_cols[1]:
            st.metric("Quote Rate", fmt_rate(job_row.get("quote_rate")))

    with tab_c:
        task_df = get_job_task_breakdown(df_scope, selected_job)
        staff_df = get_job_staff_breakdown(df_scope, selected_job)

        st.markdown("**Task Cost Leaders**")
        st.markdown(
            "<div class='sg-card'>"
            "<div class='sg-kpi-title'>Task Attribution</div>"
            "<div class='sg-kpi-sub'>Top tasks by cost; variance flags overruns vs quote.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if len(task_df) > 0:
            top_tasks = task_df.sort_values("actual_cost", ascending=False).head(10)
            st.dataframe(
                top_tasks[["task_name", "actual_hours", "actual_cost", "variance"]].rename(
                    columns={"task_name": "Task", "actual_hours": "Hours", "actual_cost": "Cost", "variance": "Variance"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("**Staff Cost Leaders**")
        st.markdown(
            "<div class='sg-card'>"
            "<div class='sg-kpi-title'>Staff Attribution</div>"
            "<div class='sg-kpi-sub'>Top staff by cost; validates mix and resourcing.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if len(staff_df) > 0:
            top_staff = staff_df.sort_values("cost", ascending=False).head(10)
            st.dataframe(
                top_staff[["staff_name", "hours", "cost", "pct_of_job"]].rename(
                    columns={"staff_name": "Staff", "hours": "Hours", "cost": "Cost", "pct_of_job": "% of Job"}
                ),
                use_container_width=True,
                hide_index=True,
                column_config={"% of Job": st.column_config.NumberColumn(format="%.1f%%")},
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================================
    # SECTION 4 — BENCHMARKS
    # ======================================================================
    with st.expander("Benchmark Panel (dept + category)"):
        st.markdown("**Benchmarks (completed jobs only)**")
        bench_cols = st.columns(3)
        with bench_cols[0]:
            st.metric("Median runtime", f"{job_row.get('median_runtime_days', np.nan):.0f} days")
        with bench_cols[1]:
            st.metric("Median margin %", fmt_percent(job_row.get("median_margin_pct")))
        with bench_cols[2]:
            st.metric(
                "Median billable share",
                fmt_percent(job_row.get("median_billable_share") * 100 if pd.notna(job_row.get("median_billable_share")) else np.nan),
            )

        df_completed = df_scope.copy()
        if "job_completed_date" in df_completed.columns:
            df_completed = df_completed[df_completed["job_completed_date"].notna()]
        elif "job_status" in df_completed.columns:
            df_completed = df_completed[df_completed["job_status"].str.lower().str.contains("completed", na=False)]
        else:
            df_completed = df_completed.iloc[0:0]

        if len(df_completed) > 0:
            df_completed = df_completed[
                (df_completed["department_final"] == job_row.get("department_final")) &
                (df_completed[category_col] == job_row.get("job_category"))
            ]

        if len(df_completed) > 0:
            task_mix = df_completed.groupby("task_name")["hours_raw"].sum().sort_values(ascending=False).head(5)
            st.markdown("**Median task mix (top 5)**")
            st.dataframe(
                task_mix.reset_index().rename(columns={"task_name": "Task", "hours_raw": "Hours"}),
                use_container_width=True,
                hide_index=True,
            )

            job_actual = df_completed.groupby("job_no")["hours_raw"].sum()
            job_task_quote = safe_quote_job_task(df_completed)
            if len(job_task_quote) > 0:
                job_quote = job_task_quote.groupby("job_no")["quoted_time_total"].sum()
                quote_ratio = (job_quote / job_actual).replace([np.inf, -np.inf], np.nan)
                st.metric("Median quote accuracy", fmt_percent(quote_ratio.median() * 100))
        else:
            st.caption("No completed job benchmarks available for this department and category.")


if __name__ == "__main__":
    main()
