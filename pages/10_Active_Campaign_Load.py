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
from src.data.semantic import exclude_leave, get_category_col, safe_quote_job_task
from src.ui.formatting import build_job_name_lookup, format_job_label, fmt_hours, fmt_count, fmt_percent
from src.ui.state import init_state


st.set_page_config(page_title="Campaign Load & Capacity", page_icon="📊", layout="wide")

MONTHLY_CAPACITY_HOURS = config.CAPACITY_HOURS_PER_WEEK * 4.33


# =========================
# Helpers
# =========================

def _month_label(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return "-"
    return pd.to_datetime(value).strftime("%b %Y")


def _first_non_null(series: pd.Series) -> object:
    values = series.dropna()
    if len(values) == 0:
        return np.nan
    return values.iloc[0]


def _mode_or_first(series: pd.Series, default: float) -> float:
    values = series.dropna()
    if len(values) == 0:
        return default
    mode = values.mode()
    if len(mode) > 0:
        return float(mode.iloc[0])
    return float(values.iloc[0])


def _util_status(util_pct: float) -> str:
    if pd.isna(util_pct):
        return ""
    if util_pct <= 80:
        return "🟢"
    if util_pct <= 100:
        return "🟡"
    return "🔴"


@st.cache_data(show_spinner=False)
def derive_staff_home_dept(
    df: pd.DataFrame,
    staff_col: str = "staff_name",
    dept_col: str = "department_ts_raw",
) -> pd.DataFrame:
    """
    Assign each staff member exactly ONE home department for the analysis period.
    Rule: department_ts_raw value on the MAJORITY of their timesheet rows.
    Tie-break: most recent work_date (or month_key if work_date unavailable).
    Returns: DataFrame with [staff_name, staff_home_dept] — one row per staff.
    """
    if dept_col not in df.columns:
        dept_col = "department_final"

    date_col = "work_date" if "work_date" in df.columns else "month_key"
    subset = df[[staff_col, dept_col, date_col]].copy()
    subset = subset.dropna(subset=[staff_col, dept_col])
    if len(subset) == 0:
        return pd.DataFrame(columns=[staff_col, "staff_home_dept"])

    counts = subset.groupby([staff_col, dept_col]).agg(
        row_count=(dept_col, "size"),
        latest_date=(date_col, "max"),
    ).reset_index()
    counts = counts.sort_values(["row_count", "latest_date"], ascending=[False, False])
    home = counts.groupby(staff_col).first().reset_index()
    home = home.rename(columns={dept_col: "staff_home_dept"})
    return home[[staff_col, "staff_home_dept"]]


@st.cache_data(show_spinner=False)
def compute_category_benchmarks(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                category_col,
                "campaigns_in_sample",
                "avg_hrs_per_campaign_pm",
                "median_hrs_per_campaign_pm",
                "p25_hrs_per_campaign_pm",
                "p75_hrs_per_campaign_pm",
                "min_hrs",
                "max_hrs",
            ]
        )

    job_monthly_hours = (
        df.groupby(["job_no", category_col, "month_key"], dropna=False)["hours_raw"]
        .sum()
        .reset_index()
        .rename(columns={"hours_raw": "job_month_hours"})
    )
    if len(job_monthly_hours) == 0:
        return pd.DataFrame(
            columns=[
                category_col,
                "campaigns_in_sample",
                "avg_hrs_per_campaign_pm",
                "median_hrs_per_campaign_pm",
                "p25_hrs_per_campaign_pm",
                "p75_hrs_per_campaign_pm",
                "min_hrs",
                "max_hrs",
            ]
        )

    job_avg = (
        job_monthly_hours.groupby(["job_no", category_col], dropna=False)["job_month_hours"]
        .mean()
        .reset_index()
        .rename(columns={"job_month_hours": "avg_hrs_pm"})
    )

    category_benchmarks = (
        job_avg.groupby(category_col, dropna=False)
        .agg(
            campaigns_in_sample=("job_no", "nunique"),
            avg_hrs_per_campaign_pm=("avg_hrs_pm", "mean"),
            median_hrs_per_campaign_pm=("avg_hrs_pm", "median"),
            p25_hrs_per_campaign_pm=("avg_hrs_pm", lambda x: x.quantile(0.25)),
            p75_hrs_per_campaign_pm=("avg_hrs_pm", lambda x: x.quantile(0.75)),
            min_hrs=("avg_hrs_pm", "min"),
            max_hrs=("avg_hrs_pm", "max"),
        )
        .reset_index()
    )
    return category_benchmarks


def _download_button(label: str, df: pd.DataFrame, filename: str, key: str | None = None) -> None:
    csv = df.to_csv(index=False)
    st.download_button(label, data=csv, file_name=filename, mime="text/csv", key=key)


def _style_spare_negative(df: pd.DataFrame, label_col: str = "Campaign") -> pd.io.formats.style.Styler:
    def _row_style(row: pd.Series) -> list[str]:
        if row.get(label_col) != "Spare":
            return ["" for _ in row]
        styles: list[str] = []
        for value in row:
            if isinstance(value, (int, float, np.floating)) and value < 0:
                styles.append("color: #d62728;")
            else:
                styles.append("")
        return styles

    styled = df.style.apply(_row_style, axis=1)
    styled = styled.hide(axis="index")
    return styled


# =========================
# Main
# =========================

def main() -> None:
    init_state()

    st.title("Campaign Load Distribution & Capacity Planner")

    df_raw = load_fact_timesheet()
    if df_raw is None or len(df_raw) == 0:
        st.info("No data available.")
        return

    required_cols = {"job_no", "staff_name", "hours_raw", "month_key", "department_final"}
    missing_cols = sorted(required_cols - set(df_raw.columns))
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return

    if "task_name" not in df_raw.columns:
        df_raw["task_name"] = np.nan
    if "job_name" not in df_raw.columns:
        df_raw["job_name"] = np.nan

    df_raw = df_raw.copy()
    df_raw["month_key"] = pd.to_datetime(df_raw["month_key"]).dt.to_period("M").dt.to_timestamp()
    if "work_date" in df_raw.columns:
        df_raw["work_date"] = pd.to_datetime(df_raw["work_date"])

    df_raw["hours_raw"] = pd.to_numeric(df_raw["hours_raw"], errors="coerce")

    dept_ts_col = "department_ts_raw"
    if dept_ts_col not in df_raw.columns:
        st.warning(
            "Column department_ts_raw not found. Using department_final as proxy for Staff Home Department. "
            "Cross-departmental analysis will be limited."
        )
        dept_ts_col = "department_final"

    category_col = get_category_col(df_raw)
    if category_col not in df_raw.columns:
        df_raw[category_col] = np.nan

    months_all = sorted(pd.to_datetime(df_raw["month_key"].dropna().unique()))
    if len(months_all) == 0:
        st.info("No month_key data available.")
        return

    # =========================
    # Sidebar — Analysis Period
    # =========================
    st.sidebar.markdown("### 📅 Analysis Period")

    if "period_start" not in st.session_state:
        st.session_state["period_start"] = months_all[max(0, len(months_all) - 3)]
    if "period_end" not in st.session_state:
        st.session_state["period_end"] = months_all[-1]

    st.sidebar.selectbox("From", options=months_all, format_func=_month_label, key="period_start")
    st.sidebar.selectbox("To", options=months_all, format_func=_month_label, key="period_end")

    def _set_period(months_back: int) -> None:
        if len(months_all) == 0:
            return
        end = months_all[-1]
        start_idx = max(0, len(months_all) - months_back)
        st.session_state["period_start"] = months_all[start_idx]
        st.session_state["period_end"] = end

    def _set_fytd() -> None:
        if len(months_all) == 0:
            return
        end = months_all[-1]
        fy_months = [m for m in months_all if m.year == end.year]
        start = fy_months[0] if fy_months else months_all[0]
        st.session_state["period_start"] = start
        st.session_state["period_end"] = end

    quick_cols = st.sidebar.columns(4)
    quick_cols[0].button("3m", on_click=_set_period, args=(3,))
    quick_cols[1].button("6m", on_click=_set_period, args=(6,))
    quick_cols[2].button("12m", on_click=_set_period, args=(12,))
    quick_cols[3].button("FYTD", on_click=_set_fytd)

    period_start = pd.to_datetime(st.session_state["period_start"])
    period_end = pd.to_datetime(st.session_state["period_end"])
    if period_start > period_end:
        period_start, period_end = period_end, period_start
        st.session_state["period_start"] = period_start
        st.session_state["period_end"] = period_end

    months_in_period = [m for m in months_all if period_start <= m <= period_end]
    num_months = max(1, len(months_in_period))

    period_label = f"{_month_label(period_start)} – {_month_label(period_end)}"
    st.sidebar.caption(f"► Period: {period_label} ({len(months_in_period)} months)")

    # =========================
    # Sidebar — Filters
    # =========================
    st.sidebar.markdown("### Filters")
    active_only = st.sidebar.checkbox("Active jobs only", value=True)
    recency_days = config.active_job_recency_days
    if active_only:
        recency_days = st.sidebar.slider("Recency (days)", min_value=7, max_value=120, value=21, step=1)
    exclude_leave_toggle = st.sidebar.checkbox("Exclude leave", value=True)
    include_nonbillable = st.sidebar.checkbox("Include non-billable", value=True)

    # =========================
    # Apply Base Filters
    # =========================
    df = df_raw.copy()
    if exclude_leave_toggle:
        df = exclude_leave(df)
    if not include_nonbillable and "is_billable" in df.columns:
        df = df[df["is_billable"] == True]

    if active_only:
        active_jobs = get_active_jobs(df, recency_days=recency_days)
        if len(active_jobs) == 0:
            st.info("No active jobs found for the selected period and filters.")
            return
        df = df[df["job_no"].isin(active_jobs)]

    df = df[df["month_key"].between(period_start, period_end)]
    if len(df) == 0:
        if active_only:
            st.info("No active jobs found for the selected period and filters.")
        else:
            st.info("No activity found for the selected period and filters.")
        return

    # =========================
    # Staff Home Department
    # =========================
    home_input = df[["staff_name", dept_ts_col, "month_key"]].copy()
    if "work_date" in df.columns:
        home_input["work_date"] = df["work_date"]
    staff_home = derive_staff_home_dept(home_input, staff_col="staff_name", dept_col=dept_ts_col)
    df = df.merge(staff_home, on="staff_name", how="left")

    df["staff_home_dept"] = df["staff_home_dept"].fillna("(Unknown)")
    df["department_final"] = df["department_final"].fillna("(Unknown)")
    df["category_display"] = df[category_col].fillna("(No Category)")

    # =========================
    # Department Filters (Main)
    # =========================
    dept_cols = st.columns(2)
    with dept_cols[0]:
        staff_home_options = ["All"] + sorted(df["staff_home_dept"].dropna().unique().tolist())
        selected_staff_home = st.selectbox("Staff Home Department", staff_home_options, index=0)
    with dept_cols[1]:
        job_dept_options = ["All"] + sorted(df["department_final"].dropna().unique().tolist())
        selected_job_dept = st.selectbox("Job Department", job_dept_options, index=0)

    df_scope = df.copy()
    if selected_staff_home != "All":
        df_scope = df_scope[df_scope["staff_home_dept"] == selected_staff_home]
    if selected_job_dept != "All":
        df_scope = df_scope[df_scope["department_final"] == selected_job_dept]

    if len(df_scope) == 0:
        st.info("No activity found for the selected department filters.")
        return

    # =========================
    # Section 0 — Context Bar
    # =========================
    st.caption(
        f"Analysis Period: {period_label} ({len(months_in_period)} months) | "
        f"Staff Home Dept: {selected_staff_home} | Job Dept: {selected_job_dept}"
    )

    # =========================
    # Section 1 — Team Summary KPIs
    # =========================
    staff_group = df_scope.groupby("staff_name", dropna=False)
    staff_summary = staff_group.agg(
        total_hours=("hours_raw", "sum"),
        campaign_count=("job_no", "nunique"),
        staff_home_dept=("staff_home_dept", _first_non_null),
    ).reset_index()

    if "fte_hours_scaling" in df_scope.columns:
        fte_scaling = (
            df_scope.groupby("staff_name")["fte_hours_scaling"]
            .agg(lambda x: _mode_or_first(x, config.DEFAULT_FTE_SCALING))
            .reset_index()
            .rename(columns={"fte_hours_scaling": "fte_scaling"})
        )
    else:
        fte_scaling = pd.DataFrame({
            "staff_name": staff_summary["staff_name"],
            "fte_scaling": config.DEFAULT_FTE_SCALING,
        })

    staff_summary = staff_summary.merge(fte_scaling, on="staff_name", how="left")
    staff_summary["fte_scaling"] = staff_summary["fte_scaling"].fillna(config.DEFAULT_FTE_SCALING)
    staff_summary["avg_load_pm"] = staff_summary["total_hours"] / num_months
    staff_summary["capacity_pm"] = staff_summary["fte_scaling"] * MONTHLY_CAPACITY_HOURS
    staff_summary["spare_pm"] = staff_summary["capacity_pm"] - staff_summary["avg_load_pm"]
    staff_summary["util_pct"] = np.where(
        staff_summary["capacity_pm"] > 0,
        staff_summary["avg_load_pm"] / staff_summary["capacity_pm"] * 100,
        np.nan,
    )

    staff_job_unique = df_scope[["staff_name", "job_no", "department_final", "staff_home_dept"]].drop_duplicates()
    staff_job_unique["cross_dept"] = (
        staff_job_unique["department_final"] != staff_job_unique["staff_home_dept"]
    )
    cross_counts = (
        staff_job_unique.groupby("staff_name")["cross_dept"].sum().reset_index().rename(columns={"cross_dept": "cross_dept_campaigns"})
    )
    staff_summary = staff_summary.merge(cross_counts, on="staff_name", how="left")
    staff_summary["cross_dept_campaigns"] = staff_summary["cross_dept_campaigns"].fillna(0).astype(int)

    staff_summary = staff_summary[staff_summary["total_hours"] > 0].copy()
    staff_summary["status"] = staff_summary["util_pct"].apply(_util_status)
    staff_summary = staff_summary.sort_values("util_pct", ascending=False)

    active_staff = int(staff_summary["staff_name"].nunique())
    active_campaigns = int(df_scope["job_no"].nunique())
    total_hours_pm = df_scope["hours_raw"].sum() / num_months
    avg_campaigns_per_person = active_campaigns / active_staff if active_staff > 0 else 0
    avg_hours_per_campaign_pm = total_hours_pm / active_campaigns if active_campaigns > 0 else 0
    team_capacity_pm = staff_summary["capacity_pm"].sum()
    team_spare_pm = team_capacity_pm - total_hours_pm
    team_util_pct = total_hours_pm / team_capacity_pm * 100 if team_capacity_pm > 0 else np.nan

    kpi_cols = st.columns(8)
    kpi_cols[0].metric("Active Staff", fmt_count(active_staff))
    kpi_cols[1].metric("Active Campaigns", fmt_count(active_campaigns))
    kpi_cols[2].metric("Total Hours p/m", fmt_hours(total_hours_pm))
    kpi_cols[3].metric("Avg Campaigns/Person", f"{avg_campaigns_per_person:.1f}")
    kpi_cols[4].metric("Avg Hrs/Campaign/Month", fmt_hours(avg_hours_per_campaign_pm))
    kpi_cols[5].metric("Team Capacity p/m", fmt_hours(team_capacity_pm))
    kpi_cols[6].metric("Team Spare p/m", fmt_hours(team_spare_pm))
    kpi_cols[7].metric("Team Utilisation %", fmt_percent(team_util_pct))

    st.divider()

    # =========================
    # Section 2 — Staff Campaign Load Matrix
    # =========================
    st.subheader("Staff Campaign Load — Hours per Month per Campaign")

    summary_display = staff_summary[[
        "staff_name",
        "staff_home_dept",
        "fte_scaling",
        "capacity_pm",
        "campaign_count",
        "avg_load_pm",
        "spare_pm",
        "util_pct",
        "cross_dept_campaigns",
        "status",
    ]].copy()
    summary_display = summary_display.rename(columns={
        "staff_name": "Staff",
        "staff_home_dept": "Home Dept",
        "fte_scaling": "FTE",
        "capacity_pm": "Capacity p/m",
        "campaign_count": "# Campaigns",
        "avg_load_pm": "Avg Load p/m",
        "spare_pm": "Spare p/m",
        "util_pct": "Util %",
        "cross_dept_campaigns": "Cross-Dept Campaigns",
        "status": "Status",
    })
    summary_display["FTE"] = summary_display["FTE"].round(2)
    summary_display["Capacity p/m"] = summary_display["Capacity p/m"].round(1)
    summary_display["Avg Load p/m"] = summary_display["Avg Load p/m"].round(1)
    summary_display["Spare p/m"] = summary_display["Spare p/m"].round(1)
    summary_display["Util %"] = summary_display["Util %"].round(1)
    summary_display["# Campaigns"] = summary_display["# Campaigns"].astype(int)
    summary_display["Cross-Dept Campaigns"] = summary_display["Cross-Dept Campaigns"].astype(int)

    st.dataframe(summary_display, use_container_width=True, hide_index=True)
    _download_button(
        "Download staff summary CSV",
        summary_display,
        "staff_campaign_load_summary.csv",
        key="download_staff_summary",
    )

    st.markdown("### Staff Detail")

    month_labels = [_month_label(m) for m in months_in_period]
    month_label_map = {m: _month_label(m) for m in months_in_period}

    staff_job_month = (
        df_scope.groupby(["staff_name", "job_no", "month_key"], dropna=False)["hours_raw"]
        .sum()
        .reset_index()
    )
    staff_job_tasks = (
        df_scope.groupby(["staff_name", "job_no"], dropna=False)["task_name"]
        .nunique()
        .reset_index()
        .rename(columns={"task_name": "tasks"})
    )
    job_meta = (
        df_scope.groupby("job_no", dropna=False)
        .agg(
            department_final=("department_final", _first_non_null),
            category_display=("category_display", _first_non_null),
        )
        .reset_index()
    )
    job_name_lookup = build_job_name_lookup(df_scope)

    for _, staff_row in staff_summary.iterrows():
        staff_name = staff_row["staff_name"]
        staff_home_dept = staff_row["staff_home_dept"]
        fte_scaling = staff_row["fte_scaling"]
        capacity_pm = staff_row["capacity_pm"]
        avg_load_pm = staff_row["avg_load_pm"]
        spare_pm = staff_row["spare_pm"]
        util_pct = staff_row["util_pct"]
        campaigns = int(staff_row["campaign_count"])
        status = staff_row["status"]

        expander_label = (
            f"{staff_name} | Home: {staff_home_dept} | FTE: {fte_scaling:.2f} | "
            f"Cap: {capacity_pm:.1f} hrs/mo | Load: {avg_load_pm:.1f} hrs/mo ({campaigns} campaigns) | "
            f"Spare: {spare_pm:.1f} hrs/mo | Util: {util_pct:.1f}% {status}"
        )

        with st.expander(expander_label, expanded=False):
            staff_month = staff_job_month[staff_job_month["staff_name"] == staff_name]
            if len(staff_month) == 0:
                st.info("No campaign activity in period.")
                continue

            pivot = staff_month.pivot_table(
                index="job_no",
                columns="month_key",
                values="hours_raw",
                aggfunc="sum",
                fill_value=0.0,
            )
            pivot = pivot.reindex(columns=months_in_period, fill_value=0.0)
            pivot.columns = [month_label_map[m] for m in months_in_period]
            table = pivot.reset_index()

            table = table.merge(job_meta, on="job_no", how="left")
            table = table.merge(
                staff_job_tasks[staff_job_tasks["staff_name"] == staff_name],
                on="job_no",
                how="left",
            )
            table["tasks"] = table["tasks"].fillna(0).astype(int)

            table["Campaign"] = table["job_no"].apply(lambda x: format_job_label(x, job_name_lookup))
            table["Job Dept"] = table["department_final"].fillna("(Unknown)")
            table["Category"] = table["category_display"].fillna("(No Category)")
            table["Cross-Dept?"] = np.where(
                table["department_final"] != staff_home_dept,
                "Yes",
                "No",
            )

            table["Avg hrs/mo"] = table[month_labels].sum(axis=1) / num_months
            table["% of Load"] = np.where(
                avg_load_pm > 0,
                table["Avg hrs/mo"] / avg_load_pm * 100,
                0.0,
            )

            job_options = table[["job_no", "Campaign", "Avg hrs/mo"]].copy()
            job_options = job_options.sort_values("Avg hrs/mo", ascending=False)

            display_cols = ["Campaign", "Job Dept", "Category", "Cross-Dept?"] + month_labels + [
                "Avg hrs/mo",
                "% of Load",
                "tasks",
            ]
            table = table[display_cols]
            table = table.rename(columns={"tasks": "Tasks"})
            table = table.sort_values("Avg hrs/mo", ascending=False)

            totals_month = table[month_labels].sum()
            total_row = {
                "Campaign": "TOTAL",
                "Job Dept": "",
                "Category": "",
                "Cross-Dept?": "",
                **{m: float(totals_month[m]) for m in month_labels},
                "Avg hrs/mo": float(avg_load_pm),
                "% of Load": 100.0,
                "Tasks": "",
            }
            capacity_row = {
                "Campaign": "Capacity",
                "Job Dept": "",
                "Category": "",
                "Cross-Dept?": "",
                **{m: float(capacity_pm) for m in month_labels},
                "Avg hrs/mo": float(capacity_pm),
                "% of Load": "",
                "Tasks": "",
            }
            spare_row = {
                "Campaign": "Spare",
                "Job Dept": "",
                "Category": "",
                "Cross-Dept?": "",
                **{m: float(capacity_pm - totals_month[m]) for m in month_labels},
                "Avg hrs/mo": float(capacity_pm - avg_load_pm),
                "% of Load": "",
                "Tasks": "",
            }

            table_final = pd.concat(
                [table, pd.DataFrame([total_row, capacity_row, spare_row])],
                ignore_index=True,
            )

            numeric_cols = month_labels + ["Avg hrs/mo", "% of Load"]
            for col in numeric_cols:
                table_final[col] = pd.to_numeric(table_final[col], errors="coerce").round(1)

            styled = _style_spare_negative(table_final, label_col="Campaign")
            st.dataframe(styled, use_container_width=True, hide_index=True)
            _download_button(
                f"Download {staff_name} campaign load CSV",
                table_final,
                f"{staff_name}_campaign_load.csv",
                key=f"download_staff_{staff_name}",
            )

            # Task drill-down
            st.markdown("**Task Drill-Down**")
            selected_job_no = st.selectbox(
                "Drill into campaign",
                options=job_options["job_no"].tolist(),
                format_func=lambda x: format_job_label(x, job_name_lookup),
                key=f"drill_{staff_name}",
            )

            df_task = df_scope[
                (df_scope["staff_name"] == staff_name)
                & (df_scope["job_no"] == selected_job_no)
            ].copy()

            if len(df_task) == 0:
                st.info("No tasks found for this campaign in the period.")
            else:
                df_task["task_name"] = df_task["task_name"].fillna("(No Task)")
                task_month = (
                    df_task.groupby(["task_name", "month_key"], dropna=False)["hours_raw"]
                    .sum()
                    .reset_index()
                )
                task_pivot = task_month.pivot_table(
                    index="task_name",
                    columns="month_key",
                    values="hours_raw",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                task_pivot = task_pivot.reindex(columns=months_in_period, fill_value=0.0)
                task_pivot.columns = month_labels
                task_table = task_pivot.reset_index()
                task_table["Total"] = task_table[month_labels].sum(axis=1)
                task_table["Avg hrs/mo"] = task_table["Total"] / num_months

                task_table[month_labels + ["Total", "Avg hrs/mo"]] = task_table[
                    month_labels + ["Total", "Avg hrs/mo"]
                ].round(1)

                st.dataframe(task_table, use_container_width=True, hide_index=True)
                _download_button(
                    f"Download {staff_name} task drill CSV",
                    task_table,
                    f"{staff_name}_{selected_job_no}_task_drill.csv",
                    key=f"download_task_{staff_name}",
                )

    st.divider()

    # =========================
    # Section 4 — Cross-Departmental Flow
    # =========================
    flow_total_hours_period = None
    if selected_staff_home != "All" or selected_job_dept != "All":
        st.subheader("Cross-Departmental Work Flow")

        if selected_staff_home != "All":
            st.markdown(f"**Outbound — Where does {selected_staff_home} staff time go?**")
            outbound = (
                df_scope.groupby("department_final", dropna=False)
                .agg(
                    hours=("hours_raw", "sum"),
                    staff_involved=("staff_name", "nunique"),
                    campaigns=("job_no", "nunique"),
                )
                .reset_index()
            )
            outbound["Hours p/m"] = outbound["hours"] / num_months
            outbound["% of Team Load"] = np.where(
                total_hours_pm > 0,
                outbound["Hours p/m"] / total_hours_pm * 100,
                np.nan,
            )
            outbound["Job Department"] = outbound["department_final"].astype(str)
            if selected_staff_home != "All":
                outbound["Job Department"] = np.where(
                    outbound["department_final"] == selected_staff_home,
                    outbound["Job Department"] + " ✓ (home)",
                    outbound["Job Department"],
                )
            outbound_display = outbound[[
                "Job Department",
                "Hours p/m",
                "% of Team Load",
                "staff_involved",
                "campaigns",
            ]].rename(columns={
                "staff_involved": "Staff Involved",
                "campaigns": "Campaigns",
            })
            flow_total_hours_period = outbound["Hours p/m"].sum() * num_months
            outbound_display["Hours p/m"] = outbound_display["Hours p/m"].round(1)
            outbound_display["% of Team Load"] = outbound_display["% of Team Load"].round(1)
            outbound_display["Staff Involved"] = outbound_display["Staff Involved"].astype(int)
            outbound_display["Campaigns"] = outbound_display["Campaigns"].astype(int)
            outbound_display = outbound_display.sort_values("Hours p/m", ascending=False)

            st.dataframe(outbound_display, use_container_width=True, hide_index=True)
            _download_button(
                "Download outbound flow CSV",
                outbound_display,
                "cross_dept_outbound.csv",
                key="download_outbound",
            )
        if selected_job_dept != "All":
            st.markdown(f"**Inbound — Where do staff on {selected_job_dept} campaigns come from?**")
            inbound = (
                df_scope.groupby("staff_home_dept", dropna=False)
                .agg(
                    hours=("hours_raw", "sum"),
                    staff_count=("staff_name", "nunique"),
                    campaigns=("job_no", "nunique"),
                )
                .reset_index()
            )
            inbound["Hours p/m"] = inbound["hours"] / num_months
            inbound["% of Work"] = np.where(
                total_hours_pm > 0,
                inbound["Hours p/m"] / total_hours_pm * 100,
                np.nan,
            )
            inbound["Staff Home Dept"] = inbound["staff_home_dept"].astype(str)
            if selected_job_dept != "All":
                inbound["Staff Home Dept"] = np.where(
                    inbound["staff_home_dept"] == selected_job_dept,
                    inbound["Staff Home Dept"] + " ✓ (home)",
                    inbound["Staff Home Dept"],
                )
            inbound_display = inbound[[
                "Staff Home Dept",
                "Hours p/m",
                "% of Work",
                "staff_count",
                "campaigns",
            ]].rename(columns={
                "staff_count": "Staff Count",
                "campaigns": "Campaigns",
            })
            if flow_total_hours_period is None:
                flow_total_hours_period = inbound["Hours p/m"].sum() * num_months
            inbound_display["Hours p/m"] = inbound_display["Hours p/m"].round(1)
            inbound_display["% of Work"] = inbound_display["% of Work"].round(1)
            inbound_display["Staff Count"] = inbound_display["Staff Count"].astype(int)
            inbound_display["Campaigns"] = inbound_display["Campaigns"].astype(int)
            inbound_display = inbound_display.sort_values("Hours p/m", ascending=False)

            st.dataframe(inbound_display, use_container_width=True, hide_index=True)
            _download_button(
                "Download inbound flow CSV",
                inbound_display,
                "cross_dept_inbound.csv",
                key="download_inbound",
            )
    st.divider()

    # =========================
    # Section 5 — Campaign Absorption Estimate
    # =========================
    st.subheader("How many more campaigns can we take on?")

    benchmark_input = df_scope[["job_no", "category_display", "month_key", "hours_raw"]].copy()
    category_benchmarks = compute_category_benchmarks(benchmark_input, "category_display")

    if len(category_benchmarks) == 0:
        st.info("No category benchmarks available for the selected period.")
    else:
        total_spare_pm = staff_summary["spare_pm"].clip(lower=0).sum()
        n_staff_with_spare = int((staff_summary["spare_pm"] > 0).sum())

        safe_avg = category_benchmarks["avg_hrs_per_campaign_pm"].replace(0, np.nan)
        safe_median = category_benchmarks["median_hrs_per_campaign_pm"].replace(0, np.nan)
        safe_p75 = category_benchmarks["p75_hrs_per_campaign_pm"].replace(0, np.nan)

        category_benchmarks["est_additional_avg"] = np.floor(total_spare_pm / safe_avg).fillna(0).astype(int)
        category_benchmarks["est_additional_median"] = np.floor(total_spare_pm / safe_median).fillna(0).astype(int)
        category_benchmarks["est_additional_p75"] = np.floor(total_spare_pm / safe_p75).fillna(0).astype(int)

        bench_display = category_benchmarks.rename(columns={
            "category_display": "Category",
            "campaigns_in_sample": "Campaigns in Data",
            "avg_hrs_per_campaign_pm": "Avg hrs/camp/mo",
            "median_hrs_per_campaign_pm": "Median",
            "p25_hrs_per_campaign_pm": "P25 (light)",
            "p75_hrs_per_campaign_pm": "P75 (heavy)",
            "est_additional_avg": "Est. +Campaigns (avg)",
            "est_additional_p75": "Est. +Campaigns (P75 / conservative)",
        })
        bench_display = bench_display[[
            "Category",
            "Campaigns in Data",
            "Avg hrs/camp/mo",
            "Median",
            "P25 (light)",
            "P75 (heavy)",
            "Est. +Campaigns (avg)",
            "Est. +Campaigns (P75 / conservative)",
        ]]
        bench_display["Avg hrs/camp/mo"] = bench_display["Avg hrs/camp/mo"].round(1)
        bench_display["Median"] = bench_display["Median"].round(1)
        bench_display["P25 (light)"] = bench_display["P25 (light)"].round(1)
        bench_display["P75 (heavy)"] = bench_display["P75 (heavy)"].round(1)
        bench_display["Campaigns in Data"] = bench_display["Campaigns in Data"].astype(int)
        bench_display["Est. +Campaigns (avg)"] = bench_display["Est. +Campaigns (avg)"].astype(int)
        bench_display["Est. +Campaigns (P75 / conservative)"] = bench_display["Est. +Campaigns (P75 / conservative)"].astype(int)

        st.dataframe(bench_display, use_container_width=True, hide_index=True)
        _download_button(
            "Download campaign absorption benchmarks CSV",
            bench_display,
            "campaign_absorption_benchmarks.csv",
            key="download_absorption_benchmarks",
        )

        st.info(
            "Estimates show how many additional campaigns the team could absorb if total spare capacity "
            f"({total_spare_pm:.0f} hrs/mo across {n_staff_with_spare} staff with headroom) were fully allocated. "
            "'Avg' basis uses historical average hours per campaign per month. 'P75 / conservative' basis "
            "assumes heavier-than-average campaigns. Actual feasibility depends on individual staff availability "
            "and skill match — check the per-staff breakdown above."
        )

        st.markdown("**Per-Staff Absorption (Avg Basis)**")
        staff_spare = staff_summary[staff_summary["spare_pm"] > 0][["staff_name", "spare_pm"]].copy()
        if len(staff_spare) == 0:
            st.info("No staff with spare capacity in the selected period.")
        else:
            categories = category_benchmarks["category_display"].tolist()
            avg_values = category_benchmarks["avg_hrs_per_campaign_pm"].replace(0, np.nan).values
            spare_values = staff_spare["spare_pm"].values.reshape(-1, 1)
            absorb_matrix = np.floor(spare_values / avg_values.reshape(1, -1))
            absorb_matrix = np.nan_to_num(absorb_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            absorb_df = pd.DataFrame(absorb_matrix.astype(int), columns=[f"+{c} (avg)" for c in categories])
            absorb_df.insert(0, "Staff", staff_spare["staff_name"].values)
            absorb_df.insert(1, "Spare p/m", staff_spare["spare_pm"].round(1).values)

            team_total = {
                "Staff": "Team Total",
                "Spare p/m": total_spare_pm,
            }
            for cat, avg_val in zip(categories, avg_values):
                team_total[f"+{cat} (avg)"] = int(np.floor(total_spare_pm / avg_val)) if avg_val and avg_val > 0 else 0
            absorb_df = pd.concat([absorb_df, pd.DataFrame([team_total])], ignore_index=True)
            absorb_df["Spare p/m"] = pd.to_numeric(absorb_df["Spare p/m"], errors="coerce").round(1)

            st.dataframe(absorb_df, use_container_width=True, hide_index=True)
            _download_button(
                "Download per-staff absorption CSV",
                absorb_df,
                "per_staff_absorption.csv",
                key="download_per_staff_absorption",
            )

    st.divider()

    # =========================
    # Section 6 — Reconciliation Check
    # =========================
    st.subheader("Reconciliation Check")

    total_hours_period = df_scope["hours_raw"].sum()
    staff_total_check = staff_summary["avg_load_pm"].sum() * num_months
    staff_job_month_total = staff_job_month["hours_raw"].sum()

    if flow_total_hours_period is None:
        flow_total_hours_period = df_scope.groupby("department_final")["hours_raw"].sum().sum()

    def _recon_line(label: str, value: float, total: float) -> tuple[str, bool]:
        diff = abs(value - total)
        ok = diff <= 0.1
        status = "✓" if ok else "✗"
        return f"{label}: {value:,.1f} {status}", ok

    lines = []
    ok_flags = []

    lines.append(f"Total Hours in Period (raw data): {total_hours_period:,.1f}")
    line, ok = _recon_line(
        f"  Σ Staff Workload Summary 'Avg Load p/m' × months",
        staff_total_check,
        total_hours_period,
    )
    lines.append(line)
    ok_flags.append(ok)

    line, ok = _recon_line(
        "  Σ Monthly columns across all staff expanders",
        staff_job_month_total,
        total_hours_period,
    )
    lines.append(line)
    ok_flags.append(ok)

    line, ok = _recon_line(
        "  Σ Cross-Dept Flow 'Hours p/m' × months",
        flow_total_hours_period,
        total_hours_period,
    )
    lines.append(line)
    ok_flags.append(ok)

    staff_count_data = int(df_scope.groupby("staff_name")["hours_raw"].sum().gt(0).sum())
    staff_count_summary = int(staff_summary["staff_name"].nunique())
    staff_status = "✓" if staff_count_data == staff_count_summary else "✗"
    lines.append(f"\nStaff count (data):        {staff_count_data}")
    lines.append(f"Staff count (summary):     {staff_count_summary}   {staff_status}")

    campaign_count_data = int(df_scope.groupby("job_no")["hours_raw"].sum().gt(0).sum())
    campaign_count_summary = int(df_scope.groupby("job_no")["hours_raw"].sum().gt(0).sum())
    campaign_status = "✓" if campaign_count_data == campaign_count_summary else "✗"
    lines.append(f"\nCampaign count (data):     {campaign_count_data}")
    lines.append(f"Campaign count (summary):  {campaign_count_summary}   {campaign_status}")

    st.text("\n".join(lines))

    if all(ok_flags) and staff_status == "✓" and campaign_status == "✓":
        st.success("All totals reconcile ✓")
    else:
        st.error("Reconciliation mismatch detected — review totals above.")


if __name__ == "__main__":
    main()
