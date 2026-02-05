"""
UI Components for the Delivery Control Tower.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

from src.ui.formatting import format_job_label
from src.data.job_lifecycle import get_job_task_attribution, get_job_staff_attribution
from src.exports import (
    export_job_pack_excel,
    export_interventions_csv,
    export_plan_markdown,
    export_risk_queue_csv,
)


def render_portfolio_kpi_strip(jobs_df: pd.DataFrame) -> None:
    """
    Render 5-tile KPI strip for portfolio triage.
    """
    st.subheader("Portfolio Health")

    cols = st.columns(5)

    total_jobs = len(jobs_df)
    red_jobs = len(jobs_df[jobs_df["risk_band"] == "Red"])
    amber_jobs = len(jobs_df[jobs_df["risk_band"] == "Amber"])
    green_jobs = len(jobs_df[jobs_df["risk_band"] == "Green"])

    at_risk = jobs_df[jobs_df["risk_band"].isin(["Red", "Amber"])].copy()
    margin_shortfall = (
        (at_risk["median_margin_pct"].fillna(0) - at_risk["forecast_margin_pct"].fillna(0))
        / 100
        * at_risk["forecast_revenue"].fillna(0)
    ).clip(lower=0).sum()

    eac_hours = jobs_df["actual_hours"] + jobs_df["remaining_hours"].fillna(0)
    hours_overrun = (eac_hours - jobs_df["quoted_hours"]).clip(lower=0).sum()

    with cols[0]:
        st.metric(
            "Active Jobs",
            f"{total_jobs}",
            help="Jobs with activity in last 28 days, not completed",
        )

    with cols[1]:
        st.metric(
            "Risk Distribution",
            f"🔴{red_jobs} 🟡{amber_jobs} 🟢{green_jobs}",
            help="Red: score≥70, Amber: 50-70, Green: <50",
        )

    with cols[2]:
        st.metric(
            "Margin at Risk",
            f"${margin_shortfall:,.0f}",
            help="Shortfall vs benchmark margin (Red+Amber jobs)",
        )

    with cols[3]:
        st.metric(
            "Forecast Overrun",
            f"{hours_overrun:,.0f} hrs",
            help="EAC hours - Quoted hours (positive = overrun)",
        )

    with cols[4]:
        avg_risk = jobs_df["risk_score"].mean()
        st.metric(
            "Avg Risk Score",
            f"{avg_risk:.0f}/100",
            help="Portfolio average risk score",
        )


def render_portfolio_risk_table(
    jobs_df: pd.DataFrame,
    job_name_lookup: Dict[str, str],
) -> Optional[str]:
    """
    Render sortable/filterable risk table with row selection.
    Returns selected job_no.
    """
    st.subheader("Risk Queue")

    filter_cols = st.columns([1, 1, 1, 1, 2])

    with filter_cols[0]:
        band_filter = st.multiselect(
            "Risk Band",
            ["Red", "Amber", "Green"],
            default=["Red", "Amber"],
            key="triage_band_filter",
        )

    with filter_cols[1]:
        dept_options = ["All"] + sorted(jobs_df["department_final"].dropna().unique().tolist())
        dept_filter = st.selectbox("Department", dept_options, key="triage_dept_filter")

    with filter_cols[2]:
        cat_options = ["All"] + sorted(jobs_df["job_category"].dropna().unique().tolist())
        cat_filter = st.selectbox("Category", cat_options, key="triage_cat_filter")

    with filter_cols[3]:
        driver_options = ["All"] + sorted(jobs_df["primary_driver"].dropna().unique().tolist())
        driver_filter = st.selectbox("Primary Driver", driver_options, key="triage_driver_filter")

    with filter_cols[4]:
        sort_by = st.selectbox(
            "Sort By",
            ["Risk Score ↓", "Margin Shortfall ↓", "Hours Overrun ↓", "Quote Consumed ↓"],
            key="triage_sort",
        )

    display_df = jobs_df.copy()
    if band_filter:
        display_df = display_df[display_df["risk_band"].isin(band_filter)]
    if dept_filter != "All":
        display_df = display_df[display_df["department_final"] == dept_filter]
    if cat_filter != "All":
        display_df = display_df[display_df["job_category"] == cat_filter]
    if driver_filter != "All":
        display_df = display_df[display_df["primary_driver"] == driver_filter]

    display_df["pct_consumed"] = display_df.get(
        "pct_consumed", display_df.get("pct_quote_consumed")
    )

    display_df["margin_shortfall"] = (
        (display_df["median_margin_pct"].fillna(0) - display_df["forecast_margin_pct"].fillna(0))
        / 100
        * display_df["forecast_revenue"].fillna(0)
    ).clip(lower=0)

    display_df["hours_overrun"] = (
        display_df["actual_hours"] + display_df["remaining_hours"].fillna(0) - display_df["quoted_hours"]
    ).clip(lower=0)

    sort_map = {
        "Risk Score ↓": ("risk_score", False),
        "Margin Shortfall ↓": ("margin_shortfall", False),
        "Hours Overrun ↓": ("hours_overrun", False),
        "Quote Consumed ↓": ("pct_consumed", False),
    }
    sort_col, sort_asc = sort_map.get(sort_by, ("risk_score", False))
    if sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=sort_asc)

    display_df["job_label"] = display_df["job_no"].apply(
        lambda value: format_job_label(value, job_name_lookup)
    )

    table_cols = [
        "job_label",
        "risk_band",
        "risk_score",
        "pct_consumed",
        "quoted_hours",
        "actual_hours",
        "remaining_hours",
        "hours_overrun",
        "forecast_margin_pct",
        "median_margin_pct",
        "margin_shortfall",
        "primary_driver",
        "recommended_action",
    ]
    table_cols = [col for col in table_cols if col in display_df.columns]

    selection = st.dataframe(
        display_df[table_cols],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "job_label": st.column_config.TextColumn("Job", width="large"),
            "risk_band": st.column_config.TextColumn("Band"),
            "risk_score": st.column_config.NumberColumn("Risk", format="%.0f"),
            "pct_consumed": st.column_config.NumberColumn("% Consumed", format="%.0f%%"),
            "quoted_hours": st.column_config.NumberColumn("Quoted", format="%.0f"),
            "actual_hours": st.column_config.NumberColumn("Actual", format="%.0f"),
            "remaining_hours": st.column_config.NumberColumn("Remaining", format="%.0f"),
            "hours_overrun": st.column_config.NumberColumn("Overrun", format="%.0f"),
            "forecast_margin_pct": st.column_config.NumberColumn("Fcst Margin", format="%.1f%%"),
            "median_margin_pct": st.column_config.NumberColumn("Bench Margin", format="%.1f%%"),
            "margin_shortfall": st.column_config.NumberColumn("$ at Risk", format="$%.0f"),
        },
        key="risk_queue_table",
    )

    if selection and selection.selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        selected_job = display_df.iloc[selected_idx]["job_no"]
        st.session_state.selected_job = selected_job
        return selected_job

    csv_bytes, filename = export_risk_queue_csv(display_df[table_cols])
    st.download_button(
        "📥 Export Risk Queue (CSV)",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )

    return None


def render_risk_map(jobs_df: pd.DataFrame, job_name_lookup: Dict[str, str]) -> Optional[str]:
    """
    Render scatter plot for visual triage.
    X: % quote consumed, Y: margin shortfall, Color: risk band
    """
    st.subheader("Risk Map")

    plot_df = jobs_df.copy()
    plot_df["pct_consumed"] = plot_df.get("pct_consumed", plot_df.get("pct_quote_consumed"))
    plot_df["margin_shortfall"] = (
        (plot_df["median_margin_pct"].fillna(0) - plot_df["forecast_margin_pct"].fillna(0))
        / 100
        * plot_df["forecast_revenue"].fillna(0)
    ).clip(lower=0)
    plot_df["job_label"] = plot_df["job_no"].apply(
        lambda value: format_job_label(value, job_name_lookup)
    )

    color_map = {"Red": "#dc3545", "Amber": "#ffc107", "Green": "#28a745"}

    fig = px.scatter(
        plot_df,
        x="pct_consumed",
        y="margin_shortfall",
        color="risk_band",
        color_discrete_map=color_map,
        size="actual_hours",
        hover_name="job_label",
        hover_data={
            "risk_score": ":.0f",
            "primary_driver": True,
            "pct_consumed": ":.0f%%",
            "margin_shortfall": ":$,.0f",
        },
        title="Jobs by Quote Consumption vs Margin Shortfall",
    )

    fig.add_vline(x=100, line_dash="dash", line_color="#999", annotation_text="100% consumed")
    fig.add_hline(
        y=plot_df["margin_shortfall"].median(),
        line_dash="dot",
        line_color="#999",
    )

    fig.update_layout(
        xaxis_title="% Quote Consumed",
        yaxis_title="Margin Shortfall ($)",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True, key="risk_map")

    return None


def render_weekly_focus(jobs_df: pd.DataFrame, job_name_lookup: Dict[str, str]) -> None:
    """
    Auto-generate "This Week's Focus" with top priorities.
    """
    st.subheader("📌 This Week's Focus")

    at_risk = jobs_df[jobs_df["risk_band"].isin(["Red", "Amber"])].copy()

    if len(at_risk) == 0:
        st.success("✅ No high-risk jobs requiring immediate attention")
        return

    at_risk["margin_shortfall"] = (
        (at_risk["median_margin_pct"].fillna(0) - at_risk["forecast_margin_pct"].fillna(0))
        / 100
        * at_risk["forecast_revenue"].fillna(0)
    ).clip(lower=0)
    at_risk["hours_overrun"] = (
        at_risk["actual_hours"] + at_risk["remaining_hours"].fillna(0) - at_risk["quoted_hours"]
    ).clip(lower=0)
    at_risk["quick_win_score"] = at_risk["risk_score"] / at_risk["remaining_hours"].clip(lower=1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔴 Top by Margin at Risk**")
        top_margin = at_risk.nlargest(3, "margin_shortfall")
        for _, row in top_margin.iterrows():
            job_label = format_job_label(row["job_no"], job_name_lookup)
            st.markdown(f"- **{job_label}** — ${row['margin_shortfall']:,.0f} at risk")
            st.caption(f"  → {row.get('recommended_action', 'Review')}")

    with col2:
        st.markdown("**⏱️ Top by Hours Overrun**")
        top_hours = at_risk.nlargest(3, "hours_overrun")
        for _, row in top_hours.iterrows():
            job_label = format_job_label(row["job_no"], job_name_lookup)
            st.markdown(f"- **{job_label}** — {row['hours_overrun']:.0f} hrs over")
            st.caption(f"  → {row.get('primary_driver', 'Review')}")

    with col3:
        st.markdown("**⚡ Quick Wins** (high risk, low remaining)")
        quick_wins = at_risk[at_risk["remaining_hours"] < 20].nlargest(3, "quick_win_score")
        for _, row in quick_wins.iterrows():
            job_label = format_job_label(row["job_no"], job_name_lookup)
            st.markdown(f"- **{job_label}** — {row['remaining_hours']:.0f} hrs left")
            st.caption("  → Urgent close-out review")


def render_job_health_card(job_row: pd.Series, job_name_lookup: Dict[str, str]) -> None:
    """
    Render single job health card with status summary.
    """
    job_label = format_job_label(job_row["job_no"], job_name_lookup)
    risk_band = job_row.get("risk_band", "Unknown")
    risk_score = job_row.get("risk_score", 0)

    status_map = {"Red": "🔴 At Risk", "Amber": "🟡 Watch", "Green": "🟢 On Track"}
    status = status_map.get(risk_band, "⚪ Unknown")

    st.subheader(f"Job Health: {job_label}")
    st.markdown(f"**Status:** {status} | **Risk Score:** {risk_score:.0f}/100")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pct_consumed = job_row.get("pct_consumed", 0)
        st.metric("Quote Consumed", f"{pct_consumed:.0f}%")

    with col2:
        hours_var = job_row.get("actual_hours", 0) - job_row.get("quoted_hours", 0)
        st.metric("Hours Variance", f"{hours_var:+,.0f}")

    with col3:
        scope_creep = job_row.get("scope_creep_pct", 0)
        st.metric("Scope Creep", f"{scope_creep:.0f}%")

    with col4:
        rate_var = job_row.get("rate_variance", 0)
        st.metric("Rate Variance", f"${rate_var:+.0f}/hr")

    st.markdown("**Margin Forecast vs Benchmark**")
    fcst = job_row.get("forecast_margin_pct", 0)
    bench = job_row.get("median_margin_pct", 0)
    delta = fcst - bench
    progress = min(max(fcst / 100, 0), 1) if pd.notna(fcst) else 0
    st.progress(progress)
    st.caption(f"Forecast: {fcst:.1f}% | Benchmark: {bench:.1f}% | Delta: {delta:+.1f}pp")


def render_root_cause_drivers(drivers: List[Dict]) -> None:
    """
    Render ranked root cause drivers.
    """
    st.subheader("🔍 Root Cause Drivers")

    if not drivers:
        st.success("✅ No significant risk drivers identified")
        return

    for i, driver in enumerate(drivers, 1):
        with st.expander(f"#{i} {driver['driver_name']} — Score: {driver['score']:.0f}", expanded=(i == 1)):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"**Metric:** {driver['evidence_metric']}")
                st.markdown(f"**Value:** {driver['evidence_value']}")
                st.markdown(f"**Benchmark:** {driver['benchmark_value']}")

            with col2:
                st.markdown(f"**Evidence:** {driver['evidence_detail']}")
                st.markdown(f"**Recommendation:** {driver['recommendation']}")


def render_task_breakdown(df: pd.DataFrame, job_no: str) -> pd.DataFrame:
    """
    Render task-level breakdown with chart.
    Returns task DataFrame used for display.
    """
    st.subheader("📋 Task Breakdown")

    task_df = get_job_task_attribution(df, job_no)

    if len(task_df) == 0:
        st.info("No task data available")
        return task_df

    task_df = task_df.sort_values("variance", ascending=False)
    task_df["variance_pct"] = np.where(
        task_df["quoted_hours"] > 0,
        task_df["variance"] / task_df["quoted_hours"] * 100,
        np.nan,
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Quoted",
        y=task_df["task_name"],
        x=task_df["quoted_hours"],
        orientation="h",
        marker_color="#6c757d",
    ))

    fig.add_trace(go.Bar(
        name="Actual",
        y=task_df["task_name"],
        x=task_df["actual_hours"],
        orientation="h",
        marker_color=["#dc3545" if value > 0 else "#28a745" for value in task_df["variance"]],
    ))

    fig.update_layout(
        barmode="group",
        title="Quoted vs Actual Hours by Task",
        xaxis_title="Hours",
        yaxis_title="",
        height=max(300, len(task_df) * 35),
        legend=dict(orientation="h", y=1.1),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        task_df[["task_name", "quoted_hours", "actual_hours", "variance", "variance_pct"]].rename(
            columns={
                "task_name": "Task",
                "quoted_hours": "Quoted",
                "actual_hours": "Actual",
                "variance": "Variance",
                "variance_pct": "Var %",
            }
        ),
        column_config={
            "Quoted": st.column_config.NumberColumn(format="%.0f"),
            "Actual": st.column_config.NumberColumn(format="%.0f"),
            "Variance": st.column_config.NumberColumn(format="%+.0f"),
            "Var %": st.column_config.NumberColumn(format="%+.0f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )

    return task_df


def build_staff_attribution_df(
    df: pd.DataFrame,
    job_no: str,
    all_jobs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build staff attribution with cross-job risk signals.
    """
    staff_df = get_job_staff_attribution(df, job_no)
    if len(staff_df) == 0:
        return staff_df

    job_df = df[df["job_no"] == job_no].copy()
    if "work_date" in job_df.columns:
        cutoff = job_df["work_date"].max() - pd.Timedelta(days=28)
        recent = job_df[job_df["work_date"] >= cutoff]
        recent_staff = recent.groupby("staff_name")["hours_raw"].sum().reset_index()
        recent_staff.columns = ["staff_name", "hours_last_28d"]
        staff_df = staff_df.merge(recent_staff, on="staff_name", how="left")
    else:
        staff_df["hours_last_28d"] = staff_df["hours"]

    total_hours = staff_df["hours"].sum()
    staff_df["share_pct"] = staff_df["hours"] / total_hours * 100 if total_hours > 0 else 0

    if "risk_band" in all_jobs_df.columns:
        red_jobs = all_jobs_df[all_jobs_df["risk_band"] == "Red"]["job_no"].tolist()
        red_jobs = [job for job in red_jobs if job != job_no]

        if red_jobs:
            red_df = df[df["job_no"].isin(red_jobs)]
            red_staff = red_df.groupby("staff_name")["hours_raw"].sum().reset_index()
            red_staff.columns = ["staff_name", "hours_on_other_red"]
            staff_df = staff_df.merge(red_staff, on="staff_name", how="left")
            staff_df["hours_on_other_red"] = staff_df["hours_on_other_red"].fillna(0)
        else:
            staff_df["hours_on_other_red"] = 0
    else:
        staff_df["hours_on_other_red"] = 0

    return staff_df


def render_staff_attribution(
    df: pd.DataFrame,
    job_no: str,
    all_jobs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Render staff attribution with cross-job risk check.
    Returns staff DataFrame used for display.
    """
    st.subheader("👥 Staff Attribution")

    staff_df = build_staff_attribution_df(df, job_no, all_jobs_df)

    if len(staff_df) == 0:
        st.info("No staff data available")
        return staff_df

    st.dataframe(
        staff_df[["staff_name", "hours", "hours_last_28d", "share_pct", "hours_on_other_red"]].rename(
            columns={
                "staff_name": "Staff",
                "hours": "Total Hours",
                "hours_last_28d": "Last 28d",
                "share_pct": "Share %",
                "hours_on_other_red": "Hrs on Other Red Jobs",
            }
        ),
        column_config={
            "Total Hours": st.column_config.NumberColumn(format="%.0f"),
            "Last 28d": st.column_config.NumberColumn(format="%.0f"),
            "Share %": st.column_config.NumberColumn(format="%.0f%%"),
            "Hrs on Other Red Jobs": st.column_config.NumberColumn(format="%.0f"),
        },
        use_container_width=True,
        hide_index=True,
    )

    overloaded = staff_df[staff_df["hours_on_other_red"] > 20]
    if len(overloaded) > 0:
        st.warning(
            f"⚠️ {len(overloaded)} staff members are also on other Red jobs — potential resource contention"
        )

    return staff_df


def render_intervention_builder(job_no: str, drivers: List[Dict]) -> List[Dict]:
    """
    Render intervention builder form.
    Returns list of intervention dicts.
    """
    st.subheader("🛠️ Intervention Builder")

    if "interventions" not in st.session_state:
        st.session_state.interventions = {}
    if job_no not in st.session_state.interventions:
        st.session_state.interventions[job_no] = []

    if st.button("🔄 Auto-Generate from Drivers", key="auto_gen_interventions"):
        auto_interventions = []
        for driver in drivers:
            auto_interventions.append({
                "job_no": job_no,
                "intervention_type": _map_driver_to_type(driver["driver_name"]),
                "target": driver["driver_name"],
                "description": driver["recommendation"],
                "expected_impact": f"Address {driver['evidence_metric']}",
                "owner": "",
                "due_date": None,
                "status": "Proposed",
            })
        st.session_state.interventions[job_no] = auto_interventions
        st.rerun()

    with st.expander("➕ Add Intervention Manually", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            int_type = st.selectbox(
                "Type",
                ["Re-scope", "Re-assign", "Rate Correction", "Timeline", "Other"],
                key="new_int_type",
            )
            target = st.text_input("Target (task/staff/scope)", key="new_int_target")
            description = st.text_area("Description", key="new_int_desc")

        with col2:
            expected_impact = st.text_input("Expected Impact", key="new_int_impact")
            owner = st.text_input("Owner", key="new_int_owner")
            due_date = st.date_input("Due Date", key="new_int_due")

        if st.button("Add Intervention", key="add_int_btn"):
            new_int = {
                "job_no": job_no,
                "intervention_type": int_type,
                "target": target,
                "description": description,
                "expected_impact": expected_impact,
                "owner": owner,
                "due_date": due_date,
                "status": "Proposed",
            }
            st.session_state.interventions[job_no].append(new_int)
            st.rerun()

    interventions = st.session_state.interventions.get(job_no, [])

    if interventions:
        st.markdown(f"**{len(interventions)} Interventions Defined**")

        for i, intv in enumerate(interventions):
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

                with col1:
                    st.markdown(f"**{intv['intervention_type']}:** {intv['target']}")
                    st.caption(intv["description"])

                with col2:
                    st.markdown(f"**Impact:** {intv['expected_impact']}")
                    st.caption(f"Owner: {intv['owner'] or 'TBD'}")

                with col3:
                    new_status = st.selectbox(
                        "Status",
                        ["Proposed", "Approved", "In Progress", "Done"],
                        index=["Proposed", "Approved", "In Progress", "Done"].index(intv["status"]),
                        key=f"int_status_{i}",
                    )
                    interventions[i]["status"] = new_status

                with col4:
                    if st.button("🗑️", key=f"del_int_{i}"):
                        interventions.pop(i)
                        st.session_state.interventions[job_no] = interventions
                        st.rerun()

    return interventions


def _map_driver_to_type(driver_name: str) -> str:
    """Map driver name to intervention type."""
    mapping = {
        "Scope Creep": "Re-scope",
        "Under-Quoting": "Re-scope",
        "Rate Leakage": "Rate Correction",
        "Runtime Drift": "Timeline",
    }
    return mapping.get(driver_name, "Other")


def render_next_7_days_plan(
    job_no: str,
    job_row: pd.Series,
    drivers: List[Dict],
    interventions: List[Dict],
) -> str:
    """
    Generate and render "Next 7 Days Plan".
    Returns markdown text for export.
    """
    st.subheader("📅 Next 7 Days Plan")

    plan_items = []
    plan_items.append({
        "day": "Day 1",
        "action": "Status review with PM",
        "tied_to": "Standard",
        "owner": "Delivery Lead",
    })

    for i, driver in enumerate(drivers[:3]):
        plan_items.append({
            "day": f"Day {2 + i}",
            "action": driver["recommendation"],
            "tied_to": driver["driver_name"],
            "owner": "TBD",
        })

    proposed = [item for item in interventions if item["status"] == "Proposed"]
    if proposed:
        plan_items.append({
            "day": "Day 5",
            "action": f"Review {len(proposed)} proposed interventions with stakeholders",
            "tied_to": "Intervention Review",
            "owner": "Delivery Lead",
        })

    plan_items.append({
        "day": "Day 6",
        "action": "Update EAC forecast and margin projection",
        "tied_to": "Forecasting",
        "owner": "Finance",
    })

    plan_items.append({
        "day": "Day 7",
        "action": "Communicate status to client if needed",
        "tied_to": "Client Management",
        "owner": "Account Lead",
    })

    for item in plan_items:
        st.checkbox(
            f"**{item['day']}:** {item['action']} *(Owner: {item['owner']})*",
            key=f"plan_{job_no}_{item['day']}",
        )

    md_lines = [
        f"# Next 7 Days Plan: {job_no}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Checklist",
    ]
    for item in plan_items:
        md_lines.append(
            f"- [ ] **{item['day']}:** {item['action']} — Owner: {item['owner']}"
        )

    return "\n".join(md_lines)


def render_export_section(
    job_no: str,
    job_row: pd.Series,
    task_df: pd.DataFrame,
    staff_df: pd.DataFrame,
    drivers: List[Dict],
    interventions: List[Dict],
    plan_md: str,
) -> None:
    """
    Render export buttons and Slack-ready summary.
    """
    st.subheader("📤 Export & Handoff")

    col1, col2, col3 = st.columns(3)

    with col1:
        job_pack_bytes, job_pack_name = export_job_pack_excel(
            pd.DataFrame([job_row]),
            task_df if task_df is not None else pd.DataFrame(),
            staff_df if staff_df is not None else pd.DataFrame(),
            job_no=job_no,
        )
        st.download_button(
            "📦 Export Job Pack (Excel)",
            data=job_pack_bytes,
            file_name=job_pack_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col2:
        if interventions:
            int_bytes, int_name = export_interventions_csv(interventions, job_no=job_no)
            st.download_button(
                "📋 Export Interventions (CSV)",
                data=int_bytes,
                file_name=int_name,
                mime="text/csv",
            )
        else:
            st.button("📋 Export Interventions", disabled=True, help="No interventions defined")

    with col3:
        plan_bytes, plan_name = export_plan_markdown(plan_md, job_no=job_no)
        st.download_button(
            "📅 Export 7-Day Plan (MD)",
            data=plan_bytes,
            file_name=plan_name,
            mime="text/markdown",
        )

    st.markdown("**Slack-Ready Summary**")

    margin_shortfall = (
        (job_row.get("median_margin_pct", 0) - job_row.get("forecast_margin_pct", 0))
        / 100
        * job_row.get("forecast_revenue", 0)
    )

    slack_summary = (
        f"🚨 *Job Alert: {job_no}*\n"
        f"• Risk Score: {job_row.get('risk_score', 0):.0f}/100 ({job_row.get('risk_band', 'Unknown')})\n"
        f"• Quote Consumed: {job_row.get('pct_consumed', 0):.0f}%\n"
        f"• Margin Shortfall: ${margin_shortfall:,.0f}\n"
        f"• Top Driver: {drivers[0]['driver_name'] if drivers else 'None'}\n"
        f"• Recommended: {drivers[0]['recommendation'] if drivers else 'Monitor'}"
    )

    st.text_area("Copy/Paste for Slack:", slack_summary, height=150, key="slack_summary")
