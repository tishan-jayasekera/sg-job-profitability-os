"""
Time Allocation Page

Descriptive view of how time is spent (no targets).
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.metrics.time_allocation import (
    compute_allocation_breakdown,
    compute_nonbillable_detail,
    compute_hhi,
    compute_crowdout_flags,
    compute_team_allocation,
)
from src.data.profiles import build_staff_profiles
from src.ui.formatting import fmt_hours, fmt_percent
from src.config import config


st.set_page_config(page_title="Time Allocation", page_icon="T", layout="wide")


def _stacked_allocation_bar(total_billable: float, breakdown: dict) -> go.Figure:
    labels = ["Billable"] + list(breakdown.keys())
    values = [total_billable] + list(breakdown.values())
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=["Allocation"],
        orientation="h",
        text=[f"{v:.0f}" for v in values],
        textposition="inside",
    ))
    fig.update_layout(
        barmode="stack",
        xaxis_title="Hours",
        yaxis_title="",
        showlegend=False,
        height=220,
    )
    return fig


def main():
    st.title("Time Allocation")
    st.caption("Descriptive view of how time was actually spent (no targets).")
    
    df = load_fact_timesheet()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    departments = ["All"] + sorted(df["department_final"].dropna().unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)
    window = st.sidebar.selectbox("Window", options=[2, 4, 8, 12], index=1, format_func=lambda x: f"{x}w")
    
    if selected_dept != "All":
        df = df[df["department_final"] == selected_dept]
    
    # Filter to window
    date_col = "work_date" if "work_date" in df.columns else "month_key"
    if date_col in df.columns:
        ref = pd.to_datetime(df[date_col]).max()
        cutoff = ref - pd.DateOffset(weeks=window)
        df = df[pd.to_datetime(df[date_col]) >= cutoff]

    if "is_billable" not in df.columns:
        df["is_billable"] = False
    
    allocation = compute_allocation_breakdown(df, group_by="staff_name")
    if len(allocation) == 0:
        st.warning("No allocation data available for this selection.")
        return
    
    # Summary KPIs
    st.subheader("Summary")
    total_hours = allocation["total_hours"].sum()
    billable_hours = allocation["billable_hours"].sum()
    nonbillable_hours = allocation["nonbillable_hours"].sum()
    billable_ratio = billable_hours / total_hours if total_hours > 0 else 0
    
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Total Hours", fmt_hours(total_hours))
    with kpi_cols[1]:
        st.metric("Billable Hours", fmt_hours(billable_hours))
    with kpi_cols[2]:
        st.metric("Non-Billable Hours", fmt_hours(nonbillable_hours))
    with kpi_cols[3]:
        st.metric("Billable Ratio", fmt_percent(billable_ratio * 100))
    
    st.info("This is descriptive - how time was actually spent. There is no target.")
    st.caption("Billable Ratio = billable hours ÷ total hours (actual). This is not capacity-based utilisation.")
    
    st.divider()
    
    # Allocation breakdown
    st.subheader("Allocation Breakdown")
    breakdown_total = {}
    for breakdown in allocation["nonbillable_by_breakdown"]:
        for key, value in breakdown.items():
            breakdown_total[key] = breakdown_total.get(key, 0) + value
    
    if breakdown_total:
        fig = _stacked_allocation_bar(billable_hours, breakdown_total)
        st.plotly_chart(fig, use_container_width=True)
        
        breakdown_options = ["All"] + sorted(breakdown_total.keys())
        selected_breakdown = st.selectbox("Non-billable breakdown detail", breakdown_options)
        if selected_breakdown != "All" and "breakdown" in df.columns:
            df_nb = df[(~df["is_billable"]) & (df["breakdown"] == selected_breakdown)]
            detail = df_nb.groupby("staff_name")["hours_raw"].sum().reset_index().rename(columns={"hours_raw": "hours"})
            st.dataframe(detail.sort_values("hours", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No non-billable breakdown data available.")
    
    st.divider()
    
    # Staff allocation table
    st.subheader("Staff Allocation Table")
    hhi = compute_hhi(df, group_col="task_name")
    profiles = build_staff_profiles(df, window, config.PROFILE_TRAINING_MONTHS)
    staff_table = allocation.merge(hhi, on="staff_name", how="left")
    staff_table = staff_table.merge(profiles[["staff_name", "archetype"]], on="staff_name", how="left")
    
    staff_table["billable_pct"] = staff_table["billable_ratio"] * 100
    staff_table["top_nonbillable"] = staff_table["nonbillable_by_breakdown"].apply(
        lambda d: max(d, key=d.get) if isinstance(d, dict) and d else "-"
    )
    
    display = staff_table.rename(columns={
        "staff_name": "Name",
        "total_hours": "Total",
        "billable_hours": "Billable",
        "billable_pct": "Bill%",
        "top_nonbillable": "Top Non-Bill",
        "hhi": "HHI",
        "archetype": "Archetype",
    })[["Name", "Total", "Billable", "Bill%", "Top Non-Bill", "HHI", "Archetype"]]
    
    selection = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "HHI": st.column_config.NumberColumn(
                "HHI",
                help="HHI = sum of squared task shares (1.0 concentrated, 0.0 fragmented)"
            )
        },
        key="allocation_table",
    )
    
    selected_staff = None
    if selection and selection.selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        selected_staff = staff_table.iloc[selected_idx]["staff_name"]
    
    if selected_staff:
        with st.expander(f"Non-billable breakdown: {selected_staff}", expanded=True):
            detail = compute_nonbillable_detail(df, selected_staff)
            if len(detail) > 0:
                st.dataframe(detail, use_container_width=True, hide_index=True)
            else:
                st.info("No non-billable detail available.")
    
    st.divider()
    
    # Crowd-out flags
    st.subheader("Crowd-Out Flags")
    flags = compute_crowdout_flags(
        staff_table,
        admin_threshold=config.CROWDOUT_ADMIN_THRESHOLD,
        internal_threshold=config.CROWDOUT_INTERNAL_THRESHOLD,
        unassigned_threshold=config.CROWDOUT_UNASSIGNED_THRESHOLD,
    )
    if flags:
        for flag in flags:
            st.warning(f"{flag['staff_name']}: {flag['detail']}")
    else:
        st.success("No crowd-out patterns detected.")
    
    st.divider()
    
    # Team comparison
    team_alloc = compute_team_allocation(df, group_by="department_final")
    if len(team_alloc) > 1:
        st.subheader("Team Comparison")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=team_alloc["department_final"],
            y=team_alloc["billable_ratio"] * 100,
            text=[f"{v:.0f}%" for v in team_alloc["billable_ratio"] * 100],
            textposition="outside",
        ))
        fig.update_layout(yaxis_title="Billable Ratio (%)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            team_alloc.rename(columns={
                "department_final": "Department",
                "total_hours": "Hours",
                "billable_ratio": "Billable Ratio",
                "staff_count": "Staff",
            })[["Department", "Hours", "Billable Ratio", "Staff"]],
            use_container_width=True,
            hide_index=True,
        )
    
    st.divider()
    
    st.subheader("Export")
    st.download_button(
        "Download Allocation CSV",
        data=staff_table.to_csv(index=False),
        file_name="time_allocation.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
