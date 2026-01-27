"""
Executive Summary Page

Company → Department → Category → Staff → Breakdown → Task drill-down with KPIs,
quote-delivery reconciliation, and rate capture analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import (
    init_state, get_state, get_drill_level,
    drill_to_department, drill_to_category, drill_to_staff,
    drill_to_breakdown, drill_to_task, get_drill_filter,
    apply_drill_filter
)
from src.ui.layout import (
    render_breadcrumb, render_filter_chips, render_sidebar_filters,
    render_kpi_strip, section_header
)
from src.ui.formatting import (
    fmt_currency, fmt_hours, fmt_percent, fmt_rate,
    format_metric_df
)
from src.ui.charts import (
    histogram, rate_scatter, horizontal_bar
)
from src.data.loader import load_fact_timesheet
from src.data.semantic import (
    full_metric_pack, profitability_rollup, quote_delivery_metrics,
    rate_rollups, utilisation_metrics, exclude_leave, get_category_col
)
from src.data.cohorts import filter_by_time_window, filter_active_jobs


st.set_page_config(page_title="Executive Summary", page_icon="📈", layout="wide")

init_state()


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all global filters to dataframe."""
    # Time window
    time_window = get_state("time_window")
    df = filter_by_time_window(df, time_window)
    
    # Active jobs only
    if get_state("active_jobs_only"):
        df = filter_active_jobs(df)
    
    # Exclude leave (for utilisation calculations - optional here)
    # Note: We don't exclude leave from revenue/cost calculations
    
    # Non-billable
    if not get_state("include_nonbillable"):
        if "is_billable" in df.columns:
            df = df[df["is_billable"]]
    
    # Client filter
    client = get_state("selected_client")
    if client and "client" in df.columns:
        df = df[df["client"] == client]
    
    # Status filter
    status = get_state("selected_status")
    if status and "job_status" in df.columns:
        df = df[df["job_status"] == status]
    
    return df




def main():
    st.title("Executive Summary")
    
    # Load data
    df = load_fact_timesheet()
    
    # Sidebar filters
    render_sidebar_filters(df)
    
    # Apply filters
    df_filtered = apply_global_filters(df)
    
    # Apply drill filter
    drill_filter = get_drill_filter()
    df_drill = apply_drill_filter(df_filtered, drill_filter)
    
    # Breadcrumb and filter chips
    render_breadcrumb()
    render_filter_chips()
    
    if len(df_drill) == 0:
        st.warning("No data matches current filters.")
        return
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION A: KPI STRIP
    # =========================================================================
    section_header("Key Metrics")
    
    # Calculate metrics
    prof = profitability_rollup(df_drill)
    util = utilisation_metrics(df_drill, exclude_leave=get_state("exclude_leave"))
    
    kpi_cols = st.columns(7)
    
    with kpi_cols[0]:
        st.metric("Revenue", fmt_currency(prof["revenue"].iloc[0]))
    
    with kpi_cols[1]:
        st.metric("Cost", fmt_currency(prof["cost"].iloc[0]))
    
    with kpi_cols[2]:
        st.metric("Margin", fmt_currency(prof["margin"].iloc[0]))
    
    with kpi_cols[3]:
        st.metric("Margin %", fmt_percent(prof["margin_pct"].iloc[0]))
    
    with kpi_cols[4]:
        st.metric("Hours", fmt_hours(prof["hours"].iloc[0]))
    
    with kpi_cols[5]:
        st.metric("Realised Rate", fmt_rate(prof["realised_rate"].iloc[0]))
    
    with kpi_cols[6]:
        st.metric("Billable Share", fmt_percent(util["utilisation"].iloc[0]))

    st.caption("Billable Share = billable hours ÷ total hours (excl. leave). This is not capacity-based utilisation.")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION B: RECONCILIATION PANELS
    # =========================================================================
    col1, col2 = st.columns(2)
    
    # -------------------------------------------------------------------------
    # B1: Quote → Delivery
    # -------------------------------------------------------------------------
    with col1:
        section_header("Quote → Delivery", "Hours variance and scope creep analysis")
        
        delivery = quote_delivery_metrics(df_drill)
        
        # KPIs
        m1, m2, m3 = st.columns(3)
        
        with m1:
            quoted = delivery["quoted_hours"].iloc[0]
            actual = delivery["actual_hours"].iloc[0]
            variance_pct = delivery["hours_variance_pct"].iloc[0] if not pd.isna(delivery["hours_variance_pct"].iloc[0]) else 0
            st.metric("Hours Variance", fmt_percent(variance_pct))
        
        with m2:
            scope_creep = delivery["unquoted_share"].iloc[0] if not pd.isna(delivery["unquoted_share"].iloc[0]) else 0
            st.metric("Scope Creep", fmt_percent(scope_creep))
        
        with m3:
            # Calculate overrun rate at job-task level
            if "job_no" in df_drill.columns:
                from src.data.semantic import safe_quote_job_task
                jt = safe_quote_job_task(df_drill)
                if len(jt) > 0 and "quoted_time_total" in jt.columns:
                    actuals = df_drill.groupby(["job_no", "task_name"])["hours_raw"].sum().reset_index()
                    jt = jt.merge(actuals, on=["job_no", "task_name"], how="left")
                    jt["is_overrun"] = jt["hours_raw"] > jt["quoted_time_total"]
                    overrun_rate = jt["is_overrun"].mean() * 100
                else:
                    overrun_rate = 0
            else:
                overrun_rate = 0
            st.metric("Overrun Rate", fmt_percent(overrun_rate))
        
        # Variance distribution (if we have job-task data)
        if "job_no" in df_drill.columns and "quoted_time_total" in df_drill.columns:
            jt_variance = df_drill.groupby(["job_no", "task_name"]).agg(
                quoted=("quoted_time_total", "first"),
                actual=("hours_raw", "sum")
            ).reset_index()
            jt_variance["variance_pct"] = np.where(
                jt_variance["quoted"] > 0,
                (jt_variance["actual"] - jt_variance["quoted"]) / jt_variance["quoted"] * 100,
                np.nan
            )
            jt_variance = jt_variance.dropna(subset=["variance_pct"])
            
            if len(jt_variance) > 10:
                fig = histogram(
                    jt_variance[jt_variance["variance_pct"].between(-100, 200)],
                    x="variance_pct",
                    title="Hours Variance Distribution (%)",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------------------------------------------------------
    # B2: Quote Rate → Realised Rate
    # -------------------------------------------------------------------------
    with col2:
        section_header("Rate Capture", "Quote rate vs realised rate analysis")
        
        rates = rate_rollups(df_drill)
        
        # KPIs
        m1, m2, m3 = st.columns(3)
        
        with m1:
            qr = rates["quote_rate"].iloc[0]
            st.metric("Quote Rate", fmt_rate(qr) if not pd.isna(qr) else "—")
        
        with m2:
            rr = rates["realised_rate"].iloc[0]
            st.metric("Realised Rate", fmt_rate(rr) if not pd.isna(rr) else "—")
        
        with m3:
            rv = rates["rate_variance"].iloc[0]
            st.metric("Rate Variance", fmt_rate(rv) if not pd.isna(rv) else "—")
        
        # Rate scatter by drill level
        level = get_drill_level()
        
        category_col = get_category_col(df_drill)
        if level == "company":
            group_col = "department_final"
        elif level == "department":
            group_col = category_col
        elif level == "category":
            group_col = "staff_name"
        else:
            group_col = "task_name"
        
        if group_col in df_drill.columns:
            rates_by_group = rate_rollups(df_drill, [group_col])
            rates_by_group = rates_by_group.dropna(subset=["quote_rate", "realised_rate"])
            
            if len(rates_by_group) > 0:
                fig = rate_scatter(
                    rates_by_group,
                    group_col=group_col,
                    title=f"Rate Capture by {group_col.replace('_', ' ').title()}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION C: DRILL TABLE
    # =========================================================================
    level = get_drill_level()
    
    category_col = get_category_col(df_drill)
    if level == "company":
        section_header("By Department", "Click a row to drill down")
        group_col = "department_final"
    elif level == "department":
        section_header("By Category", "Click a row to drill down")
        group_col = category_col
    elif level == "category":
        section_header("By Staff", "Click a row to drill down")
        group_col = "staff_name"
    elif level == "staff":
        section_header("By Breakdown", "Click a row to drill down")
        group_col = "breakdown" if "breakdown" in df_drill.columns else "task_name"
    elif level == "breakdown":
        section_header("By Task", "Click a row to drill down")
        group_col = "task_name"
    else:
        section_header("Tasks")
        group_col = "task_name"
    
    can_drill = level != "task"
    render_drill_table_section(df_drill, group_col, can_drill=can_drill)
    
    # =========================================================================
    # SECTION D: ACTION SHORTLIST
    # =========================================================================
    st.markdown("---")
    section_header("Action Shortlist", "Top hotspots requiring attention")
    
    render_action_shortlist(df_drill)


def render_drill_table_section(df: pd.DataFrame, group_col: str, can_drill: bool = True):
    """Render the drill table for a given grouping column."""
    
    if group_col not in df.columns:
        st.warning(f"Column '{group_col}' not found in data.")
        return
    
    # Calculate metrics by group
    metrics = full_metric_pack(df, [group_col], exclude_leave=get_state("exclude_leave"))
    
    # Sort by revenue descending
    metrics = metrics.sort_values("revenue", ascending=False)
    
    # Select display columns
    display_cols = [group_col, "revenue", "margin_pct", "hours", "realised_rate"]
    
    # Add optional columns if available
    if "hours_variance_pct" in metrics.columns:
        display_cols.append("hours_variance_pct")
    if "unquoted_share" in metrics.columns:
        display_cols.append("unquoted_share")
    if "utilisation" in metrics.columns:
        display_cols.append("utilisation")
    
    display_df = metrics[[c for c in display_cols if c in metrics.columns]].copy()
    
    # Format for display
    formatted_df = format_metric_df(display_df)
    
    # Rename columns for display
    col_names = {
        "department_final": "Department",
        "job_category": "Category",
        "category_rev_job": "Category",
        "task_name": "Task",
        "staff_name": "Staff",
        "breakdown": "Breakdown",
        "revenue": "Revenue",
        "margin_pct": "Margin %",
        "hours": "Hours",
        "realised_rate": "Rate",
        "hours_variance_pct": "Hrs Var %",
        "unquoted_share": "Scope Creep",
        "utilisation": "Billable %",
    }
    formatted_df = formatted_df.rename(columns=col_names)
    
    # Display table with selection
    if can_drill:
        selection = st.dataframe(
            formatted_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"drill_table_{group_col}",
        )
        
        # Handle selection
        if selection and selection.selection and selection.selection.rows:
            selected_idx = selection.selection.rows[0]
            selected_value = metrics.iloc[selected_idx][group_col]
            
            level = get_drill_level()
            if level == "company":
                drill_to_department(selected_value)
                st.rerun()
            elif level == "department":
                drill_to_category(selected_value)
                st.rerun()
            elif level == "category":
                drill_to_staff(selected_value)
                st.rerun()
            elif level == "staff":
                if group_col == "breakdown":
                    drill_to_breakdown(selected_value)
                else:
                    drill_to_task(selected_value)
                st.rerun()
            elif level == "breakdown":
                drill_to_task(selected_value)
                st.rerun()
    else:
        st.dataframe(formatted_df, use_container_width=True, hide_index=True)


def render_action_shortlist(df: pd.DataFrame):
    """Render prioritized action items."""
    
    actions = []
    
    # Find biggest overruns by department
    if "department_final" in df.columns:
        delivery = quote_delivery_metrics(df, ["department_final"])
        delivery = delivery.dropna(subset=["hours_variance_pct"])
        
        top_overruns = delivery.nlargest(3, "hours_variance_pct")
        for _, row in top_overruns.iterrows():
            if row["hours_variance_pct"] > 20:
                actions.append({
                    "type": "Hours Overrun",
                    "location": row["department_final"],
                    "detail": f"{row['hours_variance_pct']:.1f}% over quoted hours",
                    "page": "Active Delivery"
                })
    
    # Find rate leakage
    if "department_final" in df.columns:
        rates = rate_rollups(df, ["department_final"])
        rates = rates.dropna(subset=["rate_variance"])
        
        worst_rate = rates.nsmallest(3, "rate_variance")
        for _, row in worst_rate.iterrows():
            if row["rate_variance"] < -10:
                actions.append({
                    "type": "Rate Leakage",
                    "location": row["department_final"],
                    "detail": f"${abs(row['rate_variance']):.0f}/hr below quote rate",
                    "page": "Executive Summary"
                })
    
    # Flag low billable share (descriptive)
    if "department_final" in df.columns:
        util = utilisation_metrics(df, ["department_final"], exclude_leave=True)
        low_billable = util.nsmallest(3, "utilisation")
        for _, row in low_billable.iterrows():
            if row["utilisation"] < 40:
                actions.append({
                    "type": "Low Billable Share",
                    "location": row["department_final"],
                    "detail": f"{row['utilisation']:.1f}% billable share of hours",
                    "page": "Time Allocation"
                })
    
    if not actions:
        st.info("No critical action items identified.")
        return
    
    # Display as table
    action_df = pd.DataFrame(actions[:10])  # Limit to top 10
    
    st.dataframe(
        action_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "type": st.column_config.TextColumn("Issue Type"),
            "location": st.column_config.TextColumn("Location"),
            "detail": st.column_config.TextColumn("Detail"),
            "page": st.column_config.TextColumn("See Page"),
        }
    )


if __name__ == "__main__":
    main()
