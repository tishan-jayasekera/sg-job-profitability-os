"""
Executive Summary Page

Company → Department → Category → Staff → Breakdown → Task drill-down with KPIs,
quote-delivery reconciliation, and rate capture analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    rate_rollups, utilisation_metrics, exclude_leave, get_category_col, safe_quote_job_task
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
    st.markdown(
        """
        <style>
        .exec-hero {
            background: linear-gradient(135deg, #f6f1e8 0%, #fff7ea 100%);
            border: 1px solid #eadfcb;
            border-radius: 14px;
            padding: 14px 18px;
            margin: 8px 0 18px 0;
        }
        .exec-sub { color: #6b6b6b; font-size: 0.92rem; margin-top: 4px; }
        .band {
            border-radius: 14px;
            padding: 14px 16px;
            margin: 14px 0;
            border: 1px solid #e8e2d6;
            background: #fcfbf8;
        }
        .band-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 6px; }
        .band-sub { color: #6b6b6b; font-size: 0.9rem; margin-bottom: 10px; }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e6e1d5;
            border-radius: 12px;
            padding: 10px 12px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .kpi-label { font-size: 0.75rem; color: #6b6b6b; text-transform: uppercase; letter-spacing: 0.04em; }
        .kpi-value { font-size: 1.35rem; font-weight: 600; margin-top: 2px; }
        .kpi-foot { font-size: 0.8rem; color: #8a8a8a; margin-top: 4px; }
        .health-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        .health-card {
            background: #ffffff;
            border: 1px solid #e6e1d5;
            border-radius: 12px;
            padding: 12px 14px;
        }
        .health-title { font-weight: 600; margin-bottom: 6px; }
        .health-metric { display: flex; justify-content: space-between; margin: 4px 0; }
        .health-metric span:first-child { color: #6b6b6b; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
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

    st.markdown(
        """
        <div class="exec-hero">
            <div class="band-title">Executive KPI Snapshot</div>
            <div class="exec-sub">Top-line profitability and delivery alignment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prof = profitability_rollup(df_drill)
    util = utilisation_metrics(df_drill, exclude_leave=get_state("exclude_leave"))
    delivery = quote_delivery_metrics(df_drill)
    quoted = delivery["quoted_hours"].iloc[0] if "quoted_hours" in delivery.columns else 0
    actual = delivery["actual_hours"].iloc[0] if "actual_hours" in delivery.columns else 0
    quote_vs_actual = quoted / actual if actual and actual > 0 else np.nan

    kpi_html = f"""
    <div class="band">
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Revenue</div><div class="kpi-value">{fmt_currency(prof['revenue'].iloc[0])}</div></div>
            <div class="kpi-card"><div class="kpi-label">Cost</div><div class="kpi-value">{fmt_currency(prof['cost'].iloc[0])}</div></div>
            <div class="kpi-card"><div class="kpi-label">Margin</div><div class="kpi-value">{fmt_currency(prof['margin'].iloc[0])}</div></div>
            <div class="kpi-card"><div class="kpi-label">Margin %</div><div class="kpi-value">{fmt_percent(prof['margin_pct'].iloc[0])}</div></div>
            <div class="kpi-card"><div class="kpi-label">Hours</div><div class="kpi-value">{fmt_hours(prof['hours'].iloc[0])}</div></div>
            <div class="kpi-card"><div class="kpi-label">Realised Rate</div><div class="kpi-value">{fmt_rate(prof['realised_rate'].iloc[0])}</div></div>
            <div class="kpi-card"><div class="kpi-label">Billable Share</div><div class="kpi-value">{fmt_percent(util['utilisation'].iloc[0])}</div><div class="kpi-foot">Billable ÷ Total (excl. leave)</div></div>
            <div class="kpi-card"><div class="kpi-label">Quoted/Actual Hrs</div><div class="kpi-value">{f"{quote_vs_actual:.2f}x" if pd.notna(quote_vs_actual) else "—"}</div></div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # =========================================================================
    # SECTION B: HEALTH SNAPSHOT
    # =========================================================================
    st.markdown("<div class='band'><div class='band-title'>Health Snapshot</div><div class='band-sub'>Where value is leaking and why</div>", unsafe_allow_html=True)

    variance_pct = delivery["hours_variance_pct"].iloc[0] if not pd.isna(delivery["hours_variance_pct"].iloc[0]) else 0
    scope_creep = delivery["unquoted_share"].iloc[0] if not pd.isna(delivery["unquoted_share"].iloc[0]) else 0
    rates = rate_rollups(df_drill)
    qr = rates["quote_rate"].iloc[0]
    rr = rates["realised_rate"].iloc[0]
    rv = rates["rate_variance"].iloc[0]
    risk_count = 0
    if "job_no" in df_drill.columns and "quoted_time_total" in df_drill.columns:
        jt = safe_quote_job_task(df_drill)
        if len(jt) > 0 and "quoted_time_total" in jt.columns:
            actuals = df_drill.groupby(["job_no", "task_name"])["hours_raw"].sum().reset_index()
            jt = jt.merge(actuals, on=["job_no", "task_name"], how="left")
            jt["is_overrun"] = jt["hours_raw"] > jt["quoted_time_total"]
            risk_count = int(jt.groupby("job_no")["is_overrun"].mean().gt(0.5).sum())

    health_html = f"""
    <div class="health-grid">
        <div class="health-card">
            <div class="health-title">Quote → Delivery</div>
            <div class="health-metric"><span>Hours Variance</span><span>{fmt_percent(variance_pct)}</span></div>
            <div class="health-metric"><span>Scope Creep</span><span>{fmt_percent(scope_creep)}</span></div>
        </div>
        <div class="health-card">
            <div class="health-title">Rate Capture</div>
            <div class="health-metric"><span>Quote Rate</span><span>{fmt_rate(qr) if not pd.isna(qr) else "—"}</span></div>
            <div class="health-metric"><span>Realised Rate</span><span>{fmt_rate(rr) if not pd.isna(rr) else "—"}</span></div>
            <div class="health-metric"><span>Rate Variance</span><span>{fmt_rate(rv) if not pd.isna(rv) else "—"}</span></div>
        </div>
        <div class="health-card">
            <div class="health-title">Active Risk</div>
            <div class="health-metric"><span>Jobs at Risk</span><span>{risk_count:,}</span></div>
            <div class="health-metric"><span>Overrun Rate</span><span>{fmt_percent((risk_count / df_drill["job_no"].nunique() * 100) if "job_no" in df_drill.columns and df_drill["job_no"].nunique() > 0 else 0)}</span></div>
        </div>
    </div>
    </div>
    """
    st.markdown(health_html, unsafe_allow_html=True)

    # =========================================================================
    # SECTION C: PORTFOLIO PERFORMANCE
    # =========================================================================
    st.markdown("<div class='band'><div class='band-title'>Portfolio Performance by Department</div><div class='band-sub'>Where margin and revenue are concentrated</div>", unsafe_allow_html=True)

    if "department_final" in df_drill.columns:
        dept_metrics = full_metric_pack(df_drill, ["department_final"], exclude_leave=get_state("exclude_leave"))
        dept_metrics = dept_metrics.sort_values("revenue", ascending=False)

        col1, col2 = st.columns(2)
        st.caption(
            "Margin % by department is calculated as (Σ revenue − Σ cost) ÷ Σ revenue, "
            "using `rev_alloc` and `base_cost` summed within each department. "
            "This is an aggregate margin, not a simple average of job margins."
        )
        with col1:
            fig = px.bar(
                dept_metrics,
                x="department_final",
                y="margin_pct",
                title="Margin % by Department",
                labels={"department_final": "Department", "margin_pct": "Margin %"},
                text_auto=".1f",
            )
            fig.update_layout(yaxis_tickformat=".1f", xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dept_metrics["department_final"],
                y=dept_metrics["revenue"],
                name="Revenue",
                marker_color="#4c78a8",
            ))
            fig.add_trace(go.Bar(
                x=dept_metrics["department_final"],
                y=dept_metrics["margin"],
                name="Margin",
                marker_color="#54a24b",
            ))
            fig.update_layout(
                title="Revenue & Margin by Department",
                barmode="group",
                yaxis=dict(title="$"),
                xaxis_tickangle=-20,
            )
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION D: QUOTE DISCIPLINE & DELIVERY ALIGNMENT
    # =========================================================================
    st.markdown("<div class='band'><div class='band-title'>Quote Discipline & Delivery Alignment</div><div class='band-sub'>Is effort tracking the plan?</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
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

    with col2:
        if "job_no" in df_drill.columns:
            jt = safe_quote_job_task(df_drill)
            if len(jt) > 0 and "quoted_time_total" in jt.columns:
                actuals = df_drill.groupby(["job_no", "task_name"])["hours_raw"].sum().reset_index()
                jt = jt.merge(actuals, on=["job_no", "task_name"], how="left")
                job_level = jt.groupby("job_no").agg(
                    quoted_hours=("quoted_time_total", "sum"),
                    actual_hours=("hours_raw", "sum"),
                ).reset_index()
                fig = px.scatter(
                    job_level,
                    x="quoted_hours",
                    y="actual_hours",
                    title="Quoted vs Actual Hours (by Job)",
                    labels={"quoted_hours": "Quoted Hours", "actual_hours": "Actual Hours"},
                )
                max_val = max(job_level["quoted_hours"].max(), job_level["actual_hours"].max())
                fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION E: DRILL TABLE
    # =========================================================================
    st.markdown("<div class='band'><div class='band-title'>Drilldown View</div><div class='band-sub'>Click a row to drill into the delivery chain</div>", unsafe_allow_html=True)

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
    st.markdown("</div>", unsafe_allow_html=True)
    
    # =========================================================================
    # SECTION F: ACTION SHORTLIST
    # =========================================================================
    st.markdown("<div class='band'><div class='band-title'>Action Queue</div><div class='band-sub'>Top hotspots requiring immediate attention</div>", unsafe_allow_html=True)
    render_action_shortlist(df_drill)
    st.markdown("</div>", unsafe_allow_html=True)


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
                    "page": "Job Mix & Demand"
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
