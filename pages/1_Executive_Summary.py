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
    rate_rollups, utilisation_metrics, exclude_leave, get_category_col, safe_quote_job_task,
    filter_jobs_by_state
)
from src.data.cohorts import filter_by_time_window, filter_active_jobs


st.set_page_config(page_title="Executive Summary", page_icon="📈", layout="wide")

init_state()


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all global filters to dataframe."""
    # Time window
    time_window = get_state("time_window")
    df = filter_by_time_window(df, time_window)
    
    # Job state filter (Active / Completed / All)
    job_state = get_state("job_state_filter")
    df = filter_jobs_by_state(df, job_state)
    
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
    with st.expander("Metric Definitions (Executive Summary)", expanded=False):
        st.markdown(
            """
**KPI Snapshot**
- **Revenue** = Σ `rev_alloc`
- **Cost** = Σ `base_cost`
- **Margin** = Revenue − Cost
- **Margin %** = Margin ÷ Revenue
- **Hours** = Σ `hours_raw`
- **Realised Rate** = Revenue ÷ Hours
- **Billable Share** = Billable Hours ÷ Total Hours (excl. leave)
- **Quoted/Actual Hrs** = Quoted Hours ÷ Actual Hours (safe quote rollup)

**Health Snapshot**
- **Hours Variance** = (Actual Hours − Quoted Hours) ÷ Quoted Hours
- **Scope Creep** = Unquoted Hours ÷ Total Hours
- **Quote Rate** = Quoted Amount ÷ Quoted Hours (safe rollup)
- **Rate Variance** = Realised Rate − Quote Rate
- **Jobs at Risk** = jobs with a majority of job‑tasks exceeding quoted hours
            """
        )

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
            "Margin % by department = (Σ revenue − Σ cost) ÷ Σ revenue. "
            "Revenue is `rev_alloc` and cost is `base_cost`, summed within each department "
            "(aggregate margin, not a simple average)."
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
    st.markdown("<div class='band'><div class='band-title'>Quote Discipline & Delivery Alignment</div><div class='band-sub'>Is effort tracking the plan? Where is variance coming from?</div>", unsafe_allow_html=True)

    if "job_no" in df_drill.columns:
        jt = safe_quote_job_task(df_drill)
        if len(jt) > 0 and "quoted_time_total" in jt.columns:
            actuals = df_drill.groupby(["job_no", "task_name"])["hours_raw"].sum().reset_index()
            jt = jt.merge(actuals, on=["job_no", "task_name"], how="left")
            job_level = jt.groupby("job_no").agg(
                quoted_hours=("quoted_time_total", "sum"),
                actual_hours=("hours_raw", "sum"),
                quoted_value=("quoted_amount_total", "sum") if "quoted_amount_total" in jt.columns else ("quoted_time_total", "sum"),
            ).reset_index()
            job_meta = df_drill.groupby("job_no").agg(
                department_final=("department_final", "first") if "department_final" in df_drill.columns else ("job_no", "first"),
                month_key=("month_key", "max") if "month_key" in df_drill.columns else ("job_no", "first"),
            ).reset_index()
            job_level = job_level.merge(job_meta, on="job_no", how="left")
            job_level["variance_pct"] = np.where(
                job_level["quoted_hours"] > 0,
                (job_level["actual_hours"] - job_level["quoted_hours"]) / job_level["quoted_hours"] * 100,
                np.nan
            )
            job_level["delta_hours"] = job_level["actual_hours"] - job_level["quoted_hours"]
            job_level["quote_rate"] = np.where(
                job_level["quoted_hours"] > 0,
                job_level["quoted_value"] / job_level["quoted_hours"],
                np.nan,
            )
            job_level["delta_value"] = job_level["delta_hours"] * job_level["quote_rate"]
            job_level = job_level.dropna(subset=["variance_pct"])

            # Executive KPIs
            total_jobs = len(job_level)
            overrun_jobs = (job_level["variance_pct"] > 0).sum()
            severe_jobs = (job_level["variance_pct"] >= 20).sum()
            median_var = job_level["variance_pct"].median() if total_jobs else np.nan
            p75_var = job_level["variance_pct"].quantile(0.75) if total_jobs else np.nan
            total_quoted = job_level["quoted_hours"].sum()
            total_actual = job_level["actual_hours"].sum()
            total_overrun_pct = (total_actual - total_quoted) / total_quoted * 100 if total_quoted > 0 else np.nan
            avg_hours_per_job = total_actual / total_jobs if total_jobs else np.nan

            overrun_hours = job_level.loc[job_level["delta_hours"] > 0, "delta_hours"].sum()
            savings_hours = job_level.loc[job_level["delta_hours"] < 0, "delta_hours"].sum()
            overrun_value = job_level.loc[job_level["delta_value"] > 0, "delta_value"].sum()
            savings_value = job_level.loc[job_level["delta_value"] < 0, "delta_value"].sum()
            net_value = overrun_value + savings_value

            st.markdown(
                """
                <style>
                .exec-kpi-row {display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:8px 0 16px 0;}
                .exec-kpi {background:#ffffff;border:1px solid #eceae3;border-radius:12px;padding:10px 12px;}
                .exec-kpi-title{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#6b6b6b;}
                .exec-kpi-value{font-size:1.35rem;font-weight:700;margin-top:4px;}
                .exec-kpi-sub{font-size:0.78rem;color:#7a7a7a;margin-top:2px;}
                .exec-panel{background:#ffffff;border:1px solid #eceae3;border-radius:14px;padding:12px 14px;margin:10px 0;}
                </style>
                """,
                unsafe_allow_html=True,
            )
            kpi_cols = st.columns(6)
            with kpi_cols[0]:
                st.metric("% Jobs Over Quote", f"{(overrun_jobs / total_jobs * 100):.0f}%" if total_jobs else "—")
            with kpi_cols[1]:
                st.metric("% Jobs >20% Overrun", f"{(severe_jobs / total_jobs * 100):.0f}%" if total_jobs else "—")
            with kpi_cols[2]:
                st.metric("Total Hours (Actual)", f"{total_actual:,.0f}" if pd.notna(total_actual) else "—")
            with kpi_cols[3]:
                st.metric("Avg Hours / Job", f"{avg_hours_per_job:,.1f}" if pd.notna(avg_hours_per_job) else "—")
            with kpi_cols[4]:
                st.metric("Median Variance %", f"{median_var:.1f}%" if pd.notna(median_var) else "—")
            with kpi_cols[5]:
                st.metric("Net Quote Value Drift", f"${net_value:,.0f}" if pd.notna(net_value) else "—")

            st.markdown("#### Executive Readout")
            slice_cols = st.columns([1, 1])
            with slice_cols[0]:
                dept_options = ["All"] + sorted(job_level["department_final"].dropna().unique().tolist())
                selected_dept = st.selectbox("Department slice", dept_options, key="exec_quote_dept_slice")
            with slice_cols[1]:
                show_trend = st.checkbox("Show trend over time", value=True, key="exec_quote_show_trend")

            slice_df = job_level.copy()
            if selected_dept != "All":
                slice_df = slice_df[slice_df["department_final"] == selected_dept]

            col1, col2 = st.columns([1.2, 1.8])
            with col1:
                status_counts = pd.DataFrame({
                    "Status": ["On/Under Quote", "Over Quote", ">20% Overrun"],
                    "Jobs": [
                        (slice_df["variance_pct"] <= 0).sum(),
                        (slice_df["variance_pct"] > 0).sum(),
                        (slice_df["variance_pct"] >= 20).sum(),
                    ],
                })
                fig = px.bar(
                    status_counts,
                    x="Status",
                    y="Jobs",
                    title="Compliance Mix",
                    color="Status",
                    color_discrete_map={
                        "On/Under Quote": "#4c78a8",
                        "Over Quote": "#f58518",
                        ">20% Overrun": "#e45756",
                    },
                )
                fig.update_layout(showlegend=False, yaxis_title="Jobs", xaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Reconciliation panel: rate uplift vs overrun
                realised_rate = np.where(
                    total_actual > 0,
                    (slice_df["actual_hours"] * slice_df["quote_rate"]).sum() / total_actual,
                    np.nan,
                )
                quote_rate = np.where(
                    total_quoted > 0,
                    (slice_df["quoted_hours"] * slice_df["quote_rate"]).sum() / total_quoted,
                    np.nan,
                )
                rate_uplift_value = (realised_rate - quote_rate) * total_actual if pd.notna(realised_rate) and pd.notna(quote_rate) else np.nan
                overrun_cost_value = (total_actual - total_quoted) * quote_rate if pd.notna(quote_rate) else np.nan
                net_reconcile = rate_uplift_value - overrun_cost_value if pd.notna(rate_uplift_value) and pd.notna(overrun_cost_value) else np.nan

                st.markdown("**Reconciliation: Rate vs Volume**")
                rec_cols = st.columns(3)
                with rec_cols[0]:
                    st.metric("Rate Uplift ($)", f"${rate_uplift_value:,.0f}" if pd.notna(rate_uplift_value) else "—")
                with rec_cols[1]:
                    st.metric("Overrun Cost ($)", f"${overrun_cost_value:,.0f}" if pd.notna(overrun_cost_value) else "—")
                with rec_cols[2]:
                    st.metric("Net Drift ($)", f"${net_reconcile:,.0f}" if pd.notna(net_reconcile) else "—")
                st.caption(
                    "Rate uplift estimates value gained from higher realized rate vs quoted. "
                    "Overrun cost estimates value lost from hours exceeding quote. "
                    "Net drift is the balance."
                )

            # Value attribution retained (compact)
            attribution = pd.DataFrame({
                "Component": ["Savings (Value)", "Overrun (Value)", "Net Drift"],
                "Value": [savings_value, overrun_value, net_value],
            })
            fig = px.bar(
                attribution,
                x="Component",
                y="Value",
                title="Value Attribution (Quote Discipline)",
                color="Component",
                color_discrete_map={
                    "Overrun (Value)": "#e45756",
                    "Savings (Value)": "#54a24b",
                    "Net Drift": "#4c78a8",
                },
            )
            fig.update_layout(showlegend=False, yaxis_title="$", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            if "department_final" in slice_df.columns:
                dept_overrun = (
                    slice_df.groupby("department_final")["delta_hours"]
                    .sum()
                    .reset_index()
                    .sort_values("delta_hours", ascending=False)
                )
                dept_overrun = dept_overrun[dept_overrun["delta_hours"] > 0]
                if len(dept_overrun) > 0:
                    fig = px.bar(
                        dept_overrun.head(10),
                        x="department_final",
                        y="delta_hours",
                        title="Overrun Hours by Department (Top 10)",
                        labels={"department_final": "Department", "delta_hours": "Overrun Hours"},
                    )
                    fig.update_layout(xaxis_tickangle=-20)
                    st.plotly_chart(fig, use_container_width=True)

            # Department summary table
            if "department_final" in slice_df.columns:
                category_col = get_category_col(df_drill)
                if category_col not in slice_df.columns:
                    category_col = "job_category" if "job_category" in slice_df.columns else None
                group_cols = ["department_final"] + ([category_col] if category_col else [])
                dept_summary = slice_df.groupby(group_cols).agg(
                    jobs=("job_no", "nunique"),
                    quoted_hours=("quoted_hours", "sum"),
                    actual_hours=("actual_hours", "sum"),
                    variance_pct=("variance_pct", "median"),
                    overrun_jobs=("variance_pct", lambda x: (x > 0).mean() * 100),
                    severe_jobs=("variance_pct", lambda x: (x >= 20).mean() * 100),
                ).reset_index()
                dept_summary["hours_delta"] = dept_summary["actual_hours"] - dept_summary["quoted_hours"]
                dept_summary["avg_hours_per_job"] = np.where(
                    dept_summary["jobs"] > 0,
                    dept_summary["actual_hours"] / dept_summary["jobs"],
                    np.nan,
                )
                if category_col:
                    dept_summary["net_value_drift"] = dept_summary["hours_delta"] * (
                        slice_df.groupby(["department_final", category_col])["quote_rate"]
                        .median()
                        .reindex(pd.MultiIndex.from_frame(dept_summary[["department_final", category_col]]))
                        .values
                    )
                else:
                    dept_summary["net_value_drift"] = dept_summary["hours_delta"] * (
                        slice_df.groupby("department_final")["quote_rate"].median().reindex(dept_summary["department_final"]).values
                    )
                dept_summary = dept_summary.sort_values("hours_delta", ascending=False)
                st.markdown(
                    "#### Department + Category Summary (Quote Discipline)"
                    if category_col
                    else "#### Department Summary (Quote Discipline)"
                )
                st.caption(
                    "Department-category pairs are ranked by total hours variance. Use this to pinpoint where quote drift is concentrated."
                    if category_col
                    else "Departments are ranked by total hours variance. Use this to pinpoint where quote drift is concentrated."
                )
                st.dataframe(
                    dept_summary.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "jobs": st.column_config.NumberColumn("Jobs", format="%.0f"),
                        "quoted_hours": st.column_config.NumberColumn("Quoted Hours", format="%.0f"),
                        "actual_hours": st.column_config.NumberColumn("Actual Hours", format="%.0f"),
                        "hours_delta": st.column_config.NumberColumn("Hours Δ", format="%+.0f"),
                        "avg_hours_per_job": st.column_config.NumberColumn("Avg Hours/Job", format="%.1f"),
                        "variance_pct": st.column_config.NumberColumn("Median Var %", format="%.1f%%"),
                        "overrun_jobs": st.column_config.NumberColumn("% Overrun Jobs", format="%.0f%%"),
                        "severe_jobs": st.column_config.NumberColumn("% >20% Overrun", format="%.0f%%"),
                        "net_value_drift": st.column_config.NumberColumn("Net Value Drift", format="$%+.0f"),
                    },
                )
                # Visual: variance vs value drift
                fig = px.scatter(
                    dept_summary.head(12),
                    x="hours_delta",
                    y="net_value_drift",
                    size="jobs",
                    color="variance_pct",
                    hover_name="department_final",
                    title="Department + Category Hotspots: Hours Δ vs Value Drift",
                    labels={
                        "hours_delta": "Hours Δ",
                        "net_value_drift": "Net Value Drift ($)",
                        "variance_pct": "Median Var %",
                    },
                    color_continuous_scale="RdYlGn_r",
                )
                fig.update_layout(coloraxis_colorbar_title="Median Var %")
                st.plotly_chart(fig, use_container_width=True)

            # Top variance tasks (value + volume)
            if "task_name" in df_drill.columns and "quoted_time_total" in df_drill.columns:
                task_actual = df_drill.groupby(["job_no", "task_name"]).agg(
                    actual_hours=("hours_raw", "sum")
                ).reset_index()
                task_quotes = safe_quote_job_task(df_drill)
                if len(task_quotes) > 0:
                    task_quotes = task_quotes.merge(task_actual, on=["job_no", "task_name"], how="left")
                    task_quotes["delta_hours"] = task_quotes["actual_hours"] - task_quotes["quoted_time_total"]
                    task_quotes["quote_rate"] = np.where(
                        task_quotes["quoted_time_total"] > 0,
                        task_quotes["quoted_amount_total"] / task_quotes["quoted_time_total"],
                        np.nan,
                    )
                    task_quotes["delta_value"] = task_quotes["delta_hours"] * task_quotes["quote_rate"]
                    task_summary = task_quotes.groupby("task_name").agg(
                        jobs=("job_no", "nunique"),
                        delta_hours=("delta_hours", "sum"),
                        delta_value=("delta_value", "sum"),
                    ).reset_index()
                    st.markdown("#### Top Variance Tasks (Value & Volume)")
                    col_tv1, col_tv2 = st.columns(2)
                    with col_tv1:
                        st.dataframe(
                            task_summary.sort_values("delta_value", ascending=False).head(10),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "jobs": st.column_config.NumberColumn("Jobs", format="%.0f"),
                                "delta_hours": st.column_config.NumberColumn("Hours Δ", format="%+.0f"),
                                "delta_value": st.column_config.NumberColumn("Value Δ", format="$%+.0f"),
                            },
                        )
                    with col_tv2:
                        st.dataframe(
                            task_summary.sort_values("delta_hours", ascending=False).head(10),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "jobs": st.column_config.NumberColumn("Jobs", format="%.0f"),
                                "delta_hours": st.column_config.NumberColumn("Hours Δ", format="%+.0f"),
                                "delta_value": st.column_config.NumberColumn("Value Δ", format="$%+.0f"),
                            },
                        )

            if show_trend and "month_key" in slice_df.columns:
                trend_df = slice_df.copy()
                trend_df["month_key"] = pd.to_datetime(trend_df["month_key"], errors="coerce")
                trend_df = trend_df.dropna(subset=["month_key"])
                if len(trend_df) > 0:
                    trend = trend_df.groupby(pd.Grouper(key="month_key", freq="M")).agg(
                        avg_variance_pct=("variance_pct", "mean"),
                        overrun_share=("variance_pct", lambda x: (x > 0).mean() * 100),
                        severe_share=("variance_pct", lambda x: (x >= 20).mean() * 100),
                    ).reset_index()
                    fig = px.line(
                        trend,
                        x="month_key",
                        y=["avg_variance_pct", "overrun_share", "severe_share"],
                        title="Quote Discipline Trend",
                        labels={"value": "%", "month_key": "Month", "variable": "Metric"},
                    )
                    fig.update_layout(yaxis_title="%")
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
