"""
Segment profiling UI components.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

from src.metrics.segment_profiling import SEGMENT_CONFIG


def render_segment_selector(job_df: pd.DataFrame) -> Optional[str]:
    """
    Render segment pill selector with counts.
    Returns selected segment or None.
    """
    st.markdown(
        """
        <style>
        .segment-pills {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }
        .segment-pill {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        .segment-pill:hover { transform: translateY(-2px); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    segments = ["Subsidiser", "Mixed", "Margin-Erosive", "Protected Overrun"]
    cols = st.columns(len(segments))
    selected = None

    for i, seg in enumerate(segments):
        count = len(job_df[job_df["segment"] == seg])
        config = SEGMENT_CONFIG.get(seg, {})

        with cols[i]:
            if st.button(
                f"{config.get('icon', '●')} {seg} ({count})",
                key=f"seg_{seg}",
                use_container_width=True,
                type="secondary"
                if st.session_state.get("selected_segment") != seg
                else "primary",
            ):
                st.session_state.selected_segment = seg
                selected = seg

    return selected or st.session_state.get("selected_segment")


def render_segment_quadrant_legend():
    """
    Render quadrant explanation legend.
    """
    st.markdown(
        """
        ### Understanding the Quadrant

        | Position | Execution | Commercial | Segment |
        |----------|-----------|------------|---------|
        | Top-Left | ✓ Under-run | ✓ Rate premium | 🟢 Subsidiser |
        | Top-Right | ✗ Overrun | ✓ Rate premium | 🟣 Protected Overrun |
        | Bottom-Left | ✓ Under-run | ✗ Rate leakage | 🟡 Mixed |
        | Bottom-Right | ✗ Overrun | ✗ Rate leakage | 🔴 Margin-Erosive |
        """
    )


def render_composition_panel(composition: Dict):
    """
    Render segment composition: job count, revenue, hours shares.
    """
    st.subheader("📊 Segment Composition")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Jobs",
            f"{composition['job_count']}",
            f"{composition['job_share']:.0f}% of portfolio",
        )

    with col2:
        st.metric(
            "Revenue",
            f"${composition['revenue']:,.0f}",
            f"{composition['revenue_share']:.0f}% of portfolio",
        )

    with col3:
        st.metric(
            "Hours",
            f"{composition['hours']:,.0f}",
            f"{composition['hours_share']:.0f}% of portfolio",
        )

    if composition.get("by_department"):
        st.markdown("**By Department**")
        dept_df = pd.DataFrame.from_dict(composition["by_department"], orient="index")
        dept_df = dept_df.reset_index().rename(columns={"index": "Department"})

        seg = st.session_state.get("selected_segment", "Mixed")
        color = SEGMENT_CONFIG.get(seg, {}).get("color", "#666")

        fig = px.bar(
            dept_df.sort_values("revenue", ascending=True),
            y="Department",
            x="revenue",
            orientation="h",
            color_discrete_sequence=[color],
        )
        fig.update_layout(height=200, showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_driver_distributions(drivers: Dict, comparison_drivers: Optional[Dict] = None):
    """
    Render small multiples of driver distributions.
    """
    st.subheader("🔍 Why Are They Here? — Driver Distributions")

    metrics = [
        ("hours_variance_pct", "Hours Variance %", "Execution"),
        ("rate_variance", "Rate Variance ($/hr)", "Commercial"),
        ("scope_creep_pct", "Scope Creep %", "Scoping"),
        ("margin_pct", "Margin %", "Outcome"),
    ]

    cols = st.columns(4)

    for i, (key, label, category) in enumerate(metrics):
        with cols[i]:
            stats = drivers.get(key, {})
            median = stats.get("median")
            p25 = stats.get("p25")
            p75 = stats.get("p75")

            st.markdown(f"**{label}**")
            st.caption(category)

            if median is not None:
                metric_value = f"{median:+.0f}" if "variance" in key.lower() else f"{median:.0f}%"
                st.metric("Median", metric_value)
                if p25 is not None and p75 is not None:
                    st.caption(f"IQR: {p25:.0f} – {p75:.0f}")
            else:
                st.info("No data")

    leakage_pct = drivers.get("leakage_incidence", 0)
    if leakage_pct > 20:
        st.error(
            "⚠️ "
            f"{leakage_pct:.0f}% of jobs in this segment have rate leakage (> $-10/hr variance)"
        )
    elif leakage_pct > 0:
        st.warning(f"ℹ️ {leakage_pct:.0f}% of jobs have rate leakage")


def render_task_mix_divergence(task_mix: pd.DataFrame, segment: str):
    """
    Render task mix divergence chart.
    """
    st.subheader("📋 Task Mix vs Benchmark — What's Different?")
    benchmark_dept = st.session_state.get("drill_path", {}).get("department")
    benchmark_cat = st.session_state.get("drill_path", {}).get("category")
    if benchmark_dept or benchmark_cat:
        scope_parts = []
        if benchmark_dept:
            scope_parts.append(f"Dept: {benchmark_dept}")
        if benchmark_cat:
            scope_parts.append(f"Category: {benchmark_cat}")
        scope_str = " • ".join(scope_parts)
        st.caption(f"Benchmark = all other jobs in the current scope ({scope_str}), excluding this segment.")
    else:
        st.caption("Benchmark = all other jobs in the portfolio, excluding this segment.")

    if len(task_mix) == 0:
        st.info("No task data available")
        return

    top_n = 8
    top_over = task_mix.nlargest(top_n // 2, "divergence_pp")
    top_under = task_mix.nsmallest(top_n // 2, "divergence_pp")
    display_df = pd.concat([top_over, top_under]).drop_duplicates()

    colors = ["#dc3545" if x > 0 else "#28a745" for x in display_df["divergence_pp"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=display_df["task_name"],
            x=display_df["divergence_pp"],
            orientation="h",
            marker_color=colors,
            text=[f"{x:+.1f}pp" for x in display_df["divergence_pp"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Task Share Divergence: {segment} vs Benchmark",
        xaxis_title="Divergence (percentage points)",
        yaxis_title="",
        height=300,
        showlegend=False,
        margin=dict(l=0, r=40, t=40, b=40),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        """
        **Reading this chart:**
        - 🔴 Red bars = Tasks where this segment spends MORE time than benchmark
        - 🟢 Green bars = Tasks where this segment spends LESS time than benchmark
        - Large divergences reveal operational differences (e.g., erosive jobs may show outsized revisions, QA loops)
        """
    )


def render_duration_profile(duration: Dict, segment: str):
    """
    Render duration profile panel.
    """
    st.subheader("⏱️ Duration Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        median = duration.get("median_days")
        st.metric("Median Duration", f"{median:.0f} days" if median else "—")

    with col2:
        p75 = duration.get("p75_days")
        st.metric("P75 Duration", f"{p75:.0f} days" if p75 else "—")

    with col3:
        long_tail = duration.get("long_tail_pct")
        st.metric("Long Tail %", f"{long_tail:.0f}%" if long_tail else "—")
        st.caption("Jobs > P75 duration")

    if duration.get("median_days") and segment in ["Margin-Erosive", "Mixed"]:
        st.info(
            "💡 Longer durations correlate with coordination load, revision loops, and resourcing churn."
        )


def render_overrun_decomposition(decomp: Dict):
    """
    Render overrun decomposition: in-scope vs scope creep.
    """
    st.subheader("🔬 Overrun Decomposition")

    if decomp.get("total_variance_hours", 0) <= 0:
        st.success("✅ No net overrun in this segment")
        return

    col1, col2 = st.columns(2)

    with col1:
        inscope = decomp.get("inscope_overrun_share", 0)
        scope = decomp.get("scope_creep_share", 0)

        fig = go.Figure(
            data=[
                go.Pie(
                    values=[inscope, scope],
                    labels=["In-Scope Overrun", "Scope Creep"],
                    marker_colors=["#dc3545", "#ff9800"],
                    hole=0.6,
                    textinfo="label+percent",
                )
            ]
        )

        fig.update_layout(
            title="Overrun Source Split",
            height=250,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            """
            **What this tells you:**

            - **In-Scope Overrun** = Quoted tasks taking longer than expected → Execution/estimation issue
            - **Scope Creep** = Hours on unquoted work → Scoping/boundary issue

            **Action:**
            - High in-scope → Review estimation benchmarks
            - High scope creep → Review change order discipline
            """
        )

        st.metric("Total Variance", f"{decomp.get('total_variance_hours', 0):,.0f} hours")


def render_job_shortlist(shortlist: pd.DataFrame, job_name_lookup: Dict[str, str]):
    """
    Render job shortlist table with reason codes.
    """
    st.subheader("🎯 Jobs to Investigate — Ranked by Impact")

    if len(shortlist) == 0:
        st.info("No jobs in this segment")
        return

    from src.ui.formatting import format_job_label

    shortlist = shortlist.copy()
    shortlist["job_label"] = shortlist["job_no"].apply(
        lambda x: format_job_label(x, job_name_lookup)
    )

    display_cols = [
        "job_label",
        "job_category",
        "hours_variance_pct_job",
        "rate_variance_job",
        "scope_creep_pct_job",
        "margin_pct_job",
        "reason_codes",
    ]

    display_df = shortlist[display_cols].copy()
    display_df.columns = [
        "Job",
        "Category",
        "Hours Var %",
        "Rate Var",
        "Scope Creep %",
        "Margin %",
        "Why",
    ]

    display_df["Why"] = display_df["Why"].apply(lambda x: " • ".join(x) if x else "")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Job": st.column_config.TextColumn(width="large"),
            "Hours Var %": st.column_config.NumberColumn(format="%+.0f%%"),
            "Rate Var": st.column_config.NumberColumn(format="$%+.0f"),
            "Scope Creep %": st.column_config.NumberColumn(format="%.0f%%"),
            "Margin %": st.column_config.NumberColumn(format="%.0f%%"),
            "Why": st.column_config.TextColumn(width="large"),
        },
    )

    st.markdown("**Select a job to drill into:**")

    job_options = shortlist["job_no"].tolist()
    job_labels = [format_job_label(j, job_name_lookup) for j in job_options]

    selected_idx = st.selectbox(
        "Job",
        range(len(job_options)),
        format_func=lambda i: job_labels[i],
        key="job_selector_from_shortlist",
    )

    if st.button("🔎 Drill into Job", type="primary"):
        st.session_state.drill_path["job"] = job_options[selected_idx]
        st.rerun()
