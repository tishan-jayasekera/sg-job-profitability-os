"""
UI components for Three Forces Rate Bridge visualization.
"""
import streamlit as st
import plotly.graph_objects as go

from src.metrics.rate_bridge import RateBridgeResult, get_dominant_force, generate_bridge_insight

COLORS = {
    "quote": "#6c757d",
    "realised": "#1565c0",
    "delivery_pos": "#28a745",
    "delivery_neg": "#dc3545",
    "commercial_pos": "#28a745",
    "commercial_neg": "#dc3545",
    "mix_pos": "#6f42c1",
    "mix_neg": "#6f42c1",
    "underrun": "#28a745",
    "overrun": "#dc3545",
    "scope_creep": "#ff9800",
}


def render_rate_bridge_header(bridge: RateBridgeResult):
    st.subheader("📊 Quote → Realised Rate Bridge")
    st.caption("*How did we get from quoted rate to realised rate?*")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Quote Rate", f"${bridge.quote_rate:.0f}/hr", help="Quoted amount ÷ quoted hours")

    with col2:
        delta_color = "normal" if bridge.total_gap >= 0 else "inverse"
        st.metric(
            "Realised Rate",
            f"${bridge.realised_rate:.0f}/hr",
            delta=f"${bridge.total_gap:+.0f}/hr vs quote",
            delta_color=delta_color,
        )

    with col3:
        st.metric("Hours Variance", f"{bridge.hours_variance_pct:+.0f}%", help="(Actual - Quoted) / Quoted")

    with col4:
        st.metric("Revenue Capture", f"{bridge.revenue_capture_pct:.0f}%", help="Actual revenue / quoted amount")


def render_rate_bridge_waterfall(bridge: RateBridgeResult):
    categories = [
        "Quote Rate",
        "Delivery: Under-runs",
        "Delivery: Overruns",
        "Delivery: Scope Creep",
        "Commercial: Recovery",
        "Commercial: Leakage",
        "Mix Effect",
        "Realised Rate",
    ]

    values = [
        bridge.quote_rate,
        bridge.hours_underrun_effect,
        bridge.hours_overrun_effect,
        bridge.scope_creep_effect,
        bridge.revenue_recovery_effect,
        bridge.revenue_leakage_effect,
        bridge.mix_effect,
        bridge.realised_rate,
    ]

    measures = [
        "absolute",
        "relative",
        "relative",
        "relative",
        "relative",
        "relative",
        "relative",
        "total",
    ]

    fig = go.Figure(
        go.Waterfall(
            name="Rate Bridge",
            orientation="v",
            measure=measures,
            x=categories,
            y=values,
            text=[f"${v:+.0f}" if i not in [0, 7] else f"${v:.0f}" for i, v in enumerate(values)],
            textposition="outside",
            connector={"line": {"color": "#ccc", "width": 1, "dash": "dot"}},
            decreasing={"marker": {"color": COLORS["delivery_neg"]}},
            increasing={"marker": {"color": COLORS["delivery_pos"]}},
            totals={"marker": {"color": COLORS["realised"]}},
        )
    )

    fig.add_annotation(
        x=2,
        y=bridge.quote_rate + 10,
        text="Force 1: Delivery",
        showarrow=False,
        font=dict(size=10, color="#666"),
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.add_annotation(
        x=4.5,
        y=bridge.quote_rate + 10,
        text="Force 2: Commercial",
        showarrow=False,
        font=dict(size=10, color="#666"),
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.add_annotation(
        x=6,
        y=bridge.quote_rate + 10,
        text="Force 3: Mix",
        showarrow=False,
        font=dict(size=10, color="#666"),
        bgcolor="rgba(255,255,255,0.8)",
    )

    fig.update_layout(
        title="Rate Bridge: Quote → Realised",
        yaxis_title="Rate ($/hr)",
        showlegend=False,
        height=400,
        margin=dict(t=60, b=80, l=60, r=40),
        xaxis_tickangle=-30,
    )

    fig.add_hline(
        y=bridge.quote_rate,
        line_dash="dash",
        line_color="#999",
        annotation_text="Quote Rate",
        annotation_position="top left",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_three_forces_summary(bridge: RateBridgeResult):
    st.markdown("### The Three Forces")

    col1, col2, col3 = st.columns(3)

    with col1:
        effect = bridge.delivery_effect
        effect_pct = bridge.delivery_effect_pct
        color = "#28a745" if effect >= 0 else "#dc3545"
        icon = "✅" if effect >= 0 else "⚠️"

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
                        border-radius: 12px; padding: 1.25rem;
                        border-left: 4px solid {color};">
                <div style="font-size: 0.75rem; color: #666; text-transform: uppercase;
                            letter-spacing: 0.5px; margin-bottom: 0.5rem;">
                    {icon} Force 1: Delivery Efficiency
                </div>
                <div style="font-size: 1.75rem; font-weight: 700; color: {color};">
                    ${effect:+.0f}/hr
                </div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 0.25rem;">
                    {effect_pct:+.0f}% of quote rate
                </div>
                <hr style="margin: 0.75rem 0; border: none; border-top: 1px solid #eee;">
                <div style="font-size: 0.8rem; color: #888;">
                    <strong>Owner:</strong> Delivery Team<br>
                    <strong>Lever:</strong> Hours vs quote
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Breakdown", expanded=False):
            st.markdown(
                f"""
                - Under-runs: **${bridge.hours_underrun_effect:+.0f}/hr**
                - Overruns: **${bridge.hours_overrun_effect:+.0f}/hr**
                - Scope creep: **${bridge.scope_creep_effect:+.0f}/hr**
                """
            )

    with col2:
        effect = bridge.commercial_effect
        effect_pct = bridge.commercial_effect_pct
        color = "#28a745" if effect >= 0 else "#dc3545"
        icon = "✅" if effect >= 0 else "⚠️"

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
                        border-radius: 12px; padding: 1.25rem;
                        border-left: 4px solid {color};">
                <div style="font-size: 0.75rem; color: #666; text-transform: uppercase;
                            letter-spacing: 0.5px; margin-bottom: 0.5rem;">
                    {icon} Force 2: Commercial Capture
                </div>
                <div style="font-size: 1.75rem; font-weight: 700; color: {color};">
                    ${effect:+.0f}/hr
                </div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 0.25rem;">
                    {effect_pct:+.0f}% of quote rate
                </div>
                <hr style="margin: 0.75rem 0; border: none; border-top: 1px solid #eee;">
                <div style="font-size: 0.8rem; color: #888;">
                    <strong>Owner:</strong> Commercial Team<br>
                    <strong>Lever:</strong> Revenue vs quote
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Breakdown", expanded=False):
            st.markdown(
                f"""
                - Recovery (scope charges): **${bridge.revenue_recovery_effect:+.0f}/hr**
                - Leakage (write-offs): **${bridge.revenue_leakage_effect:+.0f}/hr**
                """
            )

    with col3:
        effect = bridge.mix_effect
        effect_pct = bridge.mix_effect_pct
        color = "#6f42c1"
        icon = "📊"

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
                        border-radius: 12px; padding: 1.25rem;
                        border-left: 4px solid {color};">
                <div style="font-size: 0.75rem; color: #666; text-transform: uppercase;
                            letter-spacing: 0.5px; margin-bottom: 0.5rem;">
                    {icon} Force 3: Portfolio Mix
                </div>
                <div style="font-size: 1.75rem; font-weight: 700; color: {color};">
                    ${effect:+.0f}/hr
                </div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 0.25rem;">
                    {effect_pct:+.0f}% of quote rate
                </div>
                <hr style="margin: 0.75rem 0; border: none; border-top: 1px solid #eee;">
                <div style="font-size: 0.8rem; color: #888;">
                    <strong>Owner:</strong> BD / Intake<br>
                    <strong>Lever:</strong> Job composition
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("What drives mix?", expanded=False):
            st.markdown(
                """
                Mix effect captures:
                - Shift toward higher/lower rate jobs
                - Change in category composition
                - Pricing variance from quote templates
                """
            )


def render_contradiction_explainer(bridge: RateBridgeResult):
    if bridge.delivery_effect < -5 and bridge.total_gap > 0:
        st.warning(
            f"""
            ### ⚠️ The Contradiction Explained

            **How can realised rate exceed quote when delivery leaked?**

            | Force | Impact | Direction |
            |-------|--------|-----------|
            | Delivery | ${bridge.delivery_effect:+.0f}/hr | ↓ Pushed rate DOWN |
            | Commercial | ${bridge.commercial_effect:+.0f}/hr | {'↑' if bridge.commercial_effect > 0 else '↓'} |
            | Mix | ${bridge.mix_effect:+.0f}/hr | {'↑' if bridge.mix_effect > 0 else '↓'} |
            | **Net** | **${bridge.total_gap:+.0f}/hr** | **↑ Rate still UP** |

            **Bottom line:** {'Mix' if abs(bridge.mix_effect) > abs(bridge.commercial_effect) else 'Commercial recovery'}
            masked the delivery problem. This may not be sustainable — winning higher-rate work
            can't forever compensate for delivery inefficiency.
            """
        )
    elif bridge.delivery_effect > 5 and bridge.total_gap < 0:
        st.info(
            f"""
            ### 🔍 Why Rate Is Below Quote Despite Good Delivery

            Delivery was efficient (+${bridge.delivery_effect:.0f}/hr effect), but
            {'commercial leakage' if bridge.commercial_effect < 0 else 'unfavorable mix'}
            dragged the rate down.

            **Focus area:** {'Revenue capture (write-offs, discounts)' if bridge.commercial_effect < 0 else 'Job composition (won lower-rate work)'}
            """
        )


def render_insight_summary(bridge: RateBridgeResult):
    insight = generate_bridge_insight(bridge)
    st.markdown("### 💡 Key Insight")
    st.markdown(insight)


def render_department_bridge_comparison(df):
    from src.metrics.rate_bridge import compute_rate_bridge_by_group

    st.subheader("📈 Bridge Effects by Department")
    st.caption("*Which departments are contributing what?*")

    dept_bridge = compute_rate_bridge_by_group(df, "department_final")

    if len(dept_bridge) == 0:
        st.info("No department data available")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Delivery Effect",
            x=dept_bridge["department_final"],
            y=dept_bridge["delivery_effect"],
            marker_color=[
                COLORS["delivery_pos"] if v >= 0 else COLORS["delivery_neg"]
                for v in dept_bridge["delivery_effect"]
            ],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Commercial Effect",
            x=dept_bridge["department_final"],
            y=dept_bridge["commercial_effect"],
            marker_color=[
                COLORS["commercial_pos"] if v >= 0 else COLORS["commercial_neg"]
                for v in dept_bridge["commercial_effect"]
            ],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Mix Effect",
            x=dept_bridge["department_final"],
            y=dept_bridge["mix_effect"],
            marker_color=COLORS["mix_pos"],
        )
    )

    fig.update_layout(
        barmode="group",
        title="Rate Bridge Effects by Department",
        yaxis_title="Effect on Rate ($/hr)",
        height=350,
        legend=dict(orientation="h", y=1.1),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#ccc")

    st.plotly_chart(fig, use_container_width=True)


def render_full_rate_bridge(df, job_df=None, show_department_comparison: bool = True):
    from src.metrics.rate_bridge import compute_rate_bridge

    bridge = compute_rate_bridge(df, job_df)

    render_rate_bridge_header(bridge)
    st.divider()
    render_rate_bridge_waterfall(bridge)
    st.divider()
    render_three_forces_summary(bridge)
    st.divider()
    render_contradiction_explainer(bridge)
    render_insight_summary(bridge)
    st.divider()
    if show_department_comparison:
        render_department_bridge_comparison(df)
