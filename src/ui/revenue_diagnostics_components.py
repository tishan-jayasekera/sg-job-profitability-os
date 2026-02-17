"""UI rendering components for Revenue Decline Diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.formatting import fmt_currency, fmt_percent


COLORS = {
    "primary_blue": "#2563eb",
    "green_positive": "#28a745",
    "red_negative": "#dc3545",
    "muted_gray": "#667085",
    "surface": "#f8fafc",
    "purple_accent": "#6f42c1",
    "amber_warning": "#ffc107",
}


def _is_empty(df: pd.DataFrame | None) -> bool:
    return df is None or len(df) == 0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_kpi_row(yoy_row: pd.Series) -> None:
    """Render headline KPI row from one YoY row."""
    if yoy_row is None or len(yoy_row) == 0:
        st.info("Insufficient data for KPI row")
        return

    rev_curr = _safe_float(yoy_row.get("revenue_curr", 0.0))
    rev_prev = _safe_float(yoy_row.get("revenue_prev", 0.0))
    delta = rev_curr - rev_prev
    delta_pct = (delta / rev_prev * 100) if rev_prev != 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Revenue (Current)", fmt_currency(rev_curr))
    with c2:
        st.metric("Revenue (Prior)", fmt_currency(rev_prev))
    with c3:
        st.metric("Δ Revenue", fmt_currency(delta))
    with c4:
        st.metric("Δ %", fmt_percent(delta_pct))


def render_revenue_bridge_waterfall(decomp_row: pd.Series, key_prefix: str = "rd-bridge") -> None:
    """Render volume/price/interaction waterfall."""
    if decomp_row is None or len(decomp_row) == 0:
        st.info("Insufficient data for bridge")
        return

    volume = _safe_float(decomp_row.get("volume_effect", 0.0))
    price = _safe_float(decomp_row.get("price_effect", 0.0))
    interaction = _safe_float(decomp_row.get("interaction_effect", 0.0))
    total = _safe_float(decomp_row.get("delta_revenue", volume + price + interaction))

    fig = go.Figure(
        go.Waterfall(
            x=["Volume Effect", "Price Effect", "Interaction Effect", "Total"],
            y=[volume, price, interaction, total],
            measure=["relative", "relative", "relative", "total"],
            increasing={"marker": {"color": COLORS["green_positive"]}},
            decreasing={"marker": {"color": COLORS["red_negative"]}},
            totals={"marker": {"color": COLORS["primary_blue"]}},
            connector={"line": {"color": "#94a3b8"}},
        )
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=40, b=20))
    _plot(fig, key=f"{key_prefix}-waterfall")


def render_trend_panel(monthly_df: pd.DataFrame, service_line: str, key_prefix: str = "rd-trend") -> None:
    """Render trend charts for revenue/jobs, ARPJ, and realised rate."""
    if _is_empty(monthly_df):
        st.info("Insufficient data for trend panel")
        return

    work = monthly_df.copy()
    if service_line not in ("", "All", None) and "service_line" in work.columns:
        work = work[work["service_line"] == service_line].copy()

    if len(work) == 0:
        st.info("Insufficient data for trend panel")
        return

    work = work.sort_values("month_key")

    fig_main = go.Figure()
    fig_main.add_trace(
        go.Scatter(
            x=work["month_key"],
            y=work["revenue"],
            name="Revenue",
            mode="lines",
            fill="tozeroy",
            line=dict(color=COLORS["primary_blue"], width=2),
        )
    )
    fig_main.add_trace(
        go.Scatter(
            x=work["month_key"],
            y=work["jobs"],
            name="Job Count",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=COLORS["muted_gray"], width=2),
        )
    )
    fig_main.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=30, r=30, t=30, b=20),
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="Jobs", overlaying="y", side="right"),
    )
    _plot(fig_main, key=f"{key_prefix}-main")

    c1, c2 = st.columns(2)
    with c1:
        fig_arpj = go.Figure()
        fig_arpj.add_trace(
            go.Scatter(
                x=work["month_key"],
                y=work["avg_rev_per_job"],
                mode="lines+markers",
                name="Avg Revenue / Job",
                line=dict(color=COLORS["purple_accent"], width=2),
            )
        )
        fig_arpj.update_layout(template="plotly_white", height=300, margin=dict(l=30, r=20, t=30, b=20), title="Avg Revenue per Job")
        _plot(fig_arpj, key=f"{key_prefix}-arpj")

    with c2:
        fig_rate = go.Figure()
        fig_rate.add_trace(
            go.Scatter(
                x=work["month_key"],
                y=work["realised_rate"],
                mode="lines+markers",
                name="Realised Rate",
                line=dict(color=COLORS["green_positive"], width=2),
            )
        )
        fig_rate.update_layout(template="plotly_white", height=300, margin=dict(l=30, r=20, t=30, b=20), title="Realised Rate")
        _plot(fig_rate, key=f"{key_prefix}-rate")


def render_client_bridge(
    bridge_df: pd.DataFrame,
    top_clients_df: pd.DataFrame,
    key_prefix: str = "rd-client",
) -> None:
    """Render client bridge components and top contributing clients."""
    if _is_empty(bridge_df):
        st.info("Insufficient data for client bridge")
        return

    if "bridge_component" in bridge_df.columns and (bridge_df["bridge_component"] == "No Client Data").any():
        st.info("Client column not available for the current data scope.")
        return

    chart_df = bridge_df.copy()
    fig = go.Figure(
        go.Bar(
            y=chart_df["bridge_component"],
            x=chart_df["amount"],
            orientation="h",
            marker_color=np.where(chart_df["amount"] >= 0, COLORS["green_positive"], COLORS["red_negative"]),
        )
    )
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=30, r=20, t=30, b=20), xaxis_title="Revenue Impact")
    _plot(fig, key=f"{key_prefix}-bridge")

    if _is_empty(top_clients_df):
        st.info("No client-level bridge details available.")
    else:
        st.dataframe(top_clients_df, use_container_width=True, hide_index=True)


def render_deal_size_comparison(
    deals_curr: pd.DataFrame,
    deals_prev: pd.DataFrame,
    key_prefix: str = "rd-deal",
) -> None:
    """Render overlaid deal-size histograms and supporting KPIs."""
    if _is_empty(deals_curr) and _is_empty(deals_prev):
        st.info("Insufficient data for deal-size comparison")
        return

    fig = go.Figure()
    if not _is_empty(deals_prev):
        fig.add_trace(
            go.Histogram(
                x=deals_prev["quoted_amount"],
                name="Prior",
                opacity=0.55,
                marker_color=COLORS["muted_gray"],
            )
        )
    if not _is_empty(deals_curr):
        fig.add_trace(
            go.Histogram(
                x=deals_curr["quoted_amount"],
                name="Current",
                opacity=0.55,
                marker_color=COLORS["primary_blue"],
            )
        )
    fig.update_layout(template="plotly_white", barmode="overlay", height=320, margin=dict(l=30, r=20, t=30, b=20))
    _plot(fig, key=f"{key_prefix}-hist")

    median_prev = float(deals_prev["quoted_amount"].median()) if not _is_empty(deals_prev) else np.nan
    median_curr = float(deals_curr["quoted_amount"].median()) if not _is_empty(deals_curr) else np.nan
    if not pd.isna(median_prev) and median_prev != 0:
        delta_pct = (median_curr - median_prev) / median_prev * 100
    else:
        delta_pct = np.nan

    small_prev = float((deals_prev["quoted_amount"] < 1000).mean() * 100) if not _is_empty(deals_prev) else np.nan
    small_curr = float((deals_curr["quoted_amount"] < 1000).mean() * 100) if not _is_empty(deals_curr) else np.nan

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Median Deal (Prior)", fmt_currency(median_prev))
    with c2:
        st.metric("Median Deal (Current)", fmt_currency(median_curr), delta=fmt_percent(delta_pct))
    with c3:
        st.metric("Jobs < $1,000", f"{fmt_percent(small_prev)} -> {fmt_percent(small_curr)}")


def render_staffing_panel(staffing_df: pd.DataFrame, key_prefix: str = "rd-staff") -> None:
    """Render staffing trend and continuity charts."""
    if _is_empty(staffing_df):
        st.info("Insufficient data for staffing panel")
        return

    work = staffing_df.sort_values("month_key").copy()
    c1, c2 = st.columns(2)

    with c1:
        fig_active = go.Figure()
        fig_active.add_trace(
            go.Scatter(
                x=work["month_key"],
                y=work["active_staff_count"],
                mode="lines+markers",
                line=dict(color=COLORS["primary_blue"], width=2),
                name="Active Staff",
            )
        )
        fig_active.update_layout(template="plotly_white", height=300, margin=dict(l=30, r=20, t=30, b=20), title="Active Staff Count")
        _plot(fig_active, key=f"{key_prefix}-active")

    with c2:
        fig_cont = go.Figure()
        fig_cont.add_trace(
            go.Scatter(
                x=work["month_key"],
                y=work["staff_continuity_pct"],
                mode="lines+markers",
                line=dict(color=COLORS["green_positive"], width=2),
                name="Staff Continuity %",
            )
        )
        fig_cont.update_layout(template="plotly_white", height=300, margin=dict(l=30, r=20, t=30, b=20), title="Staff Continuity")
        _plot(fig_cont, key=f"{key_prefix}-continuity")


def render_pricing_discipline(rate_trend_df: pd.DataFrame, key_prefix: str = "rd-pricing") -> None:
    """Render quote rate vs realised rate with shaded gap."""
    if _is_empty(rate_trend_df):
        st.info("Insufficient data for pricing discipline")
        return

    work = rate_trend_df.sort_values("month_key").copy()

    fig = go.Figure()
    for _, row in work.iterrows():
        if pd.isna(row.get("month_key")):
            continue
        quote_rate = _safe_float(row.get("quote_rate", np.nan), np.nan)
        realised_rate = _safe_float(row.get("realised_rate", np.nan), np.nan)
        if pd.isna(quote_rate) or pd.isna(realised_rate):
            continue
        color = "rgba(40, 167, 69, 0.16)" if realised_rate >= quote_rate else "rgba(220, 53, 69, 0.16)"
        fig.add_shape(
            type="rect",
            x0=row["month_key"] - pd.Timedelta(days=10),
            x1=row["month_key"] + pd.Timedelta(days=10),
            y0=min(quote_rate, realised_rate),
            y1=max(quote_rate, realised_rate),
            line=dict(width=0),
            fillcolor=color,
            layer="below",
        )

    fig.add_trace(
        go.Scatter(
            x=work["month_key"],
            y=work["quote_rate"],
            mode="lines+markers",
            line=dict(color=COLORS["muted_gray"], width=2),
            name="Quote Rate",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=work["month_key"],
            y=work["realised_rate"],
            mode="lines+markers",
            line=dict(color=COLORS["primary_blue"], width=2),
            name="Realised Rate",
        )
    )
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=30, r=20, t=30, b=20))
    _plot(fig, key=f"{key_prefix}-trend")


def render_task_mix_shift(task_mix_df: pd.DataFrame, key_prefix: str = "rd-taskmix") -> None:
    """Render task share divergence bar chart."""
    if _is_empty(task_mix_df):
        st.info("Insufficient data for task mix shift")
        return

    work = task_mix_df.sort_values("divergence_pp", ascending=True).copy()
    colors = np.where(work["divergence_pp"] >= 0, COLORS["green_positive"], COLORS["red_negative"])
    fig = go.Figure(
        go.Bar(
            x=work["divergence_pp"],
            y=work["task_name"],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=30, r=20, t=30, b=20), xaxis_title="Divergence (pp)")
    _plot(fig, key=f"{key_prefix}-bars")


def render_hypothesis_scorecard(scorecard_df: pd.DataFrame) -> None:
    """Render styled hypothesis table."""
    if _is_empty(scorecard_df):
        st.info("Insufficient data for hypothesis scorecard")
        return

    color_map = {
        "🔴 Strong": "#fde2e2",
        "🟡 Moderate": "#fff7d6",
        "🟢 Weak": "#e8f8ed",
        "⚪ Insufficient Data": "#eef2f7",
    }

    def _style_signal(value: str) -> str:
        bg = color_map.get(str(value), "#ffffff")
        return f"background-color: {bg}; font-weight: 600;"

    styled = scorecard_df.copy().style.map(_style_signal, subset=["signal_strength"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_diagnostic_narrative(bundle: dict, service_line: str) -> None:
    """Render auto-generated narrative bullets from bundle outputs."""
    yoy = bundle.get("yoy", pd.DataFrame())
    scorecard = bundle.get("scorecard", pd.DataFrame())

    if isinstance(yoy, pd.DataFrame) and len(yoy) > 0:
        if service_line in ("", "All", None):
            rev_curr = float(yoy["revenue_curr"].sum())
            rev_prev = float(yoy["revenue_prev"].sum())
            label = "Portfolio"
        else:
            row = yoy[yoy["service_line"] == service_line]
            if len(row) == 0:
                rev_curr = float(yoy["revenue_curr"].sum())
                rev_prev = float(yoy["revenue_prev"].sum())
                label = service_line
            else:
                rev_curr = float(row["revenue_curr"].sum())
                rev_prev = float(row["revenue_prev"].sum())
                label = service_line

        rev_delta_pct = ((rev_curr - rev_prev) / rev_prev * 100) if rev_prev != 0 else np.nan
        what_changed = f"**What changed**: {label} revenue changed {fmt_percent(rev_delta_pct)} YoY."
    else:
        what_changed = "**What changed**: Insufficient YoY data to quantify the change."

    if isinstance(scorecard, pd.DataFrame) and len(scorecard) > 0:
        strong_or_mod = scorecard[scorecard["signal_strength"].isin(["🔴 Strong", "🟡 Moderate"])].copy()
        if len(strong_or_mod) > 0:
            top = strong_or_mod.head(2)["hypothesis"].tolist()
            drivers = ", ".join(top)
            likely = f"**Most likely drivers**: {drivers}."
        else:
            likely = "**Most likely drivers**: No strong or moderate signals detected."

        insufficient = scorecard[scorecard["signal_strength"] == "⚪ Insufficient Data"]["hypothesis"].tolist()
        if len(insufficient) > 0:
            next_steps = (
                "**What to investigate next month**: Improve data capture for "
                + ", ".join(insufficient)
                + "."
            )
        else:
            next_steps = "**What to investigate next month**: Re-check if current signal hierarchy persists with fresh data."
    else:
        likely = "**Most likely drivers**: Insufficient scorecard data."
        next_steps = "**What to investigate next month**: Fill missing columns for diagnostics coverage."

    st.markdown(f"- {what_changed}")
    st.markdown(f"- {likely}")
    st.markdown(f"- {next_steps}")
