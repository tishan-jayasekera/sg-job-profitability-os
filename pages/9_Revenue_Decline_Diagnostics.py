"""Revenue Decline Diagnostics page."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cohorts import filter_by_time_window
from src.data.loader import load_fact_timesheet
from src.data.semantic import exclude_leave, get_category_col
from src.metrics.revenue_diagnostics import (
    build_diagnostics_bundle,
    normalize_service_line_labels,
)
from src.ui.revenue_diagnostics_components import (
    render_client_bridge,
    render_deal_size_comparison,
    render_diagnostic_narrative,
    render_hypothesis_scorecard,
    render_kpi_row,
    render_pricing_discipline,
    render_revenue_bridge_waterfall,
    render_staffing_panel,
    render_task_mix_shift,
    render_trend_panel,
)


st.set_page_config(page_title="Revenue Decline Diagnostics", page_icon="🔍", layout="wide")

WINDOW_SIZE = {"1m": 1, "3m": 3, "6m": 6, "12m": 12}
WINDOW_FILTER = {"1m": "3m", "3m": "6m", "6m": "12m", "12m": "24m"}


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rd-border: #e4e7ec;
            --rd-muted: #667085;
            --rd-ink: #0f172a;
        }
        .stApp {
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .rd-hero {
            background: linear-gradient(135deg, #f7faff 0%, #eef3fb 52%, #f8fafc 100%);
            border: 1px solid #dbe4f0;
            border-radius: 14px;
            padding: 18px 20px;
            margin: 0.1rem 0 1rem 0;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
        }
        .rd-eyebrow {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #1d4ed8;
            margin-bottom: 0.24rem;
        }
        .rd-title {
            font-size: clamp(1.6rem, 2.1vw, 2.35rem);
            line-height: 1.1;
            font-weight: 760;
            letter-spacing: -0.02em;
            color: var(--rd-ink);
            margin: 0 0 0.3rem 0;
        }
        .rd-subtitle {
            font-size: 0.95rem;
            color: #334155;
            margin: 0;
        }
        div[data-testid="metric-container"] {
            border: 1px solid var(--rd-border);
            border-radius: 10px;
            padding: 0.55rem 0.72rem;
            background: #ffffff;
        }
        div[data-testid="metric-container"] > label {
            font-size: 0.73rem !important;
            color: var(--rd-muted) !important;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            line-height: 1.05 !important;
            color: var(--rd-ink);
            font-weight: 680;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.3rem;
            border-bottom: 1px solid var(--rd-border);
            margin-bottom: 0.5rem;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            height: 34px;
            padding: 0 0.95rem;
            border-radius: 8px 8px 0 0;
            color: var(--rd-muted);
            font-weight: 620;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            color: #1d4ed8;
            background: #eff6ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="rd-hero">
            <div class="rd-eyebrow">Diagnostics</div>
            <div class="rd-title">Revenue Decline Diagnostics</div>
            <div class="rd-subtitle">Decomposing revenue decline drivers by service line: volume, price, clients, staffing, and mix.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _is_error_dict(value) -> bool:
    return isinstance(value, dict) and value.get("status") == "error"


def _prepare_page_df() -> pd.DataFrame:
    df = load_fact_timesheet()
    if len(df) == 0:
        return df

    work = df.copy()
    work["month_key"] = pd.to_datetime(work["month_key"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    work = exclude_leave(work).copy()
    category_col = get_category_col(work)
    work = normalize_service_line_labels(work, category_col)
    return work


def _resolve_as_of_index(all_months: list[pd.Timestamp], as_of_month: pd.Timestamp) -> int:
    if len(all_months) == 0:
        return -1
    if as_of_month in all_months:
        return all_months.index(as_of_month)

    candidates = [m for m in all_months if m <= as_of_month]
    if len(candidates) > 0:
        return all_months.index(candidates[-1])
    return len(all_months) - 1


def _compute_periods(df: pd.DataFrame, as_of_month: pd.Timestamp, analysis_window: str) -> tuple[list[pd.Timestamp], list[pd.Timestamp], pd.DataFrame]:
    window_for_filter = WINDOW_FILTER.get(analysis_window, "12m")
    scoped = filter_by_time_window(df, window_for_filter, date_col="month_key", reference_date=as_of_month)
    scoped = scoped.copy()
    scoped["month_key"] = pd.to_datetime(scoped["month_key"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    as_of_month = pd.Timestamp(as_of_month).to_period("M").to_timestamp()
    yoy_prev_month = (as_of_month - pd.DateOffset(years=1)).to_period("M").to_timestamp()

    # Ensure YoY reference month is retained even for short analysis windows.
    if "month_key" in df.columns:
        source = df.copy()
        source["month_key"] = pd.to_datetime(source["month_key"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        if yoy_prev_month in set(source["month_key"].dropna().unique().tolist()):
            scoped = pd.concat([scoped, source[source["month_key"] == yoy_prev_month].copy()], ignore_index=True)

    all_months = sorted(scoped["month_key"].dropna().unique().tolist())
    if len(all_months) == 0:
        return [], [], scoped.iloc[0:0].copy()

    as_of_idx = _resolve_as_of_index(all_months, as_of_month)
    window_size = WINDOW_SIZE.get(analysis_window, 3)

    current_months = all_months[max(0, as_of_idx - window_size + 1) : as_of_idx + 1]
    prior_months = all_months[max(0, as_of_idx - 2 * window_size + 1) : as_of_idx - window_size + 1]

    keep_months = sorted(set(current_months + prior_months + [yoy_prev_month]))
    scoped = scoped[scoped["month_key"].isin(keep_months)].copy() if len(keep_months) > 0 else scoped.iloc[0:0].copy()

    return current_months, prior_months, scoped


def _prepare_scorecard_with_guidance(scorecard_df: pd.DataFrame) -> pd.DataFrame:
    if scorecard_df is None or len(scorecard_df) == 0:
        return scorecard_df

    out = scorecard_df.copy()
    enable_map = {
        "Demand decline": "job_no",
        "Price erosion": "rev_alloc",
        "Client churn (reputation)": "client",
        "Staffing/selling changes": "staff_name",
        "Mix shift to low-value": "quoted_amount_total",
    }

    mask = out["signal_strength"] == "⚪ Insufficient Data"
    for idx in out[mask].index:
        hypothesis = str(out.at[idx, "hypothesis"])
        col = enable_map.get(hypothesis, "required")
        current = str(out.at[idx, "interpretation"])
        out.at[idx, "interpretation"] = f"{current} Enable this signal by ensuring the `{col}` column is populated."

    return out


def _render_bundle_errors(bundle: dict) -> None:
    for key, value in bundle.items():
        if _is_error_dict(value):
            st.warning(f"{key}: {value.get('reason', 'unknown error')}")


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value)).strip("-")


def main() -> None:
    _inject_theme()
    _render_header()

    df = _prepare_page_df()
    if len(df) == 0:
        st.info("No data available for diagnostics.")
        return

    st.sidebar.markdown("### Controls")

    departments = ["All"] + sorted(df["department_final"].dropna().unique().tolist()) if "department_final" in df.columns else ["All"]
    department = st.sidebar.selectbox("Department", departments)

    page_df = df.copy()
    if department != "All" and "department_final" in page_df.columns:
        page_df = page_df[page_df["department_final"] == department].copy()

    if len(page_df) == 0:
        st.info("No records after department filter.")
        return

    all_lines = sorted(page_df["service_line"].dropna().astype(str).unique().tolist())
    preferred = ["Marketing Automation", "CRM", "Landing Pages"]
    service_line_options = ["All"]
    for name in preferred + all_lines:
        if name not in service_line_options:
            service_line_options.append(name)

    service_line = st.sidebar.selectbox("Service Line", service_line_options)

    month_options = sorted(page_df["month_key"].dropna().unique().tolist())
    if len(month_options) == 0:
        st.info("No month_key values found.")
        return

    as_of_month = st.sidebar.selectbox(
        "As-of Month",
        options=month_options,
        index=len(month_options) - 1,
        format_func=lambda x: pd.Timestamp(x).strftime("%b %Y"),
    )
    as_of_month = pd.Timestamp(as_of_month).to_period("M").to_timestamp()

    analysis_window = st.sidebar.selectbox("Analysis Window", ["1m", "3m", "6m", "12m"], index=2)

    current_months, prior_months, df_window = _compute_periods(page_df, as_of_month, analysis_window)

    if len(df_window) == 0 or len(current_months) == 0:
        st.info("Insufficient data for selected period.")
        return

    bundle = build_diagnostics_bundle(df_window, as_of_month, service_line, current_months, prior_months)
    _render_bundle_errors(bundle)

    deep_line = service_line

    st.markdown("### Section 1: Overview")
    if service_line == "All":
        yoy_df = bundle.get("yoy", pd.DataFrame())
        decomp_df = bundle.get("decomp", pd.DataFrame())

        if isinstance(yoy_df, pd.DataFrame) and len(yoy_df) > 0:
            table = yoy_df[
                [
                    "service_line",
                    "revenue_curr",
                    "revenue_prev",
                    "delta_revenue",
                    "jobs_curr",
                    "jobs_prev",
                    "arpj_curr",
                    "arpj_prev",
                    "rev_yoy_pct",
                    "jobs_yoy_pct",
                    "arpj_yoy_pct",
                ]
            ].copy()
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.info("Insufficient data for all-line YoY comparison.")

        if isinstance(decomp_df, pd.DataFrame) and len(decomp_df) > 0:
            total_row = pd.Series(
                {
                    "volume_effect": float(decomp_df["volume_effect"].sum()),
                    "price_effect": float(decomp_df["price_effect"].sum()),
                    "interaction_effect": float(decomp_df["interaction_effect"].sum()),
                    "delta_revenue": float(decomp_df["delta_revenue"].sum()),
                }
            )
            render_revenue_bridge_waterfall(total_row, key_prefix="overview-bridge")
        else:
            st.info("Insufficient data for aggregate bridge.")

        if isinstance(yoy_df, pd.DataFrame) and len(yoy_df) > 0:
            drill_options = sorted(yoy_df["service_line"].dropna().astype(str).unique().tolist())
            if len(drill_options) > 0:
                deep_line = st.selectbox("Select service line to drill into Section 2", options=drill_options)
    else:
        yoy_df = bundle.get("yoy", pd.DataFrame())
        if isinstance(yoy_df, pd.DataFrame) and len(yoy_df) > 0:
            render_kpi_row(yoy_df.iloc[0])

    deep_bundle = bundle
    if deep_line not in ("", "All", None):
        deep_bundle = build_diagnostics_bundle(df_window, as_of_month, deep_line, current_months, prior_months)
        _render_bundle_errors(deep_bundle)

    st.markdown("### Section 2: Deep Dive")
    if deep_line in ("", "All", None):
        st.info("Select a specific service line to view deep-dive diagnostics.")
    else:
        tabs = st.tabs(["📊 Trends", "🔀 Volume × Price", "👥 Client Bridge", "💰 Deal Sizes", "🧑‍💼 Staffing", "💵 Pricing", "📋 Task Mix"])

        with tabs[0]:
            monthly_df = deep_bundle.get("monthly", pd.DataFrame())
            if isinstance(monthly_df, pd.DataFrame) and len(monthly_df) > 0:
                render_trend_panel(monthly_df, deep_line, key_prefix=f"deep-trend-{_slug(deep_line)}")
            else:
                st.info("Insufficient data for this section")

        with tabs[1]:
            decomp_df = deep_bundle.get("decomp", pd.DataFrame())
            if isinstance(decomp_df, pd.DataFrame) and len(decomp_df) > 0:
                render_revenue_bridge_waterfall(decomp_df.iloc[0], key_prefix=f"deep-bridge-{_slug(deep_line)}")
            else:
                st.info("Insufficient data for this section")

        with tabs[2]:
            client_bridge = deep_bundle.get("client_bridge", {})
            if _is_error_dict(client_bridge):
                st.info("Insufficient data for this section")
            else:
                render_client_bridge(
                    client_bridge.get("bridge", pd.DataFrame()),
                    client_bridge.get("top_clients", pd.DataFrame()),
                    key_prefix=f"deep-client-{_slug(deep_line)}",
                )

        with tabs[3]:
            deals_curr = deep_bundle.get("deal_sizes_curr", pd.DataFrame())
            deals_prev = deep_bundle.get("deal_sizes_prev", pd.DataFrame())
            if isinstance(deals_curr, pd.DataFrame) and isinstance(deals_prev, pd.DataFrame):
                render_deal_size_comparison(deals_curr, deals_prev, key_prefix=f"deep-deal-{_slug(deep_line)}")
            else:
                st.info("Insufficient data for this section")

        with tabs[4]:
            staffing_df = deep_bundle.get("staffing", pd.DataFrame())
            if isinstance(staffing_df, pd.DataFrame) and len(staffing_df) > 0:
                render_staffing_panel(staffing_df, key_prefix=f"deep-staff-{_slug(deep_line)}")
            else:
                st.info("Insufficient data for this section")

        with tabs[5]:
            rate_df = deep_bundle.get("rate_trend", pd.DataFrame())
            if isinstance(rate_df, pd.DataFrame) and len(rate_df) > 0:
                render_pricing_discipline(rate_df, key_prefix=f"deep-pricing-{_slug(deep_line)}")
            else:
                st.info("Insufficient data for this section")

        with tabs[6]:
            task_mix_df = deep_bundle.get("task_mix", pd.DataFrame())
            if isinstance(task_mix_df, pd.DataFrame) and len(task_mix_df) > 0:
                render_task_mix_shift(task_mix_df, key_prefix=f"deep-taskmix-{_slug(deep_line)}")
            else:
                st.info("Insufficient data for this section")

    st.markdown("### Section 3: Hypothesis Scorecard")
    scorecard_df = deep_bundle.get("scorecard", pd.DataFrame())
    if isinstance(scorecard_df, pd.DataFrame) and len(scorecard_df) > 0:
        scorecard_df = _prepare_scorecard_with_guidance(scorecard_df)
        render_hypothesis_scorecard(scorecard_df)
    else:
        st.info("Insufficient data for this section")

    render_diagnostic_narrative(deep_bundle, deep_line)


if __name__ == "__main__":
    main()
