"""
SG Job Profitability Operating System

Main entry point for Streamlit app.
"""
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.config import TABLE_FILES, config
from src.data.loader import load_fact_timesheet, get_data_status
from src.data.schema import validate_schema
from src.ui.state import init_state

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Job Profitability OS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_home_theme() -> None:
    """Inject homepage visual theme aligned with delivery page design language."""
    st.markdown(
        """
        <style>
        :root {
            --dc-border: #e4e7ec;
            --dc-muted: #667085;
            --dc-ink: #0f172a;
            --dc-selected: #2563eb;
            --dc-surface: #f8fafc;
            --dc-soft: #eef2f7;
        }
        .stApp {
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .home-hero {
            background: linear-gradient(135deg, #f7faff 0%, #eef3fb 52%, #f8fafc 100%);
            border: 1px solid #dbe4f0;
            border-radius: 14px;
            padding: 18px 20px;
            margin: 2px 0 14px 0;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }
        .home-eyebrow {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #1d4ed8;
            margin-bottom: 0.28rem;
        }
        .home-title {
            font-size: clamp(1.8rem, 2.4vw, 2.7rem);
            line-height: 1.08;
            font-weight: 760;
            letter-spacing: -0.02em;
            color: var(--dc-ink);
            margin: 0 0 0.42rem 0;
        }
        .home-subtitle {
            font-size: 0.98rem;
            color: #334155;
            margin: 0 0 0.75rem 0;
        }
        .home-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
        }
        .home-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.2rem 0.58rem;
            border: 1px solid #cbd5e1;
            background: rgba(255, 255, 255, 0.78);
            color: #334155;
            font-size: 0.73rem;
            font-weight: 650;
        }
        .home-chip-ready {
            background: #ecfdf3;
            border-color: #86efac;
            color: #166534;
        }
        .home-chip-alert {
            background: #fff1f2;
            border-color: #fecdd3;
            color: #9f1239;
        }
        .home-divider {
            height: 1px;
            margin: 0.25rem 0 0.8rem 0;
            background: linear-gradient(90deg, #dbe4f0 0%, rgba(219, 228, 240, 0.28) 100%);
        }
        .home-section-title {
            font-size: 1.55rem;
            line-height: 1.15;
            font-weight: 740;
            letter-spacing: -0.01em;
            color: #111827;
            margin-bottom: 0.2rem;
        }
        .home-section-caption {
            font-size: 0.84rem;
            color: var(--dc-muted);
            margin-bottom: 0.62rem;
        }
        .home-subsection-title {
            font-size: 1.06rem;
            line-height: 1.2;
            font-weight: 680;
            color: #1f2937;
            margin-bottom: 0.32rem;
        }
        .home-soft-divider {
            height: 1px;
            margin: 0.72rem 0;
            background: linear-gradient(90deg, #d6deea 0%, rgba(214, 222, 234, 0.18) 100%);
        }
        div[data-testid="stPageLink"] a {
            margin: 0 !important;
            padding: 0.5rem 0.62rem;
            border: 1px solid var(--dc-border);
            border-radius: 10px;
            background: #ffffff;
            text-decoration: none;
            transition: border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stPageLink"] a:hover {
            border-color: #c9d3e2;
            background: #f9fbff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stPageLink"] p {
            font-size: 0.95rem;
            color: #1f2937;
            font-weight: 610;
        }
        div[data-testid="metric-container"] {
            border: 1px solid var(--dc-border);
            border-radius: 10px;
            padding: 0.56rem 0.72rem;
            background: #ffffff;
        }
        div[data-testid="metric-container"] > label {
            font-size: 0.73rem !important;
            letter-spacing: 0.01em;
            color: var(--dc-muted) !important;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.93rem !important;
            line-height: 1.08 !important;
            color: var(--dc-ink);
            font-weight: 690;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.84rem !important;
        }
        details[data-testid="stExpander"] {
            border: 1px solid var(--dc-border);
            border-radius: 10px;
            background: #ffffff;
            overflow: hidden;
        }
        details[data-testid="stExpander"] > summary {
            background: var(--dc-surface);
        }
        @media (max-width: 900px) {
            .home-title {
                font-size: 2rem;
            }
            .home-hero {
                padding: 14px 15px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_fingerprint_rows() -> list[dict]:
    """Build file inventory rows for the data fingerprint table."""
    rows = []
    for filename in TABLE_FILES.values():
        for ext in ("parquet", "csv"):
            path = config.processed_dir / f"{filename}.{ext}"
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                rows.append(
                    {
                        "file": path.name,
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                        "modified_utc": mtime.strftime("%Y-%m-%d %H:%M"),
                    }
                )
    return rows


def main():
    """Main app entry point."""

    # Initialize session state
    init_state()
    inject_home_theme()

    # Check data availability
    status = get_data_status()
    processed_ready = sum(
        1
        for info in status["processed"].values()
        if info["parquet_exists"] or info["csv_exists"]
    )
    marts_ready = sum(
        1 for info in status["marts"].values() if info["parquet_exists"] or info["csv_exists"]
    )
    fact_available = (
        status["processed"]["fact_timesheet"]["parquet_exists"] or
        status["processed"]["fact_timesheet"]["csv_exists"]
    )

    readiness_class = "home-chip-ready" if fact_available else "home-chip-alert"
    readiness_label = "Data Layer Ready" if fact_available else "Setup Required"
    st.markdown(
        f"""
        <div class="home-hero">
            <div class="home-eyebrow">Operating Layer</div>
            <div class="home-title">Job Profitability Operating System</div>
            <div class="home-subtitle">Company → Department → Category → Task/Staff</div>
            <div class="home-chip-row">
                <span class="home-chip {readiness_class}">{readiness_label}</span>
                <span class="home-chip">Processed Tables: {processed_ready}/{len(status["processed"])}</span>
                <span class="home-chip">Precomputed Marts: {marts_ready}/{len(status["marts"])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not fact_available:
        st.error("No data found.")
        with st.container(border=True):
            st.markdown("### Setup Required")
            st.markdown(f"Please place your source files in `{config.processed_dir}`.")
            st.markdown(
                """
                **Required files**
                - `fact_timesheet_day_enriched.parquet` (or `.csv`)
                - `fact_job_task_month.parquet` (or `.csv`)

                **Optional files**
                - `audit_revenue_reconciliation_job_month.parquet`
                - `audit_unallocated_revenue.parquet`
                """
            )
            st.info(
                "To generate these files, run `01_parse_and_unify_job_profitability.ipynb`."
            )
        st.info("Once data is in place, refresh this page.")
        return

    with st.expander("Data fingerprint", expanded=False):
        rows = build_fingerprint_rows()
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info(f"No files found in {config.processed_dir}.")

    # Load and validate data
    with st.spinner("Loading data..."):
        try:
            df = load_fact_timesheet()
            result = validate_schema(df, "fact_timesheet", strict=False)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    # Show validation status
    if not result["is_valid"]:
        st.warning(f"Missing required columns: {result['missing_required']}")
        st.info("Some features may be limited.")

    total_rows = len(df)
    jobs_count = df["job_no"].nunique() if "job_no" in df.columns else None
    staff_count = df["staff_name"].nunique() if "staff_name" in df.columns else None
    date_range = "—"
    if "month_key" in df.columns and not df["month_key"].isna().all():
        min_date = df["month_key"].min()
        max_date = df["month_key"].max()
        if hasattr(min_date, "strftime") and hasattr(max_date, "strftime"):
            date_range = f"{min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}"
        else:
            date_range = f"{min_date} - {max_date}"

    total_revenue = float(df["rev_alloc"].fillna(0).sum()) if "rev_alloc" in df.columns else None
    total_cost = float(df["base_cost"].fillna(0).sum()) if "base_cost" in df.columns else None
    margin_value = None
    margin_pct = None
    if total_revenue is not None and total_cost is not None:
        margin_value = total_revenue - total_cost
        margin_pct = (margin_value / total_revenue * 100) if total_revenue > 0 else 0.0
    total_hours = float(df["hours_raw"].fillna(0).sum()) if "hours_raw" in df.columns else None

    st.markdown('<div class="home-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.05, 2.35], gap="large")

    with col1:
        st.markdown('<div class="home-section-title">Quick Links</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="home-section-caption">Navigate directly to each operating module.</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/1_Executive_Summary.py", label="Executive Summary", icon="📈")
        st.page_link("pages/2_Quote_Builder.py", label="Quote Builder", icon="📝")
        st.page_link("pages/4_Active_Delivery.py", label="Active Delivery", icon="🎯")
        st.page_link("pages/5_Revenue_Reconciliation.py", label="Revenue Reconciliation", icon="💰")
        st.page_link("pages/6_Job_Mix_and_Demand.py", label="Job Mix & Demand", icon="📊")
        st.page_link("pages/9_Revenue_Decline_Diagnostics.py", label="Revenue Decline Diagnostics", icon="🔍")
        st.page_link("pages/7_Client_Profitability_LTV.py", label="Client Profitability & LTV", icon="🧭")
        st.page_link("pages/7_Data_Quality_QA.py", label="Data Quality & QA", icon="✅")
        st.page_link("pages/8_Glossary_Method.py", label="Glossary & Method", icon="📖")

    with col2:
        st.markdown('<div class="home-section-title">Data Overview</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="home-section-caption">Current scope, key commercial totals, and operating baseline.</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Total Rows", f"{total_rows:,}")

        with c2:
            st.metric("Jobs", f"{jobs_count:,}" if jobs_count is not None else "—")

        with c3:
            st.metric("Staff", f"{staff_count:,}" if staff_count is not None else "—")

        with c4:
            st.metric("Date Range", date_range)

        st.markdown('<div class="home-soft-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="home-subsection-title">Key Metrics</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Total Revenue",
                f"${total_revenue:,.0f}" if total_revenue is not None else "—",
            )

        with m2:
            st.metric("Total Cost", f"${total_cost:,.0f}" if total_cost is not None else "—")

        with m3:
            if margin_value is not None and margin_pct is not None:
                st.metric("Margin", f"${margin_value:,.0f} ({margin_pct:.1f}%)")
            else:
                st.metric("Margin", "—")

        with m4:
            st.metric("Total Hours", f"{total_hours:,.0f}" if total_hours is not None else "—")

    # Data status
    st.markdown('<div class="home-divider"></div>', unsafe_allow_html=True)
    with st.expander("Data Status"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Processed Tables**")
            for key, info in status["processed"].items():
                icon = "✅" if info["parquet_exists"] or info["csv_exists"] else "❌"
                format_used = (
                    "parquet"
                    if info["parquet_exists"]
                    else "csv"
                    if info["csv_exists"]
                    else "missing"
                )
                st.markdown(f"{icon} `{key}` ({format_used})")

        with col2:
            st.markdown("**Precomputed Marts**")
            for key, info in status["marts"].items():
                icon = "✅" if info["parquet_exists"] or info["csv_exists"] else "⚪"
                format_used = (
                    "parquet"
                    if info["parquet_exists"]
                    else "csv"
                    if info["csv_exists"]
                    else "not built"
                )
                st.markdown(f"{icon} `{key}` ({format_used})")

            if not any(
                info["parquet_exists"] or info["csv_exists"]
                for info in status["marts"].values()
            ):
                st.info("Run `python scripts/build_marts.py` to build marts for faster loading.")


if __name__ == "__main__":
    main()
