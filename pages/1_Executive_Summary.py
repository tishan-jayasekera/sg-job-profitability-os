"""
Executive Summary: Causal Root-Cause Drill Engine (v3.1)
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.data.cohorts import filter_by_time_window
from src.data.semantic import get_category_col, safe_quote_job_task
from src.metrics.profitability import classify_department, compute_department_scorecard
from src.metrics.quote_delivery import (
    compute_scope_creep,
    compute_task_overrun_consistency,
    get_overrun_jobs_for_task,
)
from src.ui.formatting import build_job_name_lookup, format_job_label, fmt_percent, fmt_currency


# =============================================================================
# THEME
# =============================================================================

COLORS = {
    "accretive": "#28a745",
    "mixed": "#ffc107",
    "erosive": "#dc3545",
    "subsidiser": "#28a745",
    "margin_erosive": "#dc3545",
    "neutral": "#6c757d",
    "border": "#e9ecef",
    "background": "#f8f9fa",
    "primary": "#1565c0",
    "primary_light": "#e3f2fd",
    "text_dark": "#1a1a1a",
    "text_medium": "#666666",
    "text_light": "#888888",
}

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the enriched timesheet fact table."""
    try:
        return load_fact_timesheet()
    except Exception:
        data_path = Path(__file__).parent.parent / "data" / "processed" / "fact_timesheet_day_enriched.parquet"
        if not data_path.exists():
            data_path = data_path.with_suffix(".csv")

        if data_path.exists():
            if data_path.suffix == ".parquet":
                return pd.read_parquet(data_path)
            return pd.read_csv(data_path, parse_dates=["work_date"])

        st.error(
            "Could not find fact_timesheet_day_enriched in either `src/data/processed` "
            "or `data/processed`. Check DATA_DIR or place the file in the expected folder."
        )
        st.stop()


# =============================================================================
# GLOBAL CONTEXT
# =============================================================================


def render_global_control_bar() -> Tuple[pd.Timestamp, pd.Timestamp, str]:
    """
    Polished global control bar with professional styling.
    Returns (start_date, end_date, job_state).
    """
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"]:has(> div > span.control-bar-anchor) {
            position: sticky;
            top: 4rem;
            z-index: 100;
            background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }
        .control-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.35rem;
        }
        .context-pill {
            display: inline-block;
            background: #e3f2fd;
            color: #1565c0;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def segmented_or_radio(label: str, options: List[str], default: str, key: str):
        segmented = getattr(st, "segmented_control", None)
        if callable(segmented):
            return segmented(
                label,
                options=options,
                default=default,
                key=key,
                label_visibility="collapsed",
            )
        return st.radio(
            label,
            options=options,
            horizontal=True,
            index=options.index(default) if default in options else 0,
            key=key,
            label_visibility="collapsed",
        )

    legacy_map = {
        "Last 30 Days": "30D",
        "Last 90 Days": "90D",
        "Last 6 Months": "6M",
        "Last 12 Months": "12M",
    }
    if st.session_state.get("time_preset") in legacy_map:
        st.session_state["time_preset"] = legacy_map[st.session_state["time_preset"]]
    if st.session_state.get("time_preset") not in {"30D", "90D", "6M", "12M", "Custom"}:
        st.session_state["time_preset"] = "12M"
    if st.session_state.get("job_state") not in {"Active", "Completed", "All"}:
        st.session_state["job_state"] = "Completed"

    with st.container():
        st.markdown('<span class="control-bar-anchor"></span>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            st.markdown('<p class="control-label">📅 Time Period</p>', unsafe_allow_html=True)
            time_preset = segmented_or_radio(
                "Period",
                options=["30D", "90D", "6M", "12M", "Custom"],
                default="12M",
                key="time_preset",
            )

            today = pd.Timestamp.today().normalize()
            preset_map = {
                "30D": (today - pd.Timedelta(days=30), today),
                "90D": (today - pd.Timedelta(days=90), today),
                "6M": (today - pd.Timedelta(days=180), today),
                "12M": (today - pd.Timedelta(days=365), today),
            }

            if time_preset == "Custom":
                c1, c2 = st.columns(2)
                with c1:
                    start_date = pd.Timestamp(
                        st.date_input(
                            "From",
                            value=today - pd.Timedelta(days=90),
                            label_visibility="collapsed",
                        )
                    )
                with c2:
                    end_date = pd.Timestamp(
                        st.date_input("To", value=today, label_visibility="collapsed")
                    )
            else:
                start_date, end_date = preset_map.get(time_preset, preset_map["90D"])

        with col2:
            st.markdown('<p class="control-label">🏷️ Job State</p>', unsafe_allow_html=True)
            job_state = segmented_or_radio(
                "State",
                options=["Active", "Completed", "All"],
                default="Completed",
                key="job_state",
            )

        with col3:
            st.markdown('<p class="control-label">📊 Current View</p>', unsafe_allow_html=True)
            period_str = f"{start_date.strftime('%d %b')} – {end_date.strftime('%d %b %Y')}"
            st.markdown(
                f'<span class="context-pill">{job_state} • {period_str}</span>',
                unsafe_allow_html=True,
            )

    return start_date, end_date, job_state


def apply_global_context(
    df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, job_state: str
) -> pd.DataFrame:
    """
    Apply global context (time period + job state) to raw data.
    """
    date_col = "work_date" if "work_date" in df.columns else "month_key"
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    mask_period = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    df_period = df[mask_period].copy()

    if df_period.empty:
        return df_period

    # Filter by job state
    if "job_status" in df_period.columns:
        status_series = df_period["job_status"].astype(str).str.lower()
        if job_state == "Active":
            mask = ~status_series.str.contains("completed", na=False)
        elif job_state == "Completed":
            mask = status_series.str.contains("completed", na=False)
        else:
            mask = pd.Series(True, index=df_period.index)
    else:
        last_activity = df_period.groupby("job_no")[date_col].max()
        cutoff = end_date - pd.Timedelta(days=14)

        active_jobs = last_activity[last_activity >= cutoff].index
        completed_jobs = last_activity[last_activity < cutoff].index

        if job_state == "Active":
            mask = df_period["job_no"].isin(active_jobs)
        elif job_state == "Completed":
            mask = df_period["job_no"].isin(completed_jobs)
        else:
            mask = pd.Series(True, index=df_period.index)

    return df_period[mask].copy()


# =============================================================================
# DRILL STATE
# =============================================================================


def init_drill_state():
    """Initialize session state for drill tracking with context."""
    if "drill_path" not in st.session_state:
        st.session_state.drill_path = {
            "department": None,
            "department_context": None,
            "category": None,
            "category_context": None,
            "job": None,
            "task": None,
        }
    else:
        st.session_state.drill_path.setdefault("department", None)
        st.session_state.drill_path.setdefault("department_context", None)
        st.session_state.drill_path.setdefault("category", None)
        st.session_state.drill_path.setdefault("category_context", None)
        st.session_state.drill_path.setdefault("job", None)
        st.session_state.drill_path.setdefault("task", None)

    if "global_context" not in st.session_state:
        st.session_state.global_context = {
            "start_date": pd.Timestamp.today() - pd.Timedelta(days=365),
            "end_date": pd.Timestamp.today(),
            "job_state": "Completed",
        }


def reset_drill_from(level: str):
    """Reset drill state from a given level downward."""
    levels = ["department", "category", "job", "task"]
    if level not in levels:
        return
    idx = levels.index(level)
    for l in levels[idx:]:
        st.session_state.drill_path[l] = None
        if l == "department":
            st.session_state.drill_path["department_context"] = None
        if l == "category":
            st.session_state.drill_path["category_context"] = None


def get_drill_breadcrumb() -> str:
    """Generate breadcrumb string for current drill path."""
    path = st.session_state.drill_path
    parts = ["Portfolio"]
    if path["department"]:
        parts.append(path["department"])
    if path["category"] is not None:
        parts.append(path["category"])
    elif path.get("category_context") is not None:
        parts.append("(Uncategorised)")
    if path["job"]:
        parts.append(path["job"])
    if path["task"]:
        parts.append(path["task"])
    return " → ".join(parts)


def set_department_with_context(dept: str, classification: str, reasons: List[str]):
    """Set department selection WITH required context."""
    st.session_state.drill_path["department"] = dept
    st.session_state.drill_path["department_context"] = {
        "classification": classification,
        "reasons": reasons,
    }
    st.session_state.pop("selected_segment", None)
    st.session_state.drill_path["category"] = None
    st.session_state.drill_path["category_context"] = None
    st.session_state.drill_path["job"] = None
    st.session_state.drill_path["task"] = None


def set_category_with_context(cat: Optional[str], classification: str, reasons: List[str]):
    """Set category selection WITH required context."""
    st.session_state.drill_path["category"] = cat
    st.session_state.drill_path["category_context"] = {
        "classification": classification,
        "reasons": reasons,
    }
    st.session_state.pop("selected_segment", None)
    st.session_state.drill_path["job"] = None
    st.session_state.drill_path["task"] = None


def can_proceed_to_level(level: str) -> bool:
    """Check if prerequisites are met for a given level."""
    path = st.session_state.drill_path

    if level == "category":
        return path["department"] is not None and path["department_context"] is not None
    if level == "job":
        return path["category_context"] is not None
    if level == "task":
        return path["job"] is not None
    return True


def render_breadcrumb():
    """Render clickable breadcrumb that allows backtracking."""
    st.markdown(f"**📍 {get_drill_breadcrumb()}**")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("⬅️ Back", disabled=st.session_state.drill_path["department"] is None):
            for level in reversed(["task", "job", "category", "department"]):
                if st.session_state.drill_path[level] is not None:
                    reset_drill_from(level)
                    st.rerun()
                    break
    with col2:
        if st.button("🏠 Reset to Portfolio"):
            reset_drill_from("department")
            st.rerun()


# =============================================================================
# QUOTE DEDUPE
# =============================================================================


def safe_quote_rollup(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Dedupe quotes to job-task, then aggregate to requested grain.
    """
    job_task = df.groupby(["job_no", "task_name"]).agg(
        quoted_time_total=("quoted_time_total", "first"),
        quoted_amount_total=("quoted_amount_total", "first"),
    ).reset_index()

    result = job_task.groupby(group_cols).agg(
        quoted_time_total=("quoted_time_total", "sum"),
        quoted_amount_total=("quoted_amount_total", "sum"),
    ).reset_index()

    return result


# =============================================================================
# METRICS
# =============================================================================


def compute_job_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute job-level metrics with safe quote deduplication."""
    actuals = df.groupby("job_no").agg(
        actual_hours_job=("hours_raw", "sum"),
        revenue_job=("rev_alloc", "sum"),
        cost_job=("base_cost", "sum"),
        department_final=("department_final", "first"),
        job_category=("job_category", "first"),
        job_status=("job_status", "first"),
    ).reset_index()

    quotes = safe_quote_rollup(df, ["job_no"])
    quotes.columns = ["job_no", "quoted_hours_job", "quoted_amount_job"]

    job_df = actuals.merge(quotes, on="job_no", how="left")
    job_df["quoted_hours_job"] = job_df["quoted_hours_job"].fillna(0)
    job_df["quoted_amount_job"] = job_df["quoted_amount_job"].fillna(0)

    job_df["quote_rate_job"] = np.where(
        job_df["quoted_hours_job"] > 0,
        job_df["quoted_amount_job"] / job_df["quoted_hours_job"],
        np.nan,
    )
    job_df["realised_rate_job"] = np.where(
        job_df["actual_hours_job"] > 0,
        job_df["revenue_job"] / job_df["actual_hours_job"],
        np.nan,
    )

    job_df["hours_variance_job"] = job_df["actual_hours_job"] - job_df["quoted_hours_job"]
    job_df["hours_variance_pct_job"] = np.where(
        job_df["quoted_hours_job"] > 0,
        (job_df["hours_variance_job"] / job_df["quoted_hours_job"]) * 100,
        np.where(job_df["actual_hours_job"] > 0, 100, 0),
    )

    job_df["rate_variance_job"] = job_df["realised_rate_job"] - job_df["quote_rate_job"]
    job_df["rate_capture_pct_job"] = np.where(
        job_df["quote_rate_job"] > 0,
        (job_df["realised_rate_job"] / job_df["quote_rate_job"]) * 100,
        np.nan,
    )

    job_df["margin_job"] = job_df["revenue_job"] - job_df["cost_job"]
    job_df["margin_pct_job"] = np.where(
        job_df["revenue_job"] > 0,
        (job_df["margin_job"] / job_df["revenue_job"]) * 100,
        np.nan,
    )

    job_df["overrun_flag"] = job_df["hours_variance_job"] > 0
    job_df["critical_overrun_flag"] = job_df["hours_variance_pct_job"] > 20
    job_df["has_quote"] = job_df["quoted_hours_job"] > 0

    scope = compute_scope_creep(df, ["job_no"])
    if len(scope) > 0:
        scope = scope.rename(
            columns={
                "unquoted_hours": "unquoted_hours_job",
                "unquoted_share": "unquoted_pct_job",
            }
        )
        job_df = job_df.merge(
            scope[["job_no", "unquoted_hours_job", "unquoted_pct_job"]],
            on="job_no",
            how="left",
        )
        job_df["unquoted_hours_job"] = job_df["unquoted_hours_job"].fillna(0)
        job_df["unquoted_pct_job"] = job_df["unquoted_pct_job"].fillna(0)
    else:
        job_df["unquoted_hours_job"] = 0
        job_df["unquoted_pct_job"] = 0

    return job_df


def compute_portfolio_metrics(job_df: pd.DataFrame) -> Dict:
    """Compute portfolio-level KPIs with weighted rates."""
    quoted_jobs = job_df[job_df["has_quote"]].copy()

    total_quoted_hours = quoted_jobs["quoted_hours_job"].sum()
    total_quoted_amount = quoted_jobs["quoted_amount_job"].sum()
    total_actual_hours = job_df["actual_hours_job"].sum()
    total_revenue = job_df["revenue_job"].sum()
    total_cost = job_df["cost_job"].sum()

    quote_rate_portfolio = total_quoted_amount / total_quoted_hours if total_quoted_hours > 0 else 0
    realised_rate_portfolio = total_revenue / total_actual_hours if total_actual_hours > 0 else 0

    n_jobs = len(job_df)
    n_quoted = len(quoted_jobs)
    n_overrun = int(quoted_jobs["overrun_flag"].sum())
    n_critical = int(quoted_jobs["critical_overrun_flag"].sum())

    margin = total_revenue - total_cost

    return {
        "quote_rate": quote_rate_portfolio,
        "realised_rate": realised_rate_portfolio,
        "rate_variance": realised_rate_portfolio - quote_rate_portfolio,
        "margin": margin,
        "margin_pct": (margin / total_revenue * 100) if total_revenue > 0 else 0,
        "pct_jobs_overrun": (n_overrun / n_quoted * 100) if n_quoted > 0 else 0,
        "pct_jobs_critical_overrun": (n_critical / n_quoted * 100) if n_quoted > 0 else 0,
        "total_hours_variance": quoted_jobs["hours_variance_job"].sum(),
        "n_jobs": n_jobs,
        "n_quoted_jobs": n_quoted,
    }


def compute_rate_bridge(job_df: pd.DataFrame, portfolio_metrics: Dict) -> pd.DataFrame:
    """Decompose the Quote → Realised rate bridge."""
    quoted_jobs = job_df[job_df["has_quote"]].copy()

    under_run = quoted_jobs[quoted_jobs["hours_variance_job"] < 0]
    under_run_hours_saved = abs(under_run["hours_variance_job"].sum())

    overrun = quoted_jobs[quoted_jobs["hours_variance_job"] > 0]
    overrun_hours_excess = overrun["hours_variance_job"].sum()

    total_actual_hours = job_df["actual_hours_job"].sum()
    unquoted_hours = job_df["unquoted_hours_job"].sum()

    quote_rate = portfolio_metrics["quote_rate"]
    realised_rate = portfolio_metrics["realised_rate"]

    if total_actual_hours > 0:
        under_run_impact = under_run_hours_saved * quote_rate / total_actual_hours
        overrun_impact = -overrun_hours_excess * quote_rate / total_actual_hours
        unquoted_impact = -unquoted_hours * quote_rate / total_actual_hours
    else:
        under_run_impact = 0
        overrun_impact = 0
        unquoted_impact = 0

    residual = realised_rate - quote_rate - under_run_impact - overrun_impact - unquoted_impact

    return pd.DataFrame({
        "category": [
            "Quote Rate",
            "Under-run Jobs ↑",
            "Overrun Jobs ↓",
            "Unquoted Hours ↓",
            "Mix/Pricing Effect",
            "Realised Rate",
        ],
        "value": [
            quote_rate,
            under_run_impact,
            overrun_impact,
            unquoted_impact,
            residual,
            realised_rate,
        ],
        "type": ["absolute", "relative", "relative", "relative", "relative", "total"],
    })


# =============================================================================
# PORTFOLIO RENDERING
# =============================================================================


def render_kpi_strip(metrics: Dict):
    kpi_style = """
    <style>
    :root { color-scheme: light; }
    body { margin: 0; font-family: Arial, sans-serif; }
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 1rem;
        margin: 0.25rem 0 0.75rem;
    }
    .kpi-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.2s;
    }
    .kpi-card:hover {
        border-color: #1565c0;
        box-shadow: 0 4px 12px rgba(21, 101, 192, 0.1);
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a1a;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .kpi-value.positive { color: #28a745; }
    .kpi-value.negative { color: #dc3545; }
    .kpi-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-delta {
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    .kpi-delta.up { color: #28a745; }
    .kpi-delta.down { color: #dc3545; }
    @media (max-width: 900px) {
        .kpi-container { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 600px) {
        .kpi-container { grid-template-columns: 1fr; }
    }
    </style>
    """

    kpi_config = [
        ("quote_rate", "Quote Rate", "rate", None),
        ("realised_rate", "Realised Rate", "rate", "rate_variance"),
        ("rate_variance", "Rate Variance", "variance", None),
        ("margin_pct", "Margin", "percent", None),
        ("pct_jobs_overrun", "% Overrun", "percent0", None),
        ("pct_jobs_critical_overrun", "% Critical", "percent0", None),
    ]

    kpi_items = []
    for key, label, fmt_type, delta_key in kpi_config:
        value = metrics.get(key)

        if fmt_type == "rate":
            display = f"${value:,.0f}/hr" if pd.notna(value) else "—"
        elif fmt_type == "percent":
            display = fmt_percent(value) if pd.notna(value) else "—"
        elif fmt_type == "percent0":
            display = f"{value:.0f}%" if pd.notna(value) else "—"
        elif fmt_type == "variance":
            display = f"${value:+,.0f}/hr" if pd.notna(value) else "—"
        else:
            display = str(value) if pd.notna(value) else "—"

        value_class = ""
        if key in {"rate_variance"} and pd.notna(value):
            value_class = "positive" if value > 0 else "negative" if value < 0 else ""

        delta_html = ""
        if delta_key and delta_key in metrics:
            delta_val = metrics.get(delta_key)
            if pd.notna(delta_val):
                delta_class = "up" if delta_val > 0 else "down"
                delta_html = f'<div class="kpi-delta {delta_class}">{delta_val:+.0f}</div>'

        kpi_items.append(
            f"""
            <div class="kpi-card">
                <div class="kpi-value {value_class}">{display}</div>
                <div class="kpi-label">{label}</div>
                {delta_html}
            </div>
            """
        )

    kpi_html = f"{kpi_style}<div class='kpi-container'>{''.join(kpi_items)}</div>"
    components.html(kpi_html, height=260, scrolling=False)


def render_rate_bridge(bridge_df: pd.DataFrame):
    fig = go.Figure(
        go.Waterfall(
            name="Rate Bridge",
            orientation="v",
            measure=bridge_df["type"].tolist(),
            x=bridge_df["category"].tolist(),
            y=bridge_df["value"].tolist(),
            textposition="outside",
            text=[f"${v:,.0f}" for v in bridge_df["value"]],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2E7D32"}},
            decreasing={"marker": {"color": "#C62828"}},
            totals={"marker": {"color": "#1565C0"}},
        )
    )

    fig.update_layout(
        title="Quote Rate → Realised Rate Bridge",
        yaxis_title="$/hr",
        showlegend=False,
        height=400,
    )

    return fig


# =============================================================================
# ACT 2a: DEPARTMENT
# =============================================================================


def render_department_card(row: pd.Series, classification: str):
    """Render a single polished department card."""
    class_lower = classification.lower()
    badge_class = f"badge-{class_lower}"

    rate_var = row.get("rate_variance")
    rate_class = (
        "negative"
        if pd.notna(rate_var) and rate_var < 0
        else "positive"
        if pd.notna(rate_var) and rate_var > 0
        else ""
    )
    rate_display = f"${rate_var:+.0f}/hr" if pd.notna(rate_var) else "—"

    overrun_pct = row.get("pct_overrun")
    overrun_display = f"{overrun_pct:.0f}%" if pd.notna(overrun_pct) else "—"

    margin_pct = row.get("margin_pct")
    margin_display = f"{margin_pct:.0f}%" if pd.notna(margin_pct) else "—"

    reasons_html = ""
    if row.get("reasons"):
        reasons_items = "".join([f"<li>{r}</li>" for r in row["reasons"]])
        reasons_html = f'<ul class="dept-reasons">{reasons_items}</ul>'

    card_html = f"""
    <div class="dept-card {class_lower}">
        <div class="dept-header">
            <h4 class="dept-name">{row['department']}</h4>
            <span class="dept-badge {badge_class}">{classification}</span>
        </div>
        {reasons_html}
        <div class="dept-metrics">
            <div class="metric-box">
                <div class="metric-value {rate_class}">{rate_display}</div>
                <div class="metric-label">Rate Variance</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{overrun_display}</div>
                <div class="metric-label">Jobs Overrun</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{margin_display}</div>
                <div class="metric-label">Margin</div>
            </div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        if st.button("Explore →", key=f"explore_{row['department']}", type="secondary"):
            set_department_with_context(
                row["department"],
                classification,
                row.get("reasons", []),
            )
            st.rerun()


def render_department_scorecard(dept_df: pd.DataFrame):
    """Render polished department scorecard with professional styling."""
    st.subheader("1️⃣ Department Scorecard")
    st.caption("Departments ranked by attention priority. Click to explore.")

    st.markdown(
        """
        <style>
        .dept-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .dept-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        }
        .dept-card.erosive { border-left-color: #dc3545; }
        .dept-card.mixed { border-left-color: #ffc107; }
        .dept-card.accretive { border-left-color: #28a745; }

        .dept-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }
        .dept-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a1a1a;
            margin: 0;
        }
        .dept-badge {
            font-size: 0.75rem;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-erosive { background: #ffeaea; color: #dc3545; }
        .badge-mixed { background: #fff8e6; color: #b38600; }
        .badge-accretive { background: #e8f5e9; color: #28a745; }

        .dept-reasons {
            font-size: 0.85rem;
            color: #666;
            margin: 0.75rem 0;
            line-height: 1.5;
        }
        .dept-reasons li {
            margin-bottom: 0.25rem;
        }

        .dept-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
        }
        .metric-box { text-align: center; }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1a1a1a;
            line-height: 1.2;
        }
        .metric-value.negative { color: #dc3545; }
        .metric-value.positive { color: #28a745; }
        .metric-label {
            font-size: 0.75rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 0.25rem;
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid #1a1a1a;
        }
        .section-icon { font-size: 1.5rem; }
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1a1a1a;
            margin: 0;
        }
        .section-count {
            background: #f0f0f0;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            color: #666;
        }
        @media (max-width: 700px) {
            .dept-metrics { grid-template-columns: 1fr; }
            .dept-header { flex-direction: column; gap: 0.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if dept_df.empty:
        st.info("No department data available in this context.")
        return

    for classification in ["Erosive", "Mixed", "Accretive"]:
        subset = dept_df[dept_df["classification"] == classification]
        if subset.empty:
            continue

        icon = {"Erosive": "🔴", "Mixed": "🟡", "Accretive": "🟢"}[classification]
        st.markdown(
            f"""
            <div class="section-header">
                <span class="section-icon">{icon}</span>
                <h3 class="section-title">{classification} Departments</h3>
                <span class="section-count">{len(subset)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for _, row in subset.iterrows():
            render_department_card(row, classification)


def render_department_selection_gate(dept_df: pd.DataFrame):
    st.warning("⬇️ Select a department above to continue the root-cause drill.")
    if dept_df.empty:
        st.stop()


def compute_department_contribution(job_df: pd.DataFrame, portfolio_metrics: dict) -> pd.DataFrame:
    """Compute each department's contribution to portfolio realised rate."""
    portfolio_rate = portfolio_metrics["realised_rate"]
    portfolio_revenue = job_df["revenue_job"].sum()

    dept_df = job_df.groupby("department_final").agg(
        revenue=("revenue_job", "sum"),
        actual_hours=("actual_hours_job", "sum"),
        cost=("cost_job", "sum"),
        quoted_hours=("quoted_hours_job", "sum"),
        quoted_amount=("quoted_amount_job", "sum"),
        hours_variance=("hours_variance_job", "sum"),
        overrun_jobs=("overrun_flag", "sum"),
        total_jobs=("job_no", "count"),
    ).reset_index()

    dept_df["realised_rate"] = np.where(
        dept_df["actual_hours"] > 0,
        dept_df["revenue"] / dept_df["actual_hours"],
        np.nan,
    )
    dept_df["quote_rate"] = np.where(
        dept_df["quoted_hours"] > 0,
        dept_df["quoted_amount"] / dept_df["quoted_hours"],
        np.nan,
    )

    dept_df["revenue_weight"] = dept_df["revenue"] / portfolio_revenue if portfolio_revenue > 0 else 0
    dept_df["rate_contribution"] = dept_df["revenue_weight"] * (dept_df["realised_rate"] - portfolio_rate)
    dept_df["impact_type"] = np.where(dept_df["rate_contribution"] >= 0, "Subsidiser", "Erosive")

    dept_df = dept_df.rename(columns={"department_final": "department"})

    return dept_df.sort_values("rate_contribution", ascending=True)


def render_dept_contribution_bars(dept_df: pd.DataFrame):
    if dept_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No departments available",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(title="Department Contribution to Portfolio Rate", height=420)
        return fig

    df = dept_df.copy()
    total_revenue = df["revenue"].sum()
    total_hours = df["actual_hours"].sum()
    portfolio_rate = total_revenue / total_hours if total_hours > 0 else 0

    if "realised_rate" not in df.columns:
        df["realised_rate"] = np.where(
            df["actual_hours"] > 0,
            df["revenue"] / df["actual_hours"],
            np.nan,
        )

    df["rate_contribution"] = np.where(
        total_revenue > 0,
        (df["revenue"] / total_revenue) * (df["realised_rate"] - portfolio_rate),
        0,
    )
    df["impact_type"] = np.where(df["rate_contribution"] >= 0, "Subsidiser", "Erosive")
    df = df.sort_values("rate_contribution", ascending=True)

    colors = df["impact_type"].map(
        {"Subsidiser": COLORS["accretive"], "Erosive": COLORS["erosive"]}
    ).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["department"],
            x=df["rate_contribution"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Contribution: %{x:+.1f} $/hr<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Department Contribution to Portfolio Rate",
            font=dict(size=16, color=COLORS["text_dark"]),
            x=0,
        ),
        xaxis=dict(
            title="Rate Contribution ($/hr)",
            zeroline=True,
            zerolinecolor=COLORS["text_dark"],
            zerolinewidth=2,
            gridcolor="#f0f0f0",
            showgrid=True,
        ),
        yaxis=dict(title="", showgrid=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=max(300, len(df) * 50),
        margin=dict(t=60, b=40, l=150, r=40),
    )

    fig.add_annotation(
        x=1, y=1.08, xref="paper", yref="paper",
        text="🟢 Subsidiser  🔴 Erosive",
        showarrow=False,
        font=dict(size=11, color=COLORS["text_medium"]),
        xanchor="right",
    )

    return fig


def render_dept_efficiency_scatter(dept_df: pd.DataFrame):
    if dept_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No departments available",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(title="Department Efficiency vs Pricing", height=450)
        return fig

    df = dept_df.copy()
    df["hours_variance_pct"] = np.where(
        df["quoted_hours"] > 0,
        (df["actual_hours"] - df["quoted_hours"]) / df["quoted_hours"] * 100,
        0,
    )

    color_map = {
        "Accretive": COLORS["accretive"],
        "Mixed": COLORS["mixed"],
        "Erosive": COLORS["erosive"],
    }

    fig = px.scatter(
        df,
        x="hours_variance_pct",
        y="rate_variance",
        size="revenue",
        color="classification",
        color_discrete_map=color_map,
        hover_data={
            "department": True,
            "revenue": ":$,.0f",
            "margin_pct": ":.1f",
            "hours_variance_pct": ":.1f",
            "rate_variance": ":+.0f",
            "classification": False,
        },
        labels={
            "hours_variance_pct": "Hours Variance %",
            "rate_variance": "Rate Variance ($/hr)",
            "department": "Department",
        },
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="white"), opacity=0.85)
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#ccc", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#ccc", line_width=1)

    fig.add_annotation(
        x=-30, y=50, text="Efficient + Premium",
        showarrow=False, font=dict(size=10, color=COLORS["accretive"]),
        bgcolor="rgba(40, 167, 69, 0.1)", borderpad=4
    )
    fig.add_annotation(
        x=30, y=50, text="Overrun but Priced Well",
        showarrow=False, font=dict(size=10, color=COLORS["mixed"]),
        bgcolor="rgba(255, 193, 7, 0.1)", borderpad=4
    )
    fig.add_annotation(
        x=-30, y=-50, text="Efficient but Rate Leakage",
        showarrow=False, font=dict(size=10, color=COLORS["mixed"]),
        bgcolor="rgba(255, 193, 7, 0.1)", borderpad=4
    )
    fig.add_annotation(
        x=30, y=-50, text="Structural Problem",
        showarrow=False, font=dict(size=10, color=COLORS["erosive"]),
        bgcolor="rgba(220, 53, 69, 0.1)", borderpad=4
    )

    fig.update_layout(
        title=dict(
            text="Department Efficiency vs Pricing",
            font=dict(size=16, color=COLORS["text_dark"]),
            x=0,
        ),
        xaxis=dict(
            title="← Efficient | Hours Variance % | Inefficient →",
            zeroline=True,
            zerolinecolor="#ccc",
            gridcolor="#f0f0f0",
            showgrid=True,
        ),
        yaxis=dict(
            title="↓ Rate Leakage | Rate Variance ($/hr) | Premium ↑",
            zeroline=True,
            zerolinecolor="#ccc",
            gridcolor="#f0f0f0",
            showgrid=True,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title="",
        ),
        height=450,
        margin=dict(t=80, b=60, l=60, r=40),
    )
    return fig


def render_department_selector(dept_df: pd.DataFrame):
    st.subheader("Select Department to Drill Into")

    dept_options = dept_df["department"].dropna().tolist()
    selected = st.selectbox(
        "Choose department:",
        options=["-- Select --"] + dept_options,
        format_func=lambda x: x if x == "-- Select --" else f"{x} ({dept_df[dept_df['department'] == x]['impact_type'].values[0]})",
    )

    if selected != "-- Select --":
        st.session_state.drill_path["department"] = selected
        st.rerun()
    else:
        st.warning("⬇️ Select a department above to continue root-cause analysis")


# =============================================================================
# ACT 2b: CATEGORY
# =============================================================================


def compute_category_scorecard(job_df: pd.DataFrame, department: str) -> pd.DataFrame:
    """
    Compute category scorecard within a department.
    Uses job_category field from fact_timesheet_day_enriched.
    """
    dept_jobs = job_df[job_df["department_final"] == department].copy()
    if "job_category" not in dept_jobs.columns:
        # Enforce canonical source field; fall back to uncategorised if missing.
        dept_jobs["job_category"] = np.nan

    # Group by canonical job_category column only
    cat_df = dept_jobs.groupby("job_category").agg(
        revenue=("revenue_job", "sum"),
        actual_hours=("actual_hours_job", "sum"),
        cost=("cost_job", "sum"),
        quoted_hours=("quoted_hours_job", "sum"),
        quoted_amount=("quoted_amount_job", "sum"),
        hours_variance=("hours_variance_job", "sum"),
        overrun_jobs=("overrun_flag", "sum"),
        severe_overrun_jobs=("critical_overrun_flag", "sum"),
        total_jobs=("job_no", "count"),
    ).reset_index()

    cat_df.columns = [
        "category",
        "revenue",
        "actual_hours",
        "cost",
        "quoted_hours",
        "quoted_amount",
        "hours_variance",
        "overrun_jobs",
        "severe_overrun_jobs",
        "total_jobs",
    ]

    # Drop categories with no activity in the current period
    cat_df = cat_df[
        (cat_df["actual_hours"] > 0) | (cat_df["revenue"] > 0) | (cat_df["cost"] > 0)
    ].copy()

    if cat_df.empty:
        return cat_df

    cat_df["category"] = cat_df["category"].fillna("(Uncategorised)")

    cat_df["quote_rate"] = np.where(
        cat_df["quoted_hours"] > 0,
        cat_df["quoted_amount"] / cat_df["quoted_hours"],
        np.nan,
    )
    cat_df["realised_rate"] = np.where(
        cat_df["actual_hours"] > 0,
        cat_df["revenue"] / cat_df["actual_hours"],
        np.nan,
    )
    cat_df["rate_variance"] = cat_df["realised_rate"] - cat_df["quote_rate"]
    cat_df["margin"] = cat_df["revenue"] - cat_df["cost"]
    cat_df["margin_pct"] = np.where(
        cat_df["revenue"] > 0,
        cat_df["margin"] / cat_df["revenue"] * 100,
        np.nan,
    )
    cat_df["pct_overrun"] = np.where(
        cat_df["total_jobs"] > 0,
        cat_df["overrun_jobs"] / cat_df["total_jobs"] * 100,
        0,
    )
    cat_df["pct_severe"] = np.where(
        cat_df["total_jobs"] > 0,
        cat_df["severe_overrun_jobs"] / cat_df["total_jobs"] * 100,
        0,
    )

    classifications = cat_df.apply(classify_department, axis=1)
    cat_df["classification"] = [c["classification"] for c in classifications]
    cat_df["reasons"] = [c["reasons"] for c in classifications]

    return cat_df.sort_values("rate_variance", ascending=True)


def render_category_card(row: pd.Series, classification: str):
    """Render a single polished category card."""
    class_lower = classification.lower()
    badge_class = f"badge-{class_lower}"

    rate_var = row.get("rate_variance")
    rate_class = (
        "negative"
        if pd.notna(rate_var) and rate_var < 0
        else "positive"
        if pd.notna(rate_var) and rate_var > 0
        else ""
    )
    rate_display = f"${rate_var:+.0f}/hr" if pd.notna(rate_var) else "—"

    overrun_pct = row.get("pct_overrun")
    overrun_display = f"{overrun_pct:.0f}%" if pd.notna(overrun_pct) else "—"

    margin_pct = row.get("margin_pct")
    margin_display = f"{margin_pct:.0f}%" if pd.notna(margin_pct) else "—"

    reasons_html = ""
    if row.get("reasons"):
        reasons_items = "".join([f"<li>{r}</li>" for r in row["reasons"]])
        reasons_html = f'<ul class="dept-reasons">{reasons_items}</ul>'

    label = row["category"]
    cat_value = None if label == "(Uncategorised)" else label

    card_html = f"""
    <div class="dept-card {class_lower}">
        <div class="dept-header">
            <h4 class="dept-name">{label}</h4>
            <span class="dept-badge {badge_class}">{classification}</span>
        </div>
        {reasons_html}
        <div class="dept-metrics">
            <div class="metric-box">
                <div class="metric-value {rate_class}">{rate_display}</div>
                <div class="metric-label">Rate Variance</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{overrun_display}</div>
                <div class="metric-label">Jobs Overrun</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{margin_display}</div>
                <div class="metric-label">Margin</div>
            </div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        if st.button("Explore →", key=f"cat_{label}", type="secondary"):
            set_category_with_context(
                cat_value,
                classification,
                row.get("reasons", []),
            )
            st.rerun()


def render_category_scorecard(cat_df: pd.DataFrame):
    """Render category scorecard with polished styling."""
    st.subheader("2️⃣ Category Scorecard")
    st.caption("Which categories drive the department outcome?")

    if cat_df.empty:
        st.info("No category data available for this department.")
        return

    for classification in ["Erosive", "Mixed", "Accretive"]:
        subset = cat_df[cat_df["classification"] == classification]
        if subset.empty:
            continue

        icon = {"Erosive": "🔴", "Mixed": "🟡", "Accretive": "🟢"}[classification]
        st.markdown(
            f"""
            <div class="section-header">
                <span class="section-icon">{icon}</span>
                <h3 class="section-title">{classification} Categories</h3>
                <span class="section-count">{len(subset)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for _, row in subset.iterrows():
            render_category_card(row, classification)


def render_category_selection_gate(cat_df: pd.DataFrame):
    st.warning("⬇️ Select a category above to continue to job-level analysis.")
    if cat_df.empty:
        st.stop()


def compute_category_contribution(job_df: pd.DataFrame, department: str) -> pd.DataFrame:
    dept_jobs = job_df[job_df["department_final"] == department].copy()
    if dept_jobs.empty:
        return pd.DataFrame()

    dept_rate = dept_jobs["revenue_job"].sum() / dept_jobs["actual_hours_job"].sum()
    dept_revenue = dept_jobs["revenue_job"].sum()

    cat_df = dept_jobs.groupby("job_category").agg(
        revenue=("revenue_job", "sum"),
        actual_hours=("actual_hours_job", "sum"),
        quoted_hours=("quoted_hours_job", "sum"),
        quoted_amount=("quoted_amount_job", "sum"),
        hours_variance=("hours_variance_job", "sum"),
        overrun_jobs=("overrun_flag", "sum"),
        total_jobs=("job_no", "count"),
    ).reset_index()

    cat_df["realised_rate"] = np.where(
        cat_df["actual_hours"] > 0,
        cat_df["revenue"] / cat_df["actual_hours"],
        np.nan,
    )
    cat_df["quote_rate"] = np.where(
        cat_df["quoted_hours"] > 0,
        cat_df["quoted_amount"] / cat_df["quoted_hours"],
        np.nan,
    )
    cat_df["rate_variance"] = cat_df["realised_rate"] - cat_df["quote_rate"]
    cat_df["hours_variance_pct"] = np.where(
        cat_df["quoted_hours"] > 0,
        cat_df["hours_variance"] / cat_df["quoted_hours"] * 100,
        np.nan,
    )

    cat_df["revenue_weight"] = cat_df["revenue"] / dept_revenue if dept_revenue > 0 else 0
    cat_df["rate_contribution"] = cat_df["revenue_weight"] * (cat_df["realised_rate"] - dept_rate)
    cat_df["impact_type"] = np.where(cat_df["rate_contribution"] >= 0, "Subsidiser", "Erosive")

    return cat_df.sort_values("rate_contribution", ascending=True)


def render_category_contribution_bars(cat_df: pd.DataFrame):
    colors = {"Subsidiser": "#2E7D32", "Erosive": "#C62828"}
    fig = px.bar(
        cat_df,
        x="rate_contribution",
        y="job_category",
        orientation="h",
        color="impact_type",
        color_discrete_map=colors,
        labels={"rate_contribution": "Rate Contribution ($/hr)", "job_category": "Category"},
    )
    fig.update_layout(title="Category Contribution to Department Rate", height=420)
    return fig


def render_category_efficiency_scatter(cat_df: pd.DataFrame):
    if cat_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No categories available",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(title="Category Efficiency vs Pricing", height=450)
        return fig

    df = cat_df.copy()
    df["hours_variance_pct"] = np.where(
        df["quoted_hours"] > 0,
        (df["actual_hours"] - df["quoted_hours"]) / df["quoted_hours"] * 100,
        0,
    )

    color_map = {
        "Accretive": COLORS["accretive"],
        "Mixed": COLORS["mixed"],
        "Erosive": COLORS["erosive"],
    }

    fig = px.scatter(
        df,
        x="hours_variance_pct",
        y="rate_variance",
        size="revenue",
        color="classification",
        color_discrete_map=color_map,
        hover_data={
            "category": True,
            "revenue": ":$,.0f",
            "margin_pct": ":.1f",
            "hours_variance_pct": ":.1f",
            "rate_variance": ":+.0f",
            "classification": False,
        },
        labels={
            "hours_variance_pct": "Hours Variance %",
            "rate_variance": "Rate Variance ($/hr)",
            "category": "Category",
        },
    )

    fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.85))
    fig.add_hline(y=0, line_dash="dash", line_color="#ccc", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#ccc", line_width=1)

    fig.add_annotation(
        x=-30, y=50, text="Efficient + Premium",
        showarrow=False, font=dict(size=10, color=COLORS["accretive"]),
        bgcolor="rgba(40, 167, 69, 0.1)", borderpad=4
    )
    fig.add_annotation(
        x=30, y=50, text="Overrun but Priced Well",
        showarrow=False, font=dict(size=10, color=COLORS["mixed"]),
        bgcolor="rgba(255, 193, 7, 0.1)", borderpad=4
    )
    fig.add_annotation(
        x=-30, y=-50, text="Efficient but Rate Leakage",
        showarrow=False, font=dict(size=10, color=COLORS["mixed"]),
        bgcolor="rgba(255, 193, 7, 0.1)", borderpad=4
    )
    fig.add_annotation(
        x=30, y=-50, text="Structural Problem",
        showarrow=False, font=dict(size=10, color=COLORS["erosive"]),
        bgcolor="rgba(220, 53, 69, 0.1)", borderpad=4
    )

    fig.update_layout(
        title=dict(
            text="Category Efficiency vs Pricing",
            font=dict(size=16, color=COLORS["text_dark"]),
            x=0,
        ),
        xaxis=dict(
            title="← Efficient | Hours Variance % | Inefficient →",
            zeroline=True,
            zerolinecolor="#ccc",
            gridcolor="#f0f0f0",
            showgrid=True,
        ),
        yaxis=dict(
            title="↓ Rate Leakage | Rate Variance ($/hr) | Premium ↑",
            zeroline=True,
            zerolinecolor="#ccc",
            gridcolor="#f0f0f0",
            showgrid=True,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title="",
        ),
        height=450,
        margin=dict(t=80, b=60, l=60, r=40),
    )
    return fig


def render_category_selector(cat_df: pd.DataFrame):
    st.subheader(f"Select Category within {st.session_state.drill_path['department']}")

    cat_df = cat_df.copy()
    cat_df["category_display"] = cat_df["job_category"].fillna("(Uncategorised)")
    display_options = cat_df["category_display"].tolist()

    selected = st.selectbox(
        "Choose category:",
        options=["-- Select --"] + display_options,
    )

    if selected != "-- Select --":
        actual_val = None if selected == "(Uncategorised)" else selected
        st.session_state.drill_path["category"] = actual_val
        st.rerun()
    else:
        st.warning("⬇️ Select a category above to see job-level breakdown")


# =============================================================================
# RECURRING TASK OVERRUNS
# =============================================================================


def _is_uncategorised_value(category: Optional[str]) -> bool:
    if category is None:
        return False
    cat = str(category).strip().lower()
    return cat in {"(uncategorised)", "(uncategorized)", "uncategorised", "uncategorized", "__null__"}


def _scope_key(scope_label: str, department: Optional[str], category: Optional[str]) -> str:
    raw = f"{scope_label}|{department}|{category}"
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in raw)


def _default_overrun_window() -> str:
    preset = str(st.session_state.get("time_preset", "")).upper()
    preset_map = {"30D": "3m", "90D": "3m", "6M": "6m", "12M": "12m"}
    if preset in preset_map:
        return preset_map[preset]

    ctx = st.session_state.get("global_context", {})
    start_date = ctx.get("start_date")
    end_date = ctx.get("end_date")
    if isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
        days = max((end_date - start_date).days, 0)
        if days <= 120:
            return "3m"
        if days <= 220:
            return "6m"
        if days <= 420:
            return "12m"
        if days <= 820:
            return "24m"
        return "all"

    return "12m"


def _filter_recurring_overrun_scope(
    df_scope: pd.DataFrame,
    department: Optional[str],
    category: Optional[str],
    time_window: str,
) -> pd.DataFrame:
    scoped = df_scope

    if department and "department_final" in scoped.columns:
        scoped = scoped[scoped["department_final"].astype(str) == str(department)]

    if category is not None and len(scoped) > 0:
        category_col = get_category_col(scoped)
        if category_col in scoped.columns:
            if _is_uncategorised_value(category):
                scoped = scoped[scoped[category_col].isna()]
            else:
                scoped = scoped[scoped[category_col] == category]

    date_col = "month_key" if "month_key" in scoped.columns else "work_date" if "work_date" in scoped.columns else None
    if date_col and len(scoped) > 0:
        scoped = scoped.copy()
        scoped[date_col] = pd.to_datetime(scoped[date_col], errors="coerce")
        scoped = filter_by_time_window(scoped, window=time_window, date_col=date_col)

    return scoped


def _format_signed_pct(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value * 100:+.0f}%"


def render_recurring_task_overruns_section(
    df_scope: pd.DataFrame,
    scope_label: str,
    department: Optional[str],
    category: Optional[str],
) -> None:
    st.subheader("Recurring Quote Overruns (Task Margin Erosion)")
    st.caption(f"{scope_label} | Tasks ranked by repeated margin leakage, not one-off variance.")

    if "quoted_time_total" not in df_scope.columns:
        st.info("Quoted hours not available — cannot compute quote overruns.")
        return

    missing_core = [c for c in ["job_no", "task_name", "hours_raw"] if c not in df_scope.columns]
    if missing_core:
        st.info(f"Required columns missing: {', '.join(missing_core)}")
        return

    has_cost = "base_cost" in df_scope.columns
    scope_key = _scope_key(scope_label, department, category)
    time_options = ["3m", "6m", "12m", "24m", "fytd", "all"]
    default_window = _default_overrun_window()
    default_idx = time_options.index(default_window) if default_window in time_options else 2

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.0, 1.0])
    with ctrl1:
        time_window = st.selectbox(
            "Time window",
            options=time_options,
            index=default_idx,
            key=f"overrun_window_{scope_key}",
        )
    with ctrl2:
        min_jobs_with_quote = st.slider(
            "Min jobs with quote",
            min_value=3,
            max_value=30,
            value=8,
            step=1,
            key=f"overrun_min_jobs_{scope_key}",
        )
    with ctrl3:
        min_overrun_rate = st.slider(
            "Min overrun rate",
            min_value=0.0,
            max_value=0.9,
            value=0.30,
            step=0.05,
            key=f"overrun_min_rate_{scope_key}",
        )

    task_overruns = compute_task_overrun_consistency(
        df_scope,
        department=department,
        category=category,
        time_window=time_window,
        min_jobs_with_quote=min_jobs_with_quote,
        min_overrun_rate=min_overrun_rate,
    )

    if task_overruns.empty:
        st.info("No recurring task overruns meet the selected thresholds.")
        return

    scoped_for_detail = _filter_recurring_overrun_scope(
        df_scope,
        department=department,
        category=category,
        time_window=time_window,
    )

    top_rows = task_overruns.head(2)
    if has_cost and top_rows["total_overrun_cost"].notna().any():
        top_summary = [
            (
                f"**{r['task_name'] if pd.notna(r['task_name']) else '(Unspecified task)'}** "
                f"leaks {fmt_currency(r['total_overrun_cost'])} "
                f"at {r['overrun_rate'] * 100:.0f}% overrun frequency"
            )
            for _, r in top_rows.iterrows()
        ]
    else:
        top_summary = [
            (
                f"**{r['task_name'] if pd.notna(r['task_name']) else '(Unspecified task)'}** "
                f"overruns in {r['overrun_rate'] * 100:.0f}% of quoted jobs "
                f"({r['total_overrun_hours']:,.1f}h total overrun)"
            )
            for _, r in top_rows.iterrows()
        ]
    st.markdown(f"**So what:** {'; '.join(top_summary)}.")

    show_revenue_at_risk = (
        "total_revenue_at_risk" in task_overruns.columns
        and task_overruns["total_revenue_at_risk"].notna().any()
    )

    recurring_table = pd.DataFrame({
        "Task": task_overruns["task_name"].fillna("(Unspecified task)").astype(str),
        "Overrun rate (%)": task_overruns["overrun_rate"].apply(lambda v: f"{v * 100:.0f}%"),
        "Overrun jobs": task_overruns["overrun_jobs"].fillna(0).astype(int),
        "Jobs with quote": task_overruns["jobs_with_quote"].fillna(0).astype(int),
        "Total overrun hours": task_overruns["total_overrun_hours"].apply(lambda v: f"{v:,.1f}"),
        "Est. margin erosion ($)": (
            task_overruns["total_overrun_cost"].apply(fmt_currency)
            if has_cost
            else "—"
        ),
        "Avg overrun (%)": task_overruns["avg_overrun_pct"].apply(_format_signed_pct),
    })
    if show_revenue_at_risk:
        recurring_table["Revenue at risk ($)"] = task_overruns["total_revenue_at_risk"].apply(fmt_currency)

    st.markdown("**Recurring Margin-Leak Tasks**")
    st.dataframe(recurring_table, use_container_width=True, hide_index=True)

    task_selector_df = task_overruns.reset_index(drop=True).copy()
    task_selector_df["task_label"] = task_selector_df["task_name"].fillna("(Unspecified task)").astype(str)
    selected_label = st.selectbox(
        "Deep dive a task",
        options=task_selector_df["task_label"].tolist(),
        key=f"overrun_task_{scope_key}",
    )
    selected_idx = task_selector_df["task_label"].tolist().index(selected_label)
    selected_task_row = task_selector_df.iloc[selected_idx]
    selected_task = selected_task_row["task_name"]

    top_jobs = get_overrun_jobs_for_task(
        df_scope,
        task_name=selected_task,
        department=department,
        category=category,
        time_window=time_window,
        n=15,
    )

    st.markdown("**Top Offending Jobs**")
    if top_jobs.empty:
        st.info("No overrun jobs found for this task in the selected scope/window.")
    else:
        client_col = None
        if "client_name" in top_jobs.columns and top_jobs["client_name"].notna().any():
            client_col = "client_name"
        elif "client_group" in top_jobs.columns and top_jobs["client_group"].notna().any():
            client_col = "client_group"

        jobs_table = pd.DataFrame({
            "Job": top_jobs["job_no"].astype(str),
            "Quoted hours": top_jobs["quoted_hours"].apply(lambda v: f"{v:,.1f}"),
            "Actual hours": top_jobs["actual_hours"].apply(lambda v: f"{v:,.1f}"),
            "Overrun hours": top_jobs["overrun_hours"].apply(lambda v: f"{v:,.1f}"),
            "Overrun cost ($)": top_jobs["overrun_cost"].apply(fmt_currency) if has_cost else "—",
            "Avg cost rate": (
                top_jobs["avg_cost_rate"].apply(lambda v: f"${v:,.0f}/hr" if pd.notna(v) else "—")
                if has_cost
                else "—"
            ),
        })
        if client_col:
            jobs_table.insert(1, "Client", top_jobs[client_col].fillna("—").astype(str))
        if "department_final" in top_jobs.columns and top_jobs["department_final"].notna().any():
            jobs_table.insert(2 if client_col else 1, "Department", top_jobs["department_final"].fillna("—").astype(str))
        if "category" in top_jobs.columns and top_jobs["category"].notna().any():
            jobs_table.insert(3 if client_col else 2, "Category", top_jobs["category"].fillna("—").astype(str))
        if "quote_rate" in top_jobs.columns and top_jobs["quote_rate"].notna().any():
            jobs_table["Quote rate"] = top_jobs["quote_rate"].apply(lambda v: f"${v:,.0f}/hr" if pd.notna(v) else "—")
        if "revenue_at_risk" in top_jobs.columns and top_jobs["revenue_at_risk"].notna().any():
            jobs_table["Revenue at risk ($)"] = top_jobs["revenue_at_risk"].apply(fmt_currency)

        st.dataframe(jobs_table, use_container_width=True, hide_index=True)

    if "staff_name" in scoped_for_detail.columns and not top_jobs.empty:
        staff_scope = scoped_for_detail.copy()
        if pd.isna(selected_task):
            staff_scope = staff_scope[staff_scope["task_name"].isna()]
        else:
            staff_scope = staff_scope[staff_scope["task_name"] == selected_task]
        staff_scope = staff_scope[staff_scope["job_no"].isin(top_jobs["job_no"].tolist())]

        if not staff_scope.empty:
            if has_cost:
                staff_table = staff_scope.groupby("staff_name", dropna=False).agg(
                    hours=("hours_raw", "sum"),
                    cost=("base_cost", "sum"),
                ).reset_index()
                staff_table = staff_table.sort_values("hours", ascending=False).head(15)
                staff_display = pd.DataFrame({
                    "Staff": staff_table["staff_name"].fillna("(Unassigned)").astype(str),
                    "Hours": staff_table["hours"].apply(lambda v: f"{v:,.1f}"),
                    "Cost": staff_table["cost"].apply(fmt_currency),
                })
            else:
                staff_table = staff_scope.groupby("staff_name", dropna=False).agg(
                    hours=("hours_raw", "sum"),
                ).reset_index()
                staff_table = staff_table.sort_values("hours", ascending=False).head(15)
                staff_display = pd.DataFrame({
                    "Staff": staff_table["staff_name"].fillna("(Unassigned)").astype(str),
                    "Hours": staff_table["hours"].apply(lambda v: f"{v:,.1f}"),
                    "Cost": "—",
                })

            st.markdown("**Top Contributing Staff (Selected Task + Top Jobs)**")
            st.dataframe(staff_display, use_container_width=True, hide_index=True)

    if pd.isna(selected_task):
        task_scope = scoped_for_detail[scoped_for_detail["task_name"].isna()].copy()
    else:
        task_scope = scoped_for_detail[scoped_for_detail["task_name"] == selected_task].copy()

    overrun_rate = float(selected_task_row.get("overrun_rate", 0) or 0)
    avg_overrun_pct = float(selected_task_row.get("avg_overrun_pct", 0) or 0)
    rule_1_trigger = overrun_rate >= 0.60 and avg_overrun_pct >= 0.20

    mismatch_share = np.nan
    rule_2_evaluable = "quote_match_flag" in task_scope.columns and not task_scope.empty
    rule_2_trigger = False
    if rule_2_evaluable:
        mismatch_share = task_scope["quote_match_flag"].astype(str).str.lower().ne("matched").mean()
        rule_2_trigger = mismatch_share >= 0.20

    quote_rate = np.nan
    realised_rate = np.nan
    rate_gap_pct = np.nan
    rule_3_evaluable = False
    rule_3_trigger = False
    needed_for_rate = {"quoted_time_total", "quoted_amount_total", "rev_alloc", "hours_raw", "job_no", "task_name"}
    if needed_for_rate.issubset(task_scope.columns) and not task_scope.empty:
        task_quotes = safe_quote_job_task(task_scope)
        if (
            not task_quotes.empty
            and "quoted_time_total" in task_quotes.columns
            and "quoted_amount_total" in task_quotes.columns
        ):
            quote_hours = pd.to_numeric(task_quotes["quoted_time_total"], errors="coerce").fillna(0).sum()
            quote_amount = pd.to_numeric(task_quotes["quoted_amount_total"], errors="coerce").sum(min_count=1)
            actual_hours = pd.to_numeric(task_scope["hours_raw"], errors="coerce").sum()
            actual_revenue = pd.to_numeric(task_scope["rev_alloc"], errors="coerce").sum(min_count=1)

            if quote_hours > 0 and pd.notna(quote_amount):
                quote_rate = quote_amount / quote_hours
            if actual_hours > 0 and pd.notna(actual_revenue):
                realised_rate = actual_revenue / actual_hours

            if pd.notna(quote_rate) and quote_rate > 0 and pd.notna(realised_rate):
                rate_gap_pct = (quote_rate - realised_rate) / quote_rate
                rule_3_evaluable = True
                rule_3_trigger = rate_gap_pct >= 0.15

    card_1 = {
        "title": "Fix the quote baseline",
        "body": (
            f"Overruns hit {overrun_rate * 100:.0f}% of quoted jobs with {avg_overrun_pct * 100:.0f}% average overshoot. "
            "Increase standard hours for this task, add complexity drivers, and update quote builder defaults."
            if rule_1_trigger
            else "Refresh baseline hours from the latest delivery data and tighten estimate guardrails in the quote template."
        ),
    }

    if rule_2_evaluable and rule_2_trigger:
        card_2_body = (
            f"{mismatch_share * 100:.0f}% of task rows are not quote-matched. Enforce variation approval before any extra "
            "delivery effort and track exceptions weekly."
        )
    elif rule_2_evaluable:
        card_2_body = (
            f"Quote mismatch share is {mismatch_share * 100:.0f}%. Keep change-control gates active and review mismatches monthly."
        )
    else:
        card_2_body = "Introduce scope variation control on this task with mandatory approval and a weekly scope-change log."

    if rule_3_evaluable and rule_3_trigger:
        card_3_body = (
            f"Realised rate ({fmt_currency(realised_rate)}/hr) is {rate_gap_pct * 100:.0f}% below quote rate "
            f"({fmt_currency(quote_rate)}/hr). Rebalance staffing mix and tighten write-down control."
        )
    elif rule_3_evaluable:
        card_3_body = (
            f"Realised rate ({fmt_currency(realised_rate)}/hr) is close to quote rate ({fmt_currency(quote_rate)}/hr). "
            "Maintain current staffing mix and monitor rate slippage."
        )
    else:
        card_3_body = "Review execution playbook and QA gates for this task to improve delivery efficiency and rate capture."

    actions = [
        card_1,
        {"title": "Stop scope creep", "body": card_2_body},
        {"title": "Fix staffing / rate capture", "body": card_3_body},
    ]

    st.markdown("**Recommended actions**")
    action_cols = st.columns(3)
    for col, action in zip(action_cols, actions):
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid #e9ecef;border-radius:10px;padding:0.85rem;background:#fafbfc;min-height:170px;">
                    <div style="font-weight:600;color:#1a1a1a;">{action['title']}</div>
                    <div style="font-size:0.86rem;color:#555;margin-top:0.4rem;line-height:1.35;">{action['body']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# ACT 2c: JOB QUADRANT
# =============================================================================


def render_job_quadrant_scatter(
    job_df: pd.DataFrame,
    impact_mode: str,
    job_name_lookup: Optional[Dict[str, str]] = None,
):
    if job_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No jobs available",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(title="Job Outcome Quadrant", height=500)
        return fig

    job_df = job_df.copy()
    job_df["job_label"] = job_df["job_no"].apply(
        lambda x: format_job_label(x, job_name_lookup)
    )
    job_df["margin_pct_job"] = job_df["margin_pct_job"].replace([np.inf, -np.inf], np.nan)
    job_df["revenue_display"] = job_df["revenue_job"].apply(fmt_currency)
    job_df["margin_pct_display"] = job_df["margin_pct_job"].apply(lambda x: fmt_percent(x, 1))

    job_df["impact_type"] = np.where(
        (job_df["rate_variance_job"] >= 0) & (job_df["hours_variance_pct_job"] <= 0),
        "Subsidiser",
        np.where(
            (job_df["rate_variance_job"] < 0) & (job_df["hours_variance_pct_job"] > 20),
            "Margin Erosive",
            "Mixed",
        ),
    )

    size_col = {
        "Revenue Exposure": "revenue_job",
        "Rate Impact": "rate_variance_job",
        "Margin Impact": "margin_job",
    }.get(impact_mode, "revenue_job")

    job_df["size"] = (
        job_df[size_col].abs().replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    max_points = 2000
    if len(job_df) > max_points:
        job_df = job_df.nlargest(max_points, "size")
        st.caption(f"Showing top {max_points:,} jobs by impact for chart responsiveness.")

    fig = px.scatter(
        job_df,
        x="hours_variance_pct_job",
        y="rate_variance_job",
        size="size",
        color="impact_type",
        color_discrete_map={
            "Subsidiser": COLORS["accretive"],
            "Margin Erosive": COLORS["erosive"],
            "Mixed": COLORS["mixed"],
        },
        hover_data=None,
        custom_data=["job_label", "revenue_display", "margin_pct_display"],
        labels={
            "hours_variance_pct_job": "Hours Variance %",
            "rate_variance_job": "Rate Variance ($/hr)",
            "size": impact_mode,
        },
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Revenue: %{customdata[1]}<br>"
            "Margin: %{customdata[2]}<br>"
            "Hours Var: %{x:+.0f}%<br>"
            "Rate Var: %{y:+.0f} $/hr<br>"
            "<extra></extra>"
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#ccc", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#ccc", line_width=1)

    fig.add_annotation(
        x=-30, y=50, text="Efficient + Premium",
        showarrow=False, font=dict(size=10, color=COLORS["accretive"]),
        bgcolor="rgba(40, 167, 69, 0.1)", borderpad=4
    )
    fig.add_annotation(
        x=30, y=50, text="Overrun but Priced Well",
        showarrow=False, font=dict(size=10, color=COLORS["mixed"]),
        bgcolor="rgba(255, 193, 7, 0.1)", borderpad=4
    )
    fig.add_annotation(
        x=-30, y=-50, text="Efficient but Rate Leakage",
        showarrow=False, font=dict(size=10, color=COLORS["mixed"]),
        bgcolor="rgba(255, 193, 7, 0.1)", borderpad=4
    )
    fig.add_annotation(
        x=30, y=-50, text="Structural Problem",
        showarrow=False, font=dict(size=10, color=COLORS["erosive"]),
        bgcolor="rgba(220, 53, 69, 0.1)", borderpad=4
    )

    fig.update_layout(
        title=dict(
            text=f"Job Quadrant — Sized by {impact_mode}",
            font=dict(size=16, color=COLORS["text_dark"]),
            x=0,
        ),
        xaxis_title="← Under-run | Hours Variance % | Overrun →",
        yaxis_title="↓ Rate Leakage | Rate Variance | Premium ↑",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
    )

    return fig


def render_job_selector(
    job_df: pd.DataFrame,
    job_name_lookup: Optional[Dict[str, str]] = None,
):
    st.markdown("**Select job to drill into:**")
    sort_df = job_df.copy()
    total_revenue = sort_df["revenue_job"].sum()
    category_rate = total_revenue / sort_df["actual_hours_job"].sum() if sort_df["actual_hours_job"].sum() > 0 else 0
    sort_df["revenue_weight"] = sort_df["revenue_job"] / total_revenue if total_revenue > 0 else 0
    sort_df["rate_contribution"] = sort_df["revenue_weight"] * (sort_df["realised_rate_job"] - category_rate)
    sort_df["priority_score"] = sort_df["rate_contribution"].abs()

    job_options = sort_df.sort_values("priority_score", ascending=False)["job_no"].tolist()
    label_map = {job_no: format_job_label(job_no, job_name_lookup) for job_no in job_options}

    selected = st.selectbox(
        "Choose job:",
        options=["-- Select --"] + job_options,
        format_func=lambda x: "-- Select --" if x == "-- Select --" else label_map.get(x, str(x)),
    )

    if selected != "-- Select --":
        st.session_state.drill_path["job"] = selected
        st.rerun()


# =============================================================================
# ACT 3a: JOB DIAGNOSTIC
# =============================================================================


def render_job_diagnostic_card(
    df: pd.DataFrame,
    job_no: str,
    job_name_lookup: Optional[Dict[str, str]] = None,
):
    job_data = df[df["job_no"] == job_no]
    job_data = job_data.copy()

    unquoted_flag = pd.Series(False, index=job_data.index)
    if "quote_match_flag" in job_data.columns:
        unquoted_flag |= job_data["quote_match_flag"].astype(str).str.lower().ne("matched")
    if "quoted_time_total" in job_data.columns:
        unquoted_flag |= job_data["quoted_time_total"].fillna(0) <= 0
    if "quoted_amount_total" in job_data.columns:
        unquoted_flag |= job_data["quoted_amount_total"].fillna(0) <= 0
    job_data["is_unquoted"] = unquoted_flag

    actual_hours = job_data["hours_raw"].sum()
    revenue = job_data["rev_alloc"].sum()
    cost = job_data["base_cost"].sum()

    task_quotes = job_data.groupby("task_name").agg(
        quoted_time_total=("quoted_time_total", "first"),
        quoted_amount_total=("quoted_amount_total", "first"),
    )
    quoted_hours = task_quotes["quoted_time_total"].sum()
    quoted_amount = task_quotes["quoted_amount_total"].sum()

    quote_rate = quoted_amount / quoted_hours if quoted_hours > 0 else 0
    realised_rate = revenue / actual_hours if actual_hours > 0 else 0

    hours_variance = actual_hours - quoted_hours
    hours_variance_pct = (hours_variance / quoted_hours * 100) if quoted_hours > 0 else 0
    rate_variance = realised_rate - quote_rate

    margin = revenue - cost
    margin_pct = (margin / revenue * 100) if revenue > 0 else 0

    unquoted_tasks = job_data[job_data["is_unquoted"]]
    unquoted_hours = unquoted_tasks["hours_raw"].sum() if not unquoted_tasks.empty else 0
    unquoted_pct = (unquoted_hours / actual_hours * 100) if actual_hours > 0 else 0

    job_label = format_job_label(job_no, job_name_lookup)
    st.subheader(f"📋 Job Diagnostic: {job_label}")

    st.markdown("### A. Volume Efficiency (Time)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quoted Hours", f"{quoted_hours:,.0f}")
    col2.metric("Actual Hours", f"{actual_hours:,.0f}", delta=f"{hours_variance:+,.0f}")
    col3.metric("Hours Variance", f"{hours_variance_pct:+.0f}%")
    col4.metric("Scope Creep", f"{unquoted_pct:.0f}%", help="Unquoted hours as % of actual hours")

    st.markdown("### B. Rate Integrity (Commercial)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Quote Rate", f"${quote_rate:,.0f}/hr")
    col2.metric("Realised Rate", f"${realised_rate:,.0f}/hr", delta=f"${rate_variance:+,.0f}")
    col3.metric("Rate Capture", f"{(realised_rate / quote_rate * 100) if quote_rate > 0 else 0:.0f}%")

    st.markdown("### C. Margin Outcome")
    cost_rate = cost / actual_hours if actual_hours > 0 else 0
    cost_delta_from_hours = (actual_hours - quoted_hours) * cost_rate

    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue", f"${revenue:,.0f}")
    col2.metric("Cost", f"${cost:,.0f}")
    col3.metric("Margin", f"${margin:,.0f}", delta=f"{margin_pct:.1f}%")

    st.caption(f"**Margin delta from hours overrun:** ${cost_delta_from_hours:,.0f}")


# =============================================================================
# ACT 3b: TASK WATERFALL
# =============================================================================


def compute_task_waterfall(df: pd.DataFrame, job_no: str) -> pd.DataFrame:
    job_data = df[df["job_no"] == job_no].copy()

    unquoted_flag = pd.Series(False, index=job_data.index)
    if "quote_match_flag" in job_data.columns:
        unquoted_flag |= job_data["quote_match_flag"].astype(str).str.lower().ne("matched")
    if "quoted_time_total" in job_data.columns:
        unquoted_flag |= job_data["quoted_time_total"].fillna(0) <= 0
    if "quoted_amount_total" in job_data.columns:
        unquoted_flag |= job_data["quoted_amount_total"].fillna(0) <= 0
    job_data["is_unquoted"] = unquoted_flag

    task_df = job_data.groupby("task_name").agg(
        hours_raw=("hours_raw", "sum"),
        quoted_time_total=("quoted_time_total", "first"),
        is_unquoted=("is_unquoted", "max"),
        base_cost=("base_cost", "sum"),
    ).reset_index()

    task_df["quoted_hours"] = task_df["quoted_time_total"].fillna(0)
    task_df["actual_hours"] = task_df["hours_raw"]
    task_df["hours_variance"] = task_df["actual_hours"] - task_df["quoted_hours"]

    task_df["variance_type"] = np.where(
        task_df["is_unquoted"],
        "Scope Creep (Unquoted)",
        np.where(
            task_df["hours_variance"] > 0,
            "Overrun (Inefficiency)",
            np.where(
                task_df["hours_variance"] < 0,
                "Under-run (Efficiency)",
                "On Target",
            ),
        ),
    )

    total_abs_variance = task_df["hours_variance"].abs().sum()
    task_df["contribution_pct"] = (
        task_df["hours_variance"].abs() / total_abs_variance * 100 if total_abs_variance > 0 else 0
    )

    return task_df.sort_values("hours_variance", ascending=False)


def render_task_waterfall(task_df: pd.DataFrame, quoted_hours_total: float):
    """
    Render task waterfall with top contributors and an 'Other tasks' bucket.
    This avoids visually confusing jumps when many small tasks are included.
    """
    if task_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No task variance available",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(title="Task Contribution to Hours Variance", height=400)
        return fig

    task_df = task_df.copy()
    task_df = task_df[task_df["hours_variance"].notna()]
    task_df["abs_variance"] = task_df["hours_variance"].abs()

    max_tasks = 12
    if len(task_df) > max_tasks:
        display_df = task_df.nlargest(max_tasks, "abs_variance").copy()
        other_variance = task_df["hours_variance"].sum() - display_df["hours_variance"].sum()
        if abs(other_variance) > 0.01:
            display_df = pd.concat(
                [
                    display_df,
                    pd.DataFrame(
                        [{
                            "task_name": "Other tasks",
                            "hours_variance": other_variance,
                            "abs_variance": abs(other_variance),
                        }]
                    ),
                ],
                ignore_index=True,
            )
    else:
        display_df = task_df.copy()

    display_df = display_df.sort_values("hours_variance", ascending=False)

    waterfall_data = [{"task": "Quoted Hours", "value": quoted_hours_total, "type": "absolute"}]

    for _, row in display_df.iterrows():
        if row["hours_variance"] != 0:
            task_label = row["task_name"] if row["task_name"] == "Other tasks" else row["task_name"][:25]
            waterfall_data.append(
                {
                    "task": task_label,
                    "value": row["hours_variance"],
                    "type": "relative",
                }
            )

    total_variance = task_df["hours_variance"].sum()
    waterfall_data.append(
        {
            "task": "Actual Hours",
            "value": quoted_hours_total + total_variance,
            "type": "total",
        }
    )

    wf_df = pd.DataFrame(waterfall_data)

    fig = go.Figure(
        go.Waterfall(
            name="Hours",
            orientation="v",
            measure=wf_df["type"].tolist(),
            x=wf_df["task"].tolist(),
            y=wf_df["value"].tolist(),
            textposition="outside",
            text=[
                f"{v:+,.0f}" if t == "relative" else f"{v:,.0f}"
                for v, t in zip(wf_df["value"], wf_df["type"])
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#C62828"}},
            decreasing={"marker": {"color": "#2E7D32"}},
            totals={"marker": {"color": "#1565C0"}},
        )
    )

    fig.update_layout(
        title="Task Contribution to Hours Variance",
        yaxis_title="Hours",
        showlegend=False,
        height=400,
        xaxis_tickangle=-30,
    )

    return fig


def render_task_table(task_df: pd.DataFrame):
    display_df = task_df[[
        "task_name",
        "quoted_hours",
        "actual_hours",
        "hours_variance",
        "variance_type",
        "contribution_pct",
        "base_cost",
    ]].copy()

    display_df.columns = [
        "Task",
        "Quoted Hrs",
        "Actual Hrs",
        "Variance",
        "Type",
        "Contribution %",
        "Cost",
    ]

    def highlight_type(row):
        if "Scope Creep" in row["Type"]:
            return ["background-color: #FFCDD2"] * len(row)
        if "Overrun" in row["Type"]:
            return ["background-color: #FFE0B2"] * len(row)
        if "Under-run" in row["Type"]:
            return ["background-color: #C8E6C9"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style
            .apply(highlight_type, axis=1)
            .format({
                "Quoted Hrs": "{:,.0f}",
                "Actual Hrs": "{:,.0f}",
                "Variance": "{:+,.0f}",
                "Contribution %": "{:.0f}%",
                "Cost": "${:,.0f}",
            }),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# ACT 3c: FTE ATTRIBUTION
# =============================================================================


def render_fte_attribution(
    df: pd.DataFrame,
    job_no: str,
    task_name: Optional[str] = None,
    job_name_lookup: Optional[Dict[str, str]] = None,
):
    job_data = df[df["job_no"] == job_no]

    job_label = format_job_label(job_no, job_name_lookup)
    if task_name:
        job_data = job_data[job_data["task_name"] == task_name]
        st.subheader(f"👥 Staff Attribution: {task_name} — {job_label}")
    else:
        st.subheader(f"👥 Staff Attribution (All Tasks) — {job_label}")

    staff_df = job_data.groupby("staff_name").agg(
        hours_raw=("hours_raw", "sum"),
        base_cost=("base_cost", "sum"),
    ).reset_index()

    staff_df.columns = ["Staff", "Hours", "Cost"]

    total_hours = staff_df["Hours"].sum()
    total_cost = staff_df["Cost"].sum()

    staff_df["Hours %"] = staff_df["Hours"] / total_hours * 100 if total_hours > 0 else 0
    staff_df["Cost %"] = staff_df["Cost"] / total_cost * 100 if total_cost > 0 else 0
    staff_df["Effective Rate"] = np.where(
        staff_df["Hours"] > 0,
        staff_df["Cost"] / staff_df["Hours"],
        np.nan,
    )

    staff_df = staff_df.sort_values("Hours", ascending=False)

    st.dataframe(
        staff_df.style.format({
            "Hours": "{:,.1f}",
            "Cost": "${:,.0f}",
            "Hours %": "{:.0f}%",
            "Cost %": "{:.0f}%",
            "Effective Rate": "${:,.0f}/hr",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if not staff_df.empty:
        top_staff = staff_df.iloc[0]
        st.info(
            f"**{top_staff['Staff']}** contributed {top_staff['Hours %']:.0f}% of hours "
            f"at ${top_staff['Effective Rate']:,.0f}/hr effective rate"
        )


# =============================================================================
# RECONCILIATION
# =============================================================================


def render_reconciliation_panel(df: pd.DataFrame, job_df: pd.DataFrame, portfolio_metrics: Dict):
    with st.expander("🔍 Data Reconciliation (QA)", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Raw Data Totals**")
            st.write(f"- Total rows: {len(df):,}")
            st.write(f"- Total hours (sum hours_raw): {df['hours_raw'].sum():,.1f}")
            st.write(f"- Total revenue (sum rev_alloc): ${df['rev_alloc'].sum():,.0f}")
            st.write(f"- Unique jobs: {df['job_no'].nunique():,}")

        with col2:
            st.markdown("**Job Aggregation Totals**")
            st.write(f"- Jobs in summary: {len(job_df):,}")
            st.write(f"- Total hours (sum job hours): {job_df['actual_hours_job'].sum():,.1f}")
            st.write(f"- Total revenue (sum job revenue): ${job_df['revenue_job'].sum():,.0f}")

        hours_delta = abs(df["hours_raw"].sum() - job_df["actual_hours_job"].sum())
        rev_delta = abs(df["rev_alloc"].sum() - job_df["revenue_job"].sum())

        if hours_delta < 0.01 and rev_delta < 0.01:
            st.success("✅ Totals reconcile correctly")
        else:
            st.error(
                f"❌ Reconciliation mismatch: Hours delta={hours_delta:.2f}, Revenue delta=${rev_delta:.2f}"
            )


# =============================================================================
# MAIN
# =============================================================================


def main():
    st.set_page_config(
        page_title="Executive Summary | Root Cause Engine",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Executive Summary: Margin & Delivery Diagnosis")

    init_drill_state()

    # ===========================================
    # GLOBAL CONTROL BAR (NEW)
    # ===========================================
    start_date, end_date, job_state = render_global_control_bar()

    st.session_state.global_context = {
        "start_date": start_date,
        "end_date": end_date,
        "job_state": job_state,
    }

    df_raw = load_data()
    df = apply_global_context(df_raw, start_date, end_date, job_state)

    if df.empty:
        st.error(
            f"No data found for: {job_state} jobs from "
            f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
        )
        st.stop()

    job_df = compute_job_metrics(df)
    job_name_lookup = build_job_name_lookup(df, "job_no", "job_name")
    portfolio_metrics = compute_portfolio_metrics(job_df)

    render_breadcrumb()
    st.divider()

    # ===========================================
    # LEVEL 0: PORTFOLIO (Always visible)
    # ===========================================
    st.header("Level 0: Portfolio Overview")
    st.markdown(
        f"<span style='color:#6c757d;font-style:italic;'>"
        f"{job_state} jobs | {start_date.strftime('%d %b')} → {end_date.strftime('%d %b %Y')}"
        f"</span>",
        unsafe_allow_html=True,
    )

    render_kpi_strip(portfolio_metrics)

    st.divider()

    # ===========================================
    # RATE BRIDGE SECTION (Toggle)
    # ===========================================
    st.header("Quote → Realised Rate Bridge")
    st.caption("*Choose a view: mechanical decomposition or three‑forces ownership*")

    scope_view = st.radio(
        "Scope",
        options=["Company (All)", "Department"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="rate_bridge_scope",
    )

    df_bridge = df
    job_df_bridge = job_df
    selected_dept = None

    if scope_view == "Department":
        if "department_final" in job_df.columns:
            dept_options = (
                job_df["department_final"].dropna().sort_values().unique().tolist()
            )
        else:
            dept_options = []

        selected_dept = st.selectbox(
            "Select department",
            options=["-- Select --"] + dept_options,
            key="rate_bridge_department",
        )

        if selected_dept == "-- Select --":
            st.info("Select a department to view its rate bridge.")
            st.stop()

        df_bridge = df[df["department_final"] == selected_dept].copy()
        job_df_bridge = job_df[job_df["department_final"] == selected_dept].copy()

        if df_bridge.empty or job_df_bridge.empty:
            st.warning("No data available for the selected department in this context.")
            st.stop()

    bridge_view = st.radio(
        "Bridge view",
        options=["Mechanical (Quote → Realised)", "Three‑Forces (Owners)"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if bridge_view.startswith("Mechanical"):
        portfolio_metrics_bridge = compute_portfolio_metrics(job_df_bridge)
        col1, col2 = st.columns([2, 1])
        with col1:
            bridge_df = compute_rate_bridge(job_df_bridge, portfolio_metrics_bridge)
            st.plotly_chart(render_rate_bridge(bridge_df), use_container_width=True)
            st.markdown(
                """
**How the Quote → Realised Rate bridge is calculated (plain English)**

We start with the **quoted hourly rate** for the portfolio, then explain how actual delivery changes the realised rate.

1. **Quote Rate (starting point)**  
   Total quoted value divided by total quoted hours, across all quoted jobs.

2. **Under‑run impact (efficiency wins)**  
   Jobs that finished **under** their quoted hours create “saved hours.”  
   We value those saved hours at the **quote rate** and spread the impact across total actual hours.

3. **Overrun impact (execution leakage)**  
   Jobs that ran **over** their quoted hours consume extra time at the quote rate.  
   This reduces the realised rate once spread across total actual hours.

4. **Unquoted hours impact (scope creep)**  
   Hours with no matching quote are also valued at the quote rate and treated as leakage.

5. **Mix/Pricing effect (the residual)**  
   Whatever difference remains after steps 2–4 is attributed to **mix and pricing**.

**Bottom line:** Realised rate = Quote rate + (under‑runs) − (overruns) − (unquoted hours) + (mix/pricing).
"""
            )
        with col2:
            st.info(
                """
**The Contradiction Explained**

Realised rate can exceed quote rate even when most jobs overrun.
This is due to **mix effects** — some jobs subsidise others.

👇 **Switch to Three‑Forces view** to see owner‑level drivers.
"""
            )
    else:
        from src.ui.rate_bridge_components import render_full_rate_bridge

        render_full_rate_bridge(
            df_bridge,
            job_df_bridge,
            show_department_comparison=(scope_view != "Department"),
        )

    st.divider()

    # ===========================================
    # LEVEL 1: DEPARTMENT DIAGNOSTIC
    # ===========================================
    st.header("Level 1: Department Diagnostic")
    st.caption("*Who is creating vs eroding value — and WHY?*")

    dept_df = compute_department_scorecard(job_df)

    col1, col2 = st.columns(2)
    with col1:
        render_department_scorecard(dept_df)
    with col2:
        st.plotly_chart(render_dept_efficiency_scatter(dept_df), use_container_width=True)
        st.plotly_chart(render_dept_contribution_bars(dept_df), use_container_width=True)

    if "department_final" in df.columns and df["department_final"].notna().any():
        dept_options = sorted(df["department_final"].dropna().astype(str).unique().tolist())
        if dept_options:
            default_dept = st.session_state.drill_path.get("department")
            if default_dept not in dept_options:
                default_dept = dept_options[0]
            selected_overrun_dept = st.selectbox(
                "Department scope for recurring overruns",
                options=dept_options,
                index=dept_options.index(default_dept),
                key="l1_overrun_department_scope",
            )
            dept_scope_df = df[df["department_final"].astype(str) == selected_overrun_dept].copy()
            render_recurring_task_overruns_section(
                df_scope=dept_scope_df,
                scope_label=f"Department: {selected_overrun_dept}",
                department=selected_overrun_dept,
                category=None,
            )

    if not can_proceed_to_level("category"):
        render_department_selection_gate(dept_df)
        st.stop()

    st.divider()

    # ===========================================
    # LEVEL 2: CATEGORY DIAGNOSTIC
    # ===========================================
    dept = st.session_state.drill_path["department"]
    dept_ctx = st.session_state.drill_path["department_context"]

    st.header(f"Level 2: Category Diagnostic — {dept}")
    st.caption("*Is this structural or job-specific?*")

    emoji = {"Accretive": "🟢", "Mixed": "🟡", "Erosive": "🔴"}[dept_ctx["classification"]]
    st.info(
        f"""
**Analysing: {emoji} {dept}** ({dept_ctx['classification']})

You're here because: {dept_ctx['reasons'][0] if dept_ctx['reasons'] else 'Selected for analysis'}
"""
    )

    cat_df = compute_category_scorecard(job_df, dept)

    col1, col2 = st.columns(2)
    with col1:
        render_category_scorecard(cat_df)
    with col2:
        st.plotly_chart(render_category_efficiency_scatter(cat_df), use_container_width=True)

    dept_scope_df = df[df["department_final"] == dept].copy() if "department_final" in df.columns else df.copy()
    if not dept_scope_df.empty:
        category_col = get_category_col(dept_scope_df)
        category_map: Dict[str, object] = {}
        if category_col in dept_scope_df.columns:
            non_null_values = sorted(
                dept_scope_df[category_col].dropna().unique().tolist(),
                key=lambda x: str(x),
            )
            for value in non_null_values:
                category_map[str(value)] = value
            if dept_scope_df[category_col].isna().any():
                category_map["(Uncategorised)"] = "(Uncategorised)"

        if category_map:
            category_labels = list(category_map.keys())
            selected_cat = st.session_state.drill_path.get("category")
            if selected_cat is None and st.session_state.drill_path.get("category_context") is not None:
                default_label = "(Uncategorised)" if "(Uncategorised)" in category_labels else category_labels[0]
            elif selected_cat is not None and str(selected_cat) in category_labels:
                default_label = str(selected_cat)
            else:
                default_label = category_labels[0]

            selected_category_label = st.selectbox(
                "Category scope for recurring overruns",
                options=category_labels,
                index=category_labels.index(default_label),
                key=f"l2_overrun_category_scope_{dept}",
            )
            selected_category_value = category_map[selected_category_label]

            render_recurring_task_overruns_section(
                df_scope=dept_scope_df,
                scope_label=f"Category: {dept} : {selected_category_label}",
                department=dept,
                category=selected_category_value,
            )

    if not can_proceed_to_level("job"):
        render_category_selection_gate(cat_df)
        st.stop()

    st.divider()

    # ===========================================
    # LEVEL 3: JOB QUADRANT
    # ===========================================
    cat = st.session_state.drill_path["category"]
    cat_ctx = st.session_state.drill_path["category_context"]

    st.header(f"Level 3: Job Analysis — {cat or '(Uncategorised)'}")
    st.caption("*Which jobs subsidise vs destroy margin?*")

    emoji = {"Accretive": "🟢", "Mixed": "🟡", "Erosive": "🔴"}[cat_ctx["classification"]]
    st.info(
        f"""
**Analysing: {emoji} {cat or '(Uncategorised)'}** ({cat_ctx['classification']})

You're here because: {cat_ctx['reasons'][0] if cat_ctx['reasons'] else 'Selected for analysis'}
"""
    )

    filtered_jobs = job_df[
        (job_df["department_final"] == dept)
        & ((job_df["job_category"] == cat) | (job_df["job_category"].isna() & (cat is None)))
    ].copy()

    impact_mode = st.radio(
        "Size bubbles by:",
        options=["Revenue Exposure", "Rate Impact", "Margin Impact"],
        horizontal=True,
    )

    st.plotly_chart(
        render_job_quadrant_scatter(filtered_jobs, impact_mode, job_name_lookup),
        use_container_width=True,
    )

    st.divider()

    # ===========================================
    # LEVEL 3b: SEGMENT PROFILING
    # ===========================================
    st.header("Level 3b: Segment Analysis — What Differentiates Performance?")
    st.caption("*Understand the profile before drilling into individual jobs*")

    from src.metrics.segment_profiling import (
        SEGMENT_CONFIG,
        compute_job_segments,
        compute_job_shortlist,
        compute_segment_profile,
    )
    from src.ui.segment_components import (
        render_segment_selector,
        render_segment_quadrant_legend,
        render_composition_panel,
        render_driver_distributions,
        render_task_mix_divergence,
        render_duration_profile,
        render_overrun_decomposition,
        render_job_shortlist,
    )

    job_df_with_segments = compute_job_segments(df, filtered_jobs)

    render_segment_quadrant_legend()
    selected_segment = render_segment_selector(job_df_with_segments)

    if not selected_segment:
        st.info(
            "👆 **Select a segment above** to see its profile and understand what makes these jobs different."
        )
        st.stop()

    profile = compute_segment_profile(
        df,
        job_df_with_segments,
        selected_segment,
        benchmark_dept=dept,
        benchmark_category=cat,
    )

    if not profile:
        st.warning(f"No jobs in the '{selected_segment}' segment for current filters.")
        st.stop()

    config = SEGMENT_CONFIG.get(selected_segment, {})
    st.markdown(
        f"""
### {config.get('icon', '●')} {selected_segment}: {config.get('short_desc', '')}

> {config.get('explanation', '')}
"""
    )

    render_composition_panel(profile["composition"])
    st.divider()
    render_driver_distributions(profile["drivers"])
    st.divider()
    render_task_mix_divergence(profile["task_mix"], selected_segment)
    st.divider()
    render_duration_profile(profile["duration"], selected_segment)
    st.divider()
    render_overrun_decomposition(profile["overrun_decomposition"])
    st.divider()

    shortlist = compute_job_shortlist(job_df_with_segments, selected_segment, n=10)
    render_job_shortlist(shortlist, job_name_lookup)

    if not can_proceed_to_level("task"):
        st.stop()

    st.divider()

    # ===========================================
    # LEVEL 4: TASK & FTE ROOT CAUSE
    # ===========================================
    job_no = st.session_state.drill_path["job"]

    job_label = format_job_label(job_no, job_name_lookup)
    st.header(f"Level 4: Root Cause — {job_label}")
    render_job_diagnostic_card(df, job_no, job_name_lookup)

    st.divider()

    st.subheader("Task Contribution Waterfall")
    task_df = compute_task_waterfall(df, job_no)
    quoted_hours = task_df["quoted_hours"].sum()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(render_task_waterfall(task_df, quoted_hours), use_container_width=True)
    with col2:
        render_task_table(task_df)

    st.divider()

    task_selection = st.selectbox(
        "View staff for:",
        options=["All Tasks"] + task_df["task_name"].tolist(),
    )

    selected_task = None if task_selection == "All Tasks" else task_selection
    render_fte_attribution(df, job_no, selected_task, job_name_lookup)

    st.divider()
    render_reconciliation_panel(df, job_df, portfolio_metrics)


if __name__ == "__main__":
    main()
