"""
Service Load Analyzer page.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.data.semantic import get_category_col
from src.metrics.service_load import (
    build_job_description_index,
    define_service_scope,
    compute_staff_service_load,
    compute_staff_client_breakdown,
    compute_staff_task_breakdown,
    compute_scope_task_breakdown,
    compute_staff_task_capacity_flow,
    compute_scope_budget_comparison,
    compute_new_client_absorption,
    compute_scope_weekly_trend,
)
from src.ui.formatting import fmt_hours, fmt_percent


st.set_page_config(page_title="Service Load Analyzer", page_icon="⚖️", layout="wide")

PLOT_MARGINS = dict(l=40, r=20, t=40, b=40)
LOOKBACK_OPTIONS = {
    "Last 1 Month": 1,
    "Last 2 Months": 2,
    "Last 3 Months": 3,
    "Last 6 Months": 6,
}
STATUS_STYLES = {
    "Under Budget": "background: #dcfce7; color: #166534;",
    "On Track": "background: #f0fdf4; color: #15803d;",
    "Over Budget": "background: #fff7ed; color: #9a3412;",
    "Way Over": "background: #fef2f2; color: #991b1b;",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .dc-section-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #667085;
            margin: 0.2rem 0 0.45rem 0;
        }
        .dc-section-title {
            font-size: 1rem;
            font-weight: 650;
            color: #111827;
            margin: 0 0 0.5rem 0;
        }
        .dc-soft-divider {
            height: 1px;
            background: linear-gradient(90deg, #e2e8f0 0%, rgba(226, 232, 240, 0.25) 100%);
            margin: 0.7rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _parse_keywords(raw_text: str) -> list[str]:
    return [part.strip().lower() for part in (raw_text or "").split(",") if part.strip()]


def _truncate_text(value, max_len: int = 120) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _unique_truncated_label_map(values: list[str], max_len: int = 40) -> dict[str, str]:
    """
    Build a stable value -> display label mapping with unique truncated labels.
    """
    mapping: dict[str, str] = {}
    used_labels: set[str] = set()

    for raw in values:
        text = str(raw)
        base = _truncate_text(text, max_len=max_len)
        label = base
        suffix_idx = 2
        while label in used_labels:
            label = f"{base} ({suffix_idx})"
            suffix_idx += 1
        mapping[text] = label
        used_labels.add(label)

    return mapping


def _status_from_budget_pct(used_pct: float) -> str:
    if pd.isna(used_pct):
        return "N/A"
    if used_pct < 80:
        return "Under Budget"
    if used_pct <= 100:
        return "On Track"
    if used_pct <= 120:
        return "Over Budget"
    return "Way Over"


def _render_explainer_expander(
    title: str,
    points: list[str],
    expanded: bool = False,
    as_expander: bool = True,
) -> None:
    if as_expander:
        with st.expander(f"ℹ️ {title}", expanded=expanded):
            for point in points:
                st.markdown(f"- {point}")
        return

    st.markdown(f"**ℹ️ {title}**")
    for point in points:
        st.markdown(f"- {point}")


def _staff_load_display(staff_df: pd.DataFrame, scope_budget: float) -> pd.io.formats.style.Styler:
    display = pd.DataFrame(
        {
            "Staff": staff_df["staff_name"],
            "In-Scope Hrs/Mo": staff_df["in_scope_hours_per_month"].apply(fmt_hours),
            "In-Scope Hrs/Wk": staff_df["in_scope_hours_per_week"].apply(fmt_hours),
            "Other Hrs/Mo": staff_df["out_of_scope_hours_per_month"].apply(fmt_hours),
            "Total Hrs/Mo": staff_df["total_hours_per_month"].apply(fmt_hours),
            "Capacity": staff_df.apply(
                lambda r: f"{fmt_hours(r['monthly_capacity'])}{' (assumed)' if bool(r.get('fte_assumed', False)) else ''}",
                axis=1,
            ),
            "Capacity Used %": staff_df["capacity_used_pct"].apply(fmt_percent),
            "Scope Budget %": (
                staff_df["scope_budget_used_pct"].apply(fmt_percent) if scope_budget > 0 else "N/A"
            ),
            "Headroom Hrs/Mo": staff_df["headroom_hours_per_month"].apply(fmt_hours),
            "Can Absorb": staff_df["absorption_estimate"]
            .fillna(0)
            .astype(int)
            .map(lambda x: f"{x} clients"),
            "Status": staff_df["scope_status"],
        }
    )

    headroom_map = staff_df["headroom_hours_per_month"].to_dict()
    status_map = staff_df["scope_status"].to_dict()

    def _row_style(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        idx = row.name
        headroom_idx = display.columns.get_loc("Headroom Hrs/Mo")
        status_idx = display.columns.get_loc("Status")

        if idx in headroom_map and pd.notna(headroom_map[idx]) and headroom_map[idx] < 0:
            styles[headroom_idx] = "color: #991b1b; font-weight: 600;"

        status = status_map.get(idx)
        if status in STATUS_STYLES:
            styles[status_idx] = STATUS_STYLES[status]
        return styles

    return display.style.apply(_row_style, axis=1)


def _build_scope_trend_chart(
    trend_df: pd.DataFrame,
    scope_budget: float,
    in_scope_staff_count: int,
) -> go.Figure:
    fig = go.Figure()
    for staff_name in trend_df.columns:
        hours = trend_df[staff_name]
        if pd.to_numeric(hours, errors="coerce").fillna(0).sum() <= 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=trend_df.index,
                y=hours,
                mode="lines",
                stackgroup="one",
                name=str(staff_name),
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.1f}h<extra></extra>",
            )
        )

    if scope_budget > 0 and in_scope_staff_count > 0:
        team_weekly_budget = scope_budget * in_scope_staff_count / 4.33
        fig.add_hline(
            y=team_weekly_budget,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="Team weekly budget",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Weekly In-Scope Hours by Team Member",
        template="plotly_white",
        margin=PLOT_MARGINS,
        hovermode="x unified",
        xaxis_title="Week",
        yaxis_title="Hours",
    )
    return fig


def _scenario_row_style(row: pd.Series) -> list[str]:
    cap_raw = str(row["Projected Capacity %"]).replace("%", "").replace(",", "").strip()
    cap_pct = pd.to_numeric(cap_raw, errors="coerce")
    base_style = ""
    if pd.notna(cap_pct):
        if cap_pct > 100:
            base_style = "background: #fef2f2;"
        elif cap_pct >= 90:
            base_style = "background: #fff7ed;"
        else:
            base_style = "background: #f0fdf4;"
    return [base_style] * len(row)


def _build_staff_so_what_summary(
    selected_staff: str,
    selected_row: pd.Series,
    client_breakdown: pd.DataFrame,
) -> dict:
    capacity_used_pct = float(pd.to_numeric(selected_row.get("capacity_used_pct"), errors="coerce"))
    headroom_hours = float(pd.to_numeric(selected_row.get("headroom_hours_per_month"), errors="coerce"))
    scope_share_pct = float(pd.to_numeric(selected_row.get("scope_share_pct"), errors="coerce"))
    in_scope_month = float(pd.to_numeric(selected_row.get("in_scope_hours_per_month"), errors="coerce"))
    out_scope_month = float(pd.to_numeric(selected_row.get("out_of_scope_hours_per_month"), errors="coerce"))
    absorption_raw = pd.to_numeric(selected_row.get("absorption_estimate"), errors="coerce")
    absorption = int(max(absorption_raw, 0)) if pd.notna(absorption_raw) else 0

    top_driver_label = "No dominant client/job detected"
    top_driver_share = np.nan
    top_driver_hours = np.nan
    if len(client_breakdown) > 0:
        top_driver = client_breakdown.sort_values("total_hours", ascending=False).iloc[0]
        top_label = (
            f"{top_driver.get('client', 'Unknown')} — "
            f"{top_driver.get('job_name') if pd.notna(top_driver.get('job_name')) else top_driver.get('job_no')}"
        )
        top_driver_label = _truncate_text(str(top_label), max_len=64)
        top_driver_share = float(pd.to_numeric(top_driver.get("share_of_staff_total_pct"), errors="coerce"))
        top_driver_hours = float(pd.to_numeric(top_driver.get("hours_per_month"), errors="coerce"))

    if headroom_hours < 0 or capacity_used_pct > 100:
        signal = "Critical"
        message = (
            f"So what: {selected_staff} is overloaded by {fmt_hours(abs(headroom_hours))}h/mo. "
            "Rebalance work before assigning new scope."
        )
    elif capacity_used_pct >= 90:
        signal = "Tight"
        message = (
            f"So what: {selected_staff} has limited buffer ({fmt_hours(max(headroom_hours, 0))}h/mo). "
            f"Likely can absorb at most {absorption} new client(s) at current assumptions."
        )
    else:
        signal = "Available"
        message = (
            f"So what: {selected_staff} has usable capacity ({fmt_hours(max(headroom_hours, 0))}h/mo). "
            f"Can likely absorb around {absorption} new client(s) without overload."
        )

    return {
        "signal": signal,
        "message": message,
        "capacity_used_pct": capacity_used_pct,
        "headroom_hours": headroom_hours,
        "scope_share_pct": scope_share_pct,
        "in_scope_month": in_scope_month,
        "out_scope_month": out_scope_month,
        "absorption": absorption,
        "top_driver_label": top_driver_label,
        "top_driver_share": top_driver_share,
        "top_driver_hours": top_driver_hours,
    }


def _render_staff_so_what_summary(summary: dict) -> None:
    signal = summary["signal"]
    if signal == "Critical":
        st.error(summary["message"])
    elif signal == "Tight":
        st.warning(summary["message"])
    else:
        st.success(summary["message"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capacity Used", fmt_percent(summary["capacity_used_pct"]))
    c1.caption(f"Headroom: {fmt_hours(summary['headroom_hours'])}h/mo")

    c2.metric("Scope Share", fmt_percent(summary["scope_share_pct"]))
    c2.caption(f"In-scope: {fmt_hours(summary['in_scope_month'])}h/mo")

    c3.metric(
        "Top Driver Share",
        fmt_percent(summary["top_driver_share"]) if pd.notna(summary["top_driver_share"]) else "—",
    )
    top_hours_text = (
        f"{fmt_hours(summary['top_driver_hours'])}h/mo"
        if pd.notna(summary["top_driver_hours"])
        else "—"
    )
    c3.caption(f"{summary['top_driver_label']} ({top_hours_text})")

    c4.metric("Absorption Signal", f"{summary['absorption']} clients")
    c4.caption(f"Out-of-scope: {fmt_hours(summary['out_scope_month'])}h/mo")


def main() -> None:
    _inject_styles()

    st.title("⚖️ Service Load Analyzer")
    st.caption(
        "Search job descriptions to define a service scope \u2192 see who's working on it \u2192 assess capacity for new engagements"
    )
    _render_explainer_expander(
        "How To Use This Page",
        [
            "Define scope using description keywords first, then narrow with category, client, and task filters.",
            "Review matched jobs before analysis to remove false positives using `Exclude jobs from scope`.",
            "Treat `In-Scope Hrs/Mo` as service-load demand and `Total Hrs/Mo` as the full workload context.",
            "Use `Task Capacity Flow` to see which tasks are consuming the most scoped capacity.",
            "Use `Scope Budget %` to compare in-scope demand against your target monthly manager allocation.",
            "Use `Capacity Used %` and `Headroom Hrs/Mo` to assess overload risk from all work, not just this scope.",
            "Use the scenario planner to test assignment choices before accepting new client load.",
        ],
        expanded=False,
    )

    df = load_fact_timesheet()
    job_index = build_job_description_index(df)

    has_job_description = "job_description" in df.columns
    has_job_name = "job_name" in df.columns
    has_work_date = "work_date" in df.columns and pd.to_datetime(df["work_date"], errors="coerce").notna().any()
    keyword_disabled = not has_job_description and not has_job_name

    with st.container(border=True):
        st.markdown("<div class='dc-section-label'>Inputs</div>", unsafe_allow_html=True)
        st.markdown("<div class='dc-section-title'>🔍 Define Service Scope</div>", unsafe_allow_html=True)
        st.caption("Start with description keywords, then narrow with category, client, or task filters.")
        _render_explainer_expander(
            "Scope Definition Logic",
            [
                "Keywords are the primary mechanism and use OR logic (a job matches if any keyword appears).",
                "Category, client, and task filters are secondary refinements.",
                "When keywords and secondary filters are both provided, secondary filters narrow keyword matches via intersection.",
                "If you only use secondary filters, scope is built directly from those filters.",
                "Matched jobs include a `Match Reason` trail so you can verify exactly why each job was included.",
            ],
            expanded=False,
        )

        keyword_input = st.text_input(
            "Search job descriptions",
            placeholder="e.g. organic social, content scheduling, community management",
            disabled=keyword_disabled,
        )
        st.caption(
            "Searches [Job] Description and Job Name fields. Use commas to separate multiple keywords (matched with OR logic)."
        )

        if not has_job_description and has_job_name:
            st.info(
                "Job descriptions not available — searching job names only. Run the latest parsing notebook to add descriptions."
            )
        if keyword_disabled:
            st.warning(
                "Job names and descriptions are unavailable, so keyword search is disabled. Use category/client filters to define scope."
            )

        category_col = get_category_col(df) if len(df) > 0 else "job_category"
        category_options = (
            sorted(df[category_col].dropna().astype(str).unique().tolist())
            if category_col in df.columns
            else []
        )
        client_options = (
            sorted(df["client"].dropna().astype(str).unique().tolist())
            if "client" in df.columns
            else []
        )
        task_options = (
            sorted(df["task_name"].dropna().astype(str).unique().tolist())
            if "task_name" in df.columns
            else []
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            selected_categories = st.multiselect("Narrow by Category", options=category_options)
        with c2:
            selected_clients = st.multiselect("Narrow by Client", options=client_options)
        with c3:
            selected_tasks = st.multiselect("Narrow by Job Task", options=task_options)
        with c4:
            lookback_label = st.selectbox(
                "Lookback Period",
                options=list(LOOKBACK_OPTIONS.keys()),
                index=2,
            )

        p1, p2 = st.columns(2)
        with p1:
            scope_budget = st.number_input(
                "Scope budget (hrs/month per manager)",
                min_value=0,
                value=80,
                step=10,
            )
        with p2:
            avg_new_client_hours = st.number_input(
                "Avg hrs/month per new client",
                min_value=1,
                value=20,
                step=5,
            )

    keywords = _parse_keywords(keyword_input)
    has_secondary_filters = (
        len(selected_categories) > 0 or len(selected_clients) > 0 or len(selected_tasks) > 0
    )

    if keyword_disabled and not has_secondary_filters:
        st.warning("Enter keywords or select filters to define a service scope.")
        st.stop()

    if len(keywords) == 0 and not has_secondary_filters:
        st.warning("Enter keywords or select filters above to define a service scope.")
        st.stop()

    lookback_months = LOOKBACK_OPTIONS[lookback_label]
    scope_mask, matched_jobs_df = define_service_scope(
        df,
        job_index,
        description_keywords=keywords,
        categories=selected_categories,
        clients=selected_clients,
        task_names=selected_tasks,
    )

    if len(matched_jobs_df) == 0 or not scope_mask.any():
        keyword_text = ", ".join(keywords) if keywords else "(no keywords)"
        st.info(f"No jobs found matching '{keyword_text}'. Try broader terms.")
        with st.expander("Browse available job descriptions", expanded=False):
            browse_df = job_index.copy()
            browse_df["job_description"] = browse_df["job_description"].apply(_truncate_text)
            st.dataframe(
                browse_df.rename(
                    columns={
                        "job_no": "Job No",
                        "job_name": "Job Name",
                        "job_description": "Description",
                    }
                )[["Job No", "Job Name", "Description"]],
                use_container_width=True,
                hide_index=True,
            )
        st.stop()

    st.markdown("<div class='dc-section-label'>Section A</div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-title'>Matched Jobs Review</div>", unsafe_allow_html=True)
    st.caption("Validate what the scope filter captured before acting on capacity analytics.")
    _render_explainer_expander(
        "How To Validate Matches",
        [
            "Check each row's `Match Reason` first to catch accidental matches.",
            "Use `Exclude jobs from scope` to quickly remove jobs that mention similar language but are not the intended service.",
            "If too few jobs are matched, broaden keywords; if too many are matched, add task/category/client refinements.",
            "Analytics below are computed only from this final matched set after exclusions.",
        ],
        expanded=False,
    )
    with st.expander(f"📋 Matched Jobs ({len(matched_jobs_df)} jobs)", expanded=True):
        matched_view = matched_jobs_df.copy()
        matched_view["job_description"] = matched_view["job_description"].apply(_truncate_text)
        matched_view = matched_view.rename(
            columns={
                "job_no": "Job No",
                "job_name": "Job Name",
                "job_description": "Description",
                "match_reason": "Match Reason",
            }
        )
        st.dataframe(
            matched_view[["Job No", "Job Name", "Description", "Match Reason"]],
            use_container_width=True,
            hide_index=True,
        )

        matched_job_nos = sorted(matched_jobs_df["job_no"].dropna().astype(str).unique().tolist())
        excluded_job_nos = st.multiselect(
            "Exclude jobs from scope",
            options=matched_job_nos,
        )

        client_count = (
            df[df["job_no"].astype(str).isin(matched_job_nos)]["client"].nunique()
            if "client" in df.columns
            else 0
        )
        st.caption(f"Matched {len(matched_job_nos)} jobs across {client_count} clients")

    if len(excluded_job_nos) > 0:
        exclude_set = set(excluded_job_nos)
        scope_mask = scope_mask & ~df["job_no"].astype(str).isin(exclude_set)
        matched_jobs_df = matched_jobs_df[~matched_jobs_df["job_no"].astype(str).isin(exclude_set)].copy()

    if len(matched_jobs_df) == 0 or not scope_mask.any():
        st.warning("All matched jobs were excluded. Refine your scope filters and try again.")
        st.stop()

    staff_load_df = compute_staff_service_load(
        df,
        scope_mask=scope_mask,
        lookback_months=lookback_months,
    )
    staff_load_df = staff_load_df[
        (staff_load_df["in_scope_hours"] > 0) & (staff_load_df["total_hours"] > 0)
    ].copy()

    if len(staff_load_df) == 0:
        st.warning("No staff logged in-scope hours for the selected scope and lookback window.")
        st.stop()

    staff_load_df = compute_scope_budget_comparison(staff_load_df, scope_budget_hours_per_month=float(scope_budget))
    staff_load_df = compute_new_client_absorption(
        staff_load_df,
        avg_hours_per_new_client_per_month=float(avg_new_client_hours),
    )
    staff_load_df = staff_load_df.sort_values("in_scope_hours_per_month", ascending=False)

    months_covered = int(staff_load_df["months_in_window"].iloc[0]) if "months_in_window" in staff_load_df.columns else 0
    st.markdown("<div class='dc-soft-divider'></div>", unsafe_allow_html=True)
    st.caption(f"Lookback window covers {months_covered} month(s) of available data.")

    st.markdown("<div class='dc-section-label'>Section B</div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-title'>Portfolio Overview</div>", unsafe_allow_html=True)
    st.caption("High-level pulse of team load, monthly scope effort, and budget pressure.")
    _render_explainer_expander(
        "How To Read Portfolio KPIs",
        [
            "`Staff Involved` counts people with non-zero in-scope hours in the selected window.",
            "`Total Scope Hours/Month` is aggregate in-scope hours normalized by months covered.",
            "`Avg Capacity Used` reflects total workload pressure (`Total Hrs/Mo / Capacity`), not just this scope.",
            "`Avg Scope Budget Used` reflects service-pressure against target (`In-Scope Hrs/Mo / Scope Budget`).",
        ],
        expanded=False,
    )

    in_scope_staff_count = int((staff_load_df["in_scope_hours"] > 0).sum())
    total_scope_hours_month = float(staff_load_df["in_scope_hours_per_month"].sum())
    avg_capacity_used = float(staff_load_df["capacity_used_pct"].mean())
    avg_scope_budget_used = (
        float(staff_load_df["scope_budget_used_pct"].mean())
        if float(scope_budget) > 0 and staff_load_df["scope_budget_used_pct"].notna().any()
        else np.nan
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Staff Involved", f"{in_scope_staff_count}")
    k1.caption("People who logged time on matched jobs")
    k2.metric("Total Scope Hours/Month", fmt_hours(total_scope_hours_month))
    k2.caption("Aggregate monthly effort on this service")
    k3.metric("Avg Capacity Used", fmt_percent(avg_capacity_used))
    k3.caption("How loaded the team is overall (all work, not just scope)")
    k4.metric("Avg Scope Budget Used", fmt_percent(avg_scope_budget_used) if pd.notna(avg_scope_budget_used) else "N/A")
    k4.caption("How much of the per-manager scope budget is consumed")

    st.markdown("<div class='dc-soft-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-label'>Section C</div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-title'>Staff Load Table</div>", unsafe_allow_html=True)
    st.caption("Primary operating view of in-scope load, total load, and individual headroom.")
    _render_explainer_expander(
        "How To Read Staff Load",
        [
            "`In-Scope Hrs/Mo` is monthly demand from matched jobs for this service scope.",
            "`Other Hrs/Mo` is all non-scope work and shows what is already consuming time.",
            "`Capacity Used %` can be high even when `Scope Budget %` is low, indicating non-scope pressure.",
            "`Headroom Hrs/Mo` below zero means the person is already overloaded.",
            "`Can Absorb` is a conservative estimate based on both capacity headroom and remaining scope budget (when budget > 0).",
            "`Status` is driven by scope budget thresholds: <80 Under Budget, 80-100 On Track, 100-120 Over Budget, >120 Way Over.",
        ],
        expanded=False,
    )

    styled_staff_table = _staff_load_display(staff_load_df, scope_budget=float(scope_budget))
    st.dataframe(styled_staff_table, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Staff Load CSV",
        data=staff_load_df.to_csv(index=False).encode("utf-8"),
        file_name="service_load_staff_table.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.markdown("<div class='dc-soft-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-label'>Section D</div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-title'>Scope Trend</div>", unsafe_allow_html=True)
    st.caption("Trend of in-scope effort by person to see ramp-up, handoffs, and workload shifts.")
    _render_explainer_expander(
        "How To Interpret Trend",
        [
            "Each stacked band shows a staff member's in-scope hours by week (or by month when work dates are unavailable).",
            "A rising total stack indicates growing service demand.",
            "A widening individual band indicates concentration risk on that person.",
            "The red dashed line is the implied team weekly scope budget benchmark.",
        ],
        expanded=False,
    )

    trend_df = compute_scope_weekly_trend(
        df,
        scope_mask=scope_mask,
        staff_names=staff_load_df["staff_name"].tolist(),
        lookback_months=lookback_months,
    )
    if len(trend_df) == 0:
        st.info("No in-scope trend data available for the selected filters.")
    else:
        trend_fig = _build_scope_trend_chart(
            trend_df=trend_df,
            scope_budget=float(scope_budget),
            in_scope_staff_count=in_scope_staff_count,
        )
        st.plotly_chart(trend_fig, use_container_width=True)

    if not has_work_date:
        st.info("Work-date detail is unavailable, so trend granularity is monthly.")

    st.markdown("<div class='dc-soft-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-label'>Section E</div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-title'>Task Capacity Flow</div>", unsafe_allow_html=True)
    st.caption("Track which tasks across scoped jobs consume the most team capacity.")
    _render_explainer_expander(
        "How To Read Task Capacity Flow",
        [
            "Task rows aggregate hours only from jobs currently included in scope.",
            "`Hrs/Mo` shows monthly-normalized demand by task over the selected lookback window.",
            "`Share of Scope %` indicates concentration; high top-task share means delivery risk is task-concentrated.",
            "`Staff` and `Jobs` show how distributed each task is across people and engagements.",
            "`Billable %` highlights tasks absorbing effort without direct billable recovery.",
        ],
        expanded=False,
    )

    task_breakdown_df = compute_scope_task_breakdown(
        df,
        scope_mask=scope_mask,
        lookback_months=lookback_months,
    )
    if len(task_breakdown_df) == 0:
        st.info("No scoped task records available for this selection.")
    else:
        top_task = task_breakdown_df.iloc[0]
        top_three_share = float(task_breakdown_df["share_of_scope_pct"].head(3).sum())
        multi_staff_tasks = int((task_breakdown_df["staff_count"] >= 2).sum())

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Tracked Tasks", f"{len(task_breakdown_df):,}")
        t1.caption("Distinct tasks across scoped jobs")
        t2.metric("Top Task Share", fmt_percent(float(top_task["share_of_scope_pct"])))
        t2.caption(
            f"{_truncate_text(str(top_task['task_name']), max_len=38)} ({fmt_hours(float(top_task['hours_per_month']))}h/mo)"
        )
        t3.metric("Top 3 Task Share", fmt_percent(top_three_share))
        t3.caption("Concentration across highest-load tasks")
        t4.metric("Multi-Staff Tasks", f"{multi_staff_tasks}")
        t4.caption("Tasks touched by 2+ staff members")

        max_top_n = int(min(25, len(task_breakdown_df)))
        default_top_n = int(min(12, max_top_n))
        top_n_tasks = st.slider(
            "Top tasks to display",
            min_value=1,
            max_value=max_top_n,
            value=default_top_n,
        )

        chart_df = task_breakdown_df.head(top_n_tasks).copy().sort_values("hours_per_month", ascending=True)
        chart_df["task_label"] = chart_df["task_name"].astype(str).apply(lambda x: _truncate_text(x, max_len=42))
        fig_task = go.Figure(
            data=[
                go.Bar(
                    y=chart_df["task_label"],
                    x=chart_df["hours_per_month"],
                    orientation="h",
                    marker_color="#0ea5e9",
                    hovertemplate=(
                        "%{y}<br>Hrs/Mo: %{x:.1f}"
                        "<br>Share: %{customdata[0]:.1f}%"
                        "<br>Staff: %{customdata[1]}"
                        "<br>Jobs: %{customdata[2]}"
                        "<br>Billable: %{customdata[3]:.1f}%<extra></extra>"
                    ),
                    customdata=np.column_stack(
                        [
                            chart_df["share_of_scope_pct"],
                            chart_df["staff_count"],
                            chart_df["job_count"],
                            chart_df["billable_pct"],
                        ]
                    ),
                )
            ]
        )
        fig_task.update_layout(
            template="plotly_white",
            margin=PLOT_MARGINS,
            xaxis_title="Hours / Month",
            yaxis_title="Task",
        )
        st.plotly_chart(fig_task, use_container_width=True)

        task_table = pd.DataFrame(
            {
                "Task": task_breakdown_df["task_name"].astype(str),
                "Hrs/Mo": task_breakdown_df["hours_per_month"].apply(fmt_hours),
                "Share of Scope %": task_breakdown_df["share_of_scope_pct"].apply(fmt_percent),
                "Staff": task_breakdown_df["staff_count"].astype(int),
                "Jobs": task_breakdown_df["job_count"].astype(int),
                "Clients": task_breakdown_df["client_count"].astype(int),
                "Billable %": task_breakdown_df["billable_pct"].apply(fmt_percent),
            }
        )
        st.dataframe(task_table, use_container_width=True, hide_index=True)

    st.markdown("<div class='dc-soft-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-label'>Section F</div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-title'>Staff Deep Dive</div>", unsafe_allow_html=True)
    st.caption("Drill from person to client/job and task detail to understand what consumes their time.")
    _render_explainer_expander(
        "How To Use Deep Dive",
        [
            "Start with `So What Summary` for the immediate decision signal.",
            "Use `Client × Job Breakdown` to identify what specifically drives that person's load.",
            "Use `Staff Task Capacity Flow` to see which scoped tasks are consuming this person's capacity.",
            "Use `Task → Constituent Job Summary` to trace each task back to the jobs creating that load.",
            "Use `Hours Composition` to compare scope demand versus other work and client concentration.",
            "Use `Task Drill-Down` to identify task-level opportunities for reallocation, automation, or scope reset.",
        ],
        expanded=False,
    )

    staff_options = staff_load_df.sort_values("in_scope_hours_per_month", ascending=False)["staff_name"].tolist()
    selected_staff_list = st.multiselect(
        "Select staff member(s) for deep dive",
        options=staff_options,
        default=staff_options[:1],
    )
    if len(selected_staff_list) == 0:
        st.warning("Select at least one staff member for deep dive.")
        st.stop()

    selected_staff_df = staff_load_df[staff_load_df["staff_name"].isin(selected_staff_list)].copy()
    selected_staff_label = (
        selected_staff_list[0] if len(selected_staff_list) == 1 else f"{len(selected_staff_list)} staff members"
    )
    selected_row = pd.Series(
        {
            "capacity_used_pct": (
                selected_staff_df["total_hours_per_month"].sum() / selected_staff_df["monthly_capacity"].sum() * 100
                if selected_staff_df["monthly_capacity"].sum() > 0
                else np.nan
            ),
            "headroom_hours_per_month": selected_staff_df["headroom_hours_per_month"].sum(),
            "scope_share_pct": (
                selected_staff_df["in_scope_hours"].sum() / selected_staff_df["total_hours"].sum() * 100
                if selected_staff_df["total_hours"].sum() > 0
                else 0.0
            ),
            "in_scope_hours_per_month": selected_staff_df["in_scope_hours_per_month"].sum(),
            "out_of_scope_hours_per_month": selected_staff_df["out_of_scope_hours_per_month"].sum(),
            "absorption_estimate": selected_staff_df["absorption_estimate"].fillna(0).sum(),
        }
    )

    with st.container(border=True):
        if len(selected_staff_list) == 1:
            client_breakdown = compute_staff_client_breakdown(
                df,
                staff_name=selected_staff_list[0],
                scope_mask=scope_mask,
                lookback_months=lookback_months,
            )
        else:
            client_parts = []
            for staff_name in selected_staff_list:
                part = compute_staff_client_breakdown(
                    df,
                    staff_name=staff_name,
                    scope_mask=scope_mask,
                    lookback_months=lookback_months,
                )
                if len(part) > 0:
                    client_parts.append(part)
            if len(client_parts) == 0:
                client_breakdown = pd.DataFrame(
                    columns=[
                        "client",
                        "job_no",
                        "job_name",
                        "job_category",
                        "total_hours",
                        "in_scope_hours",
                        "out_of_scope_hours",
                        "hours_per_month",
                        "share_of_staff_total_pct",
                        "is_in_scope",
                    ]
                )
            else:
                combined_client = pd.concat(client_parts, ignore_index=True)
                client_breakdown = (
                    combined_client.groupby(
                        ["client", "job_no", "job_name", "job_category"],
                        dropna=False,
                    )
                    .agg(
                        total_hours=("total_hours", "sum"),
                        in_scope_hours=("in_scope_hours", "sum"),
                        out_of_scope_hours=("out_of_scope_hours", "sum"),
                        hours_per_month=("hours_per_month", "sum"),
                    )
                    .reset_index()
                )
                total_combined_hours = float(client_breakdown["total_hours"].sum())
                client_breakdown["share_of_staff_total_pct"] = np.where(
                    total_combined_hours > 0,
                    client_breakdown["total_hours"] / total_combined_hours * 100,
                    0.0,
                )
                client_breakdown["is_in_scope"] = client_breakdown["in_scope_hours"] > 0
                client_breakdown = client_breakdown.sort_values("total_hours", ascending=False)

        st.markdown("**So What Summary**")
        summary = _build_staff_so_what_summary(selected_staff_label, selected_row, client_breakdown)
        _render_staff_so_what_summary(summary)
        st.markdown("<div class='dc-soft-divider'></div>", unsafe_allow_html=True)

        st.markdown("**Client × Job Breakdown**")

        if len(client_breakdown) == 0:
            st.info("No client/job records for this staff member in the selected window.")
        else:
            chart_df = client_breakdown.copy()
            chart_df["client_job"] = (
                chart_df["client"].fillna("Unknown").astype(str)
                + " — "
                + chart_df["job_name"].fillna(chart_df["job_no"]).astype(str)
            ).apply(lambda x: _truncate_text(x, max_len=40))
            chart_df = chart_df.sort_values("total_hours", ascending=True)

            fig_client = go.Figure()
            fig_client.add_trace(
                go.Bar(
                    y=chart_df["client_job"],
                    x=chart_df["out_of_scope_hours"] / max(months_covered, 1),
                    name="Out-of-Scope",
                    orientation="h",
                    marker_color="#94a3b8",
                    hovertemplate="%{y}<br>Out-of-scope: %{x:.1f}h/mo<extra></extra>",
                )
            )
            fig_client.add_trace(
                go.Bar(
                    y=chart_df["client_job"],
                    x=chart_df["in_scope_hours"] / max(months_covered, 1),
                    name="In-Scope",
                    orientation="h",
                    marker_color="#2563eb",
                    hovertemplate="%{y}<br>In-scope: %{x:.1f}h/mo<extra></extra>",
                )
            )
            fig_client.update_layout(
                barmode="stack",
                template="plotly_white",
                margin=PLOT_MARGINS,
                yaxis_title="Client — Job",
                xaxis_title="Hours / Month",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_client, use_container_width=True)

            table_client = pd.DataFrame(
                {
                    "Client": client_breakdown["client"],
                    "Job": client_breakdown["job_name"].fillna(client_breakdown["job_no"]),
                    "Category": client_breakdown["job_category"],
                    "Hrs/Mo": client_breakdown["hours_per_month"].apply(fmt_hours),
                    "% of Total": client_breakdown["share_of_staff_total_pct"].apply(fmt_percent),
                    "In Scope?": client_breakdown["is_in_scope"].map({True: "Yes", False: "No"}),
                }
            )
            st.dataframe(table_client, use_container_width=True, hide_index=True)

            st.markdown("**Hours Composition**")
            d1, d2 = st.columns(2)
            with d1:
                scope_vals = [
                    max(float(selected_row["in_scope_hours_per_month"]), 0.0),
                    max(float(selected_row["out_of_scope_hours_per_month"]), 0.0),
                ]
                fig_scope_split = go.Figure(
                    data=[
                        go.Pie(
                            labels=["In-Scope", "Out-of-Scope"],
                            values=scope_vals,
                            hole=0.62,
                            marker=dict(colors=["#2563eb", "#e2e8f0"]),
                            textinfo="label+percent",
                        )
                    ]
                )
                fig_scope_split.update_layout(
                    template="plotly_white",
                    margin=PLOT_MARGINS,
                    annotations=[
                        dict(
                            text=f"{selected_row['scope_share_pct']:.0f}% in scope",
                            showarrow=False,
                            x=0.5,
                            y=0.5,
                            font=dict(size=14),
                        )
                    ],
                )
                st.plotly_chart(fig_scope_split, use_container_width=True)

            with d2:
                client_dist = (
                    client_breakdown.groupby("client", dropna=False)
                    .agg(
                        total_hours=("total_hours", "sum"),
                        in_scope_hours=("in_scope_hours", "sum"),
                    )
                    .reset_index()
                    .sort_values("total_hours", ascending=False)
                )
                if len(client_dist) > 8:
                    top = client_dist.head(8).copy()
                    other = client_dist.iloc[8:]
                    top = pd.concat(
                        [
                            top,
                            pd.DataFrame(
                                {
                                    "client": ["Other"],
                                    "total_hours": [other["total_hours"].sum()],
                                    "in_scope_hours": [other["in_scope_hours"].sum()],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
                    client_dist = top

                client_dist["client"] = client_dist["client"].fillna("Unknown").astype(str)
                palette = px.colors.qualitative.Plotly
                colors = []
                color_idx = 0
                for _, row in client_dist.iterrows():
                    if row["client"] == "Other":
                        colors.append("#e2e8f0")
                    elif row["in_scope_hours"] > 0:
                        colors.append(palette[color_idx % len(palette)])
                        color_idx += 1
                    else:
                        colors.append("#cbd5e1")

                fig_client_donut = go.Figure(
                    data=[
                        go.Pie(
                            labels=client_dist["client"],
                            values=client_dist["total_hours"],
                            hole=0.62,
                            marker=dict(colors=colors),
                            textinfo="label+percent",
                        )
                    ]
                )
                fig_client_donut.update_layout(
                    template="plotly_white",
                    margin=PLOT_MARGINS,
                )
                st.plotly_chart(fig_client_donut, use_container_width=True)

            st.markdown("**Staff Task Capacity Flow (Scoped Jobs)**")
            st.caption("Task-level view across selected staff and scoped jobs, with direct linkage back to contributing jobs.")
            staff_task_summary, staff_task_jobs = compute_staff_task_capacity_flow(
                df,
                staff_name=selected_staff_list,
                scope_mask=scope_mask,
                lookback_months=lookback_months,
            )

            if len(staff_task_summary) == 0:
                st.info("No scoped task activity found for the selected staff in the selected window.")
            else:
                top_task_row = staff_task_summary.iloc[0]
                top_three_share = float(staff_task_summary["share_of_staff_scope_pct"].head(3).sum())
                top_task_name = str(top_task_row["task_name"])
                top_task_share = float(top_task_row["share_of_staff_scope_pct"])
                top_task_hours = float(top_task_row["hours_per_month"])
                top_job_label = (
                    str(top_task_row["top_job_name"])
                    if pd.notna(top_task_row["top_job_name"])
                    else str(top_task_row["top_job_no"])
                )
                top_job_share_of_task = float(
                    pd.to_numeric(top_task_row["top_job_share_of_task_pct"], errors="coerce")
                )

                st.info(
                    f"So what: {selected_staff_label}'s scoped task load is led by `{top_task_name}` "
                    f"({fmt_percent(top_task_share)} of scoped load, {fmt_hours(top_task_hours)}h/mo), "
                    f"primarily from `{_truncate_text(top_job_label, max_len=38)}` "
                    f"({fmt_percent(top_job_share_of_task)} of that task). Top 3 tasks account for {fmt_percent(top_three_share)}."
                )

                tf1, tf2, tf3, tf4 = st.columns(4)
                tf1.metric("Scoped Tasks", f"{len(staff_task_summary):,}")
                tf1.caption("Distinct tasks in selected scoped workload")
                tf2.metric("Top Task Share", fmt_percent(top_task_share))
                tf2.caption(f"{_truncate_text(top_task_name, max_len=32)} ({fmt_hours(top_task_hours)}h/mo)")
                tf3.metric("Top 3 Task Share", fmt_percent(top_three_share))
                tf3.caption("Concentration across leading tasks")
                tf4.metric(
                    "Multi-Job Tasks",
                    f"{int((staff_task_summary['job_count'] > 1).sum()):,}",
                )
                tf4.caption("Tasks spread across more than one job")

                staff_task_chart_df = staff_task_summary.head(12).copy().sort_values("hours_per_month", ascending=True)
                staff_task_chart_df["task_label"] = staff_task_chart_df["task_name"].astype(str).apply(
                    lambda x: _truncate_text(x, max_len=40)
                )
                fig_staff_task = go.Figure(
                    data=[
                        go.Bar(
                            y=staff_task_chart_df["task_label"],
                            x=staff_task_chart_df["hours_per_month"],
                            orientation="h",
                            marker_color="#2563eb",
                            customdata=np.column_stack(
                                [
                                    staff_task_chart_df["share_of_staff_scope_pct"],
                                    staff_task_chart_df["job_count"],
                                    staff_task_chart_df["client_count"],
                                ]
                            ),
                            hovertemplate=(
                                "%{y}<br>Hrs/Mo: %{x:.1f}"
                                "<br>Share of scoped load: %{customdata[0]:.1f}%"
                                "<br>Jobs: %{customdata[1]}"
                                "<br>Clients: %{customdata[2]}<extra></extra>"
                            ),
                        )
                    ]
                )
                fig_staff_task.update_layout(
                    template="plotly_white",
                    margin=PLOT_MARGINS,
                    xaxis_title="Hours / Month",
                    yaxis_title="Task",
                )
                st.plotly_chart(fig_staff_task, use_container_width=True)

                staff_task_table = pd.DataFrame(
                    {
                        "Task": staff_task_summary["task_name"].astype(str),
                        "Hrs/Mo": staff_task_summary["hours_per_month"].apply(fmt_hours),
                        "Share of Scoped Load %": staff_task_summary["share_of_staff_scope_pct"].apply(fmt_percent),
                        "Jobs": staff_task_summary["job_count"].astype(int),
                        "Clients": staff_task_summary["client_count"].astype(int),
                        "Top Job": staff_task_summary["top_job_name"]
                        .fillna(staff_task_summary["top_job_no"])
                        .astype(str)
                        .apply(lambda x: _truncate_text(x, max_len=42)),
                        "Top Job Share of Task %": staff_task_summary["top_job_share_of_task_pct"].apply(fmt_percent),
                    }
                )
                st.dataframe(staff_task_table, use_container_width=True, hide_index=True)

                st.markdown("**Task → Constituent Job Summary**")
                task_job_view = staff_task_jobs.copy().sort_values(["task_name", "hours_per_month"], ascending=[True, False])
                if len(task_job_view) == 0:
                    st.info("No constituent job linkage found for scoped tasks.")
                else:
                    budget_pool = float(scope_budget) * len(selected_staff_list) if float(scope_budget) > 0 else np.nan
                    scoped_task_hours_month = float(staff_task_summary["hours_per_month"].sum())
                    budget_used_pct = (
                        scoped_task_hours_month / budget_pool * 100
                        if pd.notna(budget_pool) and budget_pool > 0
                        else np.nan
                    )
                    budget_gap = (
                        budget_pool - scoped_task_hours_month
                        if pd.notna(budget_pool) and budget_pool > 0
                        else np.nan
                    )

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Scope Budget Pool", fmt_hours(budget_pool) if pd.notna(budget_pool) else "N/A")
                    r1.caption(f"{len(selected_staff_list)} × {fmt_hours(float(scope_budget))}h/mo")
                    r2.metric("Scoped Task Hrs/Mo", fmt_hours(scoped_task_hours_month))
                    r2.caption("Reconciles to total selected scoped task flow")
                    r3.metric("Budget Gap (Hrs/Mo)", fmt_hours(budget_gap) if pd.notna(budget_gap) else "N/A")
                    r3.caption("Positive = remaining capacity in scope budget")
                    r4.metric("Budget Used %", fmt_percent(budget_used_pct) if pd.notna(budget_used_pct) else "N/A")
                    r4.caption("Scoped task hours divided by budget pool")

                    max_task_rows = int(min(12, len(staff_task_summary)))
                    task_rows_to_show = st.slider(
                        "Tasks shown in task-to-job stack",
                        min_value=1,
                        max_value=max_task_rows,
                        value=max_task_rows,
                        key=f"task_job_stack_rows_{len(selected_staff_list)}",
                    )
                    shown_tasks = staff_task_summary.head(task_rows_to_show)["task_name"].astype(str).tolist()
                    shown_task_set = set(shown_tasks)
                    task_label_map = _unique_truncated_label_map(shown_tasks, max_len=40)

                    stack_df = task_job_view[task_job_view["task_name"].astype(str).isin(shown_task_set)].copy()
                    stack_df["task_label"] = stack_df["task_name"].astype(str).map(task_label_map)
                    stack_df["job_label"] = (
                        stack_df["client"].fillna("Unknown").astype(str)
                        + " — "
                        + stack_df["job_name"].fillna(stack_df["job_no"]).astype(str)
                    ).apply(lambda x: _truncate_text(x, max_len=42))

                    top_jobs = (
                        stack_df.groupby("job_label", dropna=False)["hours_per_month"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(12)
                        .index
                    )
                    stack_df["job_group"] = np.where(
                        stack_df["job_label"].isin(top_jobs),
                        stack_df["job_label"],
                        "Other scoped jobs",
                    )
                    stack_df = (
                        stack_df.groupby(["task_label", "job_group"], dropna=False)["hours_per_month"]
                        .sum()
                        .reset_index()
                    )

                    task_order = [task_label_map[str(task)] for task in shown_tasks]
                    stack_df["task_label"] = pd.Categorical(
                        stack_df["task_label"],
                        categories=list(reversed(task_order)),
                        ordered=True,
                    )

                    fig_task_job_stack = go.Figure()
                    job_order = (
                        stack_df.groupby("job_group", dropna=False)["hours_per_month"]
                        .sum()
                        .sort_values(ascending=False)
                        .index
                    )
                    for job_group in job_order:
                        sub = stack_df[stack_df["job_group"] == job_group]
                        fig_task_job_stack.add_trace(
                            go.Bar(
                                y=sub["task_label"],
                                x=sub["hours_per_month"],
                                orientation="h",
                                name=str(job_group),
                                hovertemplate=(
                                    "%{y}<br>Constituent job: %{fullData.name}"
                                    "<br>Hrs/Mo: %{x:.1f}<extra></extra>"
                                ),
                            )
                        )

                    fig_task_job_stack.update_layout(
                        barmode="stack",
                        template="plotly_white",
                        margin=PLOT_MARGINS,
                        xaxis_title="Hours / Month",
                        yaxis_title="Task",
                        legend_title_text="Scoped Jobs",
                    )
                    st.plotly_chart(fig_task_job_stack, use_container_width=True)

                    task_recon = staff_task_summary.copy()
                    if pd.notna(budget_pool) and budget_pool > 0:
                        task_recon["share_of_budget_pct"] = task_recon["hours_per_month"] / budget_pool * 100
                        task_recon["cumulative_hours_per_month"] = task_recon["hours_per_month"].cumsum()
                        task_recon["budget_remaining_after_task"] = budget_pool - task_recon["cumulative_hours_per_month"]
                    else:
                        task_recon["share_of_budget_pct"] = np.nan
                        task_recon["cumulative_hours_per_month"] = task_recon["hours_per_month"].cumsum()
                        task_recon["budget_remaining_after_task"] = np.nan

                    task_recon_table = pd.DataFrame(
                        {
                            "Task": task_recon["task_name"].astype(str),
                            "Hrs/Mo": task_recon["hours_per_month"].apply(fmt_hours),
                            "Share of Scoped Load %": task_recon["share_of_staff_scope_pct"].apply(fmt_percent),
                            "Share of Budget %": task_recon["share_of_budget_pct"].apply(fmt_percent),
                            "Cumulative Hrs/Mo": task_recon["cumulative_hours_per_month"].apply(fmt_hours),
                            "Budget Remaining After Task (Hrs/Mo)": task_recon["budget_remaining_after_task"].apply(fmt_hours),
                        }
                    )
                    st.dataframe(task_recon_table, use_container_width=True, hide_index=True)

            st.markdown("**Task Drill-Down**")
            job_option_map: Dict[str, str] = {}
            for _, row in client_breakdown.sort_values("total_hours", ascending=False).iterrows():
                label = (
                    f"{row['client']} — {row['job_name'] if pd.notna(row['job_name']) else row['job_no']}"
                    f" ({fmt_hours(row['hours_per_month'])}h/mo)"
                )
                job_option_map[label] = str(row["job_no"])

            selected_job_label = st.selectbox("Drill into job", options=list(job_option_map.keys()))
            selected_job_no = job_option_map[selected_job_label]

            if len(selected_staff_list) == 1:
                task_breakdown = compute_staff_task_breakdown(
                    df,
                    staff_name=selected_staff_list[0],
                    job_no=selected_job_no,
                    scope_mask=scope_mask,
                    lookback_months=lookback_months,
                )
            else:
                task_parts = []
                for staff_name in selected_staff_list:
                    part = compute_staff_task_breakdown(
                        df,
                        staff_name=staff_name,
                        job_no=selected_job_no,
                        scope_mask=scope_mask,
                        lookback_months=lookback_months,
                    )
                    if len(part) > 0:
                        task_parts.append(part)
                if len(task_parts) == 0:
                    task_breakdown = pd.DataFrame(
                        columns=["task_name", "total_hours", "hours_per_month", "cost", "share_pct"]
                    )
                else:
                    task_breakdown = (
                        pd.concat(task_parts, ignore_index=True)
                        .groupby("task_name", dropna=False)
                        .agg(
                            total_hours=("total_hours", "sum"),
                            hours_per_month=("hours_per_month", "sum"),
                            cost=("cost", "sum"),
                        )
                        .reset_index()
                    )
                    total_task_hours = float(task_breakdown["total_hours"].sum())
                    task_breakdown["share_pct"] = np.where(
                        total_task_hours > 0,
                        task_breakdown["total_hours"] / total_task_hours * 100,
                        0.0,
                    )
                    task_breakdown = task_breakdown.sort_values("total_hours", ascending=False)
            if len(task_breakdown) == 0:
                st.info("No task-level hours found for the selected job.")
            else:
                task_table = pd.DataFrame(
                    {
                        "Task": task_breakdown["task_name"].fillna("Unspecified"),
                        "Hours": task_breakdown["total_hours"].apply(fmt_hours),
                        "Hrs/Month": task_breakdown["hours_per_month"].apply(fmt_hours),
                        "Share %": task_breakdown["share_pct"].apply(fmt_percent),
                    }
                )
                st.dataframe(task_table, use_container_width=True, hide_index=True)

    st.markdown("<div class='dc-soft-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-label'>Section G</div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-title'>Absorption Scenario Planner</div>", unsafe_allow_html=True)
    st.caption("Test how new client load distribution changes projected capacity and budget pressure.")
    with st.expander("🧮 Absorption Scenario Planner", expanded=False):
        st.caption("Simulate adding new clients and compare projected load, capacity, and budget outcomes.")
        _render_explainer_expander(
            "Scenario Assumptions",
            [
                "Each simulated client adds the same fixed monthly hours from `Avg hrs/month per new client`.",
                "Auto assignment places each new client on the person with the most remaining headroom at that step.",
                "Manual assignment places all simulated clients onto one selected person.",
                "Projected capacity and scope-budget percentages are recalculated after each simulated assignment.",
            ],
            expanded=False,
            as_expander=False,
        )
        s1, s2 = st.columns(2)
        with s1:
            simulate_clients = st.slider("New clients to simulate", min_value=0, max_value=5, value=1)
        with s2:
            strategy = st.selectbox(
                "Assignment strategy",
                options=["Auto (least loaded first)", "Manual — assign all to one person"],
            )

        manual_staff = None
        if strategy == "Manual — assign all to one person":
            manual_staff = st.selectbox("Assign all clients to", options=staff_options)

        scenario = staff_load_df.copy()
        scenario["new_load_hours"] = 0.0
        scenario["projected_total_hours_per_month"] = scenario["total_hours_per_month"]
        scenario["projected_in_scope_hours_per_month"] = scenario["in_scope_hours_per_month"]
        scenario["headroom_hours_per_month"] = scenario["monthly_capacity"] - scenario["projected_total_hours_per_month"]

        if simulate_clients > 0:
            if strategy == "Auto (least loaded first)":
                for _ in range(simulate_clients):
                    target_idx = scenario["headroom_hours_per_month"].idxmax()
                    scenario.at[target_idx, "new_load_hours"] += float(avg_new_client_hours)
                    scenario.at[target_idx, "projected_total_hours_per_month"] += float(avg_new_client_hours)
                    scenario.at[target_idx, "projected_in_scope_hours_per_month"] += float(avg_new_client_hours)
                    scenario["headroom_hours_per_month"] = (
                        scenario["monthly_capacity"] - scenario["projected_total_hours_per_month"]
                    )
            else:
                target_idx = scenario[scenario["staff_name"] == manual_staff].index[0]
                added = float(avg_new_client_hours) * simulate_clients
                scenario.at[target_idx, "new_load_hours"] += added
                scenario.at[target_idx, "projected_total_hours_per_month"] += added
                scenario.at[target_idx, "projected_in_scope_hours_per_month"] += added
                scenario["headroom_hours_per_month"] = (
                    scenario["monthly_capacity"] - scenario["projected_total_hours_per_month"]
                )

        scenario["projected_capacity_pct"] = np.where(
            scenario["monthly_capacity"] > 0,
            scenario["projected_total_hours_per_month"] / scenario["monthly_capacity"] * 100,
            np.nan,
        )

        if float(scope_budget) > 0:
            scenario["projected_scope_budget_pct"] = (
                scenario["projected_in_scope_hours_per_month"] / float(scope_budget) * 100
            )
            scenario["projected_status"] = scenario["projected_scope_budget_pct"].apply(_status_from_budget_pct)
        else:
            scenario["projected_scope_budget_pct"] = np.nan
            scenario["projected_status"] = np.where(
                scenario["projected_capacity_pct"] > 100,
                "Over Capacity",
                "Within Capacity",
            )

        scenario["status_change"] = scenario["scope_status"].astype(str) + " → " + scenario["projected_status"].astype(str)

        scenario_table = pd.DataFrame(
            {
                "Staff": scenario["staff_name"],
                "Current Hrs/Mo": scenario["total_hours_per_month"].apply(fmt_hours),
                "+ New Load": scenario["new_load_hours"].apply(fmt_hours),
                "Projected Hrs/Mo": scenario["projected_total_hours_per_month"].apply(fmt_hours),
                "Projected Capacity %": scenario["projected_capacity_pct"],
                "Projected Scope Budget %": scenario["projected_scope_budget_pct"],
                "Status Change": scenario["status_change"],
            }
        )
        scenario_table["Projected Capacity %"] = scenario_table["Projected Capacity %"].apply(fmt_percent)
        scenario_table["Projected Scope Budget %"] = (
            scenario_table["Projected Scope Budget %"].apply(fmt_percent)
            if float(scope_budget) > 0
            else "N/A"
        )

        scenario_styler = scenario_table.style.apply(_scenario_row_style, axis=1)
        st.dataframe(scenario_styler, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
