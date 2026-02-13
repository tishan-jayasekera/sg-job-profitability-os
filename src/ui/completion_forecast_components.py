"""UI rendering for the Active Delivery completion forecast section."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.semantic import get_category_col
from src.modeling.completion_forecast import (
    build_peer_lifecycle_profiles,
    compute_job_timeline,
    estimate_remaining_work,
    forecast_completion,
)
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent


_PALETTE = [
    "#2563eb",
    "#7db7e8",
    "#ef4444",
    "#8b5cf6",
    "#14b8a6",
    "#f59e0b",
    "#22c55e",
    "#fb7185",
    "#94a3b8",
    "#0ea5e9",
]


def _resolve_date_col(df: pd.DataFrame) -> str | None:
    if "work_date" in df.columns and pd.to_datetime(df["work_date"], errors="coerce").notna().any():
        return "work_date"
    if "month_key" in df.columns and pd.to_datetime(df["month_key"], errors="coerce").notna().any():
        return "month_key"
    return None


def _completed_mask(df: pd.DataFrame) -> pd.Series:
    if "job_completed_date" in df.columns:
        return df["job_completed_date"].notna()
    if "job_status" in df.columns:
        return df["job_status"].astype(str).str.lower().str.contains("completed", na=False)
    return pd.Series(False, index=df.index)


def _compute_job_decile_distribution(df_job: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if len(df_job) == 0:
        return pd.DataFrame(columns=["decile_bucket", "share"])

    job = df_job.copy()
    job[date_col] = pd.to_datetime(job[date_col], errors="coerce")
    job = job[job[date_col].notna()].copy()
    if len(job) == 0:
        return pd.DataFrame(columns=["decile_bucket", "share"])

    job["hours_raw"] = pd.to_numeric(job.get("hours_raw"), errors="coerce").fillna(0.0)
    total_hours = job["hours_raw"].sum()
    if total_hours <= 0:
        return pd.DataFrame(columns=["decile_bucket", "share"])

    start_date = job[date_col].min()
    end_date = job[date_col].max()
    duration = max((end_date - start_date).days + 1, 1)

    rel = (job[date_col] - start_date).dt.days / duration
    job["decile_bucket"] = np.floor(rel.clip(lower=0, upper=1) * 10).astype(int).clip(0, 9)

    decile_hours = job.groupby("decile_bucket")["hours_raw"].sum().reindex(range(10), fill_value=0.0)
    share = decile_hours / total_hours
    return pd.DataFrame({"decile_bucket": share.index.astype(int), "share": share.values})


def _compute_peer_decile_band(
    df_all: pd.DataFrame,
    department: str,
    category: str,
    selected_job: str,
) -> pd.DataFrame:
    if len(df_all) == 0 or "job_no" not in df_all.columns:
        return pd.DataFrame(columns=["decile_bucket", "median", "p25", "p75"])

    cat_col = get_category_col(df_all)
    peers = df_all[
        _completed_mask(df_all)
        & (df_all["department_final"] == department)
        & (df_all[cat_col] == category)
        & (df_all["job_no"].astype(str) != str(selected_job))
    ].copy()
    if len(peers) == 0:
        return pd.DataFrame(columns=["decile_bucket", "median", "p25", "p75"])

    date_col = _resolve_date_col(peers)
    if date_col is None:
        return pd.DataFrame(columns=["decile_bucket", "median", "p25", "p75"])

    rows = []
    for job_no, job_df in peers.groupby("job_no"):
        deciles = _compute_job_decile_distribution(job_df, date_col)
        if len(deciles) == 0:
            continue
        for row in deciles.itertuples(index=False):
            rows.append(
                {
                    "job_no": str(job_no),
                    "decile_bucket": int(row.decile_bucket),
                    "share": float(row.share),
                }
            )

    if len(rows) == 0:
        return pd.DataFrame(columns=["decile_bucket", "median", "p25", "p75"])

    peer_deciles = pd.DataFrame(rows)
    band = (
        peer_deciles.groupby("decile_bucket")
        .agg(
            median=("share", "median"),
            p25=("share", lambda s: s.quantile(0.25)),
            p75=("share", lambda s: s.quantile(0.75)),
        )
        .reindex(range(10), fill_value=0.0)
        .reset_index()
    )
    return band


def _bucket_labels() -> list[str]:
    return [f"{i*10}-{(i+1)*10}%" for i in range(10)]


def _status_for_task(row: pd.Series) -> str:
    if bool(row.get("unmodeled_flag", False)):
        return "Unmodeled"
    if bool(row.get("outlier_flag", False)):
        return "Over (outlier)"

    actual = pd.to_numeric(row.get("actual_hours"), errors="coerce")
    exp_p25 = pd.to_numeric(row.get("expected_hours_p25"), errors="coerce")
    exp_p75 = pd.to_numeric(row.get("expected_hours_p75"), errors="coerce")

    if pd.notna(actual) and pd.notna(exp_p25) and actual < exp_p25:
        return "Ahead"
    if pd.notna(actual) and pd.notna(exp_p75) and actual <= exp_p75:
        return "On Track"
    return "On Track"


def _margin_color(margin_pct: float, benchmark_pct: float) -> str:
    m = pd.to_numeric(margin_pct, errors="coerce")
    b = pd.to_numeric(benchmark_pct, errors="coerce")
    if pd.isna(m) or pd.isna(b):
        return "#475467"
    if m > b:
        return "#16a34a"
    if m >= (b - 5):
        return "#d97706"
    return "#dc2626"


def _peer_overlay_summary(peer_band: pd.DataFrame, current_deciles: pd.DataFrame) -> str:
    """Return a plain-language interpretation of the peer lifecycle overlay."""
    if len(peer_band) == 0 or len(current_deciles) == 0:
        return "Not enough data to compare this job's delivery pattern to peers."

    peer = peer_band.set_index("decile_bucket").reindex(range(10), fill_value=0.0)
    curr = current_deciles.set_index("decile_bucket").reindex(range(10), fill_value=0.0)

    peer_pct = pd.to_numeric(peer["median"], errors="coerce").fillna(0.0) * 100
    curr_pct = pd.to_numeric(curr["share"], errors="coerce").fillna(0.0) * 100
    delta = curr_pct - peer_pct

    early_delta = float(delta.iloc[0:4].sum())
    late_delta = float(delta.iloc[6:10].sum())
    avg_abs_delta = float(np.abs(delta).mean())

    if avg_abs_delta <= 3:
        return "This job's effort pattern is very close to the typical completed peer lifecycle."
    if early_delta >= 8:
        return (
            "This job is front-loaded versus peers: more effort has happened early than usual. "
            "That can indicate early heavy discovery/setup or accelerated start."
        )
    if late_delta >= 8:
        return (
            "This job is back-loaded versus peers: more effort is concentrated in later phases than usual. "
            "That can indicate downstream rework, review pressure, or delayed completion work."
        )
    return (
        "This job's phase mix differs from peers in specific deciles, but without a strong early/late skew. "
        "Use the bars above/below the blue band to identify where effort diverges."
    )


def _default_forecast_task_scope(selected_df: pd.DataFrame) -> tuple[list[str], str, bool]:
    """
    Build default forecast task scope from latest job activity date.

    Returns:
        (task_list, date_label, used_fallback_all_tasks)
    """
    if len(selected_df) == 0:
        return [], "latest activity", True

    scope_df = selected_df.copy()
    scope_df["task_name"] = scope_df.get("task_name", "Unspecified").fillna("Unspecified").astype(str)
    scope_df["hours_raw"] = pd.to_numeric(scope_df.get("hours_raw"), errors="coerce").fillna(0.0)
    date_col = _resolve_date_col(scope_df)

    all_executed = sorted(scope_df.loc[scope_df["hours_raw"] > 0, "task_name"].dropna().unique().tolist())
    if date_col is None:
        return all_executed, "all history", True

    scope_df[date_col] = pd.to_datetime(scope_df[date_col], errors="coerce")
    scope_df = scope_df[scope_df[date_col].notna()].copy()
    if len(scope_df) == 0:
        return all_executed, "all history", True

    latest_date = scope_df[date_col].max()
    if date_col == "work_date":
        latest_mask = scope_df[date_col].dt.normalize() == latest_date.normalize()
        label = latest_date.strftime("%Y-%m-%d")
    else:
        latest_mask = scope_df[date_col] == latest_date
        label = latest_date.strftime("%Y-%m")

    latest_tasks = sorted(
        scope_df.loc[latest_mask & (scope_df["hours_raw"] > 0), "task_name"].dropna().unique().tolist()
    )
    if len(latest_tasks) > 0:
        return latest_tasks, label, False

    return all_executed, label, True


def render_completion_forecast_section(
    df_all: pd.DataFrame,
    df_scope: pd.DataFrame,
    jobs_df: pd.DataFrame,
    selected_job: str,
    job_name_lookup: Dict[str, str],
) -> None:
    """Render the completion forecast section in the selected-job right panel."""
    _ = (df_scope, job_name_lookup)

    st.markdown('<div class="dc-soft-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dc-section-label">Completion Forecast</div>', unsafe_allow_html=True)

    if df_all is None or len(df_all) == 0 or len(jobs_df) == 0:
        st.info("Insufficient data for completion forecast analysis.")
        return

    job_match = jobs_df[jobs_df["job_no"].astype(str) == str(selected_job)]
    if len(job_match) == 0:
        st.info("Insufficient data for completion forecast analysis.")
        return
    job_row = job_match.iloc[0]

    selected_df = df_all[df_all["job_no"].astype(str) == str(selected_job)].copy()
    if len(selected_df) == 0:
        st.info("Insufficient data for completion forecast analysis.")
        return

    department = str(job_row.get("department_final", ""))
    category = job_row.get("job_category")
    if pd.isna(category) or category is None or str(category).strip() == "":
        cat_col = get_category_col(selected_df)
        category = selected_df[cat_col].dropna().iloc[0] if selected_df[cat_col].notna().any() else ""
    category = str(category)

    with st.expander("Timeline & Forecast Analysis", expanded=True):
        # Sub-section A: Timeline
        st.markdown('<div class="dc-section-title">Cost & Hours Timeline</div>', unsafe_allow_html=True)
        st.caption(
            "Why this chart: shows where time and spend are accumulating across the delivery. "
            "How to read: stacked areas = task hours by period; black line = cumulative cost-to-date."
        )

        date_col = _resolve_date_col(selected_df)
        if date_col is None:
            st.info("Insufficient date data to build a timeline.")
            return

        if date_col == "month_key":
            granularity = "monthly"
            st.caption("`work_date` unavailable; timeline granularity is set to monthly.")
        else:
            granularity_choice = st.radio(
                "Timeline Granularity",
                ["Weekly", "Monthly"],
                horizontal=True,
                key=f"completion_forecast_granularity_{selected_job}",
            )
            granularity = granularity_choice.lower()

        timeline_df = compute_job_timeline(df_all=df_all, job_no=str(selected_job), granularity=granularity)
        if len(timeline_df) == 0:
            st.info("No timeline data available for this job.")
            return

        timeline_fig = go.Figure()
        task_order = (
            timeline_df.groupby("task_name")["hours"].sum().sort_values(ascending=False).index.tolist()
        )
        for i, task_name in enumerate(task_order):
            task_df = timeline_df[timeline_df["task_name"] == task_name].sort_values("period_start")
            timeline_fig.add_trace(
                go.Scatter(
                    x=task_df["period_start"],
                    y=task_df["hours"],
                    mode="lines",
                    stackgroup="hours",
                    name=str(task_name),
                    line=dict(width=1.0, color=_PALETTE[i % len(_PALETTE)]),
                    hovertemplate=(
                        "Period: %{x|%Y-%m-%d}<br>"
                        f"Task: {task_name}<br>"
                        "Hours: %{y:.1f}h<extra></extra>"
                    ),
                )
            )

        period_totals = (
            timeline_df.groupby("period_start", as_index=False)
            .agg(cumulative_cost=("cumulative_cost", "max"))
            .sort_values("period_start")
        )
        timeline_fig.add_trace(
            go.Scatter(
                x=period_totals["period_start"],
                y=period_totals["cumulative_cost"],
                mode="lines+markers",
                name="Cumulative Cost",
                line=dict(color="#111827", width=2),
                marker=dict(size=5),
                yaxis="y2",
                hovertemplate="Period: %{x|%Y-%m-%d}<br>Cumulative Cost: $%{y:,.0f}<extra></extra>",
            )
        )

        timeline_fig.update_layout(
            template="plotly_white",
            height=360,
            margin=dict(l=30, r=30, t=20, b=40),
            xaxis_title="Time Period",
            yaxis_title="Hours",
            yaxis2=dict(
                title="Cumulative Cost",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        # Sub-section B: Peer lifecycle overlay
        st.markdown('<div class="dc-soft-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="dc-section-title">Peer Lifecycle Overlay</div>', unsafe_allow_html=True)
        st.caption(
            "What this chart does: compares where this job is spending effort across its lifecycle "
            "vs completed similar jobs. Each bar is this job's % of total hours in that phase; "
            "the blue band/line is the normal peer range/median."
        )

        lifecycle_df, task_summary_df = build_peer_lifecycle_profiles(
            df_all=df_all,
            department=department,
            category=category,
            exclude_job_no=str(selected_job),
        )

        peer_count = int(task_summary_df["peer_job_count"].max()) if len(task_summary_df) > 0 else 0

        if peer_count == 0:
            st.warning("No completed peer jobs found for benchmarking.")
            return
        if peer_count < 3:
            st.warning("⚠️ Fewer than 3 completed peer jobs available. Forecast confidence is low.")

        peer_band = _compute_peer_decile_band(df_all, department, category, str(selected_job))
        current_deciles = _compute_job_decile_distribution(selected_df, date_col)

        if len(peer_band) == 0 or len(current_deciles) == 0:
            st.caption("Unable to build lifecycle overlay from the available data.")
        else:
            labels = _bucket_labels()
            peer_band = peer_band.sort_values("decile_bucket")
            current_deciles = current_deciles.set_index("decile_bucket").reindex(range(10), fill_value=0.0).reset_index()

            overlay_fig = go.Figure()
            overlay_fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=(peer_band["p75"] * 100).values,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            overlay_fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=(peer_band["p25"] * 100).values,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(37, 99, 235, 0.18)",
                    line=dict(width=0),
                    name="Peer IQR (P25-P75)",
                    hovertemplate="Peer P25: %{y:.1f}%<extra></extra>",
                )
            )
            overlay_fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=(peer_band["median"] * 100).values,
                    mode="lines+markers",
                    line=dict(color="#1d4ed8", width=2),
                    name="Peer Median",
                    hovertemplate="Peer Median: %{y:.1f}%<extra></extra>",
                )
            )
            overlay_fig.add_trace(
                go.Bar(
                    x=labels,
                    y=(current_deciles["share"] * 100).values,
                    marker_color="#7db7e8",
                    opacity=0.75,
                    name="Current Job",
                    hovertemplate="Current Job: %{y:.1f}%<extra></extra>",
                )
            )

            overlay_fig.update_layout(
                template="plotly_white",
                height=320,
                margin=dict(l=30, r=30, t=20, b=40),
                xaxis_title="Lifecycle Decile",
                yaxis_title="% Share of Total Hours",
                barmode="overlay",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(overlay_fig, use_container_width=True)
            st.caption(
                "Interpretation: bars above the blue band = heavier effort than peers in that phase; "
                "bars below = lighter effort than peers."
            )
            st.info(_peer_overlay_summary(peer_band, current_deciles))

        # Sub-section C: Remaining work estimate
        st.markdown('<div class="dc-soft-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="dc-section-title">Remaining Work Estimate</div>', unsafe_allow_html=True)
        st.caption(
            "Why this chart: converts peer behavior into task-level remaining-hour expectations for planning. "
            "How to read: dark bar = actual hours, light extension = expected remaining (P50), whiskers = P25-P75 range."
        )

        default_scope_tasks, scope_date_label, used_scope_fallback = _default_forecast_task_scope(selected_df)
        peer_only_tasks = sorted(
            set(task_summary_df["task_name"].astype(str).tolist()) - set(default_scope_tasks)
        ) if len(task_summary_df) > 0 else []

        st.caption(
            "Default task scope uses tasks executed on the latest job activity date "
            f"({scope_date_label})."
        )
        if used_scope_fallback:
            st.caption("No positive-hour tasks found on that date; using all executed tasks to date.")

        selected_peer_tasks = st.multiselect(
            "Add peer tasks not yet executed on this job",
            options=peer_only_tasks,
            default=[],
            key=f"completion_forecast_peer_task_scope_{selected_job}",
            help=(
                "Select additional tasks seen in completed peers to include in remaining-work "
                "and completion forecasting."
            ),
        )
        scoped_tasks = tuple(sorted(set(default_scope_tasks) | set(selected_peer_tasks)))
        if len(scoped_tasks) == 0:
            st.info("No tasks available for forecast scope selection.")
            return

        st.caption(f"Forecast task scope ({len(scoped_tasks)}): {', '.join(scoped_tasks)}")

        remaining_df, remaining_summary = estimate_remaining_work(
            df_all=df_all,
            job_no=str(selected_job),
            department=department,
            category=category,
            job_row=job_row,
            included_tasks=scoped_tasks,
        )

        quoted_hours = pd.to_numeric(job_row.get("quoted_hours"), errors="coerce")
        if (pd.isna(quoted_hours) or quoted_hours <= 0) and remaining_summary.get("peer_count", 0) > 0:
            st.caption("Estimate baseline uses peer median total hours because quoted hours are unavailable.")

        if len(remaining_df) == 0:
            st.info("Unable to estimate remaining work from available benchmark data.")
            return

        work_df = remaining_df.copy()
        work_df["task_display"] = work_df["task_name"]
        work_df.loc[work_df["outlier_flag"], "task_display"] = work_df.loc[work_df["outlier_flag"], "task_name"] + " ⚠️"
        work_df.loc[work_df["unmodeled_flag"], "task_display"] = work_df.loc[work_df["unmodeled_flag"], "task_name"] + " ℹ️"

        work_df = work_df.sort_values(["remaining_hours_median", "actual_hours"], ascending=[False, False])

        range_plus = (work_df["remaining_hours_p75"] - work_df["remaining_hours_median"]).clip(lower=0.0)
        range_minus = (work_df["remaining_hours_median"] - work_df["remaining_hours_p25"]).clip(lower=0.0)

        remaining_fig = go.Figure()
        remaining_fig.add_trace(
            go.Bar(
                y=work_df["task_display"],
                x=work_df["actual_hours"],
                orientation="h",
                marker_color="#2563eb",
                name="Actual Hours",
                hovertemplate="Task: %{y}<br>Actual: %{x:.1f}h<extra></extra>",
            )
        )
        remaining_fig.add_trace(
            go.Bar(
                y=work_df["task_display"],
                x=work_df["remaining_hours_median"],
                base=work_df["actual_hours"],
                orientation="h",
                marker_color="#7db7e8",
                name="Expected Remaining (P50)",
                error_x=dict(
                    type="data",
                    array=range_plus,
                    arrayminus=range_minus,
                    color="#334155",
                    thickness=1,
                ),
                hovertemplate="Task: %{y}<br>Remaining (P50): %{x:.1f}h<extra></extra>",
            )
        )

        remaining_fig.update_layout(
            template="plotly_white",
            height=max(320, 38 * len(work_df) + 140),
            margin=dict(l=180, r=30, t=20, b=40),
            xaxis_title="Hours",
            yaxis_title="Task",
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        remaining_fig.update_yaxes(autorange="reversed")
        st.plotly_chart(remaining_fig, use_container_width=True)
        st.caption("⚠️ = outlier task. ℹ️ = unmodeled task with no peer equivalent.")

        table_df = work_df.copy()
        table_df["Status"] = table_df.apply(_status_for_task, axis=1)
        table_df["Range (P25-P75)"] = table_df.apply(
            lambda r: f"{fmt_hours(r['remaining_hours_p25'])} - {fmt_hours(r['remaining_hours_p75'])}",
            axis=1,
        )

        display = pd.DataFrame(
            {
                "Task": table_df["task_name"],
                "Actual Hrs": table_df["actual_hours"].apply(fmt_hours),
                "Expected Total (P50)": table_df["expected_hours_median"].apply(fmt_hours),
                "Remaining (P50)": table_df["remaining_hours_median"].apply(fmt_hours),
                "Range (P25-P75)": table_df["Range (P25-P75)"],
                "Status": table_df["Status"],
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Sub-section D: Forecast summary
        st.markdown('<div class="dc-soft-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="dc-section-title">Forecast Summary</div>', unsafe_allow_html=True)
        st.caption(
            "Why this section: translates estimated remaining work into completion date, total cost, and margin outcomes "
            "across optimistic/expected/conservative scenarios."
        )

        forecast = forecast_completion(
            job_row=job_row,
            remaining_summary=remaining_summary,
            df_all=df_all,
            job_no=str(selected_job),
        )

        if forecast.get("is_stalled", False):
            st.warning("⚠️ No recent activity detected — forecast cannot be computed.")

        burn_rate = pd.to_numeric(forecast.get("burn_rate_per_day"), errors="coerce")
        last_activity = pd.to_datetime(forecast.get("last_activity_date"), errors="coerce")
        burn_text = f"{fmt_hours(burn_rate)} hrs/day" if pd.notna(burn_rate) else "—"
        last_activity_text = last_activity.strftime("%Y-%m-%d") if pd.notna(last_activity) else "—"
        st.caption(f"Burn rate basis: {burn_text}. Last activity: {last_activity_text}.")

        actual_hours_total = pd.to_numeric(job_row.get("actual_hours"), errors="coerce")
        if pd.isna(actual_hours_total):
            actual_hours_total = pd.to_numeric(selected_df.get("hours_raw"), errors="coerce").fillna(0.0).sum()
        actual_hours_total = float(actual_hours_total) if pd.notna(actual_hours_total) else 0.0

        eac_baseline = pd.to_numeric(remaining_summary.get("eac_baseline"), errors="coerce")
        est_completion_pct = pd.to_numeric(remaining_summary.get("actual_completion_pct"), errors="coerce")
        if pd.notna(est_completion_pct):
            est_completion_pct = float(est_completion_pct)
        elif pd.notna(eac_baseline) and eac_baseline > 0:
            est_completion_pct = float(actual_hours_total / eac_baseline * 100)
        else:
            rem_median = pd.to_numeric(remaining_summary.get("total_remaining_median"), errors="coerce")
            rem_median = float(rem_median) if pd.notna(rem_median) else 0.0
            denom = actual_hours_total + rem_median
            est_completion_pct = float((actual_hours_total / denom) * 100) if denom > 0 else np.nan

        lifecycle_adjusted = bool(remaining_summary.get("lifecycle_adjustment_applied", False))
        eac_pre = pd.to_numeric(remaining_summary.get("eac_pre_adjustment"), errors="coerce")
        runtime_progress_pct = pd.to_numeric(remaining_summary.get("runtime_progress_pct"), errors="coerce")
        lifecycle_expected_pct = pd.to_numeric(
            remaining_summary.get("lifecycle_expected_completion_pct"), errors="coerce"
        )

        kpi_a, kpi_b, kpi_c = st.columns(3)
        with kpi_a:
            st.metric("Estimated % Complete (Lifecycle-Informed)", fmt_percent(est_completion_pct))
        with kpi_b:
            st.metric("Peer Jobs Used", f"{int(remaining_summary.get('peer_count', 0))}")
        with kpi_c:
            st.metric("EAC Baseline Hours", fmt_hours(eac_baseline))

        if lifecycle_adjusted:
            st.caption(
                "Lifecycle adjustment applied: baseline moved from "
                f"{fmt_hours(eac_pre)}h to {fmt_hours(eac_baseline)}h "
                f"(runtime progress: {fmt_percent(runtime_progress_pct)}; "
                f"peer-expected completion at this phase: {fmt_percent(lifecycle_expected_pct)})."
            )
        else:
            st.caption(
                "Lifecycle adjustment not applied (insufficient peer coverage or too little runtime history)."
            )

        benchmark_margin = pd.to_numeric(job_row.get("median_margin_pct"), errors="coerce")
        scenario_labels = [
            ("optimistic", "Optimistic (P25)"),
            ("expected", "Expected (P50)"),
            ("conservative", "Conservative (P75)"),
        ]

        cols = st.columns(3)
        for col, (key, label) in zip(cols, scenario_labels):
            scen = forecast.get("scenarios", {}).get(key, {})
            with col:
                st.markdown(f"**{label}**")
                st.metric("Remaining Hours", fmt_hours(scen.get("remaining_hours")))

                end_date = pd.to_datetime(scen.get("forecast_end_date"), errors="coerce")
                end_date_txt = end_date.strftime("%Y-%m-%d") if pd.notna(end_date) else "—"
                st.metric("Est. Completion Date", end_date_txt)

                st.metric("Forecast Total Cost", fmt_currency(scen.get("forecast_total_cost")))
                margin_pct = pd.to_numeric(scen.get("forecast_margin_pct"), errors="coerce")
                st.metric("Forecast Margin %", fmt_percent(margin_pct))

                color = _margin_color(margin_pct, benchmark_margin)
                st.markdown(
                    (
                        "<div style='font-size:0.78rem;font-weight:600;"
                        f"color:{color};margin-top:-0.25rem;'>"
                        "Margin vs benchmark"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
