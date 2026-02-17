"""Tests for revenue diagnostics metrics and orchestrator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.revenue_diagnostics import (
    build_diagnostics_bundle,
    build_hypothesis_scorecard,
    compute_client_bridge,
    compute_deal_size_distribution,
    compute_market_signals,
    compute_personnel_signals,
    compute_quoted_vs_actual_rate_trend,
    compute_reputation_signals,
    compute_service_line_monthly,
    compute_staff_selling_analysis,
    compute_task_mix_shift,
    compute_yoy_snapshot,
    decompose_revenue_change,
    normalize_service_line_labels,
)


def _make_row(
    month: pd.Timestamp,
    job_no: str,
    category: str,
    client: str,
    staff: str,
    role: str,
    function: str,
    state: str,
    task: str,
    hours: float,
    revenue: float,
) -> dict:
    return {
        "month_key": month,
        "job_no": job_no,
        "department_final": "Digital",
        "category_rev_job": category,
        "client": client,
        "staff_name": staff,
        "role": role,
        "function": function,
        "state": state,
        "task_name": task,
        "hours_raw": hours,
        "base_cost": hours * 70,
        "rev_alloc": revenue,
        "is_billable": True,
        "quote_match_flag": "matched",
        "quoted_time_total": hours,
        "quoted_amount_total": revenue * 1.05,
        "breakdown": "Billable",
    }


@pytest.fixture
def synthetic_fact_df():
    """Build 24-month synthetic fact_timesheet with controlled data for deterministic assertions."""
    months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
    clients = [f"Client {i:02d}" for i in range(1, 13)]
    staff = ["Ava", "Ben", "Cara", "Dion", "Eli", "Faye"]
    roles = {
        "Ava": ("Sales", "Growth"),
        "Ben": ("BDM", "Business Development"),
        "Cara": ("Account Manager", "Client Growth"),
        "Dion": ("Consultant", "Delivery"),
        "Eli": ("Consultant", "Delivery"),
        "Faye": ("Sales", "Growth"),
    }
    states = ["NSW", "VIC", "QLD", "WA", "SA"]

    ma_jobs_curve = np.linspace(30, 10, len(months)).round().astype(int)
    ma_arpj_curve = np.linspace(3200, 1400, len(months))
    crm_jobs_curve = np.linspace(8, 24, len(months)).round().astype(int)
    crm_arpj_curve = np.linspace(2400, 1200, len(months))

    rows: list[dict] = []

    for i, month in enumerate(months):
        ma_jobs = int(ma_jobs_curve[i])
        ma_arpj = float(ma_arpj_curve[i])
        crm_jobs = int(crm_jobs_curve[i])
        crm_arpj = float(crm_arpj_curve[i])

        lp_jobs = 0
        lp_arpj = 180.0
        if month >= pd.Timestamp("2025-07-01"):
            lp_jobs = int(np.linspace(15, 30, 6).round()[month.month - 7])

        for j in range(ma_jobs):
            client = clients[(j + i) % 8] if i < 12 else clients[(j + i + 4) % 8]
            staff_name = staff[(j + i) % len(staff)]
            role, function = roles[staff_name]
            state = states[(j + i) % len(states)]
            category = "MA" if i % 2 == 0 else "mkt automation"
            job_no = f"MA-{i:02d}-{j:03d}"
            rev = ma_arpj
            rows.append(_make_row(month, job_no, category, client, staff_name, role, function, state, "Strategy", 6.0, rev * 0.5))
            rows.append(_make_row(month, job_no, category, client, staff_name, role, function, state, "Execution", 6.0, rev * 0.5))

        for j in range(crm_jobs):
            client = clients[(j + 2 * i) % len(clients)]
            staff_name = staff[(j + 2 * i) % len(staff)]
            role, function = roles[staff_name]
            state = states[(j + 2 * i) % len(states)]
            job_no = f"CRM-{i:02d}-{j:03d}"
            rev = crm_arpj
            rows.append(_make_row(month, job_no, "crm", client, staff_name, role, function, state, "Ops", 5.0, rev * 0.5))
            rows.append(_make_row(month, job_no, "crm", client, staff_name, role, function, state, "Automation", 5.0, rev * 0.5))

        for j in range(lp_jobs):
            client = clients[(j + i) % len(clients)]
            staff_name = staff[(j + i + 3) % len(staff)]
            role, function = roles[staff_name]
            state = states[(j + i + 1) % len(states)]
            job_no = f"LP-{i:02d}-{j:03d}"
            rows.append(_make_row(month, job_no, "landing page", client, staff_name, role, function, state, "Build", 2.0, lp_arpj * 0.5))
            rows.append(_make_row(month, job_no, "landing page", client, staff_name, role, function, state, "QA", 2.0, lp_arpj * 0.5))

    df = pd.DataFrame(rows)
    df["month_key"] = pd.to_datetime(df["month_key"]).dt.to_period("M").dt.to_timestamp()
    return df


@pytest.fixture
def monthly_df(synthetic_fact_df):
    """Pre-computed monthly rollup for snapshot/decomp tests."""
    return compute_service_line_monthly(synthetic_fact_df)


def test_normalize_service_line_labels_maps_synonyms():
    """Verify 'ma', 'MA', 'mkt automation' all map to 'Marketing Automation'."""
    df = pd.DataFrame({"category_rev_job": ["ma", "MA", "mkt automation"]})
    out = normalize_service_line_labels(df, "category_rev_job")
    assert (out["service_line"] == "Marketing Automation").all()


def test_normalize_service_line_labels_preserves_unmapped():
    """Verify unmapped categories retain original value."""
    df = pd.DataFrame({"category_rev_job": ["Custom Advisory"]})
    out = normalize_service_line_labels(df, "category_rev_job")
    assert out["service_line"].iloc[0] == "Custom Advisory"


def test_normalize_service_line_labels_never_drops_rows():
    """Verify output has same row count as input."""
    df = pd.DataFrame({"category_rev_job": ["ma", "crm", "other", None]})
    out = normalize_service_line_labels(df, "category_rev_job")
    assert len(out) == len(df)


def test_compute_service_line_monthly_output_columns(monthly_df):
    """Verify exact output column set matches contract."""
    assert list(monthly_df.columns) == [
        "month_key",
        "service_line",
        "jobs",
        "revenue",
        "hours",
        "cost",
        "margin",
        "margin_pct",
        "avg_rev_per_job",
        "avg_hours_per_job",
        "realised_rate",
        "active_clients",
        "active_staff",
    ]


def test_compute_service_line_monthly_basic_metrics(synthetic_fact_df):
    """Verify revenue = sum(rev_alloc), jobs = nunique(job_no) for known fixture data."""
    monthly = compute_service_line_monthly(synthetic_fact_df)
    prepared = normalize_service_line_labels(synthetic_fact_df, "category_rev_job")

    month = prepared["month_key"].min()
    expected = prepared[(prepared["month_key"] == month) & (prepared["service_line"] == "Marketing Automation")]

    row = monthly[(monthly["month_key"] == month) & (monthly["service_line"] == "Marketing Automation")].iloc[0]
    assert row["revenue"] == pytest.approx(expected["rev_alloc"].sum())
    assert row["jobs"] == expected["job_no"].nunique()


def test_compute_service_line_monthly_empty_input():
    """Verify empty df returns empty DataFrame with correct columns."""
    out = compute_service_line_monthly(pd.DataFrame())
    assert list(out.columns) == [
        "month_key",
        "service_line",
        "jobs",
        "revenue",
        "hours",
        "cost",
        "margin",
        "margin_pct",
        "avg_rev_per_job",
        "avg_hours_per_job",
        "realised_rate",
        "active_clients",
        "active_staff",
    ]
    assert len(out) == 0


def test_compute_yoy_snapshot_handles_missing_prior_year():
    """Verify service_line with no prior-year data gets 0-filled priors and NaN YoY pcts."""
    monthly = pd.DataFrame(
        {
            "month_key": [pd.Timestamp("2025-12-01")],
            "service_line": ["Landing Pages"],
            "jobs": [10],
            "revenue": [1000],
            "hours": [40],
            "cost": [600],
            "margin": [400],
            "margin_pct": [0.4],
            "avg_rev_per_job": [100],
            "avg_hours_per_job": [4],
            "realised_rate": [25],
            "active_clients": [5],
            "active_staff": [3],
        }
    )
    out = compute_yoy_snapshot(monthly, pd.Timestamp("2025-12-01"))
    row = out.iloc[0]
    assert row["revenue_prev"] == 0
    assert row["jobs_prev"] == 0
    assert pd.isna(row["rev_yoy_pct"])
    assert pd.isna(row["jobs_yoy_pct"])


def test_decompose_revenue_change_identity_holds(monthly_df):
    """Verify volume_effect + price_effect + interaction_effect == delta_revenue within 1e-6."""
    yoy = compute_yoy_snapshot(monthly_df, pd.Timestamp("2025-12-01"))
    decomp = decompose_revenue_change(yoy)
    max_diff = (decomp["check_total"] - decomp["delta_revenue"]).abs().max()
    assert max_diff < 1e-6


def test_decompose_revenue_change_zero_jobs_prior():
    """Verify no crash when prior period has zero jobs for a service line."""
    yoy = pd.DataFrame(
        {
            "service_line": ["Landing Pages"],
            "month_curr": [pd.Timestamp("2025-12-01")],
            "month_prev": [pd.Timestamp("2024-12-01")],
            "revenue_curr": [1000.0],
            "revenue_prev": [0.0],
            "jobs_curr": [5.0],
            "jobs_prev": [0.0],
            "arpj_curr": [200.0],
            "arpj_prev": [0.0],
            "hours_curr": [20.0],
            "hours_prev": [0.0],
            "rev_yoy_pct": [np.nan],
            "jobs_yoy_pct": [np.nan],
            "arpj_yoy_pct": [np.nan],
            "hours_yoy_pct": [np.nan],
            "delta_revenue": [1000.0],
        }
    )
    decomp = decompose_revenue_change(yoy)
    assert len(decomp) == 1


def test_client_bridge_total_matches_delta(synthetic_fact_df):
    """Verify bridge Total Delta == (curr revenue - prev revenue) within 1e-6."""
    current = [pd.Timestamp("2025-10-01"), pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-01")]
    prior = [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01")]

    result = compute_client_bridge(synthetic_fact_df, "Marketing Automation", current, prior)
    bridge = result["bridge"]

    total_delta = float(bridge.loc[bridge["bridge_component"] == "Total Delta", "amount"].iloc[0])

    prepared = normalize_service_line_labels(synthetic_fact_df, "category_rev_job")
    ma = prepared[prepared["service_line"] == "Marketing Automation"]
    curr_rev = float(ma[ma["month_key"].isin(current)]["rev_alloc"].sum())
    prev_rev = float(ma[ma["month_key"].isin(prior)]["rev_alloc"].sum())

    assert abs(total_delta - (curr_rev - prev_rev)) < 1e-6


def test_client_bridge_handles_missing_client_column(synthetic_fact_df):
    """Verify returns 'No Client Data' row when client col is absent."""
    df = synthetic_fact_df.drop(columns=["client"], errors="ignore")
    result = compute_client_bridge(df, "Marketing Automation", [pd.Timestamp("2025-12-01")], [pd.Timestamp("2025-11-01")])
    assert result["bridge"].iloc[0]["bridge_component"] == "No Client Data"
    assert len(result["top_clients"]) == 0


def test_reputation_signals_status_logic():
    """Verify 'supported' when repeat_share drops ≥10pp AND lost revenue ≥ 25% of delta."""
    months = pd.date_range("2025-04-01", "2025-12-01", freq="MS")
    rows = []
    lookback_clients = ["A", "B", "C", "D", "E"]

    for m in months[:-2]:
        for c in lookback_clients:
            rows.append(_make_row(m, f"J-{m.month}-{c}", "MA", c, "Ava", "Sales", "Growth", "NSW", "Task", 1.0, 50.0))

    prior_month = pd.Timestamp("2025-11-01")
    for c in ["A", "B", "C", "D", "E"]:
        rows.append(_make_row(prior_month, f"P-{c}", "MA", c, "Ava", "Sales", "Growth", "NSW", "Task", 1.0, 100.0))

    curr_month = pd.Timestamp("2025-12-01")
    for c in ["D", "E", "F", "G", "H"]:
        rev = 60.0 if c in ["D", "E"] else 40.0
        rows.append(_make_row(curr_month, f"C-{c}", "MA", c, "Ava", "Sales", "Growth", "NSW", "Task", 1.0, rev))

    df = pd.DataFrame(rows)
    out = compute_reputation_signals(df, "Marketing Automation", [curr_month], [prior_month], repeat_window_months=6)
    assert out["status"] == "supported"


def test_reputation_signals_insufficient_data(synthetic_fact_df):
    """Verify 'insufficient_data' when client column missing."""
    df = synthetic_fact_df.drop(columns=["client"], errors="ignore")
    out = compute_reputation_signals(df, "Marketing Automation", [pd.Timestamp("2025-12-01")], [pd.Timestamp("2024-12-01")])
    assert out["status"] == "insufficient_data"


def test_personnel_signals_proxy_mode_when_no_role_function(synthetic_fact_df):
    """Verify proxy_mode=True and status != error when role/function columns absent."""
    df = synthetic_fact_df.drop(columns=["role", "function"], errors="ignore")
    out = compute_personnel_signals(
        df,
        "Marketing Automation",
        [pd.Timestamp("2025-10-01"), pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-01")],
        [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01")],
    )
    assert out["proxy_mode"] is True
    assert out["status"] in {"supported", "not_supported", "insufficient_data"}


def test_personnel_signals_supported_logic():
    """Verify 'supported' when staff down ≥20% and jobs_per_staff increased."""
    rows = []
    prev_month = pd.Timestamp("2025-06-01")
    curr_month = pd.Timestamp("2025-12-01")

    prev_staff = ["S1", "S2", "S3", "S4", "S5"]
    curr_staff = ["S1", "S2", "S3"]

    for i, s in enumerate(prev_staff):
        for j in range(2):
            rows.append(_make_row(prev_month, f"P-{s}-{j}", "MA", f"C{i}", s, "Sales", "Growth", "NSW", "Task", 1.0, 100.0))

    for i, s in enumerate(curr_staff):
        for j in range(6):
            rows.append(_make_row(curr_month, f"C-{s}-{j}", "MA", f"X{i}", s, "Sales", "Growth", "NSW", "Task", 1.0, 120.0))

    df = pd.DataFrame(rows)
    out = compute_personnel_signals(df, "Marketing Automation", [curr_month], [prev_month])
    assert out["status"] == "supported"


def test_market_signals_insufficient_without_state(synthetic_fact_df):
    """Verify 'insufficient_data' when state column missing."""
    df = synthetic_fact_df.drop(columns=["state"], errors="ignore")
    out = compute_market_signals(df, "Marketing Automation", [pd.Timestamp("2025-12-01")], [pd.Timestamp("2024-12-01")])
    assert out["status"] == "insufficient_data"


def test_market_signals_supported_logic():
    """Verify 'supported' when negative_state_share ≥ 0.60."""
    rows = []
    prev = pd.Timestamp("2025-06-01")
    curr = pd.Timestamp("2025-12-01")
    states = ["NSW", "VIC", "QLD", "WA", "SA"]
    prev_vals = [200, 200, 200, 200, 200]
    curr_vals = [100, 90, 80, 210, 210]

    for s, p, c in zip(states, prev_vals, curr_vals):
        rows.append(_make_row(prev, f"P-{s}", "MA", "C1", "Ava", "Sales", "Growth", s, "Task", 1.0, float(p)))
        rows.append(_make_row(curr, f"C-{s}", "MA", "C1", "Ava", "Sales", "Growth", s, "Task", 1.0, float(c)))

    df = pd.DataFrame(rows)
    out = compute_market_signals(df, "Marketing Automation", [curr], [prev])
    assert out["status"] == "supported"


def test_build_hypothesis_scorecard_shape_and_columns(monthly_df):
    """Verify output has 5 rows and exact column set."""
    yoy = compute_yoy_snapshot(monthly_df, pd.Timestamp("2025-12-01"))
    churn = {"pct_small_jobs_delta_pp": 12.0}
    reputation = {
        "status": "not_supported",
        "repeat_share_delta_pp": -6.0,
        "lost_client_revenue": 100.0,
    }
    personnel = {
        "status": "not_supported",
        "staff_yoy_pct": -12.0,
        "jobs_per_staff_prev": 2.0,
        "jobs_per_staff_curr": 2.5,
    }
    market = {"negative_state_share": 70.0, "top2_decline_contribution": 55.0}

    out = build_hypothesis_scorecard(yoy, churn, reputation, personnel, market)
    assert len(out) == 5
    assert list(out.columns) == ["hypothesis", "signal_strength", "evidence_1", "evidence_2", "interpretation"]


def test_build_diagnostics_bundle_returns_all_keys(synthetic_fact_df):
    """Verify all expected keys present in bundle dict."""
    current = [pd.Timestamp("2025-10-01"), pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-01")]
    prior = [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01")]
    out = build_diagnostics_bundle(synthetic_fact_df, pd.Timestamp("2025-12-01"), "Marketing Automation", current, prior)
    assert set(out.keys()) == {
        "monthly",
        "yoy",
        "decomp",
        "client_bridge",
        "reputation",
        "personnel",
        "market",
        "deal_sizes_curr",
        "deal_sizes_prev",
        "staffing",
        "task_mix",
        "rate_trend",
        "scorecard",
    }


def test_dataframe_immutability(synthetic_fact_df):
    """Verify input DataFrame is not mutated by any function."""
    original = synthetic_fact_df.copy(deep=True)

    current = [pd.Timestamp("2025-10-01"), pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-01")]
    prior = [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01")]

    monthly = compute_service_line_monthly(synthetic_fact_df)
    yoy = compute_yoy_snapshot(monthly, pd.Timestamp("2025-12-01"))
    _ = decompose_revenue_change(yoy)
    _ = compute_client_bridge(synthetic_fact_df, "Marketing Automation", current, prior)
    _ = compute_reputation_signals(synthetic_fact_df, "Marketing Automation", current, prior)
    _ = compute_personnel_signals(synthetic_fact_df, "Marketing Automation", current, prior)
    _ = compute_market_signals(synthetic_fact_df, "Marketing Automation", current, prior)
    _ = compute_deal_size_distribution(synthetic_fact_df, "Marketing Automation", current)
    _ = compute_staff_selling_analysis(synthetic_fact_df, "Marketing Automation")
    _ = compute_task_mix_shift(synthetic_fact_df, "Marketing Automation", current, prior)
    _ = compute_quoted_vs_actual_rate_trend(synthetic_fact_df, "Marketing Automation")
    _ = build_diagnostics_bundle(synthetic_fact_df, pd.Timestamp("2025-12-01"), "Marketing Automation", current, prior)

    assert_frame_equal(synthetic_fact_df, original)
