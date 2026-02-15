import pandas as pd
import pytest

from src.metrics.client_group_subsidy import compute_client_group_subsidy_context


def _make_df(
    selected_margin: float,
    peer_margins: list[float],
    group_columns: dict | None = None,
) -> pd.DataFrame:
    if group_columns is None:
        group_columns = {
            "client_group_rev_job_month": "G1",
            "client_group_rev_job": "G1",
            "client_group": "G1",
            "client": "G1",
        }

    rows = []
    all_jobs = [("SEL", selected_margin)] + [
        (f"P{i+1}", margin) for i, margin in enumerate(peer_margins)
    ]

    for i, (job_no, margin) in enumerate(all_jobs):
        revenue = float(abs(margin) + 100.0)
        cost = float(revenue - margin)
        row = {
            "job_no": job_no,
            "job_name": f"Job {job_no}",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-01-01") + pd.DateOffset(days=i),
            "hours_raw": float(10 + i),
            "base_cost": cost,
            "rev_alloc": revenue,
        }
        row.update(group_columns)
        rows.append(row)

    return pd.DataFrame(rows)


def _make_jobs_df(job_nos: list[str]) -> pd.DataFrame:
    rows = []
    for i, job_no in enumerate(job_nos):
        rows.append(
            {
                "job_no": job_no,
                "risk_band": "Red" if i == 0 else "Amber",
                "risk_score": 85 - (i * 10),
            }
        )
    return pd.DataFrame(rows)


def test_full_subsidy_verdict():
    df = _make_df(selected_margin=-100, peer_margins=[100, 80])
    jobs_df = _make_jobs_df(["SEL", "P1", "P2"])

    result = compute_client_group_subsidy_context(df, jobs_df, "SEL", lookback_months=12, scope="all")

    assert result["status"] == "ok"
    assert result["summary"]["verdict"] == "Fully Subsidized"
    assert result["summary"]["coverage_ratio"] == pytest.approx(1.8)
    assert result["summary"]["buffer_after_subsidy"] == pytest.approx(80.0)


def test_partial_subsidy_verdict():
    df = _make_df(selected_margin=-200, peer_margins=[140])
    jobs_df = _make_jobs_df(["SEL", "P1"])

    result = compute_client_group_subsidy_context(df, jobs_df, "SEL", lookback_months=12, scope="all")

    assert result["status"] == "ok"
    # Guardrail: when the overall group is still loss-making, do not label as subsidized.
    assert result["summary"]["verdict"] == "Weak Subsidy"
    assert result["summary"]["coverage_ratio"] == pytest.approx(0.7)
    assert result["summary"]["buffer_after_subsidy"] == pytest.approx(-60.0)


def test_weak_subsidy_verdict():
    df = _make_df(selected_margin=-200, peer_margins=[60])
    jobs_df = _make_jobs_df(["SEL", "P1"])

    result = compute_client_group_subsidy_context(df, jobs_df, "SEL", lookback_months=12, scope="all")

    assert result["status"] == "ok"
    assert result["summary"]["verdict"] == "Weak Subsidy"
    assert result["summary"]["coverage_ratio"] == pytest.approx(0.3)


def test_not_subsidized_verdict():
    df = _make_df(selected_margin=-150, peer_margins=[0, -50])
    jobs_df = _make_jobs_df(["SEL", "P1", "P2"])

    result = compute_client_group_subsidy_context(df, jobs_df, "SEL", lookback_months=12, scope="all")

    assert result["status"] == "ok"
    assert result["summary"]["verdict"] == "Not Subsidized"
    assert result["summary"]["positive_peer_margin_pool"] == 0


def test_no_subsidy_needed_verdict():
    df = _make_df(selected_margin=50, peer_margins=[25, -10])
    jobs_df = _make_jobs_df(["SEL", "P1", "P2"])

    result = compute_client_group_subsidy_context(df, jobs_df, "SEL", lookback_months=12, scope="all")

    assert result["status"] == "ok"
    assert result["summary"]["verdict"] == "No Subsidy Needed"


def test_group_negative_prevents_fully_subsidized_label():
    df = _make_df(selected_margin=-100, peer_margins=[120, -40])
    jobs_df = _make_jobs_df(["SEL", "P1", "P2"])

    result = compute_client_group_subsidy_context(df, jobs_df, "SEL", lookback_months=12, scope="all")

    assert result["status"] == "ok"
    assert result["summary"]["group_margin"] < 0
    assert result["summary"]["coverage_ratio"] > 1.0
    assert result["summary"]["verdict"] == "Weak Subsidy"
    assert result["summary"]["buffer_after_subsidy"] == pytest.approx(-20.0)


def test_group_column_fallback_to_client():
    df = _make_df(
        selected_margin=-100,
        peer_margins=[120],
        group_columns={"client": "ACME"},
    )
    jobs_df = _make_jobs_df(["SEL", "P1"])

    result = compute_client_group_subsidy_context(df, jobs_df, "SEL", lookback_months=12, scope="all")

    assert result["status"] == "ok"
    assert result["group_col"] == "client"


def test_preferred_group_col_client_uses_client_level_membership():
    rows = [
        {
            "job_no": "SEL",
            "job_name": "Job SEL",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-01-01"),
            "hours_raw": 8.0,
            "base_cost": 120.0,
            "rev_alloc": 100.0,
            "client_group_rev_job_month": "G1",
            "client": "ACME",
        },
        {
            "job_no": "P1",
            "job_name": "Job P1",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-01-02"),
            "hours_raw": 7.0,
            "base_cost": 60.0,
            "rev_alloc": 100.0,
            "client_group_rev_job_month": "G2",
            "client": "ACME",
        },
        {
            "job_no": "P2",
            "job_name": "Job P2",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-01-03"),
            "hours_raw": 6.0,
            "base_cost": 40.0,
            "rev_alloc": 100.0,
            "client_group_rev_job_month": "G1",
            "client": "OTHER",
        },
    ]
    df = pd.DataFrame(rows)
    jobs_df = _make_jobs_df(["SEL", "P1", "P2"])

    result = compute_client_group_subsidy_context(
        df,
        jobs_df,
        "SEL",
        lookback_months=None,
        scope="all",
        preferred_group_col="client",
    )

    assert result["status"] == "ok"
    assert result["group_col"] == "client"
    assert set(result["jobs"]["job_no"].tolist()) == {"SEL", "P1"}


def test_active_only_scope_keeps_active_and_forces_selected():
    df = _make_df(selected_margin=-100, peer_margins=[150, 200])
    jobs_df = _make_jobs_df(["P1"])  # SEL and P2 are inactive in this context

    result = compute_client_group_subsidy_context(
        df,
        jobs_df,
        "SEL",
        lookback_months=12,
        scope="active_only",
    )

    assert result["status"] == "ok"
    jobs = set(result["jobs"]["job_no"].tolist())
    assert jobs == {"SEL", "P1"}
    assert bool(result["jobs"].loc[result["jobs"]["job_no"] == "SEL", "is_selected"].iloc[0]) is True


def test_active_only_scope_uses_jobs_df_as_authoritative_active_set():
    rows = [
        {
            "job_no": "SEL",
            "job_name": "Job SEL",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-01"),
            "hours_raw": 8.0,
            "base_cost": 150.0,
            "rev_alloc": 100.0,
            "job_status": "In Progress",
            "job_completed_date": pd.NaT,
            "client_group_rev_job_month": "G1",
        },
        {
            "job_no": "P1",
            "job_name": "Job P1",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-02"),
            "hours_raw": 7.0,
            "base_cost": 60.0,
            "rev_alloc": 100.0,
            "job_status": "In Progress",
            "job_completed_date": pd.NaT,
            "client_group_rev_job_month": "G1",
        },
        {
            "job_no": "P2",
            "job_name": "Job P2",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-03"),
            "hours_raw": 5.0,
            "base_cost": 80.0,
            "rev_alloc": 50.0,
            "job_status": "Completed",
            "job_completed_date": pd.Timestamp("2025-12-04"),
            "client_group_rev_job_month": "G1",
        },
    ]
    df = pd.DataFrame(rows)
    # jobs_df intentionally missing P1; jobs_df should remain authoritative.
    jobs_df = _make_jobs_df(["SEL"])

    result = compute_client_group_subsidy_context(
        df,
        jobs_df,
        "SEL",
        lookback_months=12,
        scope="active_only",
    )

    assert result["status"] == "ok"
    jobs = set(result["jobs"]["job_no"].tolist())
    assert jobs == {"SEL"}


def test_active_only_scope_falls_back_to_status_when_jobs_df_empty():
    rows = [
        {
            "job_no": "SEL",
            "job_name": "Job SEL",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-01"),
            "hours_raw": 8.0,
            "base_cost": 150.0,
            "rev_alloc": 100.0,
            "job_status": "In Progress",
            "job_completed_date": pd.NaT,
            "client_group_rev_job_month": "G1",
        },
        {
            "job_no": "P1",
            "job_name": "Job P1",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-02"),
            "hours_raw": 7.0,
            "base_cost": 60.0,
            "rev_alloc": 100.0,
            "job_status": "In Progress",
            "job_completed_date": pd.NaT,
            "client_group_rev_job_month": "G1",
        },
        {
            "job_no": "P2",
            "job_name": "Job P2",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-03"),
            "hours_raw": 5.0,
            "base_cost": 80.0,
            "rev_alloc": 50.0,
            "job_status": "Completed",
            "job_completed_date": pd.Timestamp("2025-12-04"),
            "client_group_rev_job_month": "G1",
        },
    ]
    df = pd.DataFrame(rows)
    jobs_df = pd.DataFrame(columns=["job_no", "risk_band", "risk_score"])

    result = compute_client_group_subsidy_context(
        df,
        jobs_df,
        "SEL",
        lookback_months=12,
        scope="active_only",
    )

    assert result["status"] == "ok"
    jobs = set(result["jobs"]["job_no"].tolist())
    assert jobs == {"SEL", "P1"}


def test_active_only_scope_excludes_job_when_any_row_is_completed_status():
    rows = [
        {
            "job_no": "SEL",
            "job_name": "Job SEL",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-01"),
            "hours_raw": 8.0,
            "base_cost": 120.0,
            "rev_alloc": 100.0,
            "job_status": "In Progress",
            "job_completed_date": pd.NaT,
            "client_group_rev_job_month": "G1",
        },
        {
            "job_no": "P1",
            "job_name": "Job P1",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-02"),
            "hours_raw": 7.0,
            "base_cost": 60.0,
            "rev_alloc": 100.0,
            "job_status": "In Progress",
            "job_completed_date": pd.NaT,
            "client_group_rev_job_month": "G1",
        },
        {
            "job_no": "P2",
            "job_name": "Job P2",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-11-15"),
            "hours_raw": 3.0,
            "base_cost": 10.0,
            "rev_alloc": 15.0,
            "job_status": "In Progress",
            "job_completed_date": pd.NaT,
            "client_group_rev_job_month": "G1",
        },
        {
            "job_no": "P2",
            "job_name": "Job P2",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-03"),
            "hours_raw": 5.0,
            "base_cost": 20.0,
            "rev_alloc": 20.0,
            "job_status": "Completed",
            "job_completed_date": pd.NaT,
            "client_group_rev_job_month": "G1",
        },
    ]
    df = pd.DataFrame(rows)
    jobs_df = _make_jobs_df(["SEL", "P1", "P2"])

    result = compute_client_group_subsidy_context(
        df,
        jobs_df,
        "SEL",
        lookback_months=12,
        scope="active_only",
    )

    assert result["status"] == "ok"
    jobs = set(result["jobs"]["job_no"].tolist())
    assert jobs == {"SEL", "P1"}


def test_group_membership_is_job_level_and_keeps_full_history_margin():
    rows = [
        # Selected job: older row has no group tag, newer row does.
        {
            "job_no": "SEL",
            "job_name": "Job SEL",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2024-01-15"),
            "hours_raw": 8.0,
            "base_cost": 220.0,
            "rev_alloc": 100.0,
            "client_group_rev_job_month": None,
        },
        {
            "job_no": "SEL",
            "job_name": "Job SEL",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-15"),
            "hours_raw": 6.0,
            "base_cost": 40.0,
            "rev_alloc": 100.0,
            "client_group_rev_job_month": "G1",
        },
        # Peer in same current group.
        {
            "job_no": "P1",
            "job_name": "Job P1",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2024-02-01"),
            "hours_raw": 7.0,
            "base_cost": 30.0,
            "rev_alloc": 50.0,
            "client_group_rev_job_month": None,
        },
        {
            "job_no": "P1",
            "job_name": "Job P1",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-12-10"),
            "hours_raw": 7.0,
            "base_cost": 20.0,
            "rev_alloc": 50.0,
            "client_group_rev_job_month": "G1",
        },
        # Stale job: same group but no activity in lookback window.
        {
            "job_no": "STALE",
            "job_name": "Job STALE",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2023-06-01"),
            "hours_raw": 5.0,
            "base_cost": 20.0,
            "rev_alloc": 80.0,
            "client_group_rev_job_month": "G1",
        },
    ]
    df = pd.DataFrame(rows)
    jobs_df = _make_jobs_df(["SEL", "P1"])

    result = compute_client_group_subsidy_context(
        df,
        jobs_df,
        "SEL",
        lookback_months=12,
        scope="all",
    )

    assert result["status"] == "ok"
    # SEL full-history margin% = (200 - 260) / 200 * 100 = -30%
    assert result["summary"]["selected_margin_pct"] == pytest.approx(-30.0)

    jobs = result["jobs"].set_index("job_no")
    assert jobs.loc["SEL", "margin_pct"] == pytest.approx(-30.0)
    assert jobs.loc["P1", "margin_pct"] == pytest.approx(50.0)
    assert "STALE" not in set(result["jobs"]["job_no"].tolist())


def test_blank_job_names_fall_back_to_job_no():
    rows = [
        {
            "job_no": "SEL",
            "job_name": "",
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-01-01"),
            "hours_raw": 5.0,
            "base_cost": 120.0,
            "rev_alloc": 100.0,
            "client_group_rev_job_month": "G1",
        },
        {
            "job_no": "P1",
            "job_name": None,
            "department_final": "Tax",
            "job_category": "Compliance",
            "work_date": pd.Timestamp("2025-01-02"),
            "hours_raw": 4.0,
            "base_cost": 40.0,
            "rev_alloc": 80.0,
            "client_group_rev_job_month": "G1",
        },
    ]
    df = pd.DataFrame(rows)
    jobs_df = _make_jobs_df(["SEL", "P1"])

    result = compute_client_group_subsidy_context(
        df,
        jobs_df,
        "SEL",
        lookback_months=None,
        scope="all",
    )

    assert result["status"] == "ok"
    names = dict(zip(result["jobs"]["job_no"], result["jobs"]["job_name"]))
    assert names["SEL"] == "SEL"
    assert names["P1"] == "P1"
