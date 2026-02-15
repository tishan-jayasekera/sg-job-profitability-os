import numpy as np
import pandas as pd
import pytest

from src.modeling.completion_forecast import (
    build_peer_lifecycle_profiles,
    compute_job_timeline,
    estimate_remaining_work,
    forecast_completion,
)


def _make_fact_row(
    job_no: str,
    work_date: str,
    task_name: str,
    hours: float,
    department: str = "Tax",
    category: str = "Compliance",
    job_status: str = "Completed",
    completed_date: str | None = "2025-12-31",
    quoted_hours_task: float = 0.0,
    quoted_amount_task: float = 0.0,
    base_cost_rate: float = 100.0,
    rev_rate: float = 150.0,
) -> dict:
    ts = pd.Timestamp(work_date)
    return {
        "job_no": job_no,
        "department_final": department,
        "category_rev_job": category,
        "job_category": category,
        "task_name": task_name,
        "work_date": ts,
        "month_key": ts.to_period("M").to_timestamp(),
        "hours_raw": float(hours),
        "base_cost": float(hours * base_cost_rate),
        "rev_alloc": float(hours * rev_rate),
        "quoted_time_total": float(quoted_hours_task),
        "quoted_amount_total": float(quoted_amount_task),
        "quote_match_flag": "matched",
        "job_status": job_status,
        "job_completed_date": pd.Timestamp(completed_date) if completed_date else pd.NaT,
    }


def _make_completed_peer_job(job_no: str, start_date: str, dept: str = "Tax", cat: str = "Compliance") -> list[dict]:
    rows: list[dict] = []
    start = pd.Timestamp(start_date)
    for i in range(10):
        day = start + pd.Timedelta(days=i)
        rows.append(
            _make_fact_row(
                job_no=job_no,
                work_date=str(day.date()),
                task_name="Prep",
                hours=6.0,
                department=dept,
                category=cat,
                job_status="Completed",
                completed_date="2025-12-31",
                quoted_hours_task=60,
                quoted_amount_task=6000,
            )
        )
        rows.append(
            _make_fact_row(
                job_no=job_no,
                work_date=str(day.date()),
                task_name="Review",
                hours=4.0,
                department=dept,
                category=cat,
                job_status="Completed",
                completed_date="2025-12-31",
                quoted_hours_task=40,
                quoted_amount_task=4000,
            )
        )
    return rows


def _make_active_job_rows(
    job_no: str = "A1",
    dept: str = "Tax",
    cat: str = "Compliance",
    prep_hours: float = 20.0,
    review_hours: float = 10.0,
    include_unmodeled: bool = False,
) -> list[dict]:
    rows = [
        _make_fact_row(
            job_no=job_no,
            work_date="2026-01-01",
            task_name="Prep",
            hours=prep_hours,
            department=dept,
            category=cat,
            job_status="In Progress",
            completed_date=None,
            quoted_hours_task=60,
            quoted_amount_task=6000,
        ),
        _make_fact_row(
            job_no=job_no,
            work_date="2026-01-08",
            task_name="Review",
            hours=review_hours,
            department=dept,
            category=cat,
            job_status="In Progress",
            completed_date=None,
            quoted_hours_task=40,
            quoted_amount_task=4000,
        ),
    ]
    if include_unmodeled:
        rows.append(
            _make_fact_row(
                job_no=job_no,
                work_date="2026-01-10",
                task_name="Client Mgmt",
                hours=8.0,
                department=dept,
                category=cat,
                job_status="In Progress",
                completed_date=None,
                quoted_hours_task=0,
                quoted_amount_task=0,
            )
        )
    return rows


def _job_row(**overrides) -> pd.Series:
    base = {
        "job_no": "A1",
        "department_final": "Tax",
        "job_category": "Compliance",
        "quoted_hours": 100.0,
        "quoted_amount": 10000.0,
        "actual_hours": 30.0,
        "actual_cost": 3000.0,
        "actual_revenue": 4500.0,
        "burn_rate_per_day": 5.0,
        "median_margin_pct": 35.0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_build_peer_lifecycle_profiles_basic():
    rows = []
    for idx in range(5):
        rows.extend(_make_completed_peer_job(f"P{idx+1}", f"2025-01-{idx+1:02d}"))
    rows.extend(_make_active_job_rows(job_no="A1"))
    df = pd.DataFrame(rows)

    lifecycle_df, task_summary_df = build_peer_lifecycle_profiles(
        df_all=df,
        department="Tax",
        category="Compliance",
        exclude_job_no="A1",
    )

    assert not lifecycle_df.empty
    assert not task_summary_df.empty
    assert set(lifecycle_df["decile_bucket"].unique()) == set(range(10))

    decile_sums = lifecycle_df.groupby("decile_bucket")["median_hours_share"].sum()
    assert (decile_sums > 0.99).all()
    assert (decile_sums < 1.01).all()


def test_build_peer_lifecycle_profiles_no_peers():
    df = pd.DataFrame(_make_active_job_rows(job_no="A1"))

    lifecycle_df, task_summary_df = build_peer_lifecycle_profiles(
        df_all=df,
        department="Tax",
        category="Compliance",
        exclude_job_no="A1",
    )

    assert lifecycle_df.empty
    assert task_summary_df.empty


def test_compute_job_timeline_weekly():
    rows = [
        _make_fact_row("A1", "2026-01-01", "Prep", 2.0, job_status="In Progress", completed_date=None),
        _make_fact_row("A1", "2026-01-03", "Review", 3.0, job_status="In Progress", completed_date=None),
        _make_fact_row("A1", "2026-01-10", "Prep", 4.0, job_status="In Progress", completed_date=None),
    ]
    df = pd.DataFrame(rows)

    timeline = compute_job_timeline(df, "A1", granularity="weekly")

    assert not timeline.empty
    assert timeline["period_label"].nunique() == 2
    assert timeline["hours"].sum() == pytest.approx(9.0)
    assert timeline["cost"].sum() == pytest.approx(900.0)
    assert timeline["cumulative_hours"].max() == pytest.approx(9.0)
    assert timeline["cumulative_cost"].max() == pytest.approx(900.0)


def test_estimate_remaining_work_normal():
    rows = []
    for idx in range(5):
        rows.extend(_make_completed_peer_job(f"P{idx+1}", f"2025-02-{idx+1:02d}"))
    rows.extend(_make_active_job_rows(job_no="A1", prep_hours=20.0, review_hours=10.0))
    df = pd.DataFrame(rows)

    remaining_df, summary = estimate_remaining_work(
        df_all=df,
        job_no="A1",
        department="Tax",
        category="Compliance",
        job_row=_job_row(quoted_hours=100.0),
    )

    assert not remaining_df.empty
    prep_row = remaining_df[remaining_df["task_name"] == "Prep"].iloc[0]
    review_row = remaining_df[remaining_df["task_name"] == "Review"].iloc[0]

    assert prep_row["remaining_hours_median"] > 0
    assert review_row["remaining_hours_median"] > 0
    assert summary["total_remaining_median"] > 0


def test_estimate_remaining_work_outlier_detection():
    rows = []
    for idx in range(5):
        rows.extend(_make_completed_peer_job(f"P{idx+1}", f"2025-03-{idx+1:02d}"))
    rows.extend(_make_active_job_rows(job_no="A1", prep_hours=100.0, review_hours=10.0))
    df = pd.DataFrame(rows)

    remaining_df, _summary = estimate_remaining_work(
        df_all=df,
        job_no="A1",
        department="Tax",
        category="Compliance",
        job_row=_job_row(quoted_hours=100.0, actual_hours=110.0),
    )

    prep_row = remaining_df[remaining_df["task_name"] == "Prep"].iloc[0]
    assert bool(prep_row["outlier_flag"]) is True
    assert prep_row["remaining_hours_median"] == 0


def test_estimate_remaining_work_unmodeled_tasks():
    rows = []
    for idx in range(5):
        rows.extend(_make_completed_peer_job(f"P{idx+1}", f"2025-04-{idx+1:02d}"))
    rows.extend(_make_active_job_rows(job_no="A1", include_unmodeled=True))
    df = pd.DataFrame(rows)

    remaining_df, summary = estimate_remaining_work(
        df_all=df,
        job_no="A1",
        department="Tax",
        category="Compliance",
        job_row=_job_row(quoted_hours=100.0),
    )

    unmodeled = remaining_df[remaining_df["task_name"] == "Client Mgmt"].iloc[0]
    assert bool(unmodeled["unmodeled_flag"]) is True
    assert unmodeled["remaining_hours_median"] == 0
    assert summary["unmodeled_task_count"] == 1


def test_estimate_remaining_work_task_scope_with_peer_task_addition():
    rows = []
    for idx in range(5):
        rows.extend(_make_completed_peer_job(f"P{idx+1}", f"2025-04-{idx+1:02d}"))
    rows.append(
        _make_fact_row(
            job_no="A1",
            work_date="2026-01-08",
            task_name="Prep",
            hours=12.0,
            job_status="In Progress",
            completed_date=None,
            quoted_hours_task=60.0,
            quoted_amount_task=6000.0,
        )
    )
    df = pd.DataFrame(rows)

    prep_only_df, prep_only_summary = estimate_remaining_work(
        df_all=df,
        job_no="A1",
        department="Tax",
        category="Compliance",
        job_row=_job_row(quoted_hours=100.0, actual_hours=12.0),
        included_tasks=("Prep",),
    )
    assert set(prep_only_df["task_name"].tolist()) == {"Prep"}
    assert prep_only_summary["task_scope_count"] == 1

    with_peer_df, with_peer_summary = estimate_remaining_work(
        df_all=df,
        job_no="A1",
        department="Tax",
        category="Compliance",
        job_row=_job_row(quoted_hours=100.0, actual_hours=12.0),
        included_tasks=("Prep", "Review"),
    )
    assert set(with_peer_df["task_name"].tolist()) == {"Prep", "Review"}
    review_row = with_peer_df[with_peer_df["task_name"] == "Review"].iloc[0]
    assert review_row["actual_hours"] == pytest.approx(0.0)
    assert review_row["remaining_hours_median"] > 0
    assert with_peer_summary["task_scope_count"] == 2


def test_estimate_remaining_work_lifecycle_adjustment_applies():
    rows = []
    for idx in range(5):
        rows.extend(_make_completed_peer_job(f"P{idx+1}", f"2025-05-{idx+1:02d}"))
    rows.extend(_make_active_job_rows(job_no="A1", prep_hours=24.0, review_hours=16.0))
    df = pd.DataFrame(rows)

    remaining_df, summary = estimate_remaining_work(
        df_all=df,
        job_no="A1",
        department="Tax",
        category="Compliance",
        job_row=_job_row(quoted_hours=100.0, actual_hours=40.0),
    )

    assert not remaining_df.empty
    assert bool(summary["lifecycle_adjustment_applied"]) is True
    assert summary["eac_pre_adjustment"] == pytest.approx(100.0)
    assert summary["eac_baseline"] < 100.0
    assert summary["eac_baseline"] > 40.0
    assert summary["runtime_progress_pct"] > 0
    assert summary["lifecycle_expected_completion_pct"] > 0


def test_forecast_completion_normal():
    df = pd.DataFrame(
        _make_active_job_rows(job_no="A1", prep_hours=20.0, review_hours=10.0)
    )
    remaining_summary = {
        "total_remaining_p25": 20.0,
        "total_remaining_median": 30.0,
        "total_remaining_p75": 40.0,
    }

    result = forecast_completion(
        job_row=_job_row(
            burn_rate_per_day=5.0,
            actual_cost=1000.0,
            actual_hours=50.0,
            actual_revenue=1500.0,
            quoted_amount=3000.0,
        ),
        remaining_summary=remaining_summary,
        df_all=df,
        job_no="A1",
    )

    expected = result["scenarios"]["expected"]
    last_activity = pd.Timestamp("2026-01-08")

    assert result["is_stalled"] is False
    assert expected["days_to_complete"] == pytest.approx(6.0)
    assert expected["forecast_end_date"] == last_activity + pd.Timedelta(days=6)
    assert expected["forecast_total_cost"] == pytest.approx(1600.0)
    assert expected["forecast_revenue"] == pytest.approx(3000.0)
    assert expected["forecast_margin_pct"] == pytest.approx(((3000.0 - 1600.0) / 3000.0) * 100)


def test_forecast_completion_stalled():
    df = pd.DataFrame(
        [
            _make_fact_row(
                "A1",
                "2026-01-01",
                "Prep",
                0.0,
                job_status="In Progress",
                completed_date=None,
            )
        ]
    )
    remaining_summary = {
        "total_remaining_p25": 10.0,
        "total_remaining_median": 20.0,
        "total_remaining_p75": 30.0,
    }

    result = forecast_completion(
        job_row=_job_row(burn_rate_per_day=0.0, actual_cost=1000.0, actual_hours=0.0, actual_revenue=0.0),
        remaining_summary=remaining_summary,
        df_all=df,
        job_no="A1",
    )

    assert result["is_stalled"] is True
    assert pd.isna(result["scenarios"]["expected"]["forecast_end_date"])


def test_forecast_completion_no_quote():
    df = pd.DataFrame(
        _make_active_job_rows(job_no="A1", prep_hours=25.0, review_hours=25.0)
    )
    df["quoted_amount_total"] = 0.0
    remaining_summary = {
        "total_remaining_p25": 5.0,
        "total_remaining_median": 10.0,
        "total_remaining_p75": 15.0,
    }

    result = forecast_completion(
        job_row=_job_row(
            quoted_amount=np.nan,
            burn_rate_per_day=5.0,
            actual_cost=1000.0,
            actual_hours=50.0,
            actual_revenue=1000.0,
        ),
        remaining_summary=remaining_summary,
        df_all=df,
        job_no="A1",
    )

    expected = result["scenarios"]["expected"]
    assert expected["forecast_revenue"] == pytest.approx(1200.0)


def test_forecast_completion_uses_scoped_summary_values():
    df = pd.DataFrame(
        _make_active_job_rows(job_no="A1", prep_hours=25.0, review_hours=25.0)
    )
    remaining_summary = {
        "total_remaining_p25": 5.0,
        "total_remaining_median": 10.0,
        "total_remaining_p75": 15.0,
        "included_tasks": ("Prep",),
        "scoped_actual_hours": 25.0,
        "scoped_actual_cost": 2500.0,
        "scoped_actual_revenue": 3750.0,
        "scoped_quoted_amount": 6000.0,
    }

    result = forecast_completion(
        job_row=_job_row(
            quoted_amount=10000.0,
            burn_rate_per_day=5.0,
            actual_cost=1000.0,
            actual_hours=50.0,
            actual_revenue=1000.0,
        ),
        remaining_summary=remaining_summary,
        df_all=df,
        job_no="A1",
    )

    expected = result["scenarios"]["expected"]
    # Scoped cost rate = 2500/25 = 100; 10 remaining -> +1000 cost
    assert expected["forecast_total_cost"] == pytest.approx(3500.0)
    # Scoped quoted amount should be used over job-row quote
    assert expected["forecast_revenue"] == pytest.approx(6000.0)


def test_forecast_completion_partial_scope_without_quote_does_not_use_full_job_quote():
    df = pd.DataFrame(
        _make_active_job_rows(job_no="A1", prep_hours=25.0, review_hours=25.0)
    )
    # Simulate selected scope task without usable quote.
    df.loc[df["task_name"] == "Prep", "quoted_amount_total"] = 0.0
    df.loc[df["task_name"] == "Prep", "quoted_time_total"] = 0.0

    remaining_summary = {
        "total_remaining_p25": 5.0,
        "total_remaining_median": 10.0,
        "total_remaining_p75": 15.0,
        "included_tasks": ("Prep",),
        "scope_is_full_job": False,
        "scoped_actual_hours": 25.0,
        "scoped_actual_cost": 1250.0,
        "scoped_actual_revenue": 1250.0,
        "scoped_quoted_amount": np.nan,
    }

    result = forecast_completion(
        job_row=_job_row(
            quoted_amount=10000.0,
            burn_rate_per_day=5.0,
            actual_cost=5000.0,
            actual_hours=50.0,
            actual_revenue=7500.0,
        ),
        remaining_summary=remaining_summary,
        df_all=df,
        job_no="A1",
    )

    expected = result["scenarios"]["expected"]
    # Revenue extrapolation should use scoped realised rate (1250/25 = 50/hr):
    # 50 * (25 + 10) = 1750
    assert expected["forecast_revenue"] == pytest.approx(1750.0)
