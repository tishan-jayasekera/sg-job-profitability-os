"""
Tests for service load analytics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.service_load import (
    build_job_description_index,
    define_service_scope,
    compute_staff_service_load,
    compute_staff_client_breakdown,
    compute_scope_task_breakdown,
    compute_staff_task_capacity_flow,
    compute_scope_budget_comparison,
    compute_new_client_absorption,
)


def make_test_data() -> pd.DataFrame:
    months = pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"])
    job_meta = {
        "J001": {
            "job_name": "Organic Social Retainer",
            "job_description": "Organic Social Media Management",
            "job_category": "Organic",
            "client": "Client Alpha",
            "is_billable": True,
            "task_name": "Copywriting",
        },
        "J002": {
            "job_name": "Paid Ads Campaign",
            "job_description": "Paid Advertising Campaign",
            "job_category": "Advertising",
            "client": "Client Beta",
            "is_billable": False,
            "task_name": "Media Buying",
        },
        "J003": {
            "job_name": "Social Scheduling Sprint",
            "job_description": "Social Content and Scheduling",
            "job_category": "Content",
            "client": "Client Gamma",
            "is_billable": True,
            "task_name": "Scheduling",
        },
        "J004": {
            "job_name": "Website Refresh",
            "job_description": "Website Redesign",
            "job_category": "Web",
            "client": "Client Delta",
            "is_billable": False,
            "task_name": "Design",
        },
    }

    rows: list[dict] = []

    def add_row(month: pd.Timestamp, staff: str, job_no: str, hours: float, fte: float) -> None:
        meta = job_meta[job_no]
        rows.append(
            {
                "staff_name": staff,
                "job_no": job_no,
                "job_name": meta["job_name"],
                "job_description": meta["job_description"],
                "job_category": meta["job_category"],
                "category_rev_job": meta["job_category"],
                "task_name": meta["task_name"],
                "client": meta["client"],
                "month_key": month,
                "work_date": month + pd.Timedelta(days=14),
                "hours_raw": hours,
                "base_cost": hours * 50,
                "rev_alloc": hours * 100,
                "is_billable": meta["is_billable"],
                "fte_hours_scaling": fte,
            }
        )

    for month in months:
        # Alice (3 jobs)
        add_row(month, "Alice", "J001", 20, 1.0)
        add_row(month, "Alice", "J002", 10, 1.0)
        add_row(month, "Alice", "J004", 10, 1.0)

        # Bob
        add_row(month, "Bob", "J003", 15, 0.8)
        add_row(month, "Bob", "J002", 5, 0.8)

        # Carol
        add_row(month, "Carol", "J001", 10, 1.0)
        add_row(month, "Carol", "J004", 20, 1.0)

    return pd.DataFrame(rows)


def test_build_job_description_index():
    df = make_test_data()
    job_index = build_job_description_index(df)

    assert len(job_index) == 4
    assert job_index["search_text"].str.lower().eq(job_index["search_text"]).all()
    row = job_index[job_index["job_no"] == "J001"].iloc[0]
    assert "organic social retainer" in row["search_text"]
    assert "organic social media management" in row["search_text"]


def test_define_scope_description_keywords_single():
    df = make_test_data()
    job_index = build_job_description_index(df)
    mask, matched = define_service_scope(df, job_index, description_keywords=["organic social"])

    assert set(df.loc[mask, "job_no"].unique()) == {"J001"}
    assert set(matched["job_no"]) == {"J001"}


def test_define_scope_description_keywords_multiple():
    df = make_test_data()
    job_index = build_job_description_index(df)
    mask, matched = define_service_scope(df, job_index, description_keywords=["social", "scheduling"])

    assert set(df.loc[mask, "job_no"].unique()) == {"J001", "J003"}
    assert set(matched["job_no"]) == {"J001", "J003"}


def test_define_scope_keyword_plus_category_intersection():
    df = make_test_data()
    job_index = build_job_description_index(df)
    mask, matched = define_service_scope(
        df,
        job_index,
        description_keywords=["social"],
        categories=["Organic"],
    )

    assert set(df.loc[mask, "job_no"].unique()) == {"J001"}
    assert set(matched["job_no"]) == {"J001"}


def test_define_scope_keyword_plus_task_intersection():
    df = make_test_data()
    job_index = build_job_description_index(df)
    mask, matched = define_service_scope(
        df,
        job_index,
        description_keywords=["social"],
        task_names=["Copywriting"],
    )

    assert set(df.loc[mask, "job_no"].unique()) == {"J001"}
    assert set(matched["job_no"]) == {"J001"}


def test_define_scope_no_filters_returns_all():
    df = make_test_data()
    job_index = build_job_description_index(df)
    mask, matched = define_service_scope(df, job_index)

    assert mask.all()
    assert set(matched["job_no"]) == {"J001", "J002", "J003", "J004"}


def test_define_scope_no_description_column_falls_back_to_name():
    df = make_test_data().drop(columns=["job_description"])
    job_index = build_job_description_index(df)
    mask, matched = define_service_scope(df, job_index, description_keywords=["organic"])

    assert set(df.loc[mask, "job_no"].unique()) == {"J001"}
    assert set(matched["job_no"]) == {"J001"}


def test_compute_staff_service_load_basic():
    df = make_test_data()
    scope_mask = df["job_no"] == "J001"
    load = compute_staff_service_load(
        df,
        scope_mask,
        lookback_months=3,
        reference_date=pd.Timestamp("2025-04-15"),
    )
    alice = load[load["staff_name"] == "Alice"].iloc[0]

    assert alice["in_scope_hours"] == pytest.approx(60.0)
    assert alice["in_scope_hours_per_month"] == pytest.approx(20.0)
    assert alice["monthly_capacity"] == pytest.approx(38.0 * 4.33, rel=1e-4)
    assert alice["months_in_window"] == 3


def test_compute_staff_service_load_fte_scaling():
    df = make_test_data()
    scope_mask = df["job_no"] == "J003"
    load = compute_staff_service_load(
        df,
        scope_mask,
        lookback_months=3,
        reference_date=pd.Timestamp("2025-04-15"),
    )
    bob = load[load["staff_name"] == "Bob"].iloc[0]

    assert bob["fte_scaling"] == pytest.approx(0.8)
    assert bob["monthly_capacity"] == pytest.approx(38.0 * 0.8 * 4.33, rel=1e-4)


def test_compute_staff_client_breakdown():
    df = make_test_data()
    scope_mask = df["job_no"] == "J001"
    breakdown = compute_staff_client_breakdown(df, "Alice", scope_mask, lookback_months=3)

    assert len(breakdown) == 3
    assert breakdown["share_of_staff_total_pct"].sum() == pytest.approx(100.0, rel=1e-5)
    in_scope_map = dict(zip(breakdown["job_no"], breakdown["is_in_scope"]))
    assert in_scope_map["J001"] is True
    assert in_scope_map["J002"] is False
    assert in_scope_map["J004"] is False


def test_compute_scope_task_breakdown():
    df = make_test_data()
    scope_mask = df["job_no"].isin(["J001", "J003"])
    task_breakdown = compute_scope_task_breakdown(df, scope_mask, lookback_months=3)

    assert set(task_breakdown["task_name"]) == {"Copywriting", "Scheduling"}
    assert task_breakdown["share_of_scope_pct"].sum() == pytest.approx(100.0, rel=1e-6)

    copy_row = task_breakdown[task_breakdown["task_name"] == "Copywriting"].iloc[0]
    assert copy_row["hours_per_month"] == pytest.approx(30.0)  # (Alice 20 + Carol 10) per month
    assert copy_row["staff_count"] == 2
    assert copy_row["job_count"] == 1
    assert copy_row["billable_pct"] == pytest.approx(100.0)


def test_compute_staff_task_capacity_flow():
    months = pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"])
    rows = []
    for m in months:
        rows.extend(
            [
                {
                    "staff_name": "Alice",
                    "job_no": "J010",
                    "job_name": "Client A Retainer",
                    "client": "Client A",
                    "task_name": "Copywriting",
                    "month_key": m,
                    "hours_raw": 10,
                },
                {
                    "staff_name": "Alice",
                    "job_no": "J011",
                    "job_name": "Client B Retainer",
                    "client": "Client B",
                    "task_name": "Copywriting",
                    "month_key": m,
                    "hours_raw": 5,
                },
                {
                    "staff_name": "Alice",
                    "job_no": "J012",
                    "job_name": "Client C Retainer",
                    "client": "Client C",
                    "task_name": "Scheduling",
                    "month_key": m,
                    "hours_raw": 6,
                },
                {
                    "staff_name": "Alice",
                    "job_no": "J010",
                    "job_name": "Client A Retainer",
                    "client": "Client A",
                    "task_name": "Scheduling",
                    "month_key": m,
                    "hours_raw": 5,
                },
                {
                    "staff_name": "Bob",
                    "job_no": "J013",
                    "job_name": "Other Work",
                    "client": "Client D",
                    "task_name": "Design",
                    "month_key": m,
                    "hours_raw": 8,
                },
            ]
        )
    df = pd.DataFrame(rows)
    scope_mask = df["job_no"].isin(["J010", "J011", "J012"])

    summary, detail = compute_staff_task_capacity_flow(
        df,
        staff_name="Alice",
        scope_mask=scope_mask,
        lookback_months=3,
    )

    assert set(summary["task_name"]) == {"Copywriting", "Scheduling"}
    assert summary["share_of_staff_scope_pct"].sum() == pytest.approx(100.0, rel=1e-6)

    copy_task = summary[summary["task_name"] == "Copywriting"].iloc[0]
    assert copy_task["hours_per_month"] == pytest.approx(15.0)
    assert copy_task["job_count"] == 2
    assert copy_task["top_job_no"] == "J010"
    assert copy_task["top_job_share_of_task_pct"] == pytest.approx(66.6666, rel=1e-3)

    copy_detail = detail[detail["task_name"] == "Copywriting"]
    assert set(copy_detail["job_no"]) == {"J010", "J011"}
    assert copy_detail["share_of_task_pct"].sum() == pytest.approx(100.0, rel=1e-6)
    copy_j010 = copy_detail[copy_detail["job_no"] == "J010"].iloc[0]
    assert copy_j010["share_of_job_pct"] == pytest.approx(66.6666, rel=1e-3)


def test_compute_staff_task_capacity_flow_multiple_staff():
    df = make_test_data()
    scope_mask = df["job_no"].isin(["J001", "J003"])
    summary, detail = compute_staff_task_capacity_flow(
        df,
        staff_name=["Alice", "Bob"],
        scope_mask=scope_mask,
        lookback_months=3,
    )

    assert set(summary["task_name"]) == {"Copywriting", "Scheduling"}
    assert summary["share_of_staff_scope_pct"].sum() == pytest.approx(100.0, rel=1e-6)
    assert detail["hours_per_month"].sum() == pytest.approx(summary["hours_per_month"].sum(), rel=1e-6)


def test_scope_budget_comparison():
    staff_load = pd.DataFrame(
        {
            "staff_name": ["A", "B", "C"],
            "in_scope_hours_per_month": [90.0, 60.0, 80.0],
        }
    )
    compared = compute_scope_budget_comparison(staff_load, scope_budget_hours_per_month=80.0)
    status = dict(zip(compared["staff_name"], compared["scope_status"]))

    assert status["A"] == "Over Budget"
    assert status["B"] == "Under Budget"
    assert status["C"] == "On Track"


def test_new_client_absorption():
    staff_load = pd.DataFrame(
        {
            "staff_name": ["A"],
            "headroom_hours_per_month": [30.0],
            "scope_budget_remaining": [20.0],
            "total_hours_per_month": [130.0],
            "monthly_capacity": [160.0],
        }
    )
    result = compute_new_client_absorption(
        staff_load,
        avg_hours_per_new_client_per_month=20.0,
    ).iloc[0]

    assert result["additional_clients_headroom"] == 1
    assert result["additional_clients_budget"] == 1
    assert result["absorption_estimate"] == 1
    assert result["overload_hours_if_one_more"] == pytest.approx(-10.0)


def test_matched_jobs_df_has_match_reason():
    df = make_test_data()
    job_index = build_job_description_index(df)
    _, matched = define_service_scope(df, job_index, description_keywords=["social"])

    assert "match_reason" in matched.columns
    assert matched["match_reason"].str.contains("keyword: social", na=False).all()
