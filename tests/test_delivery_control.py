import pandas as pd

from src.metrics.delivery_control import (
    compute_delivery_control_view,
    compute_root_cause_drivers,
    compute_benchmarks,
)


def test_delivery_control_risk_banding():
    df = pd.DataFrame([
        {
            "job_no": "H1",
            "department_final": "DeptA",
            "category_rev_job": "Cat1",
            "task_name": "TaskA",
            "work_date": pd.Timestamp("2024-01-15"),
            "hours_raw": 200,
            "base_cost": 45,
            "rev_alloc": 50,
            "quoted_time_total": 100,
            "quoted_amount_total": 2000,
            "quote_match_flag": "matched",
            "job_status": "Active",
        },
        {
            "job_no": "L1",
            "department_final": "DeptA",
            "category_rev_job": "Cat1",
            "task_name": "TaskA",
            "work_date": pd.Timestamp("2024-01-15"),
            "hours_raw": 40,
            "base_cost": 20,
            "rev_alloc": 80,
            "quoted_time_total": 100,
            "quoted_amount_total": 90,
            "quote_match_flag": "matched",
            "job_status": "Active",
        },
        {
            "job_no": "C1",
            "department_final": "DeptA",
            "category_rev_job": "Cat1",
            "task_name": "TaskA",
            "work_date": pd.Timestamp("2023-10-01"),
            "hours_raw": 50,
            "base_cost": 20,
            "rev_alloc": 100,
            "quoted_time_total": 60,
            "quoted_amount_total": 120,
            "quote_match_flag": "matched",
            "job_status": "Completed",
            "job_completed_date": pd.Timestamp("2023-12-15"),
        },
        {
            "job_no": "C1",
            "department_final": "DeptA",
            "category_rev_job": "Cat1",
            "task_name": "TaskA",
            "work_date": pd.Timestamp("2023-12-15"),
            "hours_raw": 10,
            "base_cost": 5,
            "rev_alloc": 20,
            "quoted_time_total": 60,
            "quoted_amount_total": 120,
            "quote_match_flag": "matched",
            "job_status": "Completed",
            "job_completed_date": pd.Timestamp("2023-12-15"),
        },
    ])

    jobs_df = compute_delivery_control_view(df, recency_days=28)

    bands = jobs_df.set_index("job_no")["risk_band"].to_dict()
    assert bands["H1"] == "Red"
    assert bands["L1"] == "Green"


def test_root_cause_driver_scope_creep():
    df = pd.DataFrame([
        {
            "job_no": "SC1",
            "department_final": "DeptA",
            "category_rev_job": "Cat1",
            "task_name": "TaskA",
            "work_date": pd.Timestamp("2024-01-15"),
            "hours_raw": 77,
            "base_cost": 35,
            "rev_alloc": 770,
            "quoted_time_total": 100,
            "quoted_amount_total": 1000,
            "quote_match_flag": "matched",
            "job_status": "Active",
        },
        {
            "job_no": "SC1",
            "department_final": "DeptA",
            "category_rev_job": "Cat1",
            "task_name": "TaskB",
            "work_date": pd.Timestamp("2024-01-15"),
            "hours_raw": 33,
            "base_cost": 15,
            "rev_alloc": 330,
            "quoted_time_total": 0,
            "quoted_amount_total": 0,
            "quote_match_flag": "unmatched",
            "job_status": "Active",
        },
        {
            "job_no": "C1",
            "department_final": "DeptA",
            "category_rev_job": "Cat1",
            "task_name": "TaskA",
            "work_date": pd.Timestamp("2023-10-01"),
            "hours_raw": 50,
            "base_cost": 20,
            "rev_alloc": 100,
            "quoted_time_total": 60,
            "quoted_amount_total": 120,
            "quote_match_flag": "matched",
            "job_status": "Completed",
            "job_completed_date": pd.Timestamp("2023-12-15"),
        },
        {
            "job_no": "C1",
            "department_final": "DeptA",
            "category_rev_job": "Cat1",
            "task_name": "TaskA",
            "work_date": pd.Timestamp("2023-12-15"),
            "hours_raw": 10,
            "base_cost": 5,
            "rev_alloc": 20,
            "quoted_time_total": 60,
            "quoted_amount_total": 120,
            "quote_match_flag": "matched",
            "job_status": "Completed",
            "job_completed_date": pd.Timestamp("2023-12-15"),
        },
    ])

    jobs_df = compute_delivery_control_view(df, recency_days=28)
    job_row = jobs_df[jobs_df["job_no"] == "SC1"].iloc[0]
    benchmarks = compute_benchmarks(df)

    drivers = compute_root_cause_drivers(df, job_row, benchmarks)

    assert len(drivers) == 1
    assert drivers[0]["driver_name"] == "Scope Creep"
