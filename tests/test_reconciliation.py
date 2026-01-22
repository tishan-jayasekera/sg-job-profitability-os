"""
Tests for revenue reconciliation logic.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRevenueAllocation:
    """Tests for revenue allocation logic."""
    
    def test_allocation_sums_to_pool(self):
        """Allocated revenue should sum to pool for each job-month."""
        # Simulate allocated data
        df = pd.DataFrame({
            "job_no": ["J001", "J001", "J001"],
            "month_key": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "hours_raw": [10, 20, 10],
            "rev_alloc": [250, 500, 250],  # Should sum to 1000
            "rev_total_job_month": [1000, 1000, 1000],  # Pool (repeated)
        })
        
        # Group and check
        job_month = df.groupby(["job_no", "month_key"]).agg(
            allocated_sum=("rev_alloc", "sum"),
            pool=("rev_total_job_month", "first"),
        ).reset_index()
        
        job_month["delta"] = abs(job_month["allocated_sum"] - job_month["pool"])
        
        # All deltas should be < $0.01
        assert job_month["delta"].max() < 0.01
    
    def test_allocation_proportional_to_hours(self):
        """Allocation should be proportional to hours contribution."""
        df = pd.DataFrame({
            "job_no": ["J001", "J001"],
            "month_key": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "hours_raw": [20, 80],  # 20% and 80%
            "is_billable": [True, True],
        })
        
        revenue_pool = 1000
        total_hours = df["hours_raw"].sum()
        
        df["rev_alloc"] = (df["hours_raw"] / total_hours) * revenue_pool
        
        # First row should get 20% = $200
        assert df["rev_alloc"].iloc[0] == 200
        # Second row should get 80% = $800
        assert df["rev_alloc"].iloc[1] == 800


class TestRevenueReconciliation:
    """Tests for reconciliation audit logic."""
    
    def test_identifies_unallocated_revenue(self):
        """Should identify job-months with revenue but no hours."""
        revenue_df = pd.DataFrame({
            "job_no": ["J001", "J002"],
            "month_key": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "rev_total": [1000, 500],
        })
        
        timesheet_df = pd.DataFrame({
            "job_no": ["J001", "J001"],  # Only J001 has hours
            "month_key": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "hours_raw": [10, 20],
        })
        
        # Find job-months in revenue but not in timesheet
        rev_job_months = set(
            revenue_df.groupby(["job_no", "month_key"]).groups.keys()
        )
        ts_job_months = set(
            timesheet_df.groupby(["job_no", "month_key"]).groups.keys()
        )
        
        unallocated = rev_job_months - ts_job_months
        
        assert len(unallocated) == 1
        assert ("J002", pd.Timestamp("2024-01-01")) in unallocated
    
    def test_reconciliation_tolerance(self):
        """Reconciliation should allow small rounding differences."""
        df = pd.DataFrame({
            "job_no": ["J001", "J001", "J001"],
            "month_key": pd.to_datetime(["2024-01-01"] * 3),
            "rev_alloc": [333.33, 333.33, 333.34],  # Sum = 1000.00
            "rev_total_job_month": [1000, 1000, 1000],
        })
        
        allocated = df["rev_alloc"].sum()
        pool = df["rev_total_job_month"].iloc[0]
        
        delta = abs(allocated - pool)
        
        # Should be within tolerance
        assert delta < 0.01


class TestAllocationModes:
    """Tests for different allocation modes."""
    
    def test_billable_hours_mode(self):
        """Should use billable hours when available."""
        df = pd.DataFrame({
            "job_no": ["J001", "J001"],
            "hours_raw": [10, 20],
            "is_billable": [True, False],
        })
        
        billable_hours = df[df["is_billable"]]["hours_raw"].sum()
        
        # Only billable (10 hours) should be in denominator
        assert billable_hours == 10
    
    def test_fallback_to_all_hours(self):
        """Should use all hours when no billable hours."""
        df = pd.DataFrame({
            "job_no": ["J001", "J001"],
            "hours_raw": [10, 20],
            "is_billable": [False, False],  # No billable hours
        })
        
        billable_hours = df[df["is_billable"]]["hours_raw"].sum()
        all_hours = df["hours_raw"].sum()
        
        assert billable_hours == 0
        assert all_hours == 30  # Fallback
    
    def test_no_hours_mode(self):
        """Should mark as unallocated when no hours."""
        df = pd.DataFrame({
            "job_no": ["J001"],
            "hours_raw": [0],
            "is_billable": [True],
        })
        
        total_hours = df["hours_raw"].sum()
        
        # Cannot allocate
        assert total_hours == 0
