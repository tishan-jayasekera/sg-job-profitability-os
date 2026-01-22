"""
Tests for aggregation safety, especially quote deduplication.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.semantic import (
    safe_quote_job_task,
    safe_quote_rollup,
    profitability_rollup,
    rate_rollups,
    leave_exclusion_mask,
    exclude_leave
)


class TestSafeQuoteJobTask:
    """Tests for quote deduplication at job-task level."""
    
    def test_dedupes_repeated_quote_values(self):
        """Quote values repeated per row should be deduped."""
        # Simulate fact table with repeated quote values
        df = pd.DataFrame({
            "job_no": ["J001", "J001", "J001", "J002", "J002"],
            "task_name": ["Task A", "Task A", "Task A", "Task B", "Task B"],
            "hours_raw": [8, 4, 6, 10, 5],
            "quoted_time_total": [20, 20, 20, 15, 15],  # Repeated!
            "quoted_amount_total": [2000, 2000, 2000, 1500, 1500],  # Repeated!
            "quote_match_flag": ["matched"] * 5,
        })
        
        result = safe_quote_job_task(df)
        
        # Should have 2 rows (one per job-task)
        assert len(result) == 2
        
        # Totals should be 20 + 15 = 35, NOT 20*3 + 15*2 = 90
        assert result["quoted_time_total"].sum() == 35
        assert result["quoted_amount_total"].sum() == 3500
    
    def test_handles_null_quotes(self):
        """Should handle null quote values gracefully."""
        df = pd.DataFrame({
            "job_no": ["J001", "J002"],
            "task_name": ["Task A", "Task B"],
            "quoted_time_total": [10, np.nan],
            "quoted_amount_total": [1000, np.nan],
        })
        
        result = safe_quote_job_task(df)
        
        assert len(result) == 2
        assert result["quoted_time_total"].sum() == 10  # nan treated as 0 in sum


class TestSafeQuoteRollup:
    """Tests for safe quote rollups with grouping."""
    
    def test_rollup_by_department(self):
        """Rollup by department should dedupe first, then aggregate."""
        df = pd.DataFrame({
            "job_no": ["J001", "J001", "J002", "J002"],
            "task_name": ["Task A", "Task A", "Task B", "Task B"],
            "department_final": ["Dept X", "Dept X", "Dept X", "Dept X"],
            "quoted_time_total": [10, 10, 20, 20],
            "quoted_amount_total": [1000, 1000, 2000, 2000],
        })
        
        result = safe_quote_rollup(df, ["department_final"])
        
        # Should have 1 row for Dept X
        assert len(result) == 1
        
        # Totals: 10 + 20 = 30 hours, NOT 40
        assert result["quoted_hours"].iloc[0] == 30
        assert result["quoted_amount"].iloc[0] == 3000
    
    def test_computes_quote_rate(self):
        """Should compute quote rate = amount / hours."""
        df = pd.DataFrame({
            "job_no": ["J001"],
            "task_name": ["Task A"],
            "department_final": ["Dept X"],
            "quoted_time_total": [10],
            "quoted_amount_total": [1500],
        })
        
        result = safe_quote_rollup(df, ["department_final"])
        
        assert result["quote_rate"].iloc[0] == 150  # 1500 / 10


class TestProfitabilityRollup:
    """Tests for profitability calculations."""
    
    def test_basic_profitability(self):
        """Should calculate margin correctly."""
        df = pd.DataFrame({
            "hours_raw": [10, 20],
            "base_cost": [500, 1000],
            "rev_alloc": [800, 1500],
        })
        
        result = profitability_rollup(df)
        
        assert result["hours"].iloc[0] == 30
        assert result["cost"].iloc[0] == 1500
        assert result["revenue"].iloc[0] == 2300
        assert result["margin"].iloc[0] == 800
        assert abs(result["margin_pct"].iloc[0] - 34.78) < 0.1
    
    def test_realised_rate(self):
        """Should calculate realised rate correctly."""
        df = pd.DataFrame({
            "hours_raw": [20],
            "base_cost": [1000],
            "rev_alloc": [3000],
        })
        
        result = profitability_rollup(df)
        
        assert result["realised_rate"].iloc[0] == 150  # 3000 / 20


class TestLeaveExclusion:
    """Tests for leave task exclusion."""
    
    def test_excludes_leave_tasks(self):
        """Should exclude tasks containing 'leave'."""
        df = pd.DataFrame({
            "task_name": ["Project Work", "Annual Leave", "sick leave", "Meeting"],
            "hours_raw": [10, 8, 4, 2],
        })
        
        mask = leave_exclusion_mask(df)
        
        assert mask.sum() == 2  # Two leave tasks
        assert mask.iloc[1] is True  # "Annual Leave"
        assert mask.iloc[2] is True  # "sick leave" (case insensitive)
    
    def test_exclude_leave_function(self):
        """Should return filtered dataframe."""
        df = pd.DataFrame({
            "task_name": ["Project Work", "Annual Leave", "Meeting"],
            "hours_raw": [10, 8, 2],
        })
        
        result = exclude_leave(df)
        
        assert len(result) == 2
        assert result["hours_raw"].sum() == 12


class TestRateRollups:
    """Tests for rate calculations with quote safety."""
    
    def test_both_rates_computed(self):
        """Should compute both realised and quote rates safely."""
        df = pd.DataFrame({
            "job_no": ["J001", "J001"],
            "task_name": ["Task A", "Task A"],
            "department_final": ["Dept X", "Dept X"],
            "hours_raw": [10, 10],
            "base_cost": [500, 500],
            "rev_alloc": [800, 800],
            "quoted_time_total": [25, 25],  # Repeated
            "quoted_amount_total": [3000, 3000],  # Repeated
        })
        
        result = rate_rollups(df, ["department_final"])
        
        # Realised rate = 1600 / 20 = 80
        assert result["realised_rate"].iloc[0] == 80
        
        # Quote rate = 3000 / 25 = 120 (deduped!)
        assert result["quote_rate"].iloc[0] == 120
        
        # Rate variance = 80 - 120 = -40
        assert result["rate_variance"].iloc[0] == -40
