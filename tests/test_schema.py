"""
Tests for schema validation.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schema import (
    validate_required_columns,
    check_optional_columns,
    validate_schema,
    SchemaValidationError
)
from src.config import REQUIRED_COLUMNS


class TestValidateRequiredColumns:
    """Tests for required column validation."""
    
    def test_all_columns_present(self):
        """All required columns present should return valid."""
        df = pd.DataFrame({
            "department_final": ["Dept A"],
            "job_category": ["Cat 1"],
            "task_name": ["Task 1"],
            "staff_name": ["Staff 1"],
            "month_key": [pd.Timestamp("2024-01-01")],
            "hours_raw": [8.0],
            "base_cost": [400.0],
            "rev_alloc": [600.0],
            "quoted_time_total": [10.0],
            "quoted_amount_total": [700.0],
            "quote_match_flag": ["matched"],
            "is_billable": [True],
            "utilisation_target": [0.8],
            "fte_hours_scaling": [1.0],
            "breakdown": ["Billable"],
        })
        
        is_valid, missing = validate_required_columns(df, "fact_timesheet")
        
        assert is_valid is True
        assert missing == []
    
    def test_missing_columns(self):
        """Missing columns should be detected."""
        df = pd.DataFrame({
            "department_final": ["Dept A"],
            "job_category": ["Cat 1"],
            # Missing other columns
        })
        
        is_valid, missing = validate_required_columns(df, "fact_timesheet")
        
        assert is_valid is False
        assert "hours_raw" in missing
        assert "rev_alloc" in missing
    
    def test_unknown_table(self):
        """Unknown table name should pass (no requirements)."""
        df = pd.DataFrame({"any_col": [1, 2, 3]})
        
        is_valid, missing = validate_required_columns(df, "unknown_table")
        
        assert is_valid is True
        assert missing == []


class TestValidateSchema:
    """Tests for full schema validation."""
    
    def test_strict_mode_raises(self):
        """Strict mode should raise on missing columns."""
        df = pd.DataFrame({
            "department_final": ["Dept A"],
        })
        
        with pytest.raises(SchemaValidationError):
            validate_schema(df, "fact_timesheet", strict=True)
    
    def test_non_strict_returns_result(self):
        """Non-strict mode should return result dict."""
        df = pd.DataFrame({
            "department_final": ["Dept A"],
        })
        
        result = validate_schema(df, "fact_timesheet", strict=False)
        
        assert result["is_valid"] is False
        assert len(result["missing_required"]) > 0
        assert result["total_rows"] == 1
        assert result["total_columns"] == 1


class TestCheckOptionalColumns:
    """Tests for optional column checking."""
    
    def test_returns_missing_optional(self):
        """Should return list of missing optional columns."""
        df = pd.DataFrame({
            "department_final": ["Dept A"],
            # Missing optional columns like job_status, client, etc.
        })
        
        missing = check_optional_columns(df, "fact_timesheet")
        
        assert "job_status" in missing
        assert "client" in missing
    
    def test_empty_for_unknown_table(self):
        """Unknown table should return empty list."""
        df = pd.DataFrame({"col": [1]})
        
        missing = check_optional_columns(df, "unknown")
        
        assert missing == []
