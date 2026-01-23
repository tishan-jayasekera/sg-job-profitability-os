"""
Schema validation and column alias mapping.
"""
import pandas as pd
import streamlit as st
from typing import List, Tuple, Dict, Optional

from src.config import REQUIRED_COLUMNS, OPTIONAL_COLUMNS


class SchemaValidationError(Exception):
    """Raised when required columns are missing."""
    pass


def validate_required_columns(df: pd.DataFrame, table_name: str) -> Tuple[bool, List[str]]:
    """
    Validate that required columns exist in dataframe.
    Returns (is_valid, missing_columns).
    """
    if table_name not in REQUIRED_COLUMNS:
        return True, []
    
    required = REQUIRED_COLUMNS[table_name]
    missing = [col for col in required if col not in df.columns]
    
    return len(missing) == 0, missing


def check_optional_columns(df: pd.DataFrame, table_name: str) -> List[str]:
    """
    Check which optional columns are missing.
    Returns list of missing optional columns.
    """
    if table_name not in OPTIONAL_COLUMNS:
        return []
    
    optional = OPTIONAL_COLUMNS[table_name]
    missing = [col for col in optional if col not in df.columns]
    
    return missing


def validate_schema(df: pd.DataFrame, table_name: str, strict: bool = True) -> Dict:
    """
    Full schema validation.
    
    Args:
        df: DataFrame to validate
        table_name: Name of table for column requirements lookup
        strict: If True, raise error on missing required columns
        
    Returns:
        Dict with validation results
    """
    is_valid, missing_required = validate_required_columns(df, table_name)
    missing_optional = check_optional_columns(df, table_name)
    
    result = {
        "is_valid": is_valid,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "total_columns": len(df.columns),
        "total_rows": len(df),
    }
    
    if strict and not is_valid:
        raise SchemaValidationError(
            f"Missing required columns in {table_name}: {missing_required}"
        )
    
    return result


def display_validation_result(result: Dict, table_name: str):
    """Display validation result in Streamlit."""
    if result["is_valid"]:
        st.success(f"{table_name}: Schema valid ({result['total_rows']:,} rows, {result['total_columns']} columns)")
    else:
        st.error(f"{table_name}: Missing required columns: {result['missing_required']}")
    
    if result["missing_optional"]:
        st.warning(f"{table_name}: Missing optional columns (will degrade gracefully): {result['missing_optional']}")


def ensure_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column types."""
    df = df.copy()
    
    # Numeric columns
    numeric_cols = [
        "hours_raw", "base_cost", "rev_alloc", "base_rate", "billable_rate",
        "quoted_time_total", "quoted_amount_total", "quote_rate",
        "fte_hours_scaling", "realised_rate_alloc"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Boolean columns
    bool_cols = ["is_billable", "has_revenue_record"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # Date columns
    date_cols = ["month_key", "work_date", "job_start_date", "job_due_date", "job_completed_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    return df


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary info about all columns."""
    info = []
    for col in df.columns:
        info.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": df[col].notna().sum(),
            "null_pct": f"{df[col].isna().mean()*100:.1f}%",
            "unique": df[col].nunique(),
        })
    return pd.DataFrame(info)
