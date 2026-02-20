"""
Data loading utilities with Streamlit caching.
"""
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, Sequence

from src.config import config, TABLE_FILES, MART_FILES


def _normalise_column_selection(columns: Optional[Sequence[str]]) -> Optional[list[str]]:
    """Deduplicate and normalise a requested column list."""
    if not columns:
        return None
    return list(dict.fromkeys(str(col) for col in columns))


def _load_file(filepath: Path, columns: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
    """Load a single file (parquet or csv), optionally selecting columns."""
    selected_cols = _normalise_column_selection(columns)
    parquet_path = filepath.with_suffix(".parquet")
    csv_path = filepath.with_suffix(".csv")

    if parquet_path.exists():
        if selected_cols:
            try:
                return pd.read_parquet(parquet_path, columns=selected_cols)
            except Exception:
                df = pd.read_parquet(parquet_path)
                keep_cols = [col for col in selected_cols if col in df.columns]
                return df[keep_cols]
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        if selected_cols:
            try:
                return pd.read_csv(csv_path, usecols=selected_cols)
            except ValueError:
                df = pd.read_csv(csv_path)
                keep_cols = [col for col in selected_cols if col in df.columns]
                return df[keep_cols]
        return pd.read_csv(csv_path, parse_dates=True)
    return None


@st.cache_data(ttl=config.cache_ttl_seconds)
def load_fact_timesheet(columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load the main fact_timesheet_day_enriched table."""
    filepath = config.processed_dir / TABLE_FILES["fact_timesheet"]
    df = _load_file(filepath, columns=columns)
    if df is None:
        st.error(f"Could not find fact_timesheet_day_enriched in {config.processed_dir}")
        st.stop()
    
    # Ensure date columns are datetime
    date_cols = ["month_key", "work_date", "job_start_date", "job_due_date", "job_completed_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    return df


@st.cache_data(ttl=config.cache_ttl_seconds)
def load_fact_job_task_month() -> pd.DataFrame:
    """Load fact_job_task_month table."""
    filepath = config.processed_dir / TABLE_FILES["fact_job_task_month"]
    df = _load_file(filepath)
    if df is None:
        st.error(f"Could not find fact_job_task_month in {config.processed_dir}")
        st.stop()
    
    if "month_key" in df.columns:
        df["month_key"] = pd.to_datetime(df["month_key"], errors="coerce")
    
    return df


@st.cache_data(ttl=config.cache_ttl_seconds)
def load_audit_reconciliation() -> pd.DataFrame:
    """Load revenue reconciliation audit table."""
    filepath = config.processed_dir / TABLE_FILES["audit_reconciliation"]
    df = _load_file(filepath)
    if df is None:
        return pd.DataFrame()
    
    if "month_key" in df.columns:
        df["month_key"] = pd.to_datetime(df["month_key"], errors="coerce")
    
    return df


@st.cache_data(ttl=config.cache_ttl_seconds)
def load_audit_unallocated() -> pd.DataFrame:
    """Load unallocated revenue audit table."""
    filepath = config.processed_dir / TABLE_FILES["audit_unallocated"]
    df = _load_file(filepath)
    if df is None:
        return pd.DataFrame()
    
    if "month_key" in df.columns:
        df["month_key"] = pd.to_datetime(df["month_key"], errors="coerce")
    
    return df


@st.cache_data(ttl=config.cache_ttl_seconds)
def load_mart(mart_name: str) -> Optional[pd.DataFrame]:
    """Load a precomputed mart if it exists."""
    if mart_name not in MART_FILES:
        return None
    
    filepath = config.marts_dir / MART_FILES[mart_name]
    df = _load_file(filepath)
    
    if df is not None and "month_key" in df.columns:
        df["month_key"] = pd.to_datetime(df["month_key"], errors="coerce")
    
    return df


def get_data_status() -> Dict[str, Any]:
    """Get status of all data files."""
    status = {
        "processed": {},
        "marts": {},
    }
    
    for key, filename in TABLE_FILES.items():
        parquet_path = config.processed_dir / f"{filename}.parquet"
        csv_path = config.processed_dir / f"{filename}.csv"
        status["processed"][key] = {
            "parquet_exists": parquet_path.exists(),
            "csv_exists": csv_path.exists(),
        }
    
    for key, filename in MART_FILES.items():
        parquet_path = config.marts_dir / f"{filename}.parquet"
        csv_path = config.marts_dir / f"{filename}.csv"
        status["marts"][key] = {
            "parquet_exists": parquet_path.exists(),
            "csv_exists": csv_path.exists(),
        }
    
    return status
