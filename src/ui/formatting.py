"""
Consistent number and display formatting.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Any

try:
    from pandas.io.formats.style import Styler
except Exception:
    Styler = Any  # type: ignore[misc]


# =============================================================================
# NUMBER FORMATTERS
# =============================================================================

def fmt_currency(value: Union[float, int, None], decimals: int = 0) -> str:
    """Format as currency: $1,234 or $1,234.56"""
    if pd.isna(value) or value is None:
        return "—"
    if decimals == 0:
        return f"${value:,.0f}"
    return f"${value:,.{decimals}f}"


def fmt_hours(value: Union[float, int, None]) -> str:
    """Format hours: 1,234.5"""
    if pd.isna(value) or value is None:
        return "—"
    return f"{value:,.1f}"


def fmt_rate(value: Union[float, int, None]) -> str:
    """Format hourly rate: $123/hr"""
    if pd.isna(value) or value is None:
        return "—"
    return f"${value:,.0f}/hr"


def fmt_percent(value: Union[float, int, None], decimals: int = 1) -> str:
    """Format percentage: 12.3%"""
    if pd.isna(value) or value is None:
        return "—"
    return f"{value:,.{decimals}f}%"


def fmt_count(value: Union[float, int, None]) -> str:
    """Format count: 1,234"""
    if pd.isna(value) or value is None:
        return "—"
    return f"{int(value):,}"


def fmt_variance(value: Union[float, int, None], is_percent: bool = False) -> str:
    """Format variance with +/- sign."""
    if pd.isna(value) or value is None:
        return "—"
    
    sign = "+" if value > 0 else ""
    if is_percent:
        return f"{sign}{value:,.1f}%"
    return f"{sign}${value:,.0f}"


# =============================================================================
# DELTA INDICATORS
# =============================================================================

def delta_indicator(value: float, threshold: float = 0, invert: bool = False) -> str:
    """
    Return delta indicator symbol.
    
    Args:
        value: The value to check
        threshold: The threshold for neutral (default 0)
        invert: If True, positive is bad (red), negative is good (green)
    """
    if pd.isna(value):
        return "●"  # Gray dot
    
    if invert:
        if value > threshold:
            return "●"  # Red (bad)
        elif value < threshold:
            return "●"  # Green (good)
        return "●"  # Gray
    else:
        if value > threshold:
            return "●"  # Green (good)
        elif value < threshold:
            return "●"  # Red (bad)
        return "●"  # Gray


def status_dot(status: str) -> str:
    """
    Return colored status dot HTML.
    
    Args:
        status: 'good', 'warning', 'bad', or custom status
    """
    colors = {
        "good": "#28a745",
        "on_track": "#28a745",
        "warning": "#ffc107",
        "watch": "#ffc107",
        "bad": "#dc3545",
        "at_risk": "#dc3545",
        "neutral": "#6c757d",
    }
    
    color = colors.get(status.lower(), colors["neutral"])
    return f'<span style="color: {color}; font-size: 1.2em;">●</span>'


# =============================================================================
# DATAFRAME FORMATTERS
# =============================================================================

def format_metric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format a metrics dataframe for display.
    
    Applies appropriate formatting to known column types.
    """
    df = df.copy()
    
    # Currency columns
    currency_cols = [
        "revenue", "cost", "margin", "base_cost", "rev_alloc",
        "quoted_amount", "actual_cost", "actual_revenue",
        "estimated_cost", "estimated_value", "estimated_margin",
        "total_quoted_amount", "avg_quoted_amount"
    ]
    
    # Hours columns
    hours_cols = [
        "hours", "hours_raw", "actual_hours", "quoted_hours",
        "billable_hours", "total_hours", "unquoted_hours",
        "hours_variance", "remaining_hours", "hours_weighted",
        "total_quoted_hours", "avg_quoted_hours"
    ]
    
    # Rate columns
    rate_cols = [
        "realised_rate", "quote_rate", "rate_variance",
        "cost_per_hour", "cost_per_hour_median", "value_per_quoted_hour"
    ]
    
    # Percent columns
    percent_cols = [
        "margin_pct", "utilisation",
        "hours_variance_pct", "unquoted_share", "scope_creep_pct",
        "inclusion_rate", "overrun_rate", "severe_overrun_rate",
        "pct_quote_consumed"
    ]
    
    # Count columns
    count_cols = [
        "job_count", "job_task_count", "staff_count", "row_count",
        "jobs_with_task", "active_job_count", "n_jobs", "n_staff"
    ]
    
    for col in df.columns:
        if col in currency_cols:
            df[col] = df[col].apply(fmt_currency)
        elif col in hours_cols:
            df[col] = df[col].apply(fmt_hours)
        elif col in rate_cols:
            df[col] = df[col].apply(fmt_rate)
        elif col in percent_cols:
            df[col] = df[col].apply(fmt_percent)
        elif col in count_cols:
            df[col] = df[col].apply(fmt_count)
    
    return df


def style_metric_df(df: pd.DataFrame) -> "Styler":
    """
    Apply conditional styling to metrics dataframe.
    """
    def color_margin(val):
        if pd.isna(val):
            return ""
        try:
            v = float(str(val).replace("$", "").replace(",", "").replace("%", ""))
            if v < 0:
                return "color: #dc3545"
            elif v > 30:
                return "color: #28a745"
            return ""
        except:
            return ""
    
    def color_variance(val):
        if pd.isna(val):
            return ""
        try:
            v = float(str(val).replace("$", "").replace(",", "").replace("%", "").replace("+", ""))
            if v > 20:
                return "color: #dc3545"  # Overrun is bad
            elif v < 0:
                return "color: #28a745"  # Under is good
            return ""
        except:
            return ""
    
    styled = df.style
    
    # Apply column-specific styling
    if "margin_pct" in df.columns:
        styled = styled.applymap(color_margin, subset=["margin_pct"])
    
    if "hours_variance_pct" in df.columns:
        styled = styled.applymap(color_variance, subset=["hours_variance_pct"])
    
    return styled


# =============================================================================
# KPI CARD HELPERS
# =============================================================================

def kpi_value(value: Union[float, int, None], format_type: str = "currency") -> str:
    """
    Format a KPI value for card display.
    
    Args:
        value: The value to format
        format_type: One of 'currency', 'hours', 'rate', 'percent', 'count'
    """
    if format_type == "currency":
        return fmt_currency(value)
    elif format_type == "hours":
        return fmt_hours(value)
    elif format_type == "rate":
        return fmt_rate(value)
    elif format_type == "percent":
        return fmt_percent(value)
    elif format_type == "count":
        return fmt_count(value)
    else:
        return str(value) if value is not None else "—"


def kpi_delta(current: float, previous: float, format_type: str = "currency") -> tuple:
    """
    Calculate and format KPI delta.
    
    Returns (delta_str, is_positive)
    """
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return "—", None
    
    delta = current - previous
    delta_pct = delta / abs(previous) * 100
    
    sign = "+" if delta > 0 else ""
    
    if format_type == "currency":
        delta_str = f"{sign}{fmt_currency(delta)} ({sign}{delta_pct:.1f}%)"
    elif format_type == "percent":
        delta_str = f"{sign}{delta:.1f}pp"
    else:
        delta_str = f"{sign}{delta:,.1f} ({sign}{delta_pct:.1f}%)"
    
    return delta_str, delta > 0
