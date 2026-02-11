"""
Rate capture metrics pack.

Single source of truth for: quote rate, realised rate, rate variance.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, List, Dict

from src.data.semantic import safe_quote_rollup
from src.data.semantic import quote_delivery_metrics, utilisation_metrics
from src.data.semantic import get_category_col


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_rate_metrics(df: pd.DataFrame,
                         group_keys: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
    """
    Compute rate capture metrics.
    
    Returns DataFrame with:
    - quote_rate: quoted_amount / quoted_hours (safe rollup)
    - realised_rate: revenue / hours
    - rate_variance: realised_rate - quote_rate
    - hours, revenue, quoted_hours, quoted_amount
    """
    keys = list(group_keys) if group_keys else []

    # Safe quote rollup
    quote = safe_quote_rollup(df, tuple(keys))
    
    # Profitability metrics for realised rate
    if keys:
        actuals = df.groupby(keys).agg(
            hours=("hours_raw", "sum"),
            revenue=("rev_alloc", "sum"),
        ).reset_index()
    else:
        actuals = pd.DataFrame([{
            "hours": df["hours_raw"].sum(),
            "revenue": df["rev_alloc"].sum(),
        }])
    
    # Merge
    if keys:
        result = actuals.merge(
            quote[keys + ["quoted_hours", "quoted_amount", "quote_rate"]],
            on=keys,
            how="outer"
        )
    else:
        result = actuals
        result["quoted_hours"] = quote["quoted_hours"].iloc[0] if len(quote) > 0 else 0
        result["quoted_amount"] = quote["quoted_amount"].iloc[0] if len(quote) > 0 else 0
        result["quote_rate"] = quote["quote_rate"].iloc[0] if len(quote) > 0 else np.nan
    
    # Realised rate
    result["realised_rate"] = np.where(
        result["hours"] > 0,
        result["revenue"] / result["hours"],
        np.nan
    )
    
    # Rate variance
    result["rate_variance"] = result["realised_rate"] - result["quote_rate"]
    
    # Rate capture % (realised as % of quote)
    result["rate_capture_pct"] = np.where(
        result["quote_rate"] > 0,
        result["realised_rate"] / result["quote_rate"] * 100,
        np.nan
    )
    
    return result


@st.cache_data(show_spinner=False)
def compute_weighted_rates(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute hours-weighted average rates for summary display.
    
    Returns dict with:
    - quote_rate_wtd: hours-weighted quote rate
    - realised_rate_wtd: hours-weighted realised rate
    - rate_variance_wtd: weighted variance
    """
    rates = compute_rate_metrics(df)
    
    return {
        "quote_rate_wtd": rates["quote_rate"].iloc[0] if len(rates) > 0 else np.nan,
        "realised_rate_wtd": rates["realised_rate"].iloc[0] if len(rates) > 0 else np.nan,
        "rate_variance_wtd": rates["rate_variance"].iloc[0] if len(rates) > 0 else np.nan,
        "rate_capture_pct": rates["rate_capture_pct"].iloc[0] if len(rates) > 0 else np.nan,
    }


@st.cache_data(show_spinner=False)
def compute_rate_distribution(df: pd.DataFrame,
                              group_key: str = "job_no") -> pd.DataFrame:
    """
    Compute rate distribution at a given grain.
    
    Returns stats for both quote rate and realised rate.
    """
    rates = compute_rate_metrics(df, (group_key,))
    
    # Filter to valid rates
    valid_quote = rates[rates["quote_rate"].notna()]
    valid_realised = rates[rates["realised_rate"].notna()]
    
    result = {
        "n_quote": len(valid_quote),
        "quote_rate_mean": valid_quote["quote_rate"].mean(),
        "quote_rate_median": valid_quote["quote_rate"].median(),
        "quote_rate_std": valid_quote["quote_rate"].std(),
        "quote_rate_p25": valid_quote["quote_rate"].quantile(0.25),
        "quote_rate_p75": valid_quote["quote_rate"].quantile(0.75),
        
        "n_realised": len(valid_realised),
        "realised_rate_mean": valid_realised["realised_rate"].mean(),
        "realised_rate_median": valid_realised["realised_rate"].median(),
        "realised_rate_std": valid_realised["realised_rate"].std(),
        "realised_rate_p25": valid_realised["realised_rate"].quantile(0.25),
        "realised_rate_p75": valid_realised["realised_rate"].quantile(0.75),
    }
    
    return pd.DataFrame([result])


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_rate_leakage(df: pd.DataFrame,
                         group_keys: Optional[tuple[str, ...]] = None,
                         threshold: float = -10) -> pd.DataFrame:
    """
    Identify rate leakage (negative rate variance).
    
    Args:
        threshold: Rate variance threshold below which is considered leakage
    
    Returns DataFrame with leakage analysis.
    """
    rates = compute_rate_metrics(df, group_keys)
    
    # Flag leakage
    rates["has_leakage"] = rates["rate_variance"] < threshold
    
    # Quantify leakage impact
    rates["leakage_per_hour"] = np.where(
        rates["rate_variance"] < 0,
        rates["rate_variance"],
        0
    )
    rates["total_leakage"] = rates["leakage_per_hour"] * rates["hours"]
    
    return rates


@st.cache_data(show_spinner=False)
def get_rate_capture_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get summary rate capture metrics as a dictionary.
    """
    rates = compute_rate_metrics(df)
    
    row = rates.iloc[0] if len(rates) > 0 else {}
    
    return {
        "quote_rate": row.get("quote_rate", np.nan),
        "realised_rate": row.get("realised_rate", np.nan),
        "rate_variance": row.get("rate_variance", np.nan),
        "rate_capture_pct": row.get("rate_capture_pct", np.nan),
        "hours": row.get("hours", 0),
        "revenue": row.get("revenue", 0),
        "quoted_hours": row.get("quoted_hours", 0),
        "quoted_amount": row.get("quoted_amount", 0),
    }


@st.cache_data(show_spinner=False)
def get_top_rate_leakage(df: pd.DataFrame,
                         group_key: str,
                         n: int = 10) -> pd.DataFrame:
    """
    Get top groups by rate leakage (worst rate capture).
    """
    rates = compute_rate_metrics(df, (group_key,))
    
    # Filter to negative variance
    leakage = rates[rates["rate_variance"] < 0].copy()
    leakage["total_leakage"] = leakage["rate_variance"] * leakage["hours"]
    
    return leakage.nsmallest(n, "total_leakage")


@st.cache_data(show_spinner=False)
def compute_rate_variance_diagnosis(df: pd.DataFrame,
                                    group_key: str = "job_no",
                                    min_hours: float = 5) -> pd.DataFrame:
    """
    Diagnose rate variance drivers with actionable labels.
    """
    if group_key not in df.columns:
        return pd.DataFrame()
    
    rates = compute_rate_metrics(df, (group_key,))
    delivery = quote_delivery_metrics(df, (group_key,))
    util = utilisation_metrics(df, [group_key], exclude_leave=True)
    
    result = rates.merge(delivery[[group_key, "hours_variance_pct", "unquoted_share"]], on=group_key, how="left")
    result = result.merge(util[[group_key, "utilisation"]], on=group_key, how="left")
    result["billable_ratio"] = result["utilisation"] / 100.0
    
    if "quoted_amount" not in result.columns:
        result["quoted_amount"] = 0
    if "revenue" not in result.columns:
        result["revenue"] = 0
    
    def classify(row: pd.Series) -> str:
        if pd.isna(row.get("quote_rate")) or row.get("quoted_hours", 0) == 0:
            return "No quote"
        if row.get("hours", 0) < min_hours:
            return "Low volume"
        if row.get("unquoted_share", 0) >= 20:
            return "Scope creep"
        if row.get("hours_variance_pct", 0) >= 20:
            return "Overrun hours"
        if row.get("revenue", 0) < row.get("quoted_amount", 0) * 0.9:
            return "Revenue shortfall"
        if row.get("billable_ratio", 1) < 0.6:
            return "Non-billable mix"
        if row.get("rate_variance", 0) < 0:
            return "Rate leakage"
        return "Above quote"
    
    result["driver"] = result.apply(classify, axis=1)
    
    return result
