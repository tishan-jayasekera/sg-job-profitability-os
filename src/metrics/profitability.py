"""
Profitability metrics pack.

Single source of truth for: hours, cost, revenue, margin, margin%, realised_rate
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any


def compute_profitability(df: pd.DataFrame, 
                          group_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute core profitability metrics.
    
    Returns DataFrame with:
    - hours: Σ hours_raw
    - cost: Σ base_cost
    - revenue: Σ rev_alloc
    - margin: revenue - cost
    - margin_pct: margin / revenue * 100
    - realised_rate: revenue / hours
    """
    agg_dict = {
        "hours": ("hours_raw", "sum"),
        "cost": ("base_cost", "sum"),
        "revenue": ("rev_alloc", "sum"),
    }
    
    if group_keys:
        result = df.groupby(group_keys).agg(**agg_dict).reset_index()
    else:
        result = pd.DataFrame([{
            "hours": df["hours_raw"].sum(),
            "cost": df["base_cost"].sum(),
            "revenue": df["rev_alloc"].sum(),
        }])
    
    # Derived metrics
    result["margin"] = result["revenue"] - result["cost"]
    
    result["margin_pct"] = np.where(
        result["revenue"] != 0,
        result["margin"] / result["revenue"] * 100,
        np.nan
    )
    
    result["realised_rate"] = np.where(
        result["hours"] > 0,
        result["revenue"] / result["hours"],
        np.nan
    )
    
    return result


def compute_profitability_comparison(df: pd.DataFrame,
                                     group_keys: List[str],
                                     compare_key: str,
                                     compare_values: List[Any]) -> pd.DataFrame:
    """
    Compute profitability metrics with comparison across a dimension.
    
    Useful for period-over-period or slice comparisons.
    """
    results = []
    
    for val in compare_values:
        df_slice = df[df[compare_key] == val]
        prof = compute_profitability(df_slice, group_keys)
        prof[compare_key] = val
        results.append(prof)
    
    return pd.concat(results, ignore_index=True)


def compute_margin_contribution(df: pd.DataFrame,
                                group_key: str) -> pd.DataFrame:
    """
    Compute margin contribution by a grouping dimension.
    
    Shows what % of total margin comes from each group.
    """
    prof = compute_profitability(df, [group_key])
    
    total_margin = prof["margin"].sum()
    total_revenue = prof["revenue"].sum()
    
    prof["margin_contribution_pct"] = np.where(
        total_margin != 0,
        prof["margin"] / total_margin * 100,
        0
    )
    
    prof["revenue_share_pct"] = np.where(
        total_revenue != 0,
        prof["revenue"] / total_revenue * 100,
        0
    )
    
    return prof.sort_values("margin", ascending=False)


def compute_profitability_trend(df: pd.DataFrame,
                                group_keys: Optional[List[str]] = None,
                                time_key: str = "month_key") -> pd.DataFrame:
    """
    Compute profitability metrics over time.
    """
    if time_key not in df.columns:
        return pd.DataFrame()
    
    all_keys = [time_key]
    if group_keys:
        all_keys = group_keys + [time_key]
    
    return compute_profitability(df, all_keys).sort_values(time_key)


def compute_realised_rate_distribution(df: pd.DataFrame,
                                       group_key: str = "job_no") -> pd.DataFrame:
    """
    Compute distribution of realised rates at a given grain.
    
    Useful for understanding rate variability.
    """
    prof = compute_profitability(df, [group_key])
    
    # Filter to valid rates
    valid = prof[prof["realised_rate"].notna() & (prof["hours"] > 0)]
    
    return pd.DataFrame([{
        "n_observations": len(valid),
        "rate_mean": valid["realised_rate"].mean(),
        "rate_median": valid["realised_rate"].median(),
        "rate_std": valid["realised_rate"].std(),
        "rate_p25": valid["realised_rate"].quantile(0.25),
        "rate_p75": valid["realised_rate"].quantile(0.75),
        "rate_min": valid["realised_rate"].min(),
        "rate_max": valid["realised_rate"].max(),
    }])


def get_profitability_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get summary profitability metrics as a dictionary.
    """
    prof = compute_profitability(df)
    
    return {
        "hours": prof["hours"].iloc[0],
        "cost": prof["cost"].iloc[0],
        "revenue": prof["revenue"].iloc[0],
        "margin": prof["margin"].iloc[0],
        "margin_pct": prof["margin_pct"].iloc[0],
        "realised_rate": prof["realised_rate"].iloc[0],
    }
