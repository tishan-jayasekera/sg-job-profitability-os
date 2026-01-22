"""
Margin bridge metrics pack.

Decomposes margin variance into:
1. Hours variance effect
2. Rate variance effect  
3. Cost variance/mix drift effect
4. Non-billable leakage effect
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

from src.data.semantic import safe_quote_rollup


def compute_margin_bridge(df: pd.DataFrame,
                          group_keys: Optional[List[str]] = None,
                          baseline_cost_per_hour: Optional[float] = None) -> pd.DataFrame:
    """
    Compute margin bridge decomposition.
    
    Decomposes variance from expected margin (based on quote) to actual margin.
    
    Returns DataFrame with:
    - quoted baseline metrics
    - actual metrics
    - variance decomposition components
    """
    # Safe quote totals
    quote = safe_quote_rollup(df, group_keys if group_keys else [])
    
    # Actual profitability
    if group_keys:
        actuals = df.groupby(group_keys).agg(
            actual_hours=("hours_raw", "sum"),
            actual_cost=("base_cost", "sum"),
            actual_revenue=("rev_alloc", "sum"),
            billable_hours=("hours_raw", lambda x: x[df.loc[x.index, "is_billable"]].sum() if "is_billable" in df.columns else x.sum()),
        ).reset_index()
    else:
        actuals = pd.DataFrame([{
            "actual_hours": df["hours_raw"].sum(),
            "actual_cost": df["base_cost"].sum(),
            "actual_revenue": df["rev_alloc"].sum(),
            "billable_hours": df[df["is_billable"]]["hours_raw"].sum() if "is_billable" in df.columns else df["hours_raw"].sum(),
        }])
    
    # Merge
    if group_keys:
        result = quote.merge(actuals, on=group_keys, how="outer")
    else:
        result = quote.copy()
        for col in actuals.columns:
            result[col] = actuals[col].iloc[0]
    
    # Fill missing
    result["quoted_hours"] = result["quoted_hours"].fillna(0)
    result["quoted_amount"] = result["quoted_amount"].fillna(0)
    result["actual_hours"] = result["actual_hours"].fillna(0)
    result["actual_cost"] = result["actual_cost"].fillna(0)
    result["actual_revenue"] = result["actual_revenue"].fillna(0)
    result["billable_hours"] = result["billable_hours"].fillna(0)
    
    # Compute baseline cost/hr if not provided
    if baseline_cost_per_hour is None:
        # Use actual cost/hr as baseline
        total_cost = result["actual_cost"].sum()
        total_hours = result["actual_hours"].sum()
        baseline_cost_per_hour = total_cost / total_hours if total_hours > 0 else 0
    
    result["baseline_cost_per_hour"] = baseline_cost_per_hour
    
    # Expected metrics (from quote)
    result["expected_revenue"] = result["quoted_amount"]
    result["expected_cost"] = result["quoted_hours"] * result["baseline_cost_per_hour"]
    result["expected_margin"] = result["expected_revenue"] - result["expected_cost"]
    
    # Actual margin
    result["actual_margin"] = result["actual_revenue"] - result["actual_cost"]
    
    # Total variance
    result["margin_variance"] = result["actual_margin"] - result["expected_margin"]
    
    # DECOMPOSITION
    
    # 1. Hours variance effect
    # Impact of hours being different from quoted, at baseline cost
    result["hours_variance"] = result["actual_hours"] - result["quoted_hours"]
    result["hours_variance_effect"] = -result["hours_variance"] * result["baseline_cost_per_hour"]
    
    # 2. Rate variance effect
    # Impact of realised rate being different from quote rate
    result["quote_rate"] = np.where(
        result["quoted_hours"] > 0,
        result["quoted_amount"] / result["quoted_hours"],
        0
    )
    result["realised_rate"] = np.where(
        result["actual_hours"] > 0,
        result["actual_revenue"] / result["actual_hours"],
        0
    )
    result["rate_variance_effect"] = (result["realised_rate"] - result["quote_rate"]) * result["actual_hours"]
    
    # 3. Cost variance effect (mix drift)
    # Impact of actual cost/hr being different from baseline
    result["actual_cost_per_hour"] = np.where(
        result["actual_hours"] > 0,
        result["actual_cost"] / result["actual_hours"],
        0
    )
    result["cost_variance_effect"] = -(result["actual_cost_per_hour"] - result["baseline_cost_per_hour"]) * result["actual_hours"]
    
    # 4. Non-billable leakage effect
    # Cost of hours spent on non-billable work
    result["non_billable_hours"] = result["actual_hours"] - result["billable_hours"]
    result["non_billable_leakage"] = -result["non_billable_hours"] * result["baseline_cost_per_hour"]
    
    # Residual (should be small)
    result["residual"] = (
        result["margin_variance"] - 
        result["hours_variance_effect"] - 
        result["rate_variance_effect"] - 
        result["cost_variance_effect"]
    )
    
    return result


def get_margin_bridge_waterfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get margin bridge data formatted for waterfall chart.
    
    Returns DataFrame with labels, values, and measure types.
    """
    bridge = compute_margin_bridge(df)
    
    if len(bridge) == 0:
        return pd.DataFrame()
    
    row = bridge.iloc[0]
    
    waterfall_data = [
        {"label": "Expected Margin", "value": row["expected_margin"], "measure": "absolute"},
        {"label": "Hours Variance", "value": row["hours_variance_effect"], "measure": "relative"},
        {"label": "Rate Variance", "value": row["rate_variance_effect"], "measure": "relative"},
        {"label": "Cost Mix", "value": row["cost_variance_effect"], "measure": "relative"},
        {"label": "Residual", "value": row["residual"], "measure": "relative"},
        {"label": "Actual Margin", "value": row["actual_margin"], "measure": "total"},
    ]
    
    return pd.DataFrame(waterfall_data)


def get_margin_bridge_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get margin bridge summary as a dictionary.
    """
    bridge = compute_margin_bridge(df)
    
    if len(bridge) == 0:
        return {}
    
    row = bridge.iloc[0]
    
    return {
        "expected_revenue": row["expected_revenue"],
        "expected_cost": row["expected_cost"],
        "expected_margin": row["expected_margin"],
        "actual_revenue": row["actual_revenue"],
        "actual_cost": row["actual_cost"],
        "actual_margin": row["actual_margin"],
        "margin_variance": row["margin_variance"],
        "hours_variance_effect": row["hours_variance_effect"],
        "rate_variance_effect": row["rate_variance_effect"],
        "cost_variance_effect": row["cost_variance_effect"],
        "non_billable_leakage": row["non_billable_leakage"],
    }


def compute_margin_drivers(df: pd.DataFrame, 
                           group_key: str,
                           n: int = 10) -> pd.DataFrame:
    """
    Identify top drivers of margin variance.
    
    Returns groups sorted by contribution to margin variance.
    """
    bridge = compute_margin_bridge(df, [group_key])
    
    # Sort by absolute variance
    bridge["abs_variance"] = bridge["margin_variance"].abs()
    
    return bridge.nlargest(n, "abs_variance")
