"""
Utilisation & Leakage metrics pack.

Single source of truth for: utilisation, target, util gap, non-billable breakdown.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

from src.data.semantic import leave_exclusion_mask


def compute_utilisation(df: pd.DataFrame,
                        group_keys: Optional[List[str]] = None,
                        exclude_leave: bool = True) -> pd.DataFrame:
    """
    Compute utilisation metrics.
    
    Returns DataFrame with:
    - total_hours (excluding leave by default)
    - billable_hours
    - utilisation (%)
    - utilisation_target (hours-weighted)
    - util_gap (target - actual)
    """
    df = df.copy()
    
    # Exclude leave
    if exclude_leave:
        df = df[~leave_exclusion_mask(df)]
    
    # Ensure is_billable exists
    if "is_billable" not in df.columns:
        df["is_billable"] = True
    
    df["billable_hours"] = np.where(df["is_billable"], df["hours_raw"], 0)
    
    if group_keys:
        result = df.groupby(group_keys).agg(
            total_hours=("hours_raw", "sum"),
            billable_hours=("billable_hours", "sum"),
        ).reset_index()
        
        # Hours-weighted target
        if "utilisation_target" in df.columns:
            weighted_target = df.groupby(group_keys).apply(
                lambda x: (x["utilisation_target"] * x["hours_raw"]).sum() / x["hours_raw"].sum()
                if x["hours_raw"].sum() > 0 else 0.8
            ).reset_index()
            weighted_target.columns = group_keys + ["utilisation_target"]
            result = result.merge(weighted_target, on=group_keys, how="left")
        else:
            result["utilisation_target"] = 0.8
    else:
        total_hours = df["hours_raw"].sum()
        result = pd.DataFrame([{
            "total_hours": total_hours,
            "billable_hours": df["billable_hours"].sum(),
            "utilisation_target": (df["utilisation_target"] * df["hours_raw"]).sum() / total_hours
            if "utilisation_target" in df.columns and total_hours > 0 else 0.8,
        }])
    
    # Derived metrics
    result["utilisation"] = np.where(
        result["total_hours"] > 0,
        result["billable_hours"] / result["total_hours"] * 100,
        0
    )
    
    result["utilisation_target_pct"] = result["utilisation_target"] * 100
    result["util_gap"] = result["utilisation_target_pct"] - result["utilisation"]
    result["non_billable_hours"] = result["total_hours"] - result["billable_hours"]
    result["non_billable_pct"] = 100 - result["utilisation"]
    
    return result


def compute_leakage_breakdown(df: pd.DataFrame,
                              group_keys: Optional[List[str]] = None,
                              breakdown_col: str = "breakdown",
                              exclude_leave: bool = True) -> pd.DataFrame:
    """
    Compute non-billable hours breakdown.
    
    Returns DataFrame with breakdown categories and their hours/share.
    """
    df = df.copy()
    
    if exclude_leave:
        df = df[~leave_exclusion_mask(df)]
    
    # Filter to non-billable only
    if "is_billable" in df.columns:
        df_nb = df[~df["is_billable"]]
    else:
        df_nb = pd.DataFrame()
    
    if len(df_nb) == 0:
        return pd.DataFrame()
    
    # Use breakdown column if available
    if breakdown_col in df_nb.columns:
        category_col = breakdown_col
    elif "task_name" in df_nb.columns:
        category_col = "task_name"
    else:
        return pd.DataFrame()
    
    # Group
    all_keys = [category_col]
    if group_keys:
        all_keys = group_keys + [category_col]
    
    result = df_nb.groupby(all_keys).agg(
        hours=("hours_raw", "sum"),
        cost=("base_cost", "sum"),
    ).reset_index()
    
    # Calculate shares
    if group_keys:
        totals = result.groupby(group_keys)["hours"].sum().reset_index()
        totals.columns = group_keys + ["total_nb_hours"]
        result = result.merge(totals, on=group_keys, how="left")
    else:
        result["total_nb_hours"] = result["hours"].sum()
    
    result["share_pct"] = np.where(
        result["total_nb_hours"] > 0,
        result["hours"] / result["total_nb_hours"] * 100,
        0
    )
    
    return result.sort_values("hours", ascending=False)


def compute_staff_utilisation(df: pd.DataFrame,
                              exclude_leave: bool = True) -> pd.DataFrame:
    """
    Compute utilisation metrics per staff member.
    """
    result = compute_utilisation(df, ["staff_name"], exclude_leave)
    
    # Add department for context
    if "department_final" in df.columns:
        dept_map = df.groupby("staff_name")["department_final"].first()
        result = result.merge(
            dept_map.reset_index(),
            on="staff_name",
            how="left"
        )
    
    return result.sort_values("utilisation", ascending=True)


def get_utilisation_summary(df: pd.DataFrame,
                            exclude_leave: bool = True) -> Dict[str, float]:
    """
    Get utilisation summary as a dictionary.
    """
    util = compute_utilisation(df, exclude_leave=exclude_leave)
    
    if len(util) == 0:
        return {}
    
    row = util.iloc[0]
    
    return {
        "total_hours": row["total_hours"],
        "billable_hours": row["billable_hours"],
        "non_billable_hours": row["non_billable_hours"],
        "utilisation": row["utilisation"],
        "utilisation_target_pct": row["utilisation_target_pct"],
        "util_gap": row["util_gap"],
    }


def get_top_leakage_categories(df: pd.DataFrame,
                               n: int = 10,
                               exclude_leave: bool = True) -> pd.DataFrame:
    """
    Get top non-billable categories by hours.
    """
    breakdown = compute_leakage_breakdown(df, exclude_leave=exclude_leave)
    
    if len(breakdown) == 0:
        return pd.DataFrame()
    
    return breakdown.nlargest(n, "hours")


def get_staff_below_target(df: pd.DataFrame,
                           threshold_gap: float = 10,
                           exclude_leave: bool = True) -> pd.DataFrame:
    """
    Get staff members significantly below utilisation target.
    
    Args:
        threshold_gap: Minimum gap (in pp) below target to include
    """
    staff_util = compute_staff_utilisation(df, exclude_leave)
    
    # Filter to significant gaps
    below = staff_util[staff_util["util_gap"] > threshold_gap]
    
    return below.sort_values("util_gap", ascending=False)


def compute_utilisation_trend(df: pd.DataFrame,
                              group_keys: Optional[List[str]] = None,
                              exclude_leave: bool = True) -> pd.DataFrame:
    """
    Compute utilisation over time.
    """
    if "month_key" not in df.columns:
        return pd.DataFrame()
    
    all_keys = ["month_key"]
    if group_keys:
        all_keys = group_keys + ["month_key"]
    
    return compute_utilisation(df, all_keys, exclude_leave).sort_values("month_key")
