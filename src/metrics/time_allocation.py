"""
Time allocation analysis - descriptive, not prescriptive.
"""
from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd

from src.data.semantic import leave_exclusion_mask


def compute_allocation_breakdown(df: pd.DataFrame,
                                 group_by: str = "staff_name") -> pd.DataFrame:
    """
    Compute time allocation breakdown.
    
    Excludes leave tasks.
    
    Returns DataFrame with:
        {group_by},
        total_hours, billable_hours, nonbillable_hours,
        billable_ratio,
        nonbillable_by_breakdown (dict: breakdown → hours)
    """
    if group_by not in df.columns or "hours_raw" not in df.columns:
        return pd.DataFrame()
    
    df_clean = df[~leave_exclusion_mask(df)].copy()
    if "is_billable" not in df_clean.columns:
        df_clean["is_billable"] = False
    
    df_clean["billable_hours"] = np.where(df_clean["is_billable"], df_clean["hours_raw"], 0)
    df_clean["nonbillable_hours"] = np.where(~df_clean["is_billable"], df_clean["hours_raw"], 0)
    
    agg = df_clean.groupby(group_by).agg(
        total_hours=("hours_raw", "sum"),
        billable_hours=("billable_hours", "sum"),
        nonbillable_hours=("nonbillable_hours", "sum"),
    ).reset_index()
    
    agg["billable_ratio"] = np.where(
        agg["total_hours"] > 0,
        agg["billable_hours"] / agg["total_hours"],
        0
    )
    
    if "breakdown" in df_clean.columns:
        nb = df_clean[~df_clean["is_billable"]].groupby([group_by, "breakdown"])["hours_raw"].sum().reset_index()
        nb = nb.dropna(subset=["breakdown"])
        
        rows = []
        for key, sub in nb.groupby(group_by):
            rows.append({
                group_by: key,
                "nonbillable_by_breakdown": dict(zip(sub["breakdown"], sub["hours_raw"]))
            })
        breakdown_map = pd.DataFrame(rows)
        agg = agg.merge(breakdown_map, on=group_by, how="left")
    else:
        agg["nonbillable_by_breakdown"] = [{} for _ in range(len(agg))]
    
    agg["nonbillable_by_breakdown"] = agg["nonbillable_by_breakdown"].apply(lambda x: x if isinstance(x, dict) else {})
    
    return agg


def compute_nonbillable_detail(df: pd.DataFrame,
                               staff_name: str = None) -> pd.DataFrame:
    """
    Detailed non-billable breakdown.
    
    Returns DataFrame with:
        breakdown (or task_name), hours, share_of_nonbillable
    """
    if "hours_raw" not in df.columns:
        return pd.DataFrame()
    
    df_clean = df[~leave_exclusion_mask(df)].copy()
    if "is_billable" not in df_clean.columns:
        df_clean["is_billable"] = False
    
    df_nb = df_clean[~df_clean["is_billable"]]
    if staff_name and "staff_name" in df_nb.columns:
        df_nb = df_nb[df_nb["staff_name"] == staff_name]
    
    if "breakdown" in df_nb.columns and df_nb["breakdown"].notna().any():
        group_col = "breakdown"
    else:
        group_col = "task_name" if "task_name" in df_nb.columns else None
    
    if group_col is None:
        return pd.DataFrame()
    
    detail = df_nb.groupby(group_col)["hours_raw"].sum().reset_index()
    detail = detail.rename(columns={"hours_raw": "hours"})
    total = detail["hours"].sum()
    detail["share_of_nonbillable"] = np.where(total > 0, detail["hours"] / total, 0)
    
    return detail.sort_values("hours", ascending=False)


def compute_hhi(df: pd.DataFrame,
                group_col: str,
                value_col: str = "hours_raw") -> pd.DataFrame:
    """
    Compute Herfindahl-Hirschman Index for concentration.
    """
    if "staff_name" not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    
    totals = df.groupby("staff_name")[value_col].sum().rename("total_hours")
    shares = df.groupby(["staff_name", group_col])[value_col].sum().reset_index()
    shares = shares.merge(totals, on="staff_name", how="left")
    shares["share"] = np.where(shares["total_hours"] > 0, shares[value_col] / shares["total_hours"], 0)
    
    hhi = shares.groupby("staff_name")["share"].apply(lambda x: (x ** 2).sum()).reset_index()
    hhi.columns = ["staff_name", "hhi"]
    
    def interpret(val: float) -> str:
        if val >= 0.6:
            return "concentrated"
        if val >= 0.3:
            return "moderate"
        return "fragmented"
    
    hhi["interpretation"] = hhi["hhi"].apply(interpret)
    return hhi


def compute_crowdout_flags(allocation_df: pd.DataFrame,
                           admin_threshold: float = 0.20,
                           internal_threshold: float = 0.30,
                           unassigned_threshold: float = 0.15) -> List[Dict]:
    """
    Flag concerning time allocation patterns.
    """
    flags: List[Dict] = []
    
    for _, row in allocation_df.iterrows():
        breakdown = row.get("nonbillable_by_breakdown", {}) or {}
        nonbillable = row.get("nonbillable_hours", 0) or 0
        billable_ratio = row.get("billable_ratio", 0)
        staff_name = row.get("staff_name", "")
        
        if nonbillable > 0 and isinstance(breakdown, dict):
            for key, value in breakdown.items():
                key_l = str(key).lower()
                ratio = value / nonbillable if nonbillable > 0 else 0
                if "admin" in key_l or "meeting" in key_l:
                    if ratio > admin_threshold:
                        flags.append({
                            "staff_name": staff_name,
                            "flag": "Admin crowd-out",
                            "detail": f"{ratio:.0%} of non-billable in admin/meetings",
                            "value": ratio,
                        })
                if "internal" in key_l:
                    if ratio > internal_threshold:
                        flags.append({
                            "staff_name": staff_name,
                            "flag": "Internal crowd-out",
                            "detail": f"{ratio:.0%} of non-billable in internal projects",
                            "value": ratio,
                        })
                if "unassigned" in key_l or "misc" in key_l:
                    if ratio > unassigned_threshold:
                        flags.append({
                            "staff_name": staff_name,
                            "flag": "Unassigned crowd-out",
                            "detail": f"{ratio:.0%} of non-billable unassigned",
                            "value": ratio,
                        })
        
        if billable_ratio < 0.40 and row.get("archetype", "") != "Ops-Heavy":
            flags.append({
                "staff_name": staff_name,
                "flag": "Low billable share",
                "detail": f"{billable_ratio:.0%} billable time",
                "value": billable_ratio,
            })

        hhi = row.get("hhi")
        if pd.notna(hhi) and hhi < 0.15:
            flags.append({
                "staff_name": staff_name,
                "flag": "High fragmentation",
                "detail": f"HHI = {hhi:.2f} (fragmented)",
                "value": hhi,
            })
    
    return flags


def compute_team_allocation(df: pd.DataFrame,
                            group_by: str = "department_final") -> pd.DataFrame:
    """
    Aggregate allocation at team/department level.
    """
    if group_by not in df.columns or "hours_raw" not in df.columns:
        return pd.DataFrame()
    
    df_clean = df[~leave_exclusion_mask(df)].copy()
    if "is_billable" not in df_clean.columns:
        df_clean["is_billable"] = False
    
    df_clean["billable_hours"] = np.where(df_clean["is_billable"], df_clean["hours_raw"], 0)
    df_clean["nonbillable_hours"] = np.where(~df_clean["is_billable"], df_clean["hours_raw"], 0)
    
    agg = df_clean.groupby(group_by).agg(
        total_hours=("hours_raw", "sum"),
        billable_hours=("billable_hours", "sum"),
        nonbillable_hours=("nonbillable_hours", "sum"),
        staff_count=("staff_name", "nunique") if "staff_name" in df_clean.columns else (group_by, "count"),
    ).reset_index()
    
    agg["billable_ratio"] = np.where(
        agg["total_hours"] > 0,
        agg["billable_hours"] / agg["total_hours"],
        0
    )
    agg["avg_hours_per_staff"] = np.where(
        agg["staff_count"] > 0,
        agg["total_hours"] / agg["staff_count"],
        0
    )
    
    if "breakdown" in df_clean.columns:
        nb = df_clean[~df_clean["is_billable"]].groupby([group_by, "breakdown"])["hours_raw"].sum().reset_index()
        nb = nb.dropna(subset=["breakdown"])
        breakdown_map = (
            nb.groupby(group_by)
            .apply(lambda x: dict(zip(x["breakdown"], x["hours_raw"])))
            .rename("nonbillable_breakdown")
            .reset_index()
        )
        agg = agg.merge(breakdown_map, on=group_by, how="left")
    else:
        agg["nonbillable_breakdown"] = [{} for _ in range(len(agg))]
    
    return agg
