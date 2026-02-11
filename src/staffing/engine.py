"""
Staffing engine: match staff to tasks based on demonstrated capability.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.config import config


@dataclass
class StaffingWarning:
    """Warning surfaced during staffing."""
    type: str
    task: str
    message: str


def _get_task_row(df: pd.DataFrame, staff_name: str, task_name: str) -> Optional[pd.Series]:
    if len(df) == 0:
        return None
    match = df[(df["staff_name"] == staff_name) & (df["task_name"] == task_name)]
    if len(match) == 0:
        return None
    return match.iloc[0]


def _get_category_row(df: pd.DataFrame, staff_name: str, category: str) -> Optional[pd.Series]:
    if len(df) == 0:
        return None
    match = df[(df["staff_name"] == staff_name) & (df["category_rev_job"] == category)]
    if len(match) == 0:
        return None
    return match.iloc[0]


@st.cache_data(show_spinner=False)
def check_eligibility(staff_name: str,
                      task_name: str,
                      category: str,
                      task_expertise_df: pd.DataFrame,
                      category_expertise_df: pd.DataFrame,
                      recency_months: int = 6,
                      min_hours: float = 10,
                      min_jobs: int = 2) -> Tuple[bool, str]:
    """
    Check if staff is eligible for a task.
    """
    task_row = _get_task_row(task_expertise_df, staff_name, task_name)
    if task_row is not None:
        if (task_row["months_since_last"] <= recency_months and
                task_row["hours_total"] >= min_hours and
                task_row["job_count"] >= min_jobs):
            return True, "task_match"
    
    cat_row = _get_category_row(category_expertise_df, staff_name, category)
    if cat_row is not None:
        if (cat_row["months_since_last"] <= recency_months and
                cat_row["hours_total"] >= min_hours and
                cat_row["job_count"] >= min_jobs):
            return True, "category_match"
    
    return False, "no_match"


@st.cache_data(
    show_spinner=False,
    hash_funcs={dict: lambda d: json.dumps(d, sort_keys=True, default=str)},
)
def score_staff_for_task(task_name: str,
                         category: str,
                         hours_needed: float,
                         task_expertise_df: pd.DataFrame,
                         category_expertise_df: pd.DataFrame,
                         headroom_df: pd.DataFrame,
                         eligibility_config: dict) -> pd.DataFrame:
    """
    Score all eligible staff for a task.
    """
    if len(headroom_df) > 0 and "staff_name" in headroom_df.columns:
        staff_names = sorted(headroom_df["staff_name"].unique().tolist())
    else:
        staff_names = sorted(set(task_expertise_df["staff_name"].tolist() + category_expertise_df["staff_name"].tolist()))
    rows = []
    
    for staff in staff_names:
        eligible, reason = check_eligibility(
            staff, task_name, category,
            task_expertise_df, category_expertise_df,
            recency_months=eligibility_config["recency_months"],
            min_hours=eligibility_config["min_hours"],
            min_jobs=eligibility_config["min_jobs"],
        )
        if not eligible:
            continue
        
        task_row = _get_task_row(task_expertise_df, staff, task_name)
        cat_row = _get_category_row(category_expertise_df, staff, category)
        
        used_category_fallback = task_row is None
        capability_score = 0
        hours_on_task = 0
        
        if task_row is not None and not np.isnan(task_row["capability_score"]):
            capability_score = task_row["capability_score"]
            hours_on_task = task_row["hours_total"]
        elif cat_row is not None and not np.isnan(cat_row["capability_score"]):
            capability_score = cat_row["capability_score"]
            hours_on_task = cat_row["hours_total"]
        
        headroom_row = headroom_df[headroom_df["staff_name"] == staff]
        headroom = headroom_row["headroom_hours"].iloc[0] if len(headroom_row) > 0 else 0
        
        if headroom >= hours_needed:
            availability_score = 100
        elif headroom > 0:
            availability_score = headroom / hours_needed * 100
        else:
            availability_score = 0
        
        match_score = capability_score * 0.6 + availability_score * 0.4
        
        rows.append({
            "staff_name": staff,
            "capability_score": capability_score,
            "availability_score": availability_score,
            "match_score": match_score,
            "hours_on_task": hours_on_task,
            "headroom_hours": headroom,
            "used_category_fallback": used_category_fallback or reason == "category_match",
        })
    
    if not rows:
        return pd.DataFrame()
    
    result = pd.DataFrame(rows).sort_values("match_score", ascending=False)
    return result


@st.cache_data(
    show_spinner=False,
    hash_funcs={dict: lambda d: json.dumps(d, sort_keys=True, default=str)},
)
def recommend_staff_for_plan(quote_plan: dict,
                             task_expertise_df: pd.DataFrame,
                             category_expertise_df: pd.DataFrame,
                             headroom_df: pd.DataFrame,
                             eligibility_config: dict,
                             top_n: int = 3) -> Tuple[pd.DataFrame, List[StaffingWarning]]:
    """
    Generate staffing recommendations for a quote plan.
    """
    warnings: List[StaffingWarning] = []
    recommendations = []
    
    for task in quote_plan.get("tasks", []):
        task_name = task["task_name"]
        hours_needed = task["hours"]
        category = quote_plan.get("category", "")
        
        scored = score_staff_for_task(
            task_name,
            category,
            hours_needed,
            task_expertise_df,
            category_expertise_df,
            headroom_df,
            eligibility_config,
        )
        
        if len(scored) == 0:
            warnings.append(StaffingWarning(
                type="no_coverage",
                task=task_name,
                message="No eligible staff found for this task."
            ))
            continue
        
        eligible_over_50 = scored[scored["match_score"] > 50]
        if len(eligible_over_50) == 1:
            warnings.append(StaffingWarning(
                type="single_point",
                task=task_name,
                message="Only one strong match available for this task."
            ))
        
        if scored["used_category_fallback"].any():
            warnings.append(StaffingWarning(
                type="category_fallback",
                task=task_name,
                message="Using category expertise for at least one recommendation."
            ))
        
        top = scored.head(top_n).copy()
        for idx, row in top.iterrows():
            if row["headroom_hours"] < 0:
                warnings.append(StaffingWarning(
                    type="overallocated",
                    task=task_name,
                    message=f"{row['staff_name']} is already overallocated."
                ))
        
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            recommendations.append({
                "task_name": task_name,
                "hours": hours_needed,
                "rank": rank,
                "staff_name": row["staff_name"],
                "capability_score": row["capability_score"],
                "availability_score": row["availability_score"],
                "match_score": row["match_score"],
                "hours_on_task": row["hours_on_task"],
                "headroom_hours": row["headroom_hours"],
                "used_category_fallback": row["used_category_fallback"],
            })
    
    return pd.DataFrame(recommendations), warnings


@st.cache_data(show_spinner=False)
def get_capability_coverage(category_expertise_df: pd.DataFrame,
                            min_score: float = 30) -> pd.DataFrame:
    """
    Analyze capability coverage by category.
    """
    if len(category_expertise_df) == 0:
        return pd.DataFrame()
    
    coverage = category_expertise_df[category_expertise_df["capability_score"] >= min_score]
    counts = coverage.groupby("category_rev_job")["staff_name"].nunique().reset_index()
    counts = counts.rename(columns={"staff_name": "staff_count"})
    
    def risk(count: int) -> str:
        if count <= config.COVERAGE_CRITICAL:
            return "Critical"
        if count <= config.COVERAGE_LOW:
            return "Low"
        if count < config.COVERAGE_GOOD:
            return "Moderate"
        return "Good"
    
    counts["coverage_risk"] = counts["staff_count"].apply(risk)
    return counts.sort_values("staff_count")
