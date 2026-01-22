"""
Export utilities for tables, plans, and reports.
"""
import pandas as pd
import json
from typing import Optional, Dict, Any
from datetime import datetime
from io import BytesIO

from src.ui.state import QuotePlan, QuotePlanTask


def export_dataframe_csv(df: pd.DataFrame, filename: Optional[str] = None) -> tuple:
    """
    Export dataframe to CSV bytes.
    
    Returns: (csv_bytes, filename)
    """
    if filename is None:
        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    
    return csv_bytes, filename


def export_dataframe_excel(df: pd.DataFrame, filename: Optional[str] = None,
                           sheet_name: str = "Data") -> tuple:
    """
    Export dataframe to Excel bytes.
    
    Returns: (excel_bytes, filename)
    """
    if filename is None:
        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    excel_bytes = buffer.getvalue()
    
    return excel_bytes, filename


def export_quote_plan_csv(plan: QuotePlan) -> tuple:
    """
    Export quote plan to CSV.
    
    Returns: (csv_bytes, filename)
    """
    rows = []
    for task in plan.tasks:
        rows.append({
            "Department": plan.department,
            "Category": plan.category,
            "Task": task.task_name,
            "Hours": task.hours,
            "Optional": task.is_optional,
            "Cost/Hr": task.cost_per_hour,
            "Quote Rate": task.quote_rate,
            "Est. Cost": task.hours * task.cost_per_hour,
            "Est. Value": task.hours * task.quote_rate,
        })
    
    df = pd.DataFrame(rows)
    
    # Add totals row
    totals = {
        "Department": "",
        "Category": "",
        "Task": "TOTAL",
        "Hours": plan.total_hours,
        "Optional": "",
        "Cost/Hr": "",
        "Quote Rate": "",
        "Est. Cost": plan.estimated_cost,
        "Est. Value": plan.estimated_value,
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    
    filename = f"quote_plan_{plan.department}_{plan.category}_{datetime.now().strftime('%Y%m%d')}.csv"
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    
    return csv_bytes, filename


def export_quote_plan_json(plan: QuotePlan) -> tuple:
    """
    Export quote plan to JSON.
    
    Returns: (json_bytes, filename)
    """
    plan_dict = {
        "department": plan.department,
        "category": plan.category,
        "benchmark_window": plan.benchmark_window,
        "recency_weighted": plan.recency_weighted,
        "created_at": plan.created_at,
        "tasks": [
            {
                "task_name": t.task_name,
                "hours": t.hours,
                "is_optional": t.is_optional,
                "cost_per_hour": t.cost_per_hour,
                "quote_rate": t.quote_rate,
            }
            for t in plan.tasks
        ],
        "totals": {
            "total_hours": plan.total_hours,
            "total_hours_with_optional": plan.total_hours_with_optional,
            "estimated_cost": plan.estimated_cost,
            "estimated_value": plan.estimated_value,
            "estimated_margin": plan.estimated_margin,
            "estimated_margin_pct": plan.estimated_margin_pct,
        }
    }
    
    filename = f"quote_plan_{plan.department}_{plan.category}_{datetime.now().strftime('%Y%m%d')}.json"
    json_bytes = json.dumps(plan_dict, indent=2).encode('utf-8')
    
    return json_bytes, filename


def export_staffing_plan_csv(recommendations: list, plan: QuotePlan) -> tuple:
    """
    Export staffing recommendations to CSV.
    
    Returns: (csv_bytes, filename)
    """
    df = pd.DataFrame(recommendations)
    
    filename = f"staffing_plan_{plan.department}_{plan.category}_{datetime.now().strftime('%Y%m%d')}.csv"
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    
    return csv_bytes, filename


def export_at_risk_jobs_csv(jobs_df: pd.DataFrame) -> tuple:
    """
    Export at-risk jobs list to CSV.
    
    Returns: (csv_bytes, filename)
    """
    # Filter to at-risk only
    at_risk = jobs_df[jobs_df["risk_flag"].isin(["at_risk", "watch"])].copy()
    
    # Select relevant columns
    cols = [
        "job_no", "department_final", "job_category",
        "quoted_hours", "actual_hours", "pct_consumed",
        "scope_creep_pct", "risk_flag"
    ]
    cols = [c for c in cols if c in at_risk.columns]
    
    export_df = at_risk[cols]
    
    filename = f"at_risk_jobs_{datetime.now().strftime('%Y%m%d')}.csv"
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    
    return csv_bytes, filename


def format_export_filename(base_name: str, extension: str = "csv",
                           include_timestamp: bool = True) -> str:
    """Generate formatted export filename."""
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{timestamp}.{extension}"
    return f"{base_name}.{extension}"
