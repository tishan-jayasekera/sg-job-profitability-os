"""
Application configuration management.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

@dataclass
def _default_data_dir() -> Path:
    env_dir = os.getenv("DATA_DIR")
    if env_dir:
        return Path(env_dir)
    if Path("./src/data").exists():
        return Path("./src/data")
    return Path("./data")


class AppConfig:
    """Application configuration with environment overrides."""
    
    # Paths
    data_dir: Path = field(default_factory=_default_data_dir)
    
    # Environment
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "dev"))
    
    # Cache settings
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))
    
    # Business logic defaults
    active_job_recency_days: int = field(default_factory=lambda: int(os.getenv("ACTIVE_JOB_RECENCY_DAYS", "21")))
    active_staff_recency_months: int = field(default_factory=lambda: int(os.getenv("ACTIVE_STAFF_RECENCY_MONTHS", "6")))
    recency_half_life_months: int = field(default_factory=lambda: int(os.getenv("RECENCY_HALF_LIFE_MONTHS", "6")))
    
    # Capacity model
    standard_weekly_hours: float = 38.0
    default_utilisation_target: float = 0.8
    default_fte_scaling: float = 1.0
    
    # Thresholds
    severe_overrun_threshold: float = 1.2  # 120% of quoted hours
    
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"
    
    @property
    def marts_dir(self) -> Path:
        return self.data_dir / "marts"
    
    @property
    def is_prod(self) -> bool:
        return self.app_env.lower() == "prod"


# Global config instance
config = AppConfig()


# Table file names
TABLE_FILES = {
    "fact_timesheet": "fact_timesheet_day_enriched",
    "fact_job_task_month": "fact_job_task_month",
    "audit_reconciliation": "audit_revenue_reconciliation_job_month",
    "audit_unallocated": "audit_unallocated_revenue",
}

# Mart file names
MART_FILES = {
    "cube_dept_month": "cube_dept_month",
    "cube_dept_category_month": "cube_dept_category_month",
    "cube_dept_category_task": "cube_dept_category_task",
    "cube_dept_category_staff": "cube_dept_category_staff",
    "active_jobs_snapshot": "active_jobs_snapshot",
    "job_mix_month": "job_mix_month",
}

# Required columns (hard fail if missing)
REQUIRED_COLUMNS = {
    "fact_timesheet": [
        "department_final",
        "job_category", 
        "task_name",
        "staff_name",
        "month_key",
        "hours_raw",
        "base_cost",
        "rev_alloc",
        "quoted_time_total",
        "quoted_amount_total",
        "quote_match_flag",
        "is_billable",
        "utilisation_target",
        "fte_hours_scaling",
        "breakdown",
    ],
    "fact_job_task_month": [
        "job_no",
        "task_name",
        "month_key",
        "department_final",
        "job_category",
        "hours_raw_sum",
        "base_cost_sum",
        "rev_alloc_sum",
        "quoted_time_total",
        "quoted_amount_total",
    ],
}

# Optional columns (soft warn if missing)
OPTIONAL_COLUMNS = {
    "fact_timesheet": [
        "job_status",
        "job_due_date",
        "job_completed_date",
        "client",
        "business_unit",
        "role",
        "function",
        "onshore_flag",
        "state",
        "job_no",
        "work_date",
        "aus_fy",
    ],
}

# Formatting constants
FORMAT_CURRENCY = "${:,.0f}"
FORMAT_CURRENCY_DECIMAL = "${:,.2f}"
FORMAT_HOURS = "{:,.1f}"
FORMAT_RATE = "${:,.0f}/hr"
FORMAT_PERCENT = "{:.1f}%"
FORMAT_COUNT = "{:,}"
