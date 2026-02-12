"""
Application configuration management.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

def _default_data_dir() -> Path:
    env_dir = os.getenv("DATA_DIR")
    if env_dir:
        return Path(env_dir)
    if Path("./src/data").exists():
        return Path("./src/data")
    return Path("./data")


@dataclass
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
    
    # Capacity model (empirical)
    CAPACITY_HOURS_PER_WEEK: float = 38.0
    DEFAULT_FTE_SCALING: float = 1.0
    PROFILE_TRAINING_MONTHS: int = 12
    LOAD_TRAILING_WEEKS: int = 4
    RECENCY_HALF_LIFE_MONTHS: int = 6
    ELIGIBILITY_RECENCY_MONTHS: int = 6
    ELIGIBILITY_MIN_HOURS: int = 10
    ELIGIBILITY_MIN_JOBS: int = 2
    ARCHETYPE_OPS_HEAVY_THRESHOLD: float = 0.50
    ARCHETYPE_SPECIALIST_THRESHOLD: float = 0.70
    ARCHETYPE_GENERALIST_MIN_CATEGORIES: int = 3
    ARCHETYPE_GENERALIST_MAX_SHARE: float = 0.50
    ARCHETYPE_SENIOR_MAX_HOURS_WEEK: float = 25.0
    CROWDOUT_ADMIN_THRESHOLD: float = 0.20
    CROWDOUT_INTERNAL_THRESHOLD: float = 0.30
    CROWDOUT_UNASSIGNED_THRESHOLD: float = 0.15
    COVERAGE_CRITICAL: int = 1
    COVERAGE_LOW: int = 2
    COVERAGE_GOOD: int = 3
    
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
        "client_group_rev_job_month",
        "client_group_rev_job",
        "client_group",
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
