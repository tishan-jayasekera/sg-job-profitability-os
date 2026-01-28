"""
Session state management for Streamlit app.
"""
import streamlit as st
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


# =============================================================================
# STATE KEYS
# =============================================================================

STATE_KEYS = {
    # Drill navigation
    "drill_level": "drill_level",  # company | department | category | staff | breakdown | task
    "selected_department": "selected_department",
    "selected_category": "selected_category",
    "selected_staff": "selected_staff",
    "selected_breakdown": "selected_breakdown",
    "selected_task": "selected_task",
    "category_subtab": "category_subtab",  # legacy
    
    # Filters
    "time_window": "time_window",
    "active_jobs_only": "active_jobs_only",
    "exclude_leave": "exclude_leave",
    "include_nonbillable": "include_nonbillable",
    "selected_client": "selected_client",
    "selected_status": "selected_status",
    
    # Quote Builder
    "quote_plan": "quote_plan",
    "recency_weighted": "recency_weighted",
    "active_staff_only": "active_staff_only",
    
    # Job Mix
    "cohort_definition": "cohort_definition",
    
    # Operational Intervention Engine
    "intervention_quadrant": "intervention_quadrant",
    "intervention_selected_job": "intervention_selected_job",
    "intervention_shortlist_size": "intervention_shortlist_size",
    "intervention_issue_filter": "intervention_issue_filter",

    # Client Profitability & LTV
    "selected_client_quadrant": "selected_client_quadrant",
    "selected_client_name": "selected_client_name",
    "client_queue_mode": "client_queue_mode",
    "client_shortlist_size": "client_shortlist_size",
}


# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULTS = {
    "drill_level": "company",
    "selected_department": None,
    "selected_category": None,
    "selected_staff": None,
    "selected_breakdown": None,
    "selected_task": None,
    "category_subtab": "tasks",
    "time_window": "12m",
    "active_jobs_only": False,
    "exclude_leave": True,
    "include_nonbillable": True,
    "selected_client": None,
    "selected_status": None,
    "quote_plan": None,
    "recency_weighted": False,
    "active_staff_only": True,
    "cohort_definition": "first_activity",
    
    # Operational Intervention Engine
    "intervention_quadrant": None,
    "intervention_selected_job": None,
    "intervention_shortlist_size": 10,
    "intervention_issue_filter": "All Issues",

    # Client Profitability & LTV
    "selected_client_quadrant": "Partners",
    "selected_client_name": None,
    "client_queue_mode": "Intervention",
    "client_shortlist_size": 10,
}


# =============================================================================
# STATE HELPERS
# =============================================================================

def init_state():
    """Initialize all session state keys with defaults."""
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_state(key: str) -> Any:
    """Get state value with default fallback."""
    init_state()
    return st.session_state.get(key, DEFAULTS.get(key))


def set_state(key: str, value: Any):
    """Set state value."""
    st.session_state[key] = value


def reset_state():
    """Reset all state to defaults."""
    for key, default in DEFAULTS.items():
        st.session_state[key] = default


# =============================================================================
# DRILL NAVIGATION
# =============================================================================

def get_drill_level() -> str:
    """Get current drill level."""
    return get_state("drill_level")


def get_breadcrumb() -> List[str]:
    """Get breadcrumb path based on current drill state."""
    level = get_drill_level()
    crumbs = ["Company"]
    
    if level in ["department", "category", "staff", "breakdown", "task"]:
        dept = get_state("selected_department")
        if dept:
            crumbs.append(dept)
    
    if level in ["category", "staff", "breakdown", "task"]:
        cat = get_state("selected_category")
        if cat:
            crumbs.append(cat)

    if level in ["staff", "breakdown", "task"]:
        staff = get_state("selected_staff")
        if staff:
            crumbs.append(staff)

    if level in ["breakdown", "task"]:
        breakdown = get_state("selected_breakdown")
        if breakdown:
            crumbs.append(breakdown)

    if level == "task":
        task = get_state("selected_task")
        if task:
            crumbs.append(task)
    
    return crumbs


def drill_to_department(department: str):
    """Drill down to department level."""
    set_state("drill_level", "department")
    set_state("selected_department", department)
    set_state("selected_category", None)
    set_state("selected_staff", None)
    set_state("selected_breakdown", None)
    set_state("selected_task", None)


def drill_to_category(category: str):
    """Drill down to category level."""
    set_state("drill_level", "category")
    set_state("selected_category", category)
    set_state("selected_staff", None)
    set_state("selected_breakdown", None)
    set_state("selected_task", None)


def drill_to_staff(staff_name: str):
    """Drill down to staff level."""
    set_state("drill_level", "staff")
    set_state("selected_staff", staff_name)
    set_state("selected_breakdown", None)
    set_state("selected_task", None)


def drill_to_breakdown(breakdown: str):
    """Drill down to breakdown level."""
    set_state("drill_level", "breakdown")
    set_state("selected_breakdown", breakdown)
    set_state("selected_task", None)


def drill_to_task(task_name: str):
    """Drill down to task level."""
    set_state("drill_level", "task")
    set_state("selected_task", task_name)


def drill_up():
    """Drill up one level."""
    level = get_drill_level()
    
    if level == "task":
        set_state("drill_level", "breakdown")
        set_state("selected_task", None)
    elif level == "breakdown":
        set_state("drill_level", "staff")
        set_state("selected_breakdown", None)
    elif level == "staff":
        set_state("drill_level", "category")
        set_state("selected_staff", None)
    elif level == "category":
        set_state("drill_level", "department")
        set_state("selected_category", None)
    elif level == "department":
        set_state("drill_level", "company")
        set_state("selected_department", None)
        set_state("selected_category", None)
        set_state("selected_staff", None)
        set_state("selected_breakdown", None)
        set_state("selected_task", None)


def reset_drill():
    """Reset to company level."""
    set_state("drill_level", "company")
    set_state("selected_department", None)
    set_state("selected_category", None)
    set_state("selected_staff", None)
    set_state("selected_breakdown", None)
    set_state("selected_task", None)


# =============================================================================
# FILTER HELPERS
# =============================================================================

def get_filters() -> Dict[str, Any]:
    """Get all current filter values."""
    return {
        "time_window": get_state("time_window"),
        "active_jobs_only": get_state("active_jobs_only"),
        "exclude_leave": get_state("exclude_leave"),
        "include_nonbillable": get_state("include_nonbillable"),
        "selected_client": get_state("selected_client"),
        "selected_status": get_state("selected_status"),
    }


def get_drill_filter() -> Dict[str, Any]:
    """Get filter dict for current drill level."""
    level = get_drill_level()
    
    filters = {}
    
    if level in ["department", "category", "staff", "breakdown", "task"]:
        dept = get_state("selected_department")
        if dept:
            filters["department_final"] = dept
    
    if level in ["category", "staff", "breakdown", "task"]:
        cat = get_state("selected_category")
        if cat:
            filters["category_rev_job"] = cat

    if level in ["staff", "breakdown", "task"]:
        staff = get_state("selected_staff")
        if staff:
            filters["staff_name"] = staff

    if level in ["breakdown", "task"]:
        breakdown = get_state("selected_breakdown")
        if breakdown:
            filters["breakdown"] = breakdown

    if level == "task":
        task = get_state("selected_task")
        if task:
            filters["task_name"] = task
    
    return filters


def apply_drill_filter(df, filters: Optional[Dict] = None):
    """Apply drill filter to dataframe."""
    if filters is None:
        filters = get_drill_filter()
    
    for col, value in filters.items():
        if value is None:
            continue
        if col in df.columns:
            df = df[df[col] == value]
            continue
        if col == "category_rev_job" and "job_category" in df.columns:
            df = df[df["job_category"] == value]
    
    return df


# =============================================================================
# QUOTE PLAN
# =============================================================================

@dataclass
class QuotePlanTask:
    """Single task in a quote plan."""
    task_name: str
    hours: float
    is_optional: bool = False
    cost_per_hour: float = 0.0
    quote_rate: float = 0.0


@dataclass 
class QuotePlan:
    """Quote plan structure."""
    department: str
    category: str
    tasks: List[QuotePlanTask] = field(default_factory=list)
    benchmark_window: str = "12m"
    recency_weighted: bool = False
    created_at: str = ""
    
    @property
    def total_hours(self) -> float:
        return sum(t.hours for t in self.tasks if not t.is_optional)
    
    @property
    def total_hours_with_optional(self) -> float:
        return sum(t.hours for t in self.tasks)
    
    @property
    def estimated_cost(self) -> float:
        return sum(t.hours * t.cost_per_hour for t in self.tasks if not t.is_optional)
    
    @property
    def estimated_value(self) -> float:
        return sum(t.hours * t.quote_rate for t in self.tasks if not t.is_optional)
    
    @property
    def estimated_margin(self) -> float:
        return self.estimated_value - self.estimated_cost
    
    @property
    def estimated_margin_pct(self) -> float:
        if self.estimated_value == 0:
            return 0
        return self.estimated_margin / self.estimated_value * 100


def get_quote_plan() -> Optional[QuotePlan]:
    """Get current quote plan from state."""
    return get_state("quote_plan")


def set_quote_plan(plan: QuotePlan):
    """Save quote plan to state."""
    set_state("quote_plan", plan)


def clear_quote_plan():
    """Clear quote plan from state."""
    set_state("quote_plan", None)
