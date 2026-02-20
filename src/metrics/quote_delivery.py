"""
Quote → Delivery metrics pack.

Single source of truth for: quote totals, hours variance, scope creep, overrun rates.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, List, Dict

from src.data.semantic import safe_quote_job_task, safe_quote_rollup, get_category_col
from src.data.cohorts import filter_by_time_window
from src.config import config


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_quote_totals(df: pd.DataFrame,
                         group_keys: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
    """
    Safely compute quote totals with proper deduplication.
    
    Returns DataFrame with:
    - quoted_hours: safe sum of quoted_time_total
    - quoted_amount: safe sum of quoted_amount_total
    - quote_rate: quoted_amount / quoted_hours
    - job_task_count: number of unique job-tasks
    - job_count: number of unique jobs
    """
    return safe_quote_rollup(df, group_keys if group_keys else ())


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_hours_variance(df: pd.DataFrame,
                           group_keys: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
    """
    Compute hours variance between actual and quoted.
    
    Returns DataFrame with:
    - quoted_hours, actual_hours
    - hours_variance, hours_variance_pct
    """
    keys = list(group_keys) if group_keys else []

    # Safe quote totals
    quote = compute_quote_totals(df, tuple(keys) if keys else None)
    
    # Actual hours
    if keys:
        actuals = df.groupby(keys).agg(
            actual_hours=("hours_raw", "sum")
        ).reset_index()
    else:
        actuals = pd.DataFrame([{"actual_hours": df["hours_raw"].sum()}])
    
    # Merge
    if keys:
        result = quote.merge(actuals, on=keys, how="outer")
    else:
        result = quote.copy()
        result["actual_hours"] = actuals["actual_hours"].iloc[0]
    
    result["quoted_hours"] = result["quoted_hours"].fillna(0)
    result["actual_hours"] = result["actual_hours"].fillna(0)
    
    # Variance
    result["hours_variance"] = result["actual_hours"] - result["quoted_hours"]
    result["hours_variance_pct"] = np.where(
        result["quoted_hours"] > 0,
        result["hours_variance"] / result["quoted_hours"] * 100,
        np.nan
    )
    
    return result


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_scope_creep(df: pd.DataFrame,
                        group_keys: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
    """
    Compute scope creep metrics.
    
    Scope creep = hours spent on tasks/jobs without quote match.
    
    Returns DataFrame with:
    - total_hours, unquoted_hours
    - unquoted_share (%)
    """
    keys = list(group_keys) if group_keys else []

    has_quote_flag = "quote_match_flag" in df.columns
    has_quote_hours = "quoted_time_total" in df.columns or "quoted_amount_total" in df.columns

    if not has_quote_flag and not has_quote_hours:
        if keys:
            result = df.groupby(keys).agg(
                total_hours=("hours_raw", "sum")
            ).reset_index()
        else:
            result = pd.DataFrame([{"total_hours": df["hours_raw"].sum()}])
        result["unquoted_hours"] = 0
        result["unquoted_share"] = 0
        return result

    df = df.copy()
    unquoted = pd.Series(False, index=df.index)
    if has_quote_flag:
        unquoted |= df["quote_match_flag"].astype(str).str.lower().ne("matched")
    if "quoted_time_total" in df.columns:
        unquoted |= df["quoted_time_total"].fillna(0) <= 0
    if "quoted_amount_total" in df.columns:
        unquoted |= df["quoted_amount_total"].fillna(0) <= 0

    df["is_unquoted"] = unquoted
    df["unquoted_hours"] = np.where(df["is_unquoted"], df["hours_raw"], 0)
    
    if keys:
        result = df.groupby(keys).agg(
            total_hours=("hours_raw", "sum"),
            unquoted_hours=("unquoted_hours", "sum"),
        ).reset_index()
    else:
        result = pd.DataFrame([{
            "total_hours": df["hours_raw"].sum(),
            "unquoted_hours": df["unquoted_hours"].sum(),
        }])
    
    result["unquoted_share"] = np.where(
        result["total_hours"] > 0,
        result["unquoted_hours"] / result["total_hours"] * 100,
        0
    )
    
    return result


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_overrun_rates(df: pd.DataFrame,
                          group_keys: Optional[tuple[str, ...]] = None,
                          severe_threshold: float = None) -> pd.DataFrame:
    """
    Compute overrun rates at job-task level.
    
    Returns DataFrame with:
    - n_job_tasks
    - n_overrun (actual > quoted)
    - n_severe_overrun (actual > quoted * threshold)
    - overrun_rate (%), severe_overrun_rate (%)
    """
    keys = list(group_keys) if group_keys else []

    if severe_threshold is None:
        severe_threshold = config.severe_overrun_threshold
    
    # Get job-task level data
    job_task = safe_quote_job_task(df)
    
    if len(job_task) == 0:
        cols = keys + [
            "n_job_tasks", "n_overrun", "n_severe_overrun",
            "overrun_rate", "severe_overrun_rate"
        ]
        return pd.DataFrame(columns=cols)
    
    # Merge actual hours
    actuals = df.groupby(["job_no", "task_name"])["hours_raw"].sum().reset_index()
    actuals.columns = ["job_no", "task_name", "actual_hours"]
    
    job_task = job_task.merge(actuals, on=["job_no", "task_name"], how="left")
    job_task["actual_hours"] = job_task["actual_hours"].fillna(0)
    
    # Flags
    if "quoted_time_total" in job_task.columns:
        job_task["is_overrun"] = job_task["actual_hours"] > job_task["quoted_time_total"]
        job_task["is_severe_overrun"] = (
            job_task["actual_hours"] > job_task["quoted_time_total"] * severe_threshold
        )
    else:
        job_task["is_overrun"] = False
        job_task["is_severe_overrun"] = False
    
    # Merge group keys if needed
    if keys:
        key_mapping = df[["job_no", "task_name"] + keys].drop_duplicates()
        job_task = job_task.merge(key_mapping, on=["job_no", "task_name"], how="left")
        
        result = job_task.groupby(keys).agg(
            n_job_tasks=("job_no", "count"),
            n_overrun=("is_overrun", "sum"),
            n_severe_overrun=("is_severe_overrun", "sum"),
        ).reset_index()
    else:
        result = pd.DataFrame([{
            "n_job_tasks": len(job_task),
            "n_overrun": job_task["is_overrun"].sum(),
            "n_severe_overrun": job_task["is_severe_overrun"].sum(),
        }])
    
    result["overrun_rate"] = np.where(
        result["n_job_tasks"] > 0,
        result["n_overrun"] / result["n_job_tasks"] * 100,
        0
    )
    result["severe_overrun_rate"] = np.where(
        result["n_job_tasks"] > 0,
        result["n_severe_overrun"] / result["n_job_tasks"] * 100,
        0
    )
    
    return result


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_quote_delivery_full(df: pd.DataFrame,
                                group_keys: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
    """
    Full quote → delivery analysis combining all metrics.
    """
    keys = list(group_keys) if group_keys else []

    # Hours variance
    variance = compute_hours_variance(df, tuple(keys) if keys else None)
    
    # Scope creep
    scope = compute_scope_creep(df, tuple(keys) if keys else None)
    
    # Overrun rates
    overrun = compute_overrun_rates(df, tuple(keys) if keys else None)
    
    # Merge
    if keys:
        result = variance.merge(
            scope[[c for c in scope.columns if c not in variance.columns or c in keys]],
            on=keys,
            how="outer"
        )
        result = result.merge(
            overrun[[c for c in overrun.columns if c not in result.columns or c in keys]],
            on=keys,
            how="outer"
        )
    else:
        result = variance.copy()
        for col in scope.columns:
            if col not in result.columns:
                result[col] = scope[col].iloc[0]
        for col in overrun.columns:
            if col not in result.columns:
                result[col] = overrun[col].iloc[0]
    
    return result


@st.cache_data(show_spinner=False)
def get_quote_delivery_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get summary quote delivery metrics as a dictionary.
    """
    full = compute_quote_delivery_full(df)
    
    return {
        "quoted_hours": full["quoted_hours"].iloc[0] if "quoted_hours" in full.columns else 0,
        "actual_hours": full["actual_hours"].iloc[0] if "actual_hours" in full.columns else 0,
        "hours_variance": full["hours_variance"].iloc[0] if "hours_variance" in full.columns else 0,
        "hours_variance_pct": full["hours_variance_pct"].iloc[0] if "hours_variance_pct" in full.columns else 0,
        "unquoted_share": full["unquoted_share"].iloc[0] if "unquoted_share" in full.columns else 0,
        "overrun_rate": full["overrun_rate"].iloc[0] if "overrun_rate" in full.columns else 0,
        "severe_overrun_rate": full["severe_overrun_rate"].iloc[0] if "severe_overrun_rate" in full.columns else 0,
    }


@st.cache_data(show_spinner=False)
def get_top_overrun_tasks(df: pd.DataFrame, 
                          n: int = 10,
                          department: Optional[str] = None,
                          category: Optional[str] = None) -> pd.DataFrame:
    """
    Get top tasks by hours variance (overruns).
    """
    df_filtered = df.copy()
    
    if department:
        df_filtered = df_filtered[df_filtered["department_final"] == department]
    if category:
        category_col = get_category_col(df_filtered)
        df_filtered = df_filtered[df_filtered[category_col] == category]
    
    # Get task-level variance
    task_variance = compute_hours_variance(df_filtered, ("task_name",))
    
    # Filter to overruns only
    overruns = task_variance[task_variance["hours_variance"] > 0]
    
    return overruns.nlargest(n, "hours_variance")


def _is_uncategorised(category: Optional[str]) -> bool:
    if category is None:
        return False
    cat = str(category).strip().lower()
    return cat in {"(uncategorised)", "(uncategorized)", "uncategorised", "uncategorized", "__null__"}


def _apply_task_scope(
    df: pd.DataFrame,
    department: Optional[str] = None,
    category: Optional[str] = None,
    time_window: str = "12m",
) -> pd.DataFrame:
    scoped = df

    if department and "department_final" in scoped.columns:
        scoped = scoped[scoped["department_final"].astype(str) == str(department)]

    if category is not None and len(scoped) > 0:
        category_col = get_category_col(scoped)
        if category_col in scoped.columns:
            if _is_uncategorised(category):
                scoped = scoped[scoped[category_col].isna()]
            else:
                scoped = scoped[scoped[category_col] == category]

    date_col = "month_key" if "month_key" in scoped.columns else "work_date" if "work_date" in scoped.columns else None
    if date_col and len(scoped) > 0:
        scoped = scoped.copy()
        scoped[date_col] = pd.to_datetime(scoped[date_col], errors="coerce")
        scoped = filter_by_time_window(scoped, window=time_window, date_col=date_col)

    return scoped


def _build_job_task_overrun_base(df: pd.DataFrame) -> pd.DataFrame:
    required = {"job_no", "task_name", "hours_raw"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    scoped = df.copy()
    has_base_cost = "base_cost" in scoped.columns
    category_col = get_category_col(scoped) if len(scoped) > 0 else "job_category"

    quote = safe_quote_job_task(scoped)
    if quote.empty:
        quote = pd.DataFrame(columns=["job_no", "task_name", "quoted_time_total", "quoted_amount_total", "quote_match_flag"])
    else:
        keep_quote_cols = ["job_no", "task_name"] + [c for c in ["quoted_time_total", "quoted_amount_total", "quote_match_flag"] if c in quote.columns]
        quote = quote[keep_quote_cols].copy()

    agg_map = {"actual_hours": ("hours_raw", "sum")}
    if has_base_cost:
        agg_map["actual_cost"] = ("base_cost", "sum")

    actual = scoped.groupby(["job_no", "task_name"], dropna=False).agg(**agg_map).reset_index()
    if "actual_cost" not in actual.columns:
        actual["actual_cost"] = np.nan

    meta_cols = ["job_no", "task_name"]
    for col in ["department_final", category_col, "client_name", "client_group"]:
        if col in scoped.columns and col not in meta_cols:
            meta_cols.append(col)
    meta = scoped[meta_cols].drop_duplicates(subset=["job_no", "task_name"])
    actual = actual.merge(meta, on=["job_no", "task_name"], how="left")

    if category_col in actual.columns:
        actual = actual.rename(columns={category_col: "category"})
    elif "category" not in actual.columns:
        actual["category"] = np.nan

    job_task = actual.merge(quote, on=["job_no", "task_name"], how="left")

    job_task["quoted_hours"] = pd.to_numeric(job_task.get("quoted_time_total", 0), errors="coerce").fillna(0.0)
    job_task["actual_hours"] = pd.to_numeric(job_task["actual_hours"], errors="coerce").fillna(0.0)

    if has_base_cost:
        job_task["actual_cost"] = pd.to_numeric(job_task["actual_cost"], errors="coerce").fillna(0.0)
        job_task["effective_cost_rate"] = np.where(
            job_task["actual_hours"] > 0,
            job_task["actual_cost"] / job_task["actual_hours"],
            0.0,
        )
    else:
        job_task["actual_cost"] = np.nan
        job_task["effective_cost_rate"] = 0.0

    job_task["overrun_hours"] = np.where(
        job_task["quoted_hours"] > 0,
        np.maximum(job_task["actual_hours"] - job_task["quoted_hours"], 0.0),
        0.0,
    )

    if has_base_cost:
        job_task["overrun_cost"] = job_task["overrun_hours"] * job_task["effective_cost_rate"]
    else:
        job_task["overrun_cost"] = np.nan

    if "quoted_amount_total" in job_task.columns:
        job_task["quoted_amount_total"] = pd.to_numeric(job_task["quoted_amount_total"], errors="coerce")
        job_task["quote_rate"] = np.where(
            (job_task["quoted_hours"] > 0) & job_task["quoted_amount_total"].notna(),
            job_task["quoted_amount_total"] / job_task["quoted_hours"],
            np.nan,
        )
        job_task["revenue_at_risk"] = job_task["overrun_hours"] * job_task["quote_rate"]
    else:
        job_task["quote_rate"] = np.nan
        job_task["revenue_at_risk"] = np.nan

    job_task["overrun_pct"] = np.where(
        job_task["quoted_hours"] > 0,
        (job_task["actual_hours"] - job_task["quoted_hours"]) / job_task["quoted_hours"],
        np.nan,
    )
    job_task["overrun_flag"] = job_task["overrun_hours"] > 0

    return job_task


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def compute_task_overrun_consistency(
    df: pd.DataFrame,
    department: Optional[str] = None,
    category: Optional[str] = None,
    time_window: str = "12m",
    min_jobs_with_quote: int = 8,
    min_overrun_rate: float = 0.30,
) -> pd.DataFrame:
    task_cols = [
        "task_name",
        "jobs_with_quote",
        "overrun_jobs",
        "overrun_rate",
        "total_quoted_hours",
        "total_actual_hours",
        "total_overrun_hours",
        "total_overrun_cost",
        "total_revenue_at_risk",
        "avg_overrun_pct",
        "leakage_score",
    ]

    required = {"job_no", "task_name", "hours_raw"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=task_cols)

    scoped = _apply_task_scope(
        df,
        department=department,
        category=category,
        time_window=time_window,
    )
    if scoped.empty:
        return pd.DataFrame(columns=task_cols)

    job_task = _build_job_task_overrun_base(scoped)
    if job_task.empty:
        return pd.DataFrame(columns=task_cols)

    quoted_job_tasks = job_task[job_task["quoted_hours"] > 0].copy()
    if quoted_job_tasks.empty:
        return pd.DataFrame(columns=task_cols)

    task = quoted_job_tasks.groupby("task_name", dropna=False).agg(
        jobs_with_quote=("job_no", "nunique"),
        overrun_jobs=("overrun_flag", "sum"),
        total_quoted_hours=("quoted_hours", "sum"),
        total_actual_hours=("actual_hours", "sum"),
        total_overrun_hours=("overrun_hours", "sum"),
        total_overrun_cost=("overrun_cost", lambda x: x.sum(min_count=1)),
        total_revenue_at_risk=("revenue_at_risk", lambda x: x.sum(min_count=1)),
        avg_overrun_pct=("overrun_pct", "mean"),
    ).reset_index()

    task["overrun_rate"] = np.where(
        task["jobs_with_quote"] > 0,
        task["overrun_jobs"] / task["jobs_with_quote"],
        0.0,
    )
    task["leakage_score"] = task["total_overrun_cost"] * task["overrun_rate"]

    task = task[
        (task["jobs_with_quote"] >= min_jobs_with_quote)
        & (task["overrun_rate"] >= min_overrun_rate)
    ].copy()
    if task.empty:
        return pd.DataFrame(columns=task_cols)

    if task["leakage_score"].notna().any():
        task = task.sort_values("leakage_score", ascending=False, na_position="last")
    else:
        task = task.sort_values(["overrun_rate", "total_overrun_hours"], ascending=[False, False])

    for col in task_cols:
        if col not in task.columns:
            task[col] = np.nan

    return task[task_cols].reset_index(drop=True)


@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: tuple(x)})
def get_overrun_jobs_for_task(
    df: pd.DataFrame,
    task_name: str,
    department: Optional[str] = None,
    category: Optional[str] = None,
    time_window: str = "12m",
    n: int = 15,
) -> pd.DataFrame:
    cols = [
        "job_no",
        "client_name",
        "client_group",
        "department_final",
        "category",
        "quoted_hours",
        "actual_hours",
        "overrun_hours",
        "overrun_cost",
        "avg_cost_rate",
        "quote_rate",
        "revenue_at_risk",
    ]

    if not task_name:
        return pd.DataFrame(columns=cols)

    required = {"job_no", "task_name", "hours_raw"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=cols)

    scoped = _apply_task_scope(
        df,
        department=department,
        category=category,
        time_window=time_window,
    )
    if scoped.empty:
        return pd.DataFrame(columns=cols)

    job_task = _build_job_task_overrun_base(scoped)
    if job_task.empty:
        return pd.DataFrame(columns=cols)

    task_jobs = job_task[
        (job_task["task_name"] == task_name)
        & (job_task["quoted_hours"] > 0)
        & (job_task["overrun_hours"] > 0)
    ].copy()
    if task_jobs.empty:
        return pd.DataFrame(columns=cols)

    task_jobs = task_jobs.rename(columns={"effective_cost_rate": "avg_cost_rate"})

    if task_jobs["overrun_cost"].notna().any():
        task_jobs = task_jobs.sort_values("overrun_cost", ascending=False)
    else:
        task_jobs = task_jobs.sort_values("overrun_hours", ascending=False)

    for col in cols:
        if col not in task_jobs.columns:
            task_jobs[col] = np.nan

    return task_jobs[cols].head(n).reset_index(drop=True)
