"""
Quote Builder page.

Empirical peer-job quoting engine with three sections:
1) Peer Job Lookup
2) Quote Composer
3) Economics Preview
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_fact_timesheet
from src.data.semantic import get_category_col
from src.metrics.quote_engine import (
    compute_quote_economics,
    compute_risk_indicators,
    get_peer_jobs,
    get_peer_pool_summary,
    get_peer_task_mix,
)
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_rate
from src.ui.layout import section_header
from src.ui.state import STATE_KEYS, get_state, init_state, set_state


st.set_page_config(page_title="Quote Builder", page_icon="📝", layout="wide")
init_state()


RECENCY_CHOICES: list[tuple[str, int]] = [("12m", 12), ("24m", 24), ("All Time", 0)]
HOURS_BASIS_OPTIONS = ["p25", "p50", "p75"]
HOURS_BASIS_LABELS = {
    "p25": "P25 (conservative)",
    "p50": "P50 (median)",
    "p75": "P75 (safe)",
}

STOP_KEYWORD_TOKENS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "via",
    "plus",
    "vs",
    "per",
    "into",
    "onto",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort scalar float conversion."""
    try:
        if value is None:
            return default
        out = float(pd.to_numeric(value, errors="coerce"))
        if np.isnan(out):
            return default
        return out
    except Exception:
        return default


def _completed_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical completed-job filter pattern."""
    if "job_completed_date" in df.columns:
        return df[df["job_completed_date"].notna()]
    if "job_status" in df.columns:
        return df[df["job_status"].astype(str).str.lower().str.contains("completed", na=False)]
    return df.iloc[0:0]


def _get_client_col(df: pd.DataFrame) -> str | None:
    """Return the preferred client grouping column (same preference as LTV page)."""
    if "client_group_rev_job" in df.columns and df["client_group_rev_job"].notna().any():
        return "client_group_rev_job"
    if "client" in df.columns and df["client"].notna().any():
        return "client"
    return None


def _normalize_text(value: Any) -> str:
    """Normalize text for robust keyword matching."""
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_keyword_terms(raw: str) -> list[str]:
    """Parse keyword input into normalized terms/phrases."""
    if not raw or not raw.strip():
        return []

    parsed = _parse_keyword_query(raw)
    return parsed["keywords"]


def _parse_keyword_query(raw: str) -> dict[str, list[str]]:
    """
    Smart keyword parser.

    Returns:
        {
            "keywords": list[str],          # individual tokens after stopword removal
            "phrases": list[str],           # cleaned phrases + soft n-grams
            "ignored_tokens": list[str],    # removed connectives/common words
        }
    """
    if not raw or not raw.strip():
        return {"keywords": [], "phrases": [], "ignored_tokens": []}

    has_delimiter = bool(re.search(r"[,\n;|]", raw))
    if not has_delimiter:
        normalized = _normalize_text(raw)
        if not normalized:
            return {"keywords": [], "phrases": [], "ignored_tokens": []}

        tokens = [token for token in normalized.split() if token]
        cleaned_tokens: list[str] = []
        ignored_tokens: list[str] = []
        seen_ignored: set[str] = set()
        for token in tokens:
            if token in STOP_KEYWORD_TOKENS:
                if token not in seen_ignored:
                    seen_ignored.add(token)
                    ignored_tokens.append(token)
                continue
            cleaned_tokens.append(token)

        phrase = " ".join(cleaned_tokens).strip()
        if not phrase:
            return {"keywords": [], "phrases": [], "ignored_tokens": ignored_tokens}
        return {
            "keywords": [phrase],
            "phrases": [phrase],
            "ignored_tokens": ignored_tokens,
        }

    parts = [chunk.strip() for chunk in re.split(r"[,\n;|]+", raw) if chunk.strip()]
    if not parts:
        parts = [raw.strip()]

    keywords: list[str] = []
    phrases: list[str] = []
    ignored_tokens: list[str] = []
    kept_stream: list[str] = []

    seen_keywords: set[str] = set()
    seen_phrases: set[str] = set()
    seen_ignored: set[str] = set()

    for part in parts:
        normalized = _normalize_text(part)
        if not normalized:
            continue

        raw_tokens = [token for token in normalized.split() if token]
        kept_tokens: list[str] = []
        for token in raw_tokens:
            # Keep short acronyms like CRM/SMS/AI; drop most 1-char noise.
            if len(token) == 1 and token not in {"x"}:
                continue
            if token in STOP_KEYWORD_TOKENS:
                if token not in seen_ignored:
                    seen_ignored.add(token)
                    ignored_tokens.append(token)
                continue

            kept_tokens.append(token)
            kept_stream.append(token)
            if token not in seen_keywords:
                seen_keywords.add(token)
                keywords.append(token)

        # Phrase candidates per user chunk.
        if len(kept_tokens) >= 2:
            phrase = " ".join(kept_tokens)
            if phrase not in seen_phrases:
                seen_phrases.add(phrase)
                phrases.append(phrase)

    return {
        "keywords": keywords,
        "phrases": phrases,
        "ignored_tokens": ignored_tokens,
    }


def _first_non_empty(series: pd.Series) -> str:
    """Return first non-empty value from a series."""
    values = series.fillna("").astype(str).str.strip()
    values = values[values != ""]
    if len(values) == 0:
        return ""
    return str(values.iloc[0])


def _effective_task_hours(task: dict[str, Any], hours_basis: str) -> float:
    """Resolve effective task hours with override precedence."""
    if task.get("hours_override") is not None:
        return max(_safe_float(task.get("hours_override"), 0.0), 0.0)
    return max(_safe_float(task.get(f"hours_{hours_basis}", task.get("hours_p50", 0.0)), 0.0), 0.0)


def _effective_line_hours(line: dict[str, Any], hours_basis: str) -> float:
    """Resolve effective line hours from task overrides or line-level defaults."""
    tasks = line.get("task_mix") or []
    if tasks:
        return float(sum(_effective_task_hours(task, hours_basis) for task in tasks))

    if line.get("line_hours_override") is not None:
        return max(_safe_float(line.get("line_hours_override"), 0.0), 0.0)

    return max(_safe_float(line.get(f"hours_{hours_basis}", line.get("hours_p50", 0.0)), 0.0), 0.0)


def _effective_line_rate(line: dict[str, Any]) -> float:
    """Resolve effective line rate with override precedence."""
    if line.get("rate_override") is not None:
        return max(_safe_float(line.get("rate_override"), 0.0), 0.0)
    return max(_safe_float(line.get("rate_suggested", 0.0), 0.0), 0.0)


def _line_has_overrides(line: dict[str, Any]) -> bool:
    """Whether a line has any task or line-level hour override."""
    if line.get("line_hours_override") is not None:
        return True
    for task in line.get("task_mix") or []:
        if task.get("hours_override") is not None:
            return True
    return False


def _rebalance_task_mix_to_line_hours(line: dict[str, Any]) -> None:
    """Scale task-hour percentiles so each percentile sums to the line percentile."""
    tasks = line.get("task_mix") or []
    if not tasks:
        return

    for basis in ("p25", "p50", "p75"):
        line_target = max(_safe_float(line.get(f"hours_{basis}"), 0.0), 0.0)
        if line_target <= 0:
            continue

        task_vals = [max(_safe_float(task.get(f"hours_{basis}"), 0.0), 0.0) for task in tasks]
        task_total = float(sum(task_vals))

        if task_total > 0:
            scale = line_target / task_total
            for task, value in zip(tasks, task_vals):
                task[f"hours_{basis}"] = float(value * scale)
            continue

        shares = [max(_safe_float(task.get("share_pct"), 0.0), 0.0) / 100.0 for task in tasks]
        share_total = float(sum(shares))
        if share_total <= 0:
            even = line_target / len(tasks)
            for task in tasks:
                task[f"hours_{basis}"] = float(even)
        else:
            for task, share in zip(tasks, shares):
                task[f"hours_{basis}"] = float(line_target * (share / share_total))


def _apply_line_total_override(line: dict[str, Any], new_total: float, hours_basis: str) -> None:
    """Scale task hours proportionally when a line-level total is overridden."""
    target = max(_safe_float(new_total, 0.0), 0.0)
    tasks = line.get("task_mix") or []

    if not tasks:
        line["line_hours_override"] = float(target)
        return

    current = [_effective_task_hours(task, hours_basis) for task in tasks]
    current_total = float(sum(current))

    if current_total <= 0:
        even = target / len(tasks)
        for task in tasks:
            task["hours_override"] = float(even)
    else:
        scale = target / current_total
        for task, hours in zip(tasks, current):
            task["hours_override"] = float(max(hours * scale, 0.0))

    line["line_hours_override"] = float(target)


def _peer_badge_html(peer_count: int) -> str:
    """Render peer-count confidence badge with threshold colors."""
    if peer_count < 5:
        bg = "#dc3545"
        label = "Low confidence"
    elif peer_count < 15:
        bg = "#f0ad4e"
        label = "Moderate confidence"
    else:
        bg = "#28a745"
        label = "Good confidence"

    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:{bg};color:#fff;font-size:12px;font-weight:600;'>"
        f"{peer_count} peers · {label}</span>"
    )


def _build_quote_line_from_peers(
    department: str,
    category: str,
    peer_jobs: pd.DataFrame,
    task_mix: pd.DataFrame,
    recency_months: int,
) -> dict[str, Any]:
    """Create a quote line payload from peer-job metrics."""
    summary = get_peer_pool_summary(peer_jobs)
    median_peer_rate = float(peer_jobs["realised_rate"].median()) if len(peer_jobs) > 0 else 0.0

    line: dict[str, Any] = {
        "line_id": str(uuid4()),
        "manual": False,
        "department": department,
        "category": category,
        "hours_p25": float(summary.get("hours_p25", 0.0) or 0.0),
        "hours_p50": float(summary.get("hours_p50", 0.0) or 0.0),
        "hours_p75": float(summary.get("hours_p75", 0.0) or 0.0),
        "line_hours_override": None,
        "rate_suggested": median_peer_rate,
        "rate_override": None,
        "cost_per_hour_peer": float(summary.get("blended_cost_per_hour", 0.0) or 0.0),
        "peer_job_count": int(summary.get("n_jobs", 0) or 0),
        "recency_window": "All Time" if recency_months == 0 else f"{recency_months}m",
        "recency_months": recency_months,
        "task_mix": [],
        "peer_job_hours": peer_jobs["total_hours"].dropna().astype(float).tolist(),
    }

    tasks: list[dict[str, Any]] = []
    if len(task_mix) > 0:
        for row in task_mix.itertuples(index=False):
            tasks.append(
                {
                    "task_name": str(getattr(row, "task_name", "Unspecified")),
                    "share_pct": float(getattr(row, "share_pct", 0.0) or 0.0),
                    "hours_p25": float(getattr(row, "hours_p25", 0.0) or 0.0),
                    "hours_p50": float(getattr(row, "hours_p50", 0.0) or 0.0),
                    "hours_p75": float(getattr(row, "hours_p75", 0.0) or 0.0),
                    "hours_override": None,
                    "median_cost_per_hour": float(getattr(row, "median_cost_per_hour", 0.0) or 0.0),
                    "job_count_with_task": int(getattr(row, "job_count_with_task", 0) or 0),
                }
            )
    line["task_mix"] = tasks

    if tasks:
        _rebalance_task_mix_to_line_hours(line)

    return line


def _build_blank_line(default_rate: float, default_cost_per_hour: float) -> dict[str, Any]:
    """Create a blank/manual quote line."""
    return {
        "line_id": str(uuid4()),
        "manual": True,
        "department": "",
        "category": "",
        "hours_p25": 0.0,
        "hours_p50": 0.0,
        "hours_p75": 0.0,
        "line_hours_override": 0.0,
        "rate_suggested": float(default_rate),
        "rate_override": float(default_rate),
        "cost_per_hour_peer": float(default_cost_per_hour),
        "peer_job_count": 0,
        "recency_window": "manual",
        "recency_months": 0,
        "task_mix": [],
        "peer_job_hours": [],
    }


def _build_export_df(quote_lines: list[dict[str, Any]], hours_basis: str) -> pd.DataFrame:
    """Create task-level export rows across all quote lines."""
    rows: list[dict[str, Any]] = []

    for line in quote_lines:
        rate = _effective_line_rate(line)
        peer_count = int(_safe_float(line.get("peer_job_count", 0), 0))
        tasks = line.get("task_mix") or []

        if tasks:
            for task in tasks:
                task_hours = _effective_task_hours(task, hours_basis)
                rows.append(
                    {
                        "department": line.get("department", ""),
                        "category": line.get("category", ""),
                        "task_name": task.get("task_name", ""),
                        "quoted_hours": float(task_hours),
                        "rate_per_hour": float(rate),
                        "quoted_value": float(task_hours * rate),
                        "peer_job_count": peer_count,
                        "percentile_basis": hours_basis,
                    }
                )
        else:
            line_hours = _effective_line_hours(line, hours_basis)
            rows.append(
                {
                    "department": line.get("department", ""),
                    "category": line.get("category", ""),
                    "task_name": "Manual line",
                    "quoted_hours": float(line_hours),
                    "rate_per_hour": float(rate),
                    "quoted_value": float(line_hours * rate),
                    "peer_job_count": peer_count,
                    "percentile_basis": hours_basis,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "department",
            "category",
            "task_name",
            "quoted_hours",
            "rate_per_hour",
            "quoted_value",
            "peer_job_count",
            "percentile_basis",
        ],
    )


def _summary_text(
    client_name: str,
    job_name: str,
    economics: dict[str, Any],
    risk_indicators: list[dict[str, Any]],
) -> str:
    """Create clipboard-friendly quote summary text."""
    lines = [
        f"Client: {client_name or '—'}",
        f"Job: {job_name or '—'}",
        f"Total Hours: {fmt_hours(economics.get('total_hours', 0.0))} hrs",
        f"Total Quoted Value: {fmt_currency(economics.get('total_value', 0.0))}",
        f"Blended Rate: {fmt_rate(economics.get('blended_rate', np.nan))}",
        f"Est. Labour Cost: {fmt_currency(economics.get('est_labour_cost', 0.0))}",
        (
            "Est. Margin: "
            f"{fmt_currency(economics.get('est_margin', 0.0))} "
            f"({fmt_percent(economics.get('est_margin_pct', np.nan))})"
        ),
    ]

    if risk_indicators:
        lines.append("")
        lines.append("Risk Indicators:")
        for item in risk_indicators:
            lines.append(f"- {item.get('message', '')}")

    return "\n".join(lines)


def _get_peer_jobs_for_pairs(
    df_scope: pd.DataFrame,
    pairs: list[tuple[str, str]],
    recency_months: int,
) -> pd.DataFrame:
    """Return peer jobs combined across one or more (department, category) pairs."""
    frames: list[pd.DataFrame] = []
    for department, category in pairs:
        pair_jobs = get_peer_jobs(
            df=df_scope,
            department=department,
            category=category,
            min_revenue=None,
            max_revenue=None,
            recency_months=recency_months,
        )
        if len(pair_jobs) == 0:
            continue
        frames.append(pair_jobs.copy())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _render_peer_lookup(df: pd.DataFrame, category_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render Section 1: Peer Job Lookup."""
    section_header("Peer Job Lookup", "Pull completed peer jobs by department, category, size, and recency.")

    if "department_final" not in df.columns:
        st.warning("Missing `department_final` in fact table.")
        return pd.DataFrame(), pd.DataFrame()

    client_col = _get_client_col(df)
    if client_col is None:
        st.warning("No client grouping field available (`client_group_rev_job` or `client`).")
        return pd.DataFrame(), pd.DataFrame()

    client_label = "Client Group" if client_col == "client_group_rev_job" else "Client"
    all_clients_label = "All Client Groups" if client_label == "Client Group" else "All Clients"
    client_options = sorted(df[client_col].dropna().astype(str).unique().tolist())
    client_options_with_all = [all_clients_label] + client_options
    current_clients_raw = get_state(STATE_KEYS["quote_lookup_client"])
    if isinstance(current_clients_raw, str):
        current_clients = [current_clients_raw] if current_clients_raw.strip() else []
    elif isinstance(current_clients_raw, list):
        current_clients = [str(c) for c in current_clients_raw]
    else:
        current_clients = []
    current_clients = [c for c in current_clients if c in client_options_with_all]
    if not current_clients:
        current_clients = [all_clients_label]
    if all_clients_label in current_clients and len(current_clients) > 1:
        current_clients = [all_clients_label]

    selected_clients_raw = st.multiselect(
        f"{client_label}(s)",
        options=client_options_with_all,
        default=current_clients,
        help=(
            f"Default is {all_clients_label}. "
            "Select one or more specific clients to narrow the peer pool."
        ),
    )

    if len(selected_clients_raw) == 0 or all_clients_label in selected_clients_raw:
        selected_clients_effective: list[str] = []
        selected_clients_state = [all_clients_label]
    else:
        selected_clients_effective = [v for v in selected_clients_raw if v != all_clients_label]
        selected_clients_state = selected_clients_effective

    if sorted(selected_clients_state) != sorted(current_clients):
        set_state(STATE_KEYS["quote_lookup_client"], selected_clients_state)
        set_state(STATE_KEYS["quote_lookup_dept"], None)
        set_state(STATE_KEYS["quote_lookup_cat"], None)
        set_state(STATE_KEYS["quote_lookup_min_rev"], None)
        set_state(STATE_KEYS["quote_lookup_max_rev"], None)
    else:
        set_state(STATE_KEYS["quote_lookup_client"], selected_clients_state)

    if len(selected_clients_effective) == 0:
        df_scope = df.copy()
    else:
        df_scope = df[df[client_col].astype(str).isin([str(v) for v in selected_clients_effective])].copy()

    if len(df_scope) == 0:
        st.info("No data found for the selected client scope.")
        return pd.DataFrame(), pd.DataFrame()

    departments = sorted(df_scope["department_final"].dropna().astype(str).unique().tolist())
    if not departments:
        st.info("No departments found in this client scope.")
        return pd.DataFrame(), pd.DataFrame()

    dept_options = ["All Departments"] + departments
    current_dept = get_state(STATE_KEYS["quote_lookup_dept"])
    if current_dept not in dept_options:
        current_dept = "All Departments"
    dept_index = dept_options.index(current_dept)
    selected_department = st.selectbox("Department", options=dept_options, index=dept_index)

    if selected_department != current_dept:
        set_state(STATE_KEYS["quote_lookup_dept"], selected_department)
        set_state(STATE_KEYS["quote_lookup_cat"], None)
        set_state(STATE_KEYS["quote_lookup_min_rev"], None)
        set_state(STATE_KEYS["quote_lookup_max_rev"], None)
    else:
        set_state(STATE_KEYS["quote_lookup_dept"], selected_department)

    selected_pairs: list[tuple[str, str]] = []
    current_cat = get_state(STATE_KEYS["quote_lookup_cat"])
    if selected_department == "All Departments":
        depcat_pairs = (
            df_scope[["department_final", category_col]]
            .dropna(subset=["department_final", category_col])
            .drop_duplicates()
            .copy()
        )
        depcat_pairs["department_final"] = depcat_pairs["department_final"].astype(str)
        depcat_pairs[category_col] = depcat_pairs[category_col].astype(str)
        depcat_pairs["pair_label"] = (
            depcat_pairs["department_final"] + " -> " + depcat_pairs[category_col]
        )
        depcat_pairs = depcat_pairs.sort_values("pair_label")

        pair_labels = depcat_pairs["pair_label"].tolist()
        pair_map = {
            row["pair_label"]: (row["department_final"], row[category_col])
            for _, row in depcat_pairs.iterrows()
        }
        stored_selected: list[str]
        if isinstance(current_cat, list):
            stored_selected = [v for v in current_cat if isinstance(v, str)]
        elif isinstance(current_cat, str):
            stored_selected = [current_cat]
        else:
            stored_selected = []
        stored_selected = [v for v in stored_selected if v in (["All Categories"] + pair_labels)]
        if not stored_selected:
            stored_selected = ["All Categories"]

        selected_category_labels = st.multiselect(
            "Categories",
            options=["All Categories"] + pair_labels,
            default=stored_selected,
            help="When all departments are selected, choose one or more categories, or select all categories.",
        )
        if not selected_category_labels:
            set_state(STATE_KEYS["quote_lookup_cat"], [])
            selected_pairs = []
        else:
            if "All Categories" in selected_category_labels:
                selected_pairs = [pair_map[label] for label in pair_labels]
                set_state(STATE_KEYS["quote_lookup_cat"], ["All Categories"])
                st.caption(f"Using all {len(pair_labels)} categories across all departments.")
            else:
                selected_pairs = [pair_map[label] for label in selected_category_labels if label in pair_map]
                set_state(STATE_KEYS["quote_lookup_cat"], selected_category_labels)
    else:
        dept_df = df_scope[df_scope["department_final"].astype(str) == str(selected_department)].copy()
        categories = sorted(dept_df[category_col].dropna().astype(str).unique().tolist())
        category_options = ["Select category"] + categories
        current_cat_single = current_cat if isinstance(current_cat, str) else None
        cat_index = category_options.index(current_cat_single) if current_cat_single in category_options else 0
        selected_category_raw = st.selectbox("Category", options=category_options, index=cat_index)
        selected_category = None if selected_category_raw == "Select category" else selected_category_raw
        set_state(STATE_KEYS["quote_lookup_cat"], selected_category)
        if selected_category:
            selected_pairs = [(str(selected_department), str(selected_category))]

    recency_values = [v for _, v in RECENCY_CHOICES]
    current_recency = int(get_state(STATE_KEYS["quote_lookup_recency"]) or 24)
    recency_index = recency_values.index(current_recency) if current_recency in recency_values else 1
    selected_recency = st.selectbox(
        "Recency Window",
        options=recency_values,
        index=recency_index,
        format_func=lambda value: "All Time" if value == 0 else f"{value}m",
    )
    set_state(STATE_KEYS["quote_lookup_recency"], selected_recency)

    if len(selected_pairs) == 0:
        st.info("Select a category to see peer-job benchmarks.")
        return pd.DataFrame(), pd.DataFrame()

    base_peer_jobs = _get_peer_jobs_for_pairs(
        df_scope=df_scope,
        pairs=selected_pairs,
        recency_months=selected_recency,
    )

    if len(base_peer_jobs) == 0:
        st.info("No peer data for this department/category/recency selection.")
        return pd.DataFrame(), pd.DataFrame()

    quote_min_bound = float(base_peer_jobs["quoted_amount"].min())
    quote_max_bound = float(base_peer_jobs["quoted_amount"].max())

    stored_min = get_state(STATE_KEYS["quote_lookup_min_rev"])
    stored_max = get_state(STATE_KEYS["quote_lookup_max_rev"])

    band_min = _safe_float(stored_min, quote_min_bound)
    band_max = _safe_float(stored_max, quote_max_bound)
    band_min = max(quote_min_bound, min(band_min, quote_max_bound))
    band_max = max(quote_min_bound, min(band_max, quote_max_bound))
    if band_max < band_min:
        band_min, band_max = quote_min_bound, quote_max_bound

    if quote_max_bound > quote_min_bound:
        step = max((quote_max_bound - quote_min_bound) / 200.0, 1.0)
        value_band = st.slider(
            "Job Value Band (Quoted Value)",
            min_value=float(quote_min_bound),
            max_value=float(quote_max_bound),
            value=(float(band_min), float(band_max)),
            step=float(step),
            format="$%.0f",
        )
        band_min, band_max = float(value_band[0]), float(value_band[1])
    else:
        st.caption(f"Peer job quoted value is constant in this pool: {fmt_currency(quote_min_bound)}")
        band_min, band_max = quote_min_bound, quote_max_bound

    set_state(STATE_KEYS["quote_lookup_min_rev"], float(band_min))
    set_state(STATE_KEYS["quote_lookup_max_rev"], float(band_max))

    peer_jobs = _get_peer_jobs_for_pairs(
        df_scope=df_scope,
        pairs=selected_pairs,
        recency_months=selected_recency,
    )
    peer_jobs = peer_jobs[
        peer_jobs["quoted_amount"].between(float(band_min), float(band_max), inclusive="both")
    ].copy()

    if len(peer_jobs) == 0:
        st.info("No peer jobs in this value band. Widen the revenue range.")
        return pd.DataFrame(), pd.DataFrame()

    description_source = None
    if "job_description" in df_scope.columns:
        description_source = "job_description"
    elif "job_name" in df_scope.columns:
        description_source = "job_name"

    peer_jobs = peer_jobs.copy()
    peer_jobs["job_no"] = peer_jobs["job_no"].astype(str)
    if description_source is not None:
        description_lookup = (
            df_scope[["job_no", description_source]]
            .dropna(subset=["job_no"])
            .copy()
        )
        description_lookup["job_no"] = description_lookup["job_no"].astype(str)
        description_lookup["job_description"] = (
            description_lookup[description_source].fillna("").astype(str)
        )
        description_lookup = (
            description_lookup.groupby("job_no", dropna=False)["job_description"]
            .agg(_first_non_empty)
            .reset_index()
        )
        peer_jobs = peer_jobs.merge(description_lookup, on="job_no", how="left")
    else:
        peer_jobs["job_description"] = ""

    keyword_col, mode_col = st.columns([3, 1.2])
    keyword_input = keyword_col.text_input(
        "Job Description Keyword Filter",
        key="quote_lookup_keyword_filter",
        placeholder="e.g. martech, lifecycle strategy, crm implementation",
        help="Use comma-separated keywords/phrases to narrow peer jobs by job description text.",
    )
    keyword_mode = mode_col.selectbox(
        "Match Mode",
        options=["Match any", "Match all"],
        key="quote_lookup_keyword_mode",
    )

    if description_source is None:
        st.caption("No `job_description` or `job_name` field available, so keyword filtering is unavailable.")
    elif description_source != "job_description":
        st.caption("Using `job_name` as fallback text source for keyword matching.")

    peer_jobs_filtered = peer_jobs.copy()
    parsed_keywords = _parse_keyword_query(keyword_input)
    keyword_terms = parsed_keywords["keywords"]
    keyword_phrases = parsed_keywords["phrases"]
    ignored_tokens = parsed_keywords["ignored_tokens"]
    has_keyword_delimiter = bool(re.search(r"[,\n;|]", keyword_input or ""))

    if keyword_input.strip():
        if not has_keyword_delimiter:
            st.warning("Use comma-separated keywords for best matching. Without commas, input is treated as one phrase.")
        preview_terms = ", ".join(keyword_terms[:18]) if keyword_terms else "—"
        if len(keyword_terms) > 18:
            preview_terms = f"{preview_terms}, …"
        st.caption(f"Parsed keywords ({len(keyword_terms)}): {preview_terms}")
        if ignored_tokens:
            ignored_preview = ", ".join(ignored_tokens[:12])
            st.caption(f"Ignored connective/common words: {ignored_preview}")

    if description_source is not None and keyword_terms:
        before_count = int(peer_jobs_filtered["job_no"].nunique())
        norm_desc = peer_jobs_filtered["job_description"].fillna("").astype(str).apply(_normalize_text)
        desc_tokens = norm_desc.apply(lambda text: set(text.split()) if text else set())

        term_hits = pd.Series(
            [
                [
                    term
                    for term in keyword_terms
                    if ((" " in term and term in text) or (" " not in term and term in tokens))
                ]
                for text, tokens in zip(norm_desc.tolist(), desc_tokens.tolist())
            ],
            index=peer_jobs_filtered.index,
        )
        phrase_hits = norm_desc.apply(
            lambda text: [phrase for phrase in keyword_phrases if phrase in text]
        )

        if keyword_mode == "Match all":
            required_terms = len(keyword_terms)
            # Practical "all": long keyword lists become unrealistic; use 70% coverage.
            if required_terms >= 8:
                required_terms = int(np.ceil(required_terms * 0.7))
            mask = term_hits.apply(len) >= required_terms
        else:
            mask = (term_hits.apply(len) > 0) | (phrase_hits.apply(len) > 0)

        peer_jobs_filtered["keyword_hits"] = term_hits.apply(lambda hits: ", ".join(hits[:8]))
        peer_jobs_filtered["keyword_hit_count"] = term_hits.apply(len)
        peer_jobs_filtered = peer_jobs_filtered[mask].copy()
        after_count = int(peer_jobs_filtered["job_no"].nunique())
        st.caption(f"Keyword filter matched {after_count:,} of {before_count:,} peer jobs.")

    if len(peer_jobs_filtered) == 0:
        st.info("No peer jobs matched the keyword filter. Broaden or clear your keywords.")
        return pd.DataFrame(), pd.DataFrame()

    pair_key_values = {f"{dept}||{cat}" for dept, cat in selected_pairs}
    peer_job_nos = peer_jobs_filtered["job_no"].astype(str).tolist()
    dept_task_scope = df_scope.copy()
    dept_task_scope["job_no"] = dept_task_scope["job_no"].astype(str)
    dept_task_scope["_pair_key"] = (
        dept_task_scope["department_final"].astype(str)
        + "||"
        + dept_task_scope[category_col].astype(str)
    )
    task_mask = (
        dept_task_scope["_pair_key"].isin(pair_key_values)
        & dept_task_scope["job_no"].isin(peer_job_nos)
    )
    scoped_for_tasks = dept_task_scope[task_mask].copy()
    if "_pair_key" in scoped_for_tasks.columns:
        scoped_for_tasks = scoped_for_tasks.drop(columns=["_pair_key"])
    task_mix = get_peer_task_mix(scoped_for_tasks, peer_job_nos)

    summary = get_peer_pool_summary(peer_jobs_filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed Jobs", f"{int(summary.get('n_jobs', 0)):,}")
    c2.metric("Median Hours", f"{fmt_hours(summary.get('hours_p50', 0.0))} hrs")
    c3.metric("P25 Hours", f"{fmt_hours(summary.get('hours_p25', 0.0))} hrs")
    c4.metric("P75 Hours", f"{fmt_hours(summary.get('hours_p75', 0.0))} hrs")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Median Revenue", fmt_currency(summary.get("median_revenue", 0.0)))
    c6.metric("Median Margin %", fmt_percent(summary.get("median_margin_pct", np.nan)))
    c7.metric("Median Cost/Hr", fmt_rate(summary.get("blended_cost_per_hour", np.nan)))
    c8.metric("Median Realised Rate", fmt_rate(summary.get("blended_realised_rate", np.nan)))

    st.caption(f"Hours P80: {fmt_hours(summary.get('hours_p80', 0.0))} hrs")

    st.markdown("**Task Mix (empirical)**")
    if len(task_mix) == 0:
        st.info("No peer task mix available for these jobs.")
    else:
        mix_view = task_mix[
            [
                "task_name",
                "share_pct",
                "hours_p50",
                "hours_p25",
                "hours_p75",
                "job_count_with_task",
            ]
        ].copy()

        st.dataframe(
            mix_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "task_name": st.column_config.TextColumn("Task"),
                "share_pct": st.column_config.NumberColumn("Share %", format="%.1f%%"),
                "hours_p50": st.column_config.NumberColumn("Median Hrs", format="%.1f hrs"),
                "hours_p25": st.column_config.NumberColumn("P25", format="%.1f hrs"),
                "hours_p75": st.column_config.NumberColumn("P75", format="%.1f hrs"),
                "job_count_with_task": st.column_config.NumberColumn("Jobs", format="%d"),
            },
        )

    with st.expander("Peer Job Detail", expanded=False):
        detail_cols = [
            "department_final",
            "category",
            "job_no",
            "client",
            "total_hours",
            "total_revenue",
            "total_cost",
            "margin_pct",
            "job_completed_date",
        ]
        if "keyword_hit_count" in peer_jobs_filtered.columns:
            detail_cols.extend(["keyword_hit_count", "keyword_hits"])
        if "job_description" in peer_jobs_filtered.columns:
            detail_cols.append("job_description")
        detail = peer_jobs_filtered[detail_cols].copy()
        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True,
            column_config={
                "department_final": st.column_config.TextColumn("Department"),
                "category": st.column_config.TextColumn("Category"),
                "job_no": st.column_config.TextColumn("Job"),
                "client": st.column_config.TextColumn("Client"),
                "total_hours": st.column_config.NumberColumn("Total Hrs", format="%.1f hrs"),
                "total_revenue": st.column_config.NumberColumn("Revenue", format="$%.0f"),
                "total_cost": st.column_config.NumberColumn("Cost", format="$%.0f"),
                "margin_pct": st.column_config.NumberColumn("Margin %", format="%.1f%%"),
                "job_completed_date": st.column_config.DatetimeColumn("Completed"),
                "keyword_hit_count": st.column_config.NumberColumn("Keyword Hits", format="%d"),
                "keyword_hits": st.column_config.TextColumn("Matched Keywords"),
                "job_description": st.column_config.TextColumn("Job Description"),
            },
        )

    if st.button("+ Add to Quote", type="primary"):
        quote_lines = list(st.session_state.get(STATE_KEYS["quote_lines"], []))
        lines_added = 0
        pair_lookup = {(dept, cat) for dept, cat in selected_pairs}
        for department, category in sorted(pair_lookup):
            pair_peer_jobs = peer_jobs_filtered[
                (peer_jobs_filtered["department_final"].astype(str) == str(department))
                & (peer_jobs_filtered["category"].astype(str) == str(category))
            ].copy()
            if len(pair_peer_jobs) == 0:
                continue

            pair_job_nos = pair_peer_jobs["job_no"].astype(str).tolist()
            pair_task_scope = df_scope.copy()
            pair_task_scope["job_no"] = pair_task_scope["job_no"].astype(str)
            pair_task_scope = pair_task_scope[
                (pair_task_scope["department_final"].astype(str) == str(department))
                & (pair_task_scope[category_col].astype(str) == str(category))
                & (pair_task_scope["job_no"].isin(pair_job_nos))
            ].copy()
            pair_task_mix = get_peer_task_mix(pair_task_scope, pair_job_nos)

            new_line = _build_quote_line_from_peers(
                department=str(department),
                category=str(category),
                peer_jobs=pair_peer_jobs,
                task_mix=pair_task_mix,
                recency_months=selected_recency,
            )
            quote_lines.append(new_line)
            lines_added += 1

        st.session_state[STATE_KEYS["quote_lines"]] = quote_lines
        if lines_added == 0:
            st.warning("No categories with peer jobs were available to add.")
        elif lines_added == 1:
            st.success("Added 1 line item to quote.")
        else:
            st.success(f"Added {lines_added} line items to quote.")
        st.rerun()

    return peer_jobs_filtered, task_mix


def _render_quote_composer(
    quote_lines: list[dict[str, Any]],
    default_manual_rate: float,
    default_manual_cost_per_hour: float,
) -> None:
    """Render Section 2: Quote Composer."""
    section_header("Quote Composer", "Build multi-department line items with empirical defaults and overrides.")

    current_basis = get_state(STATE_KEYS["hours_basis"])
    if current_basis not in HOURS_BASIS_OPTIONS:
        current_basis = "p50"
    basis_idx = HOURS_BASIS_OPTIONS.index(current_basis)

    selected_basis = st.radio(
        "Hours Basis",
        options=HOURS_BASIS_OPTIONS,
        index=basis_idx,
        format_func=lambda key: HOURS_BASIS_LABELS.get(key, key),
        horizontal=True,
    )
    set_state(STATE_KEYS["hours_basis"], selected_basis)

    actions = st.columns([1, 4])
    with actions[0]:
        if st.button("+ Add Blank Line"):
            updated = list(quote_lines)
            updated.append(
                _build_blank_line(
                    default_rate=default_manual_rate,
                    default_cost_per_hour=default_manual_cost_per_hour,
                )
            )
            st.session_state[STATE_KEYS["quote_lines"]] = updated
            st.rerun()

    if not quote_lines:
        st.info("No line items yet. Add one from Peer Job Lookup or create a blank line.")
        return

    h1, h2, h3, h4, h5, h6 = st.columns([0.5, 2.1, 2.2, 1.2, 1.2, 0.8])
    h1.markdown("**#**")
    h2.markdown("**Department**")
    h3.markdown("**Category**")
    h4.markdown("**Hours (est)**")
    h5.markdown("**Rate ($/hr)**")
    h6.markdown("**Remove**")

    remove_index: int | None = None

    for idx, line in enumerate(quote_lines):
        line_id = str(line.get("line_id", f"line_{idx}"))

        c1, c2, c3, c4, c5, c6 = st.columns([0.5, 2.1, 2.2, 1.2, 1.2, 0.8])
        c1.write(idx + 1)

        if line.get("manual", False):
            line["department"] = c2.text_input(
                f"Department {idx + 1}",
                value=str(line.get("department", "")),
                key=f"line_department_{line_id}",
                label_visibility="collapsed",
                placeholder="Department",
            )
            line["category"] = c3.text_input(
                f"Category {idx + 1}",
                value=str(line.get("category", "")),
                key=f"line_category_{line_id}",
                label_visibility="collapsed",
                placeholder="Category",
            )
        else:
            c2.markdown(str(line.get("department", "—")))
            peer_count = int(_safe_float(line.get("peer_job_count", 0), 0))
            c3.markdown(
                f"{line.get('category', '—')}<br>{_peer_badge_html(peer_count)}",
                unsafe_allow_html=True,
            )

        line_hours = _effective_line_hours(line, selected_basis)
        line_hours_key_suffix = "ovr" if _line_has_overrides(line) else selected_basis
        line_hours_seed = int(round(line_hours * 10))
        line_hours_key = f"line_hours_{line_id}_{line_hours_key_suffix}_{line_hours_seed}"
        edited_line_hours = c4.number_input(
            f"Hours {idx + 1}",
            min_value=0.0,
            value=float(round(line_hours, 2)),
            step=0.5,
            key=line_hours_key,
            label_visibility="collapsed",
        )
        if abs(edited_line_hours - line_hours) > 1e-6:
            _apply_line_total_override(line, edited_line_hours, selected_basis)

        suggested_rate = _safe_float(line.get("rate_suggested", 0.0), 0.0)
        effective_rate = _effective_line_rate(line)
        edited_rate = c5.number_input(
            f"Rate {idx + 1}",
            min_value=0.0,
            value=float(round(effective_rate, 2)),
            step=1.0,
            key=f"line_rate_{line_id}",
            label_visibility="collapsed",
        )
        if line.get("manual", False):
            line["rate_override"] = float(edited_rate)
        else:
            if abs(edited_rate - suggested_rate) < 1e-6:
                line["rate_override"] = None
            else:
                line["rate_override"] = float(edited_rate)

        if c6.button("✕", key=f"remove_line_{line_id}"):
            remove_index = idx

        with st.expander(
            f"Line {idx + 1}: {line.get('department', '—')} → {line.get('category', '—')} "
            f"({fmt_hours(_effective_line_hours(line, selected_basis))} hrs)",
            expanded=False,
        ):
            tasks = line.get("task_mix") or []

            reset_cols = st.columns([1, 5])
            with reset_cols[0]:
                if tasks and st.button("Reset Task Overrides", key=f"reset_tasks_{line_id}"):
                    for task in tasks:
                        task["hours_override"] = None
                    line["line_hours_override"] = None
                    st.rerun()

            if not tasks:
                st.caption("No empirical task mix for this line. Use line-level hours and rate inputs.")
            else:
                t1, t2, t3, t4 = st.columns([2.6, 1, 1, 1.2])
                t1.markdown("**Task**")
                t2.markdown("**Emp. Share**")
                t3.markdown(f"**Emp. Hrs ({selected_basis.upper()})**")
                t4.markdown("**Override Hrs**")

                for task_idx, task in enumerate(tasks):
                    row1, row2, row3, row4 = st.columns([2.6, 1, 1, 1.2])
                    row1.write(str(task.get("task_name", "Unspecified")))
                    share = _safe_float(task.get("share_pct", np.nan), np.nan)
                    row2.write(fmt_percent(share))

                    empirical_hours = _safe_float(task.get(f"hours_{selected_basis}", 0.0), 0.0)
                    row3.write(f"{fmt_hours(empirical_hours)} hrs")

                    task_key_suffix = "ovr" if task.get("hours_override") is not None else selected_basis
                    effective_task_hours = _effective_task_hours(task, selected_basis)
                    task_hours_seed = int(round(effective_task_hours * 10))
                    task_key = (
                        f"task_override_{line_id}_{task_idx}_{task_key_suffix}_{task_hours_seed}"
                    )
                    edited_task_hours = row4.number_input(
                        f"Task {idx + 1}-{task_idx + 1}",
                        min_value=0.0,
                        value=float(round(effective_task_hours, 2)),
                        step=0.5,
                        key=task_key,
                        label_visibility="collapsed",
                    )
                    if abs(edited_task_hours - effective_task_hours) > 1e-6:
                        task["hours_override"] = float(edited_task_hours)

                add_task_cols = st.columns([2.6, 1, 1.2, 1])
                custom_task_name = add_task_cols[0].text_input(
                    "Custom Task",
                    value="",
                    key=f"custom_task_name_{line_id}",
                    placeholder="Custom task name",
                    label_visibility="collapsed",
                )
                custom_task_hours = add_task_cols[2].number_input(
                    "Custom Task Hours",
                    min_value=0.0,
                    value=0.0,
                    step=0.5,
                    key=f"custom_task_hours_{line_id}",
                    label_visibility="collapsed",
                )
                if add_task_cols[3].button("Add", key=f"add_custom_task_{line_id}"):
                    if custom_task_name.strip():
                        tasks.append(
                            {
                                "task_name": custom_task_name.strip(),
                                "share_pct": np.nan,
                                "hours_p25": float(custom_task_hours),
                                "hours_p50": float(custom_task_hours),
                                "hours_p75": float(custom_task_hours),
                                "hours_override": float(custom_task_hours),
                                "median_cost_per_hour": _safe_float(line.get("cost_per_hour_peer"), 0.0),
                                "job_count_with_task": 0,
                            }
                        )
                        st.rerun()

                total_empirical = float(sum(_safe_float(task.get(f"hours_{selected_basis}"), 0.0) for task in tasks))
                total_override = float(sum(_effective_task_hours(task, selected_basis) for task in tasks))
                st.markdown(
                    f"**TOTAL** · Empirical: {fmt_hours(total_empirical)} hrs"
                    f" · Override: {fmt_hours(total_override)} hrs"
                )

        st.divider()

    if remove_index is not None:
        updated = list(quote_lines)
        updated.pop(remove_index)
        st.session_state[STATE_KEYS["quote_lines"]] = updated
        st.rerun()

    st.session_state[STATE_KEYS["quote_lines"]] = quote_lines


def _render_economics_preview(df: pd.DataFrame, quote_lines: list[dict[str, Any]]) -> None:
    """Render Section 3: Economics Preview."""
    section_header("Economics Preview", "Understand value, estimated labour cost, margin, and risk before finalising.")

    client_col, job_col = st.columns(2)

    current_client = str(get_state(STATE_KEYS["quote_client_name"]) or "")
    current_job = str(get_state(STATE_KEYS["quote_job_name"]) or "")

    quote_client_name = client_col.text_input("Client", value=current_client)
    quote_job_name = job_col.text_input("Job Name", value=current_job)

    set_state(STATE_KEYS["quote_client_name"], quote_client_name)
    set_state(STATE_KEYS["quote_job_name"], quote_job_name)

    if not quote_lines:
        st.info("Add quote lines to preview economics.")
        return

    hours_basis = str(get_state(STATE_KEYS["hours_basis"]) or "p50")
    economics = compute_quote_economics(quote_lines=quote_lines, hours_basis=hours_basis)

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Hours", f"{fmt_hours(economics.get('total_hours', 0.0))} hrs")
    s2.metric("Total Quoted Value", fmt_currency(economics.get("total_value", 0.0)))
    s3.metric("Blended Rate", fmt_rate(economics.get("blended_rate", np.nan)))

    s4, s5 = st.columns(2)
    s4.metric("Est. Labour Cost", fmt_currency(economics.get("est_labour_cost", 0.0)))
    s5.metric(
        "Est. Margin",
        f"{fmt_currency(economics.get('est_margin', 0.0))} ({fmt_percent(economics.get('est_margin_pct', np.nan))})",
    )

    st.markdown("**Department Breakdown**")
    line_df = pd.DataFrame(economics.get("lines", []))
    if len(line_df) > 0:
        dept_breakdown = line_df.groupby("department", dropna=False).agg(
            hours=("hours", "sum"),
            value=("value", "sum"),
            cost=("cost", "sum"),
            peer_count=("peer_count", "max"),
        ).reset_index()
        dept_breakdown["margin_pct"] = np.where(
            dept_breakdown["value"] > 0,
            (dept_breakdown["value"] - dept_breakdown["cost"]) / dept_breakdown["value"] * 100,
            np.nan,
        )
        dept_breakdown = dept_breakdown[["department", "hours", "value", "margin_pct", "peer_count"]]

        st.dataframe(
            dept_breakdown,
            use_container_width=True,
            hide_index=True,
            column_config={
                "department": st.column_config.TextColumn("Dept"),
                "hours": st.column_config.NumberColumn("Hrs", format="%.1f hrs"),
                "value": st.column_config.NumberColumn("Value", format="$%.0f"),
                "margin_pct": st.column_config.NumberColumn("Margin %", format="%.1f%%"),
                "peer_count": st.column_config.NumberColumn("Peer", format="%d"),
            },
        )

    st.markdown("**Risk Indicators**")
    risk_indicators = compute_risk_indicators(quote_lines=quote_lines, hours_basis=hours_basis)
    if not risk_indicators:
        st.success("No risk indicators triggered.")
    else:
        for risk in risk_indicators:
            level = risk.get("level")
            message = str(risk.get("message", ""))
            if level == "danger":
                st.error(f"🔴 {message}")
            elif level == "warning":
                st.warning(f"⚠ {message}")
            else:
                st.success(f"✅ {message}")

    st.markdown("**Historical Comparison**")
    selected_departments = sorted(
        {
            str(line.get("department", "")).strip()
            for line in quote_lines
            if str(line.get("department", "")).strip()
        }
    )

    completed = _completed_jobs(df)
    if len(completed) == 0 or not selected_departments:
        st.info("Not enough completed job history for historical comparison.")
    else:
        scoped = completed[completed["department_final"].astype(str).isin(selected_departments)].copy()

        recency_months = int(get_state(STATE_KEYS["quote_lookup_recency"]) or 24)
        if recency_months > 0:
            recency_col = pd.Series(pd.NaT, index=scoped.index)
            if "job_completed_date" in scoped.columns:
                recency_col = scoped["job_completed_date"]
            if "month_key" in scoped.columns:
                recency_col = recency_col.where(recency_col.notna(), scoped["month_key"])
            if recency_col.notna().any():
                anchor = recency_col.max()
                cutoff = anchor - pd.DateOffset(months=recency_months)
                scoped = scoped[recency_col >= cutoff].copy()

        hours_dist = scoped.groupby("job_no", dropna=False).agg(job_total_hours=("hours_raw", "sum")).reset_index()

        if len(hours_dist) < 2:
            st.info("Need at least 2 peer jobs for a distribution chart.")
        else:
            total_quote_hours = float(economics.get("total_hours", 0.0))
            percentile_rank = float((hours_dist["job_total_hours"] <= total_quote_hours).mean() * 100)

            nbins = int(min(40, max(10, round(np.sqrt(len(hours_dist))))))
            fig = px.histogram(
                hours_dist,
                x="job_total_hours",
                nbins=nbins,
                labels={"job_total_hours": "Peer Job Total Hours"},
                title="Peer Job Total Hours Distribution",
            )
            fig.add_vline(
                x=total_quote_hours,
                line_color="#d62728",
                line_dash="dash",
                line_width=3,
                annotation_text="Your quote",
                annotation_position="top",
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Your quote of {fmt_hours(total_quote_hours)} hrs sits at the "
                f"{percentile_rank:.1f}th percentile of similar jobs."
            )

    export_df = _build_export_df(quote_lines=quote_lines, hours_basis=hours_basis)

    download_col, copy_col = st.columns(2)
    with download_col:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "📥 Export Quote to CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"quote_export_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with copy_col:
        if st.button("📋 Copy Summary", use_container_width=True):
            st.session_state["quote_summary_text"] = _summary_text(
                client_name=quote_client_name,
                job_name=quote_job_name,
                economics=economics,
                risk_indicators=risk_indicators,
            )

    if st.session_state.get("quote_summary_text"):
        st.text_area(
            "Summary",
            value=st.session_state["quote_summary_text"],
            height=180,
        )


def main() -> None:
    """Render Quote Builder page."""
    st.title("Quote Builder")
    st.caption(
        "Empirical peer-job quoting engine grounded in completed-job hours, cost, "
        "revenue, and margin outcomes."
    )

    fact_columns = [
        "department_final",
        "category_rev_job",
        "job_category",
        "task_name",
        "job_no",
        "staff_name",
        "hours_raw",
        "base_cost",
        "rev_alloc",
        "quoted_time_total",
        "quoted_amount_total",
        "quote_match_flag",
        "month_key",
        "is_billable",
        "job_status",
        "job_completed_date",
        "client",
        "client_group_rev_job",
        "client_group",
        "job_name",
        "job_description",
        "work_date",
    ]
    df = load_fact_timesheet(columns=fact_columns)

    if len(df) == 0:
        st.warning("No timesheet data available.")
        return

    category_col = get_category_col(df)

    peer_jobs, _task_mix = _render_peer_lookup(df=df, category_col=category_col)

    st.markdown("---")

    quote_lines = list(st.session_state.get(STATE_KEYS["quote_lines"], []))
    total_hours = float(pd.to_numeric(df.get("hours_raw", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_cost = float(pd.to_numeric(df.get("base_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_revenue = float(pd.to_numeric(df.get("rev_alloc", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    default_manual_rate = (total_revenue / total_hours) if total_hours > 0 else 120.0
    default_manual_cost_per_hour = (total_cost / total_hours) if total_hours > 0 else 60.0

    _render_quote_composer(
        quote_lines=quote_lines,
        default_manual_rate=default_manual_rate,
        default_manual_cost_per_hour=default_manual_cost_per_hour,
    )

    st.markdown("---")

    quote_lines = list(st.session_state.get(STATE_KEYS["quote_lines"], []))
    _render_economics_preview(df=df, quote_lines=quote_lines)

    if len(peer_jobs) == 0:
        st.caption("Tip: Start in Peer Job Lookup and add at least one empirical line item to build the quote.")


if __name__ == "__main__":
    main()
