"""
Quote Builder Page

Build quote templates from historical benchmarks with task recommendations,
economics preview, and export capabilities.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import (
    init_state, get_state, set_state,
    QuotePlan, QuotePlanTask, get_quote_plan, set_quote_plan
)
from src.ui.layout import section_header
from src.ui.formatting import (
    fmt_currency,
    fmt_hours,
    fmt_percent,
    fmt_rate,
    build_job_name_lookup,
    format_job_label,
)
from src.data.loader import load_fact_timesheet, load_mart
from src.data.semantic import safe_quote_job_task, get_category_col
from src.data.cohorts import (
    filter_by_time_window, filter_active_staff,
    compute_recency_weights, get_benchmark_metadata
)
from src.config import config


st.set_page_config(page_title="Quote Builder", page_icon="📝", layout="wide")

init_state()


def get_task_benchmarks(df: pd.DataFrame, department: str, category: str,
                        recency_weighted: bool = False) -> pd.DataFrame:
    """
    Get task benchmarks for a department/category combination.
    """
    # Filter to department and category
    df_dept = df[df["department_final"] == department] if "department_final" in df.columns else df
    category_col = get_category_col(df_dept)
    mask = (df["department_final"] == department) & (df[category_col] == category)
    df_slice = df[mask].copy()
    
    if len(df_slice) == 0:
        return pd.DataFrame()
    
    # Get job-task level quote data
    job_task = safe_quote_job_task(df_slice)
    
    if len(job_task) == 0:
        return pd.DataFrame()
    
    # Merge actual hours
    actuals = df_slice.groupby(["job_no", "task_name"]).agg(
        actual_hours=("hours_raw", "sum"),
        actual_cost=("base_cost", "sum")
    ).reset_index()
    
    job_task = job_task.merge(actuals, on=["job_no", "task_name"], how="left")
    
    # Apply recency weighting if enabled
    if recency_weighted:
        # Get month for each job-task (use first activity)
        first_month = df_slice.groupby(["job_no", "task_name"])["month_key"].min().reset_index()
        first_month.columns = ["job_no", "task_name", "first_month"]
        job_task = job_task.merge(first_month, on=["job_no", "task_name"], how="left")
        
        job_task["weight"] = compute_recency_weights(
            job_task, date_col="first_month",
            half_life_months=config.recency_half_life_months
        )
    else:
        job_task["weight"] = 1.0
    
    # Aggregate to task level with benchmarks
    task_stats = []
    
    for task in job_task["task_name"].unique():
        task_data = job_task[job_task["task_name"] == task]
        
        # Compute metrics
        n_jobs = task_data["job_no"].nunique()
        
        # Weighted percentiles for quoted hours
        quoted = task_data["quoted_time_total"].dropna()
        weights = task_data.loc[quoted.index, "weight"]
        
        if len(quoted) > 0:
            # Simple weighted stats
            quoted_p25 = quoted.quantile(0.25)
            quoted_p50 = quoted.quantile(0.50)
            quoted_p75 = quoted.quantile(0.75)
        else:
            quoted_p25 = quoted_p50 = quoted_p75 = 0
        
        # Actual hours stats
        actual = task_data["actual_hours"].dropna()
        if len(actual) > 0:
            actual_p50 = actual.quantile(0.50)
        else:
            actual_p50 = 0
        
        # Overrun probability
        if len(task_data) > 0 and "quoted_time_total" in task_data.columns:
            task_data_valid = task_data.dropna(subset=["quoted_time_total", "actual_hours"])
            if len(task_data_valid) > 0:
                overrun_rate = (task_data_valid["actual_hours"] > task_data_valid["quoted_time_total"] * 1.2).mean()
            else:
                overrun_rate = 0
        else:
            overrun_rate = 0
        
        # Cost per hour (median)
        if "actual_cost" in task_data.columns and "actual_hours" in task_data.columns:
            task_data["cost_per_hour"] = np.where(
                task_data["actual_hours"] > 0,
                task_data["actual_cost"] / task_data["actual_hours"],
                np.nan
            )
            cost_per_hour = task_data["cost_per_hour"].median()
        else:
            cost_per_hour = 0
        
        # Quote rate (median)
        if "quote_rate" in task_data.columns:
            quote_rate = task_data["quote_rate"].median()
        else:
            quote_rate = 0
        
        # Inclusion rate (what % of jobs in this slice have this task)
        total_jobs = df_slice["job_no"].nunique()
        inclusion_rate = n_jobs / total_jobs * 100 if total_jobs > 0 else 0
        
        task_stats.append({
            "task_name": task,
            "n_jobs": n_jobs,
            "inclusion_rate": inclusion_rate,
            "quoted_hours_p25": quoted_p25,
            "quoted_hours_p50": quoted_p50,
            "quoted_hours_p75": quoted_p75,
            "actual_hours_p50": actual_p50,
            "overrun_risk": overrun_rate * 100,
            "cost_per_hour": cost_per_hour,
            "quote_rate": quote_rate,
        })
    
    result = pd.DataFrame(task_stats)
    result = result.sort_values("inclusion_rate", ascending=False)
    
    return result


def main():
    st.title("Quote Builder")
    st.caption("Build quote templates from historical benchmarks")
    
    # Load data
    df = load_fact_timesheet()
    
    # =========================================================================
    # INPUT CONTROLS (LEFT RAIL)
    # =========================================================================
    col_inputs, col_main = st.columns([1, 3])
    
    with col_inputs:
        section_header("Configuration")
        
        # Department selection
        departments = sorted(df["department_final"].dropna().unique().tolist())
        selected_dept = st.selectbox(
            "Department",
            options=departments,
            key="quote_dept"
        )
        
        # Category selection (filtered by department)
        df_dept = df[df["department_final"] == selected_dept] if selected_dept else df
        category_col = get_category_col(df_dept)
        if selected_dept and category_col in df.columns:
            categories = sorted(
                df_dept[category_col]
                .dropna().unique().tolist()
            )
        else:
            categories = []
        
        selected_cat = st.selectbox(
            "Category",
            options=categories,
            key="quote_cat"
        )
        
        st.divider()
        
        # Benchmark window
        window_options = {
            "3m": "Last 3 months",
            "6m": "Last 6 months",
            "12m": "Last 12 months",
            "24m": "Last 24 months",
            "all": "All time",
        }
        
        benchmark_window = st.selectbox(
            "Benchmark Window",
            options=list(window_options.keys()),
            format_func=lambda x: window_options[x],
            index=2,  # Default to 12m
            key="quote_window"
        )
        
        # Recency weighting
        recency_weighted = st.checkbox(
            "Apply Recency Weighting",
            value=get_state("recency_weighted"),
            help="Weight recent jobs more heavily using exponential decay"
        )
        set_state("recency_weighted", recency_weighted)
        
        # Active staff only
        active_staff_only = st.checkbox(
            "Active Staff Only",
            value=get_state("active_staff_only"),
            help="Only include staff with recent activity"
        )
        set_state("active_staff_only", active_staff_only)
    
    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    with col_main:
        if not selected_dept or not selected_cat:
            st.info("Select a department and category to build a quote template.")
            return
        
        # Filter data
        df_filtered = filter_by_time_window(df, benchmark_window)
        
        if active_staff_only:
            df_filtered = filter_active_staff(df_filtered)
        
        # Get benchmark metadata
        df_filtered_dept = df_filtered[df_filtered["department_final"] == selected_dept]
        category_col = get_category_col(df_filtered_dept)
        mask = (df_filtered["department_final"] == selected_dept) & \
               (df_filtered[category_col] == selected_cat)
        df_slice = df_filtered[mask]

        def _reset_task_state():
            st.session_state["quote_task_locked"] = False
            st.session_state["quote_task_locked_table"] = None
            st.session_state["quote_econ_ready"] = False
            st.session_state["quote_task_table_key"] = None
            st.session_state["quote_bench_key"] = None

        compare_lock_key = (
            selected_dept,
            selected_cat,
            benchmark_window,
            bool(active_staff_only),
        )
        if st.session_state.get("quote_compare_lock_key") != compare_lock_key:
            st.session_state["quote_compare_lock_key"] = compare_lock_key
            st.session_state["quote_compare_locked"] = False
            st.session_state["quote_compare_locked_jobs"] = None
            st.session_state["quote_compare_pool_jobs"] = None
            _reset_task_state()

        # =====================================================================
        # COMPARABLE JOB FILTERS
        # =====================================================================
        section_header("Comparable Jobs", "Narrow benchmarks to specific clients or jobs")
        compare_locked = st.session_state.get("quote_compare_locked", False)

        if "client" in df_slice.columns:
            client_options = sorted(df_slice["client"].dropna().unique().tolist())
        else:
            client_options = []

        selected_clients = st.multiselect(
            "Client(s) to compare",
            options=client_options,
            default=client_options,
            disabled=compare_locked,
            key="quote_compare_clients",
        )

        if selected_clients:
            df_client_slice = df_slice[df_slice["client"].isin(selected_clients)]
        else:
            df_client_slice = df_slice

        job_options = sorted(df_client_slice["job_no"].dropna().unique().tolist()) if "job_no" in df_client_slice.columns else []
        job_name_lookup = build_job_name_lookup(df_client_slice)
        job_category_lookup = {}
        if "job_category_quote" in df_client_slice.columns and job_options:
            job_category_lookup = (
                df_client_slice[["job_no", "job_category_quote"]]
                .dropna()
                .assign(job_no=lambda d: d["job_no"].astype(str))
                .groupby("job_no")["job_category_quote"]
                .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
                .to_dict()
            )
        def _parse_keywords(raw: str) -> list[str]:
            if not raw:
                return []
            parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
            return [p for p in parts if p]

        def _normalize_job_ids(values) -> list[str]:
            if values is None:
                return []
            normalized = []
            for value in values:
                if value is None:
                    continue
                if isinstance(value, float) and pd.isna(value):
                    continue
                normalized.append(str(value))
            return sorted(set(normalized))

        def _filter_jobs(df_in: pd.DataFrame, job_ids) -> pd.DataFrame:
            if "job_no" not in df_in.columns:
                return df_in.iloc[0:0]
            job_set = set(_normalize_job_ids(job_ids))
            if not job_set:
                return df_in.iloc[0:0]
            return df_in[df_in["job_no"].astype(str).isin(job_set)]

        use_all_jobs = st.checkbox(
            "Use all jobs from selected clients",
            value=True,
            disabled=compare_locked or len(selected_clients) == 0,
            help="Uncheck to pick a subset of jobs.",
            key="quote_use_all_jobs",
        )

        selected_jobs = []
        if "quote_compare_jobs_override" in st.session_state:
            st.session_state["quote_compare_jobs"] = st.session_state.pop("quote_compare_jobs_override")
        if selected_clients and not use_all_jobs and job_options and not compare_locked:
            keyword_matches = []
            if "job_description" in df_client_slice.columns:
                st.markdown("**Keyword Search (Job Description)**")
                keyword_input = st.text_area(
                    "Enter keywords or phrases (comma or new line separated)",
                    placeholder="e.g. retainer, performance marketing, SEO audit",
                    key="quote_job_keyword_input",
                )
                exclude_input = st.text_area(
                    "Exclude keywords or phrases (comma or new line separated)",
                    placeholder="e.g. brand, offline",
                    key="quote_job_keyword_exclude_input",
                )
                match_mode = st.selectbox(
                    "Match mode",
                    options=["Match any keyword", "Match all keywords"],
                    key="quote_job_keyword_mode",
                )
                use_regex = st.checkbox(
                    "Regex mode",
                    value=False,
                    help="Treat keywords as regex patterns.",
                    key="quote_job_keyword_regex",
                )
                keywords = _parse_keywords(keyword_input)
                exclude_keywords = _parse_keywords(exclude_input)

                if keywords:
                    desc_df = (
                        df_client_slice[["job_no", "job_description"]]
                        .dropna(subset=["job_no", "job_description"])
                        .drop_duplicates(subset=["job_no"])
                        .copy()
                    )
                    desc_df["job_no"] = desc_df["job_no"].astype(str)
                    descriptions = desc_df.set_index("job_no")["job_description"].astype(str).str.lower()
                    keyword_hits = []
                    for keyword in keywords:
                        keyword_l = keyword.lower()
                        hits = descriptions.index[
                            descriptions.str.contains(keyword_l, regex=use_regex, na=False)
                        ]
                        keyword_hits.append(set(hits.tolist()))

                    if keyword_hits:
                        if match_mode == "Match all keywords":
                            matched = set.intersection(*keyword_hits)
                        else:
                            matched = set.union(*keyword_hits)
                        keyword_matches = sorted(matched)

                if keyword_matches and exclude_keywords:
                    exclude_hits = set()
                    for keyword in exclude_keywords:
                        keyword_l = keyword.lower()
                        hits = descriptions.index[
                            descriptions.str.contains(keyword_l, regex=use_regex, na=False)
                        ]
                        exclude_hits.update(hits.tolist())
                    keyword_matches = sorted(set(keyword_matches) - exclude_hits)

                if keyword_matches:
                    st.caption(f"Keyword matches: {len(keyword_matches)} jobs")
                    if st.button("Select all keyword matches", key="quote_select_keyword_jobs"):
                        existing = st.session_state.get("quote_compare_jobs", []) or []
                        st.session_state["quote_compare_jobs"] = sorted(set(existing) | set(keyword_matches))
                        st.rerun()
                elif keywords:
                    st.caption("Keyword matches: 0 jobs")

                st.markdown("**Find similar jobs from a pasted description**")
                paste_desc = st.text_area(
                    "Paste a job description to find similar jobs",
                    placeholder="Paste a brief job description here...",
                    key="quote_job_similarity_input",
                )
                top_k = st.slider(
                    "Results to show",
                    min_value=3,
                    max_value=15,
                    value=8,
                    step=1,
                    key="quote_job_similarity_topk",
                )

                def _tokenize(text: str) -> list[str]:
                    if not text:
                        return []
                    text = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
                    tokens = [t for t in text.split() if len(t) > 2]
                    stop = {
                        "the", "and", "for", "with", "from", "that", "this", "into", "onto",
                        "your", "our", "their", "was", "were", "are", "is", "to", "of", "in",
                        "on", "by", "as", "an", "a", "at", "or", "be", "it", "its", "we"
                    }
                    return [t for t in tokens if t not in stop]

                if paste_desc:
                    desc_df = (
                        df_client_slice[["job_no", "job_description"]]
                        .dropna(subset=["job_no", "job_description"])
                        .drop_duplicates(subset=["job_no"])
                        .copy()
                    )
                    desc_df["job_no"] = desc_df["job_no"].astype(str)
                    query_tokens = set(_tokenize(paste_desc))
                    if len(query_tokens) == 0:
                        st.info("Paste a longer description to match.")
                    else:
                        rows = []
                        for _, row in desc_df.iterrows():
                            tokens = set(_tokenize(str(row["job_description"])))
                            if not tokens:
                                continue
                            overlap = query_tokens.intersection(tokens)
                            jaccard = len(overlap) / len(query_tokens.union(tokens))
                            if jaccard == 0:
                                continue
                            rows.append({
                                "job_no": row["job_no"],
                                "score": jaccard,
                                "overlap_terms": ", ".join(sorted(list(overlap))[:8]),
                            })

                        if rows:
                            sim_df = pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k)
                            sim_df["job_label"] = sim_df["job_no"].apply(
                                lambda j: format_job_label(j, job_name_lookup)
                            )
                            st.dataframe(
                                sim_df[["job_label", "score", "overlap_terms"]].rename(columns={
                                    "job_label": "Job",
                                    "score": "Similarity",
                                    "overlap_terms": "Why (overlap terms)",
                                }),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Similarity": st.column_config.NumberColumn(format="%.2f"),
                                },
                            )
                            if st.button("Select all similar jobs", key="quote_select_similar_jobs"):
                                existing = st.session_state.get("quote_compare_jobs", []) or []
                                st.session_state["quote_compare_jobs"] = sorted(set(existing) | set(sim_df["job_no"].tolist()))
                                st.rerun()
                        else:
                            st.caption("No close matches found for the pasted description.")

            selected_jobs = st.multiselect(
                "Select specific jobs",
                options=job_options,
                format_func=lambda j: (
                    f"{format_job_label(j, job_name_lookup)} — {job_category_lookup.get(str(j), 'Unknown')}"
                    if job_category_lookup
                    else format_job_label(j, job_name_lookup)
                ),
                key="quote_compare_jobs",
            )
        elif selected_clients and not use_all_jobs and job_options and compare_locked:
            selected_jobs = st.multiselect(
                "Select specific jobs",
                options=job_options,
                format_func=lambda j: (
                    f"{format_job_label(j, job_name_lookup)} — {job_category_lookup.get(str(j), 'Unknown')}"
                    if job_category_lookup
                    else format_job_label(j, job_name_lookup)
                ),
                key="quote_compare_jobs",
                disabled=True,
            )

        if selected_jobs:
            df_compare_draft = _filter_jobs(df_client_slice, selected_jobs)
        else:
            df_compare_draft = df_client_slice

        if len(df_compare_draft) == 0:
            st.warning("No jobs found for the selected comparable filters.")
            return

        lock_cols = st.columns([1.2, 1.2, 2])
        with lock_cols[0]:
            if st.button("Lock comparable set", disabled=compare_locked):
                lock_jobs = _normalize_job_ids(df_compare_draft["job_no"].unique().tolist())
                st.session_state["quote_compare_locked"] = True
                st.session_state["quote_compare_locked_jobs"] = lock_jobs
                st.session_state["quote_compare_pool_jobs"] = lock_jobs.copy()
                _reset_task_state()
                st.rerun()
        with lock_cols[1]:
            if st.button("Unlock comparable set", disabled=not compare_locked):
                st.session_state["quote_compare_locked"] = False
                st.session_state["quote_compare_locked_jobs"] = None
                st.session_state["quote_compare_pool_jobs"] = None
                _reset_task_state()
                st.rerun()
        with lock_cols[2]:
            if compare_locked:
                st.caption("Comparable set locked. Review dispersion/similarity below to refine the pool.")
            else:
                st.caption("Lock the comparable set to review dispersion and build the quote.")

        if not compare_locked:
            st.caption(
                f"Draft set: {df_compare_draft['job_no'].nunique()} jobs"
                + (f", {df_compare_draft['client'].nunique()} clients" if "client" in df_compare_draft.columns else "")
            )
            st.info("Lock the comparable set to continue.")
            return

        compare_locked_jobs = _normalize_job_ids(st.session_state.get("quote_compare_locked_jobs") or [])
        compare_pool_jobs = _normalize_job_ids(st.session_state.get("quote_compare_pool_jobs") or compare_locked_jobs)
        if compare_locked_jobs and not compare_pool_jobs:
            compare_pool_jobs = compare_locked_jobs.copy()
            st.session_state["quote_compare_pool_jobs"] = compare_pool_jobs

        df_compare = _filter_jobs(df_slice, compare_pool_jobs) if compare_pool_jobs else df_slice.iloc[0:0]

        if len(df_compare) == 0:
            st.warning("No jobs found in the current comparable pool.")
            if st.button("Reset pool to locked set", key="quote_pool_reset_empty"):
                st.session_state["quote_compare_pool_jobs"] = compare_locked_jobs
                _reset_task_state()
                st.rerun()
            return

        pool_job_count = df_compare["job_no"].nunique() if "job_no" in df_compare.columns else 0
        pool_client_count = df_compare["client"].nunique() if "client" in df_compare.columns else 0
        st.caption(
            f"Comparable pool: {pool_job_count} jobs"
            + (f", {pool_client_count} clients" if "client" in df_compare.columns else "")
        )
        if compare_locked_jobs and len(compare_pool_jobs) != len(compare_locked_jobs):
            st.caption(
                f"Locked set: {len(compare_locked_jobs)} jobs · Pool: {len(compare_pool_jobs)} jobs"
            )

        meta = get_benchmark_metadata(df_compare, recency_weighted=recency_weighted)

        # Display metadata
        st.markdown(f"**Benchmark:** {meta['n_jobs']} jobs, {meta['n_staff']} staff")
        if meta["date_min"] and meta["date_max"]:
            st.caption(f"Date range: {meta['date_min'].strftime('%b %Y')} - {meta['date_max'].strftime('%b %Y')}")

        st.markdown("---")

        # =====================================================================
        # DISPERSION & SIMILARITY REVIEW
        # =====================================================================
        section_header("Dispersion & Similarity", "Cherry-pick jobs before building the task template")
        st.caption("Tick jobs in the tables below to update the comparable pool used for benchmarks.")

        reset_cols = st.columns([1.3, 2.7])
        with reset_cols[0]:
            if st.button("Reset pool to locked set", key="quote_pool_reset"):
                st.session_state["quote_compare_pool_jobs"] = compare_locked_jobs.copy()
                _reset_task_state()
                st.rerun()
        with reset_cols[1]:
            st.caption(f"Current pool: {pool_job_count} jobs")

        st.markdown(
            """
            <style>
            .dispersion-highlight {
                background: #fff7cc;
                border: 1px solid #f2d675;
                border-radius: 12px;
                padding: 12px 14px;
                margin: 6px 0 12px 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="dispersion-highlight">', unsafe_allow_html=True)
        st.markdown("#### Job Dispersion (Quoted vs Actual)")
        if "job_no" in df_compare.columns:
            job_actual = df_compare.groupby("job_no").agg(
                actual_hours=("hours_raw", "sum") if "hours_raw" in df_compare.columns else ("job_no", "count"),
                actual_value=("rev_alloc", "sum") if "rev_alloc" in df_compare.columns else ("job_no", "count"),
            ).reset_index()
        else:
            job_actual = pd.DataFrame()

        job_quote = safe_quote_job_task(df_compare)
        if len(job_quote) > 0:
            job_quote = job_quote.groupby("job_no").agg(
                quoted_hours=("quoted_time_total", "sum") if "quoted_time_total" in job_quote.columns else ("job_no", "count"),
                quoted_value=("quoted_amount_total", "sum") if "quoted_amount_total" in job_quote.columns else ("job_no", "count"),
            ).reset_index()
        else:
            job_quote = pd.DataFrame()

        job_disp = None
        if len(job_actual) > 0 or len(job_quote) > 0:
            job_disp = job_actual.merge(job_quote, on="job_no", how="outer")
            metrics = {
                "Quoted Value": job_disp["quoted_value"] if "quoted_value" in job_disp.columns else pd.Series(dtype=float),
                "Quoted Hours": job_disp["quoted_hours"] if "quoted_hours" in job_disp.columns else pd.Series(dtype=float),
                "Actual Value": job_disp["actual_value"] if "actual_value" in job_disp.columns else pd.Series(dtype=float),
                "Actual Hours": job_disp["actual_hours"] if "actual_hours" in job_disp.columns else pd.Series(dtype=float),
            }
            rows = []
            for label, series in metrics.items():
                if len(series.dropna()) == 0:
                    continue
                s = series.dropna()
                p25 = s.quantile(0.25)
                p75 = s.quantile(0.75)
                med = s.median()
                iqr = p75 - p25
                spread = iqr / med if med and med != 0 else np.nan
                rows.append({
                    "Metric": label,
                    "Min": s.min(),
                    "P25": p25,
                    "Median": med,
                    "P75": p75,
                    "Max": s.max(),
                    "IQR/Median": spread,
                })
            if rows:
                dispersion_df = pd.DataFrame(rows)
                dispersion_df["Flag"] = np.where(
                    dispersion_df["IQR/Median"] >= 1.0, "Wide",
                    np.where(dispersion_df["IQR/Median"] >= 0.5, "Moderate", "Tight")
                )
                st.dataframe(
                    dispersion_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Min": st.column_config.NumberColumn(format="%.1f"),
                        "P25": st.column_config.NumberColumn(format="%.1f"),
                        "Median": st.column_config.NumberColumn(format="%.1f"),
                        "P75": st.column_config.NumberColumn(format="%.1f"),
                        "Max": st.column_config.NumberColumn(format="%.1f"),
                        "IQR/Median": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                wide_metrics = dispersion_df[dispersion_df["Flag"] == "Wide"]["Metric"].tolist()
                if wide_metrics:
                    st.warning(
                        "Dispersion alert: "
                        + ", ".join(wide_metrics)
                        + " are wide. Consider narrowing the comparable set before quoting."
                    )
                else:
                    st.caption(
                        "Dispersion looks tight/moderate across key metrics — comparable set is consistent."
                    )
            else:
                st.caption("No dispersion metrics available for the current selection.")
        else:
            st.caption("No dispersion metrics available for the current selection.")

        metric_cols = []
        if job_disp is not None and len(job_disp) > 0:
            for col in ["quoted_value", "quoted_hours", "actual_value", "actual_hours"]:
                if col in job_disp.columns:
                    metric_cols.append(col)

        if job_disp is not None and metric_cols:
            contrib_df = job_disp.copy()
            contrib_df = contrib_df[contrib_df[metric_cols].notna().any(axis=1)]
            if len(contrib_df) > 0:
                for col in metric_cols:
                    med = contrib_df[col].median()
                    denom = med if med and med != 0 else (contrib_df[col].std() or 1.0)
                    contrib_df[f"{col}_norm"] = (contrib_df[col] - med).abs() / denom
                norm_cols = [f"{c}_norm" for c in metric_cols]
                contrib_df["dispersion_score"] = contrib_df[norm_cols].mean(axis=1, skipna=True)
                contrib_df["job_no"] = contrib_df["job_no"].astype(str)
                contrib_df["job_label"] = contrib_df["job_no"].apply(
                    lambda j: format_job_label(j, job_name_lookup)
                )

                outliers = contrib_df.sort_values("dispersion_score", ascending=False).head(8)
                similar = contrib_df.sort_values("dispersion_score", ascending=True).head(8)

                st.markdown("**Dispersion Contributors (Review These Jobs)**")
                outlier_view = outliers.copy()
                outlier_view["Exclude"] = False
                outlier_view = outlier_view.set_index("job_no")
                outlier_edit = st.data_editor(
                    outlier_view[["Exclude", "job_label", "dispersion_score"] + metric_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Exclude": st.column_config.CheckboxColumn("Exclude"),
                        "job_label": st.column_config.TextColumn("Job", disabled=True),
                        "dispersion_score": st.column_config.NumberColumn("Dispersion Score", format="%.2f", disabled=True),
                        "quoted_value": st.column_config.NumberColumn("Quoted Value", format="$%.0f", disabled=True),
                        "quoted_hours": st.column_config.NumberColumn("Quoted Hours", format="%.1f", disabled=True),
                        "actual_value": st.column_config.NumberColumn("Actual Value", format="$%.0f", disabled=True),
                        "actual_hours": st.column_config.NumberColumn("Actual Hours", format="%.1f", disabled=True),
                    },
                    num_rows="fixed",
                    key="quote_outlier_editor",
                )
                if st.button("Remove selected outliers from pool", key="quote_outlier_apply"):
                    to_remove = outlier_edit[outlier_edit["Exclude"]].index.tolist()
                    if to_remove:
                        remaining = [j for j in compare_pool_jobs if j not in set(map(str, to_remove))]
                        st.session_state["quote_compare_pool_jobs"] = sorted(set(remaining))
                        _reset_task_state()
                        st.rerun()
                    else:
                        st.info("Select at least one outlier to remove.")

                st.markdown("**Most Similar Jobs (Closest to Median)**")
                similar_view = similar.copy()
                similar_view["Include"] = False
                similar_view = similar_view.set_index("job_no")
                similar_edit = st.data_editor(
                    similar_view[["Include", "job_label", "dispersion_score"] + metric_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Include": st.column_config.CheckboxColumn("Include"),
                        "job_label": st.column_config.TextColumn("Job", disabled=True),
                        "dispersion_score": st.column_config.NumberColumn("Similarity Score", format="%.2f", disabled=True),
                        "quoted_value": st.column_config.NumberColumn("Quoted Value", format="$%.0f", disabled=True),
                        "quoted_hours": st.column_config.NumberColumn("Quoted Hours", format="%.1f", disabled=True),
                        "actual_value": st.column_config.NumberColumn("Actual Value", format="$%.0f", disabled=True),
                        "actual_hours": st.column_config.NumberColumn("Actual Hours", format="%.1f", disabled=True),
                    },
                    num_rows="fixed",
                    key="quote_similar_editor",
                )
                st.caption("Apply replaces the pool with the selected similar jobs.")
                if st.button("Use selected similar jobs as pool", key="quote_similar_apply"):
                    selected = similar_edit[similar_edit["Include"]].index.tolist()
                    if selected:
                        st.session_state["quote_compare_pool_jobs"] = sorted(set(map(str, selected)))
                        _reset_task_state()
                        st.rerun()
                    else:
                        st.warning("Select at least one similar job to update the pool.")
            else:
                st.caption("No job-level metrics available to score similarity.")
        elif job_disp is not None:
            st.caption("No job-level metrics available to score similarity.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        
        pool_job_key = tuple(compare_pool_jobs)
        bench_key = (
            "quote_benchmarks",
            selected_dept,
            selected_cat,
            benchmark_window,
            pool_job_key,
            bool(recency_weighted),
        )
        if st.session_state.get("quote_bench_key") != bench_key:
            # Get task benchmarks
            benchmarks = get_task_benchmarks(
                df_compare, selected_dept, selected_cat,
                recency_weighted=recency_weighted
            )
            st.session_state["quote_bench_key"] = bench_key
            st.session_state["quote_benchmarks"] = benchmarks
        else:
            benchmarks = st.session_state.get("quote_benchmarks", pd.DataFrame())
        
        if len(benchmarks) == 0:
            st.warning("No task data available for this selection.")
            return
        
        # =====================================================================
        # TASK TEMPLATE TABLE
        # =====================================================================
        section_header("Task Template", "Select tasks, adjust hours, and build a usable quote plan fast")
        
        # Initialize table in session state if not exists or context changes
        table_key = (selected_dept, selected_cat, benchmark_window, pool_job_key)
        if st.session_state.get("quote_task_table_key") != table_key:
            default_tasks = benchmarks[benchmarks["inclusion_rate"] >= 50]["task_name"].tolist()
            base_table = benchmarks.rename(columns={
                "task_name": "Task",
                "inclusion_rate": "Inclusion %",
                "quoted_hours_p50": "Median (p50)",
                "quoted_hours_p25": "Low (p25)",
                "quoted_hours_p75": "High (p75)",
                "overrun_risk": "Overrun Risk %",
                "cost_per_hour": "Cost/hr",
                "quote_rate": "Quote Rate",
            })[[
                "Task", "Inclusion %", "Median (p50)", "Low (p25)", "High (p75)",
                "Overrun Risk %", "Cost/hr", "Quote Rate"
            ]].copy()
            base_table["Include"] = base_table["Task"].isin(default_tasks)
            base_table["Hours"] = np.where(
                base_table["Include"],
                base_table["Median (p50)"],
                0.0,
            )
            st.session_state["quote_task_table"] = base_table
            st.session_state["quote_task_table_key"] = table_key
            st.session_state["quote_task_locked"] = False
            st.session_state["quote_task_locked_table"] = None
            st.session_state["quote_econ_ready"] = False

        task_table = st.session_state["quote_task_table"].copy()

        # Usability controls
        st.caption(
            "Legend: ✅ Editable (your quote) · 📊 Empirical benchmarks (read‑only). "
            "Use Hours + Include to build the quote; benchmarks show historical ranges."
        )
        control_cols = st.columns([1.2, 1, 1, 1, 1.4])
        with control_cols[0]:
            task_search = st.text_input("Search tasks", value="", placeholder="Type to filter tasks")
        with control_cols[1]:
            min_inclusion = st.slider("Min inclusion %", 0, 100, 0, step=10)
        with control_cols[2]:
            show_only_selected = st.checkbox("Show selected only", value=False)
        with control_cols[3]:
            sort_by = st.selectbox("Sort by", ["Inclusion %", "Overrun Risk %", "Hours"], index=0)
        with control_cols[4]:
            bulk_set = st.selectbox("Set hours to", ["Keep current", "Low (p25)", "Median (p50)", "High (p75)"])

        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("Select common tasks (≥50%)"):
                task_table["Include"] = task_table["Inclusion %"] >= 50
                task_table.loc[task_table["Include"], "Hours"] = task_table.loc[task_table["Include"], "Median (p50)"]
                st.session_state["quote_task_table"] = task_table
                st.session_state["quote_econ_ready"] = False
                st.rerun()
        with action_cols[1]:
            if st.button("Select all"):
                task_table["Include"] = True
                task_table.loc[task_table["Include"], "Hours"] = task_table.loc[task_table["Include"], "Median (p50)"]
                st.session_state["quote_task_table"] = task_table
                st.session_state["quote_econ_ready"] = False
                st.rerun()
        with action_cols[2]:
            if st.button("Clear all"):
                task_table["Include"] = False
                task_table["Hours"] = 0.0
                st.session_state["quote_task_table"] = task_table
                st.session_state["quote_econ_ready"] = False
                st.rerun()

        if bulk_set != "Keep current":
            task_table.loc[task_table["Include"], "Hours"] = task_table.loc[
                task_table["Include"], bulk_set
            ]
            st.session_state["quote_task_table"] = task_table
            st.session_state["quote_econ_ready"] = False

        # Apply filters for display
        view_table = task_table.copy()
        if task_search:
            view_table = view_table[view_table["Task"].str.contains(task_search, case=False, na=False)]
        view_table = view_table[view_table["Inclusion %"] >= min_inclusion]
        if show_only_selected:
            view_table = view_table[view_table["Include"]]
        view_table = view_table.sort_values(sort_by, ascending=False)

        st.markdown(
            """
            <style>
            /* Highlight editable columns in the task editor (Include + Hours) */
            [data-testid="stDataEditor"] div[role="columnheader"]:nth-child(1),
            [data-testid="stDataEditor"] div[role="columnheader"]:nth-child(3),
            [data-testid="stDataEditor"] div[role="gridcell"]:nth-child(1),
            [data-testid="stDataEditor"] div[role="gridcell"]:nth-child(3) {
                background: #fff7cc;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        locked = st.session_state.get("quote_task_locked", False)
        edited_view = None
        view_editor_table = view_table[[
                    "Include", "Task", "Hours", "Inclusion %",
                    "Low (p25)", "Median (p50)", "High (p75)",
                    "Overrun Risk %", "Cost/hr", "Quote Rate"
                ]].copy().reset_index(drop=True)
        st.session_state["quote_task_view"] = view_editor_table
        with st.form("quote_task_editor_form", clear_on_submit=False):
            edited_view = st.data_editor(
                view_editor_table,
                column_config={
                    "Include": st.column_config.CheckboxColumn("✅ Include", disabled=locked),
                    "Task": st.column_config.TextColumn("Task", disabled=True),
                    "Hours": st.column_config.NumberColumn("✅ Quote Hours", min_value=0, step=0.5, disabled=locked),
                    "Inclusion %": st.column_config.NumberColumn("📊 Inclusion %", format="%.1f%%", disabled=True),
                    "Low (p25)": st.column_config.NumberColumn("📊 Low (p25)", format="%.1f", disabled=True),
                    "Median (p50)": st.column_config.NumberColumn("📊 Median (p50)", format="%.1f", disabled=True),
                    "High (p75)": st.column_config.NumberColumn("📊 High (p75)", format="%.1f", disabled=True),
                    "Overrun Risk %": st.column_config.NumberColumn("📊 Overrun Risk %", format="%.0f%%", disabled=True),
                    "Cost/hr": st.column_config.NumberColumn("📊 Cost/hr", format="$%.0f", disabled=True),
                    "Quote Rate": st.column_config.NumberColumn("📊 Quote Rate", format="$%.0f/hr", disabled=True),
                },
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="task_editor"
            )
            apply_edits = st.form_submit_button(
                "Apply task edits",
                disabled=locked,
                help="Apply your Include/Hours changes without recomputing until you lock.",
            )

        # Merge edits back into full table only on submit
        if apply_edits and edited_view is not None and len(edited_view) > 0 and not locked:
            update_cols = ["Include", "Hours"]
            task_table.set_index("Task", inplace=True)
            edited_view.set_index("Task", inplace=True)
            task_table.loc[edited_view.index, update_cols] = edited_view[update_cols]
            task_table.reset_index(inplace=True)
            st.session_state["quote_task_table"] = task_table
            st.session_state["quote_econ_ready"] = False

        lock_cols = st.columns([1.2, 1, 1, 1.8])
        with lock_cols[0]:
            if st.button("Lock task selection", disabled=locked):
                editor_state = st.session_state.get("task_editor")
                update_cols = ["Include", "Hours"]
                if isinstance(editor_state, pd.DataFrame) and len(editor_state) > 0:
                    if st.session_state.get("quote_lock_visible_only", True):
                        task_table["Include"] = False
                        task_table["Hours"] = 0.0
                    task_table.set_index("Task", inplace=True)
                    editor_state = editor_state.set_index("Task")
                    task_table.loc[editor_state.index, update_cols] = editor_state[update_cols]
                    task_table.reset_index(inplace=True)
                    st.session_state["quote_task_table"] = task_table
                elif isinstance(editor_state, dict) and "edited_rows" in editor_state:
                    view_snapshot = st.session_state.get("quote_task_view")
                    if isinstance(view_snapshot, pd.DataFrame) and len(view_snapshot) > 0:
                        view_snapshot = view_snapshot.copy()
                        if st.session_state.get("quote_lock_visible_only", True):
                            task_table["Include"] = False
                            task_table["Hours"] = 0.0
                        for row_idx, changes in editor_state.get("edited_rows", {}).items():
                            if 0 <= int(row_idx) < len(view_snapshot):
                                for col, val in changes.items():
                                    if col in update_cols:
                                        view_snapshot.at[int(row_idx), col] = val
                        task_table.set_index("Task", inplace=True)
                        view_snapshot.set_index("Task", inplace=True)
                        task_table.loc[view_snapshot.index, update_cols] = view_snapshot[update_cols]
                        task_table.reset_index(inplace=True)
                        st.session_state["quote_task_table"] = task_table
                st.session_state["quote_task_locked"] = True
                st.session_state["quote_task_locked_table"] = task_table.copy()
                st.session_state["quote_econ_ready"] = True
                st.rerun()
        with lock_cols[1]:
            if st.button("Unlock", disabled=not locked):
                st.session_state["quote_task_locked"] = False
                st.session_state["quote_task_locked_table"] = None
                st.session_state["quote_econ_ready"] = False
                st.rerun()
        with lock_cols[2]:
            if st.button("Compute quote economics", disabled=not locked):
                st.session_state["quote_econ_ready"] = True
                st.rerun()
        with lock_cols[3]:
            if locked:
                st.caption("Selection locked. Unlock to edit tasks or hours.")
            else:
                st.caption("Edit tasks/hours, then lock to compute economics.")
            st.checkbox(
                "Lock only tasks shown in the table",
                value=True,
                key="quote_lock_visible_only",
                help="When enabled, locking will include only tasks visible in the current table view.",
            )

        st.markdown("---")
        
        # =====================================================================
        # ECONOMICS PREVIEW
        # =====================================================================
        section_header("Quote Economics")
        
        locked_table = st.session_state.get("quote_task_locked_table")
        if not st.session_state.get("quote_task_locked", False) or locked_table is None:
            st.info("Lock your task selection to update totals.")
            return

        locked_table = locked_table.copy()
        locked_table["Hours"] = pd.to_numeric(locked_table["Hours"], errors="coerce").fillna(0.0)
        edited_df = locked_table[locked_table["Include"]].copy()

        # Calculate totals from edited data
        total_hours = float(locked_table.loc[locked_table["Include"], "Hours"].sum())
        
        # Merge back cost/rate info
        edited_df = edited_df.merge(
            benchmarks[["task_name", "cost_per_hour", "quote_rate"]].rename(
                columns={"task_name": "Task"}
            ),
            on="Task",
            how="left",
            suffixes=("", "_bench")
        )
        
        # Use benchmark values for cost and rate
        cost_col = (
            "cost_per_hour_bench"
            if "cost_per_hour_bench" in edited_df.columns
            else ("cost_per_hour" if "cost_per_hour" in edited_df.columns else "Cost/hr")
        )
        rate_col = (
            "quote_rate_bench"
            if "quote_rate_bench" in edited_df.columns
            else ("quote_rate" if "quote_rate" in edited_df.columns else "Quote Rate")
        )
        edited_df["task_cost"] = edited_df["Hours"] * edited_df[cost_col].fillna(
            benchmarks["cost_per_hour"].median()
        )
        edited_df["task_value"] = edited_df["Hours"] * edited_df[rate_col].fillna(
            benchmarks["quote_rate"].median()
        )
        
        total_cost = edited_df["task_cost"].sum()
        total_value = edited_df["task_value"].sum()
        total_margin = total_value - total_cost
        margin_pct = total_margin / total_value * 100 if total_value > 0 else 0

        # Scenario analytics
        scenario_hours = {
            "Low (p25)": edited_df["Low (p25)"].fillna(0.0),
            "Median (p50)": edited_df["Median (p50)"].fillna(0.0),
            "High (p75)": edited_df["High (p75)"].fillna(0.0),
        }
        scenario_rows = []
        for label, hours_series in scenario_hours.items():
            scenario_cost = (hours_series * edited_df[cost_col].fillna(benchmarks["cost_per_hour"].median())).sum()
            scenario_value = (hours_series * edited_df[rate_col].fillna(benchmarks["quote_rate"].median())).sum()
            scenario_margin = scenario_value - scenario_cost
            scenario_margin_pct = scenario_margin / scenario_value * 100 if scenario_value > 0 else 0
            scenario_rows.append({
                "Scenario": label,
                "Total Hours": hours_series.sum(),
                "Est. Cost": scenario_cost,
                "Est. Value": scenario_value,
                "Est. Margin": scenario_margin,
                "Margin %": scenario_margin_pct,
            })
        scenario_df = pd.DataFrame(scenario_rows)

        # Task-level diagnostics
        edited_df["Hours Delta vs Median"] = edited_df["Hours"] - edited_df["Median (p50)"].fillna(0.0)
        edited_df["Quote Bias"] = np.where(
            edited_df["Hours Delta vs Median"] > 0.01, "Over median",
            np.where(edited_df["Hours Delta vs Median"] < -0.01, "Under median", "On median")
        )
        edited_df["Overrun Risk Band"] = np.where(
            edited_df["Overrun Risk %"] >= 60, "High",
            np.where(edited_df["Overrun Risk %"] >= 30, "Medium", "Low")
        )

        # Display
        st.markdown("#### Snapshot")
        econ_cols = st.columns(5)
        with econ_cols[0]:
            st.metric("Total Hours", fmt_hours(total_hours))
        with econ_cols[1]:
            st.metric("Est. Cost", fmt_currency(total_cost))
        with econ_cols[2]:
            st.metric("Est. Value", fmt_currency(total_value))
        with econ_cols[3]:
            st.metric("Est. Margin", fmt_currency(total_margin))
        with econ_cols[4]:
            st.metric("Margin %", fmt_percent(margin_pct))

        # What-if: build economics from selected jobs (avg-based)
        job_count = df_compare["job_no"].nunique() if "job_no" in df_compare.columns else 0
        actual_by_job = df_compare.groupby("job_no").agg(
            actual_hours=("hours_raw", "sum") if "hours_raw" in df_compare.columns else ("job_no", "count"),
            actual_cost=("base_cost", "sum") if "base_cost" in df_compare.columns else ("job_no", "count"),
            actual_value=("rev_alloc", "sum") if "rev_alloc" in df_compare.columns else ("job_no", "count"),
        ).reset_index() if job_count > 0 else pd.DataFrame()

        avg_actual_hours = actual_by_job["actual_hours"].mean() if "actual_hours" in actual_by_job.columns else np.nan
        total_actual_cost = actual_by_job["actual_cost"].sum() if "actual_cost" in actual_by_job.columns else np.nan
        total_actual_hours = actual_by_job["actual_hours"].sum() if "actual_hours" in actual_by_job.columns else np.nan
        avg_actual_cost_per_hour = (
            total_actual_cost / total_actual_hours
            if pd.notna(total_actual_cost) and pd.notna(total_actual_hours) and total_actual_hours > 0
            else np.nan
        )

        quote_job_task = safe_quote_job_task(df_compare)
        if len(quote_job_task) > 0:
            quoted_by_job = quote_job_task.groupby("job_no").agg(
                quoted_hours=("quoted_time_total", "sum") if "quoted_time_total" in quote_job_task.columns else ("job_no", "count"),
                quoted_value=("quoted_amount_total", "sum") if "quoted_amount_total" in quote_job_task.columns else ("job_no", "count"),
            ).reset_index()
        else:
            quoted_by_job = pd.DataFrame()

        avg_quoted_hours = quoted_by_job["quoted_hours"].mean() if "quoted_hours" in quoted_by_job.columns else np.nan
        avg_quoted_value = quoted_by_job["quoted_value"].mean() if "quoted_value" in quoted_by_job.columns else np.nan
        avg_quoted_rate = avg_quoted_value / avg_quoted_hours if pd.notna(avg_quoted_value) and avg_quoted_hours > 0 else np.nan

        what_if_est_cost = (
            avg_actual_cost_per_hour * avg_quoted_hours
            if pd.notna(avg_actual_cost_per_hour) and pd.notna(avg_quoted_hours)
            else np.nan
        )
        what_if_est_value = avg_quoted_rate * avg_quoted_hours if pd.notna(avg_quoted_rate) and pd.notna(avg_quoted_hours) else np.nan
        what_if_margin = what_if_est_value - what_if_est_cost if pd.notna(what_if_est_value) and pd.notna(what_if_est_cost) else np.nan
        what_if_margin_pct = what_if_margin / what_if_est_value * 100 if pd.notna(what_if_margin) and what_if_est_value and what_if_est_value > 0 else np.nan

        if job_count > 0:
            st.markdown("#### What‑If (Avg per Job from Selected Jobs)")
            st.caption(
                "Explainer: This section estimates what an average job looks like based on the selected comparable "
                "jobs. We use average quoted hours and average quoted $/hour to estimate value, and average actual "
                "cost to estimate cost. This makes the what‑if comparable to the quote you are building."
            )
            what_if_cols = st.columns(6)
            with what_if_cols[0]:
                st.metric("Avg Quoted Hours", fmt_hours(avg_quoted_hours))
            with what_if_cols[1]:
                st.metric("Avg Actual Hours", fmt_hours(avg_actual_hours))
            with what_if_cols[2]:
                st.metric("Est. Cost (Weighted Actual)", fmt_currency(what_if_est_cost))
            with what_if_cols[3]:
                st.metric("Est. Value (Quoted Rate × Hours)", fmt_currency(what_if_est_value))
            with what_if_cols[4]:
                st.metric("Est. Margin", fmt_currency(what_if_margin))
            with what_if_cols[5]:
                st.metric("Margin %", fmt_percent(what_if_margin_pct))

            # Avg time to completion + range (days)
            if "job_start_date" in df_compare.columns and "job_completed_date" in df_compare.columns:
                durations = (
                    df_compare[["job_no", "job_start_date", "job_completed_date"]]
                    .dropna(subset=["job_no", "job_start_date", "job_completed_date"])
                    .drop_duplicates(subset=["job_no"])
                    .copy()
                )
                if len(durations) > 0:
                    durations["job_start_date"] = pd.to_datetime(durations["job_start_date"], errors="coerce")
                    durations["job_completed_date"] = pd.to_datetime(durations["job_completed_date"], errors="coerce")
                    durations = durations.dropna(subset=["job_start_date", "job_completed_date"])
                    durations["duration_days"] = (
                        durations["job_completed_date"] - durations["job_start_date"]
                    ).dt.days
                    durations = durations[durations["duration_days"] >= 0]
                if len(durations) > 0:
                    avg_days = durations["duration_days"].mean()
                    p25_days = durations["duration_days"].quantile(0.25)
                    p75_days = durations["duration_days"].quantile(0.75)
                    expected_end = pd.Timestamp.today().normalize() + pd.Timedelta(days=int(round(avg_days)))
                    st.caption(
                        f"Avg time to completion: {avg_days:.0f} days "
                        f"(p25–p75: {p25_days:.0f}–{p75_days:.0f}). "
                        f"Imputed expected end date: {expected_end.strftime('%b %d, %Y')}."
                    )
                else:
                    st.caption("Avg time to completion: — (insufficient completed jobs with start/end dates).")
            else:
                st.caption("Avg time to completion: — (job start/completed dates not available).")

        st.markdown("#### Margin Scenarios")
        st.dataframe(
            scenario_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Total Hours": st.column_config.NumberColumn(format="%.1f"),
                "Est. Cost": st.column_config.NumberColumn(format="$%.0f"),
                "Est. Value": st.column_config.NumberColumn(format="$%.0f"),
                "Est. Margin": st.column_config.NumberColumn(format="$%.0f"),
                "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        st.markdown("#### Task Diagnostics")
        diag_df = edited_df[[
            "Task",
            "Hours",
            "Low (p25)",
            "Median (p50)",
            "High (p75)",
            "Hours Delta vs Median",
            "Quote Bias",
            "Overrun Risk %",
            "Overrun Risk Band",
            "Cost/hr",
            "Quote Rate",
        ]].copy()
        st.dataframe(
            diag_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Hours": st.column_config.NumberColumn(format="%.1f"),
                "Low (p25)": st.column_config.NumberColumn(format="%.1f"),
                "Median (p50)": st.column_config.NumberColumn(format="%.1f"),
                "High (p75)": st.column_config.NumberColumn(format="%.1f"),
                "Hours Delta vs Median": st.column_config.NumberColumn(format="%+.1f"),
                "Overrun Risk %": st.column_config.NumberColumn(format="%.0f%%"),
                "Cost/hr": st.column_config.NumberColumn(format="$%.0f"),
                "Quote Rate": st.column_config.NumberColumn(format="$%.0f/hr"),
            },
        )

        st.markdown("#### Prevalent Tasks Not Selected")
        prevalence_threshold = st.slider(
            "Flag tasks with inclusion % ≥",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
        )
        missing_df = locked_table[
            (locked_table["Include"] == False) &
            (locked_table["Inclusion %"] >= prevalence_threshold)
        ].copy()
        if len(missing_df) > 0:
            st.dataframe(
                missing_df[[
                    "Task",
                    "Inclusion %",
                    "Low (p25)",
                    "Median (p50)",
                    "High (p75)",
                    "Overrun Risk %",
                ]].sort_values("Inclusion %", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Inclusion %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Low (p25)": st.column_config.NumberColumn(format="%.1f"),
                    "Median (p50)": st.column_config.NumberColumn(format="%.1f"),
                    "High (p75)": st.column_config.NumberColumn(format="%.1f"),
                    "Overrun Risk %": st.column_config.NumberColumn(format="%.0f%%"),
                },
            )
            st.caption("These tasks are empirically common but currently excluded (possible human error).")
        else:
            st.caption("No high-prevalence tasks excluded.")

        st.markdown("#### Target Margin Solver")
        solver_cols = st.columns([1.1, 1.1, 1, 1.2])
        with solver_cols[0]:
            target_margin_pct = st.number_input(
                "Target margin %",
                min_value=0.0,
                max_value=100.0,
                value=float(round(margin_pct, 1)),
                step=0.5,
                format="%.1f",
            )
        with solver_cols[1]:
            lever = st.selectbox(
                "Lever",
                options=["Adjust quote rates", "Adjust hours", "Adjust hours + rates"],
                index=0,
                help="Adjust rates and/or hours within bounds to reach the target margin.",
            )
        with solver_cols[2]:
            max_uplift = st.slider("Max rate uplift %", 0, 80, 30, step=5)
        with solver_cols[3]:
            min_discount = st.slider("Max rate discount %", 0, 50, 0, step=5)

        risk_weight = st.slider(
            "Overrun risk weighting",
            min_value=0.0,
            max_value=1.5,
            value=0.5,
            step=0.1,
            help="Higher values push more uplift to high-risk tasks.",
        )

        solver_df = edited_df.copy()
        solver_df["Hours"] = solver_df["Hours"].fillna(0.0)
        solver_df["Base Rate"] = solver_df[rate_col].fillna(benchmarks["quote_rate"].median())
        solver_df["Cost/hr"] = solver_df[cost_col].fillna(benchmarks["cost_per_hour"].median())
        solver_df["Overrun Risk %"] = solver_df["Overrun Risk %"].fillna(0.0)
        solver_df["Low (p25)"] = solver_df["Low (p25)"].fillna(0.0)
        solver_df["Median (p50)"] = solver_df["Median (p50)"].fillna(solver_df["Hours"])
        solver_df["High (p75)"] = solver_df["High (p75)"].fillna(solver_df["Hours"])

        # Risk weighting (normalized to mean 1)
        risk_scale = 1 + (solver_df["Overrun Risk %"] / 100.0) * risk_weight
        risk_scale = risk_scale / risk_scale.mean() if risk_scale.mean() else 1.0

        def _solve_hours_mix() -> pd.Series:
            base_hours = solver_df["Hours"].where(solver_df["Hours"] > 0, solver_df["Median (p50)"])
            min_hours = solver_df["Low (p25)"].clip(lower=0.0)
            max_hours = solver_df["High (p75)"].clip(lower=min_hours)
            margin_per_hour = (solver_df["Base Rate"] - solver_df["Cost/hr"]).fillna(0.0)
            if margin_per_hour.std() == 0:
                return base_hours.clip(lower=min_hours, upper=max_hours)
            margin_z = (margin_per_hour - margin_per_hour.mean()) / (margin_per_hour.std() + 1e-9)

            def _margin_pct(hours: pd.Series) -> float:
                value = (hours * solver_df["Base Rate"]).sum()
                cost = (hours * solver_df["Cost/hr"]).sum()
                return (value - cost) / value * 100 if value > 0 else 0.0

            target = target_margin_pct
            lo, hi = -1.5, 1.5
            best = base_hours.copy()
            for _ in range(40):
                mid = (lo + hi) / 2
                scaled = base_hours * (1 + mid * margin_z)
                scaled = scaled.clip(lower=min_hours, upper=max_hours)
                pct = _margin_pct(scaled)
                best = scaled
                if pct < target:
                    lo = mid
                else:
                    hi = mid
            return best

        suggested_hours = solver_df["Hours"]
        if lever in ["Adjust hours", "Adjust hours + rates"]:
            suggested_hours = _solve_hours_mix()
            solver_df["Suggested Hours"] = suggested_hours
        else:
            solver_df["Suggested Hours"] = solver_df["Hours"]

        if lever in ["Adjust quote rates", "Adjust hours + rates"]:
            total_cost = (solver_df["Suggested Hours"] * solver_df["Cost/hr"]).sum()
            target_margin = target_margin_pct / 100.0
            target_value = total_cost / (1 - target_margin) if target_margin < 0.99 else None

            min_mult = 1 - (min_discount / 100.0)
            max_mult = 1 + (max_uplift / 100.0)

            if target_value is None:
                st.warning("Target margin too high to solve safely.")
                solver_df["Suggested Rate"] = solver_df["Base Rate"]
            else:
                def _total_value(scale: float) -> float:
                    multipliers = (scale * risk_scale).clip(lower=min_mult, upper=max_mult)
                    return float((solver_df["Suggested Hours"] * solver_df["Base Rate"] * multipliers).sum())

                min_value = _total_value(0.0)
                max_value = _total_value(10.0)

                if target_value <= min_value:
                    st.info("Target margin below achievable range at current bounds. Showing minimum rates.")
                    chosen_scale = 0.0
                elif target_value >= max_value:
                    st.info("Target margin above achievable range at current bounds. Showing maximum rates.")
                    chosen_scale = 10.0
                else:
                    lo, hi = 0.0, 10.0
                    for _ in range(40):
                        mid = (lo + hi) / 2
                        if _total_value(mid) < target_value:
                            lo = mid
                        else:
                            hi = mid
                    chosen_scale = (lo + hi) / 2

                multipliers = (chosen_scale * risk_scale).clip(lower=min_mult, upper=max_mult)
                solver_df["Suggested Rate"] = solver_df["Base Rate"] * multipliers
        else:
            solver_df["Suggested Rate"] = solver_df["Base Rate"]

        solver_df["Suggested Value"] = solver_df["Suggested Hours"] * solver_df["Suggested Rate"]
        solver_df["Suggested Cost"] = solver_df["Suggested Hours"] * solver_df["Cost/hr"]
        suggested_value = solver_df["Suggested Value"].sum()
        suggested_cost = solver_df["Suggested Cost"].sum()
        suggested_margin = suggested_value - suggested_cost
        suggested_margin_pct = suggested_margin / suggested_value * 100 if suggested_value > 0 else 0

        st.caption(
            f"Suggested margin: {suggested_margin_pct:.1f}% "
            f"(target {target_margin_pct:.0f}%). "
            f"Avg rate change: {((solver_df['Suggested Rate'] / solver_df['Base Rate']).mean() - 1) * 100:.1f}%. "
            f"Avg hours change: {((solver_df['Suggested Hours'] / solver_df['Hours'].replace(0, np.nan)).mean() - 1) * 100:.1f}%"
        )

        solver_view_cols = ["Task", "Hours", "Suggested Hours", "Base Rate", "Suggested Rate", "Overrun Risk %"]
        solver_view = solver_df[solver_view_cols].copy()
        st.dataframe(
            solver_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Hours": st.column_config.NumberColumn(format="%.1f"),
                "Suggested Hours": st.column_config.NumberColumn(format="%.1f"),
                "Base Rate": st.column_config.NumberColumn(format="$%.0f/hr"),
                "Suggested Rate": st.column_config.NumberColumn(format="$%.0f/hr"),
                "Overrun Risk %": st.column_config.NumberColumn(format="%.0f%%"),
            },
        )
        
        st.markdown("---")
        
        # =====================================================================
        # ACTIONS
        # =====================================================================
        action_cols = st.columns(4)
        
        with action_cols[0]:
            if st.button("Save Quote Plan", type="primary"):
                # Build plan object
                tasks = []
                for _, row in edited_df.iterrows():
                    if row["Hours"] > 0:
                        tasks.append(QuotePlanTask(
                            task_name=row["Task"],
                            hours=row["Hours"],
                            cost_per_hour=row.get(cost_col, 0) or 0,
                            quote_rate=row.get(rate_col, 0) or 0,
                        ))
                
                plan = QuotePlan(
                    department=selected_dept,
                    category=selected_cat,
                    tasks=tasks,
                    benchmark_window=benchmark_window,
                    recency_weighted=recency_weighted,
                    created_at=datetime.now().isoformat(),
                )
                
                set_quote_plan(plan)
                st.success("Quote plan saved!")
        
        with action_cols[1]:
            if st.button("Send to Capacity Planner"):
                # Save and redirect
                tasks = []
                for _, row in edited_df.iterrows():
                    if row["Hours"] > 0:
                        tasks.append(QuotePlanTask(
                            task_name=row["Task"],
                            hours=row["Hours"],
                            cost_per_hour=row.get(cost_col, 0) or 0,
                            quote_rate=row.get(rate_col, 0) or 0,
                        ))
                
                plan = QuotePlan(
                    department=selected_dept,
                    category=selected_cat,
                    tasks=tasks,
                    benchmark_window=benchmark_window,
                    recency_weighted=recency_weighted,
                    created_at=datetime.now().isoformat(),
                )
                
                set_quote_plan(plan)
                st.switch_page("pages/3_Capacity_Profiles.py")
        
        with action_cols[2]:
            # Export as CSV
            export_df = edited_df[["Task", "Hours"]].copy()
            export_df["Department"] = selected_dept
            export_df["Category"] = selected_cat
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                "Export CSV",
                data=csv,
                file_name=f"quote_plan_{selected_dept}_{selected_cat}.csv",
                mime="text/csv"
            )
        
        with action_cols[3]:
            if st.button("Clear Plan"):
                st.session_state.pop(plan_key, None)
                st.rerun()


if __name__ == "__main__":
    main()
