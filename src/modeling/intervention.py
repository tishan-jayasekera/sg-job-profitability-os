"""
Operational Intervention Engine - Risk scoring and job ranking for delivery leaders.

This module computes a simple, explainable risk score and identifies top reason codes
for active jobs to help delivery leaders prioritize interventions.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional


def compute_intervention_risk_score(
    row: pd.Series,
    margin_threshold: float = 15.0,
    revenue_coverage_threshold: float = 0.7,
    hours_overrun_threshold: float = 10.0,
    rate_leakage_threshold: float = 0.85,
) -> Tuple[float, List[str]]:
    """
    Compute risk score (0–100) and reason codes for a job.
    
    Args:
        row: Job row with fields:
             - margin_pct_to_date: Actual margin % to date
             - quote_to_revenue: Revenue earned / Quoted revenue
             - hours_overrun_pct: (Actual - Quoted) / Quoted * 100
             - realised_rate: Revenue to date / Actual hours
             - quote_rate: Quoted amount / Quoted hours
             - runtime_days: Days since job start
             - peer_median_runtime_days: Peer median runtime
        - margin_threshold: If margin % is below this, flag
        - revenue_coverage_threshold: If revenue / quote is below this, flag
        - hours_overrun_threshold: If hours overrun % is above this, flag
        - rate_leakage_threshold: If realised_rate / quote_rate is below this, flag
    
    Returns:
        (risk_score (0-100), list of reason codes (human readable))
    """
    reason_codes = []
    risk_score = 0.0
    
    # 1. Margin erosion
    margin = row.get('margin_pct_to_date', np.nan)
    if pd.notna(margin):
        if margin < margin_threshold:
            reason_codes.append("Low margin %")
            risk_score += max(0, (margin_threshold - margin) / margin_threshold * 30)
    
    # 2. Revenue lag
    rev_coverage = row.get('quote_to_revenue', np.nan)
    if pd.notna(rev_coverage):
        if rev_coverage < revenue_coverage_threshold:
            reason_codes.append("Revenue lagging quote")
            risk_score += max(0, (revenue_coverage_threshold - rev_coverage) / revenue_coverage_threshold * 25)
    
    # 3. Scope creep (hours overrun)
    hours_overrun = row.get('hours_overrun_pct', np.nan)
    if pd.notna(hours_overrun):
        if hours_overrun > hours_overrun_threshold:
            reason_codes.append("Hours overrun vs quote")
            risk_score += max(0, (hours_overrun - hours_overrun_threshold) / 50 * 25)
    
    # 4. Rate leakage
    real_rate = row.get('realised_rate', np.nan)
    quote_rate = row.get('quote_rate', np.nan)
    if pd.notna(real_rate) and pd.notna(quote_rate) and quote_rate > 0:
        rate_ratio = real_rate / quote_rate
        if rate_ratio < rate_leakage_threshold:
            reason_codes.append("Realized rate below quote")
            risk_score += max(0, (rate_leakage_threshold - rate_ratio) / rate_leakage_threshold * 20)
    
    # 5. Runtime over peer median (zombie risk)
    runtime = row.get('runtime_days', np.nan)
    peer_median = row.get('peer_median_runtime_days', np.nan)
    if pd.notna(runtime) and pd.notna(peer_median) and peer_median > 0:
        if runtime > peer_median * 1.5:  # 50% over median
            reason_codes.append("Runtime exceeds peers")
            risk_score += max(0, min((runtime - peer_median) / peer_median / 1.5 * 20, 20))
    
    # Cap risk score at 100
    risk_score = min(risk_score, 100.0)
    
    # Return top 2-3 reason codes
    top_reasons = reason_codes[:3]
    
    return risk_score, top_reasons


def build_intervention_queue(
    quadrant_jobs: pd.DataFrame,
    active_only: bool = True,
    sort_by: str = "risk_score",  # or "margin", "revenue_lag", "hours_overrun"
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Build ranked intervention queue for a quadrant.
    
    Args:
        quadrant_jobs: Jobs in the quadrant with all financials and metadata
        active_only: Filter to active jobs only
        sort_by: Column to sort by
        top_n: Return only top N jobs
        
    Returns:
        DataFrame with columns:
        - job_no
        - risk_score
        - primary_issue
        - margin_delta
        - hours_delta
        - rate_delta
        - job_age_days
        - owner
        (plus any other columns from input)
    """
    df = quadrant_jobs.copy()
    
    # Filter active jobs
    if active_only:
        if "is_active" in df.columns:
            df = df[df["is_active"] == True]
        elif "job_status" in df.columns:
            df = df[~df["job_status"].str.lower().str.contains("completed", na=False)]
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Compute risk scores and reasons
    results = df.apply(
        lambda row: compute_intervention_risk_score(row),
        axis=1,
        result_type="expand"
    )
    results.columns = ["risk_score", "reason_codes"]
    
    df["risk_score"] = results["risk_score"]
    df["primary_issue"] = results["reason_codes"].apply(lambda x: "; ".join(x) if x else "Monitor")
    
    # Compute deltas for table columns
    df["margin_delta"] = df.get("margin_pct_to_date", 0) - df.get("margin_pct_quote", 0)
    df["hours_delta"] = df.get("hours_overrun_pct", 0)
    df["rate_delta"] = df.get("realised_rate", 0) - df.get("quote_rate", 0)
    df["job_age_days"] = df.get("runtime_days", np.nan)
    df["owner"] = df.get("delivery_lead", "—")
    
    # Sort by risk score descending (highest risk first)
    df = df.sort_values("risk_score", ascending=False)
    
    # Return top N
    return df.head(top_n)


def compute_quadrant_health_summary(quadrant_jobs: pd.DataFrame) -> Dict[str, any]:
    """
    Compute health KPIs for a quadrant.
    
    Returns dict with keys:
    - job_count
    - quoted_revenue_exposure
    - median_margin_pct
    - median_margin_pct_quote
    - median_realized_rate
    - median_quoted_rate
    - pct_breaching_guardrails
    - avg_risk_score
    """
    if len(quadrant_jobs) == 0:
        return {
            'job_count': 0,
            'quoted_revenue_exposure': 0,
            'median_margin_pct': np.nan,
            'median_margin_pct_quote': np.nan,
            'median_realized_rate': np.nan,
            'median_quoted_rate': np.nan,
            'pct_breaching_guardrails': np.nan,
            'avg_risk_score': np.nan,
        }
    
    df = quadrant_jobs.copy()
    
    # Compute risk scores if not present
    if "risk_score" not in df.columns:
        results = df.apply(
            lambda row: compute_intervention_risk_score(row)[0],
            axis=1
        )
        df["risk_score"] = results
    
    return {
        'job_count': len(df),
        'quoted_revenue_exposure': df.get("quoted_amount", 0).sum(),
        'median_margin_pct': df.get("margin_pct_to_date", np.nan).median(),
        'median_margin_pct_quote': df.get("margin_pct_quote", np.nan).median(),
        'median_realized_rate': df.get("realised_rate", np.nan).median(),
        'median_quoted_rate': df.get("quote_rate", np.nan).median(),
        'pct_breaching_guardrails': (df["risk_score"] > 50).sum() / len(df) * 100 if len(df) > 0 else np.nan,
        'avg_risk_score': df["risk_score"].mean(),
    }


def get_peer_context(
    job_row: pd.Series,
    peer_segment: pd.DataFrame,
) -> Dict[str, any]:
    """
    Compute job's position vs peers.
    
    Args:
        job_row: Single job
        peer_segment: All peers in same (quadrant, dept, category, ...)
        
    Returns:
        Dict with percentile info
    """
    if len(peer_segment) == 0:
        return {}
    
    result = {}
    
    # Runtime percentile
    if "runtime_days" in job_row.index and "runtime_days" in peer_segment.columns:
        runtime_pct = (peer_segment["runtime_days"] <= job_row["runtime_days"]).sum() / len(peer_segment) * 100
        result['runtime_percentile'] = runtime_pct
    
    # Margin percentile
    if "margin_pct_to_date" in job_row.index and "margin_pct_to_date" in peer_segment.columns:
        margin_pct = (peer_segment["margin_pct_to_date"] <= job_row["margin_pct_to_date"]).sum() / len(peer_segment) * 100
        result['margin_percentile'] = margin_pct
    
    # Realized rate percentile
    if "realised_rate" in job_row.index and "realised_rate" in peer_segment.columns:
        rate_pct = (peer_segment["realised_rate"] <= job_row["realised_rate"]).sum() / len(peer_segment) * 100
        result['rate_percentile'] = rate_pct
    
    return result
