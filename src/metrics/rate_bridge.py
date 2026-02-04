"""
Three Forces Rate Bridge decomposition.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.data.semantic import safe_quote_rollup
from src.metrics.quote_delivery import compute_scope_creep


@dataclass
class RateBridgeResult:
    """Container for rate bridge decomposition results."""
    quote_rate: float
    realised_rate: float
    total_gap: float

    delivery_effect: float
    delivery_effect_pct: float
    hours_underrun_effect: float
    hours_overrun_effect: float
    scope_creep_effect: float

    commercial_effect: float
    commercial_effect_pct: float
    revenue_recovery_effect: float
    revenue_leakage_effect: float

    mix_effect: float
    mix_effect_pct: float

    reconciliation_check: float

    total_quoted_hours: float
    total_actual_hours: float
    total_quoted_amount: float
    total_actual_revenue: float
    hours_variance: float
    hours_variance_pct: float
    revenue_capture_pct: float


def compute_rate_bridge(df: pd.DataFrame, job_df: Optional[pd.DataFrame] = None) -> RateBridgeResult:
    """
    Compute the three forces rate bridge decomposition.
    """
    quote_totals = safe_quote_rollup(df, [])
    total_quoted_hours = quote_totals["quoted_hours"].iloc[0] if len(quote_totals) > 0 else 0
    total_quoted_amount = quote_totals["quoted_amount"].iloc[0] if len(quote_totals) > 0 else 0

    total_actual_hours = df["hours_raw"].sum()
    total_actual_revenue = df["rev_alloc"].sum()

    quote_rate = total_quoted_amount / total_quoted_hours if total_quoted_hours > 0 else 0
    realised_rate = total_actual_revenue / total_actual_hours if total_actual_hours > 0 else 0
    total_gap = realised_rate - quote_rate

    hours_variance = total_actual_hours - total_quoted_hours
    hours_variance_pct = (
        hours_variance / total_quoted_hours * 100 if total_quoted_hours > 0 else 0
    )

    scope = compute_scope_creep(df)
    unquoted_hours = scope["unquoted_hours"].iloc[0] if len(scope) > 0 else 0
    quoted_work_variance = hours_variance - unquoted_hours

    if job_df is not None and "hours_variance_job" in job_df.columns:
        underrun_hours = job_df[job_df["hours_variance_job"] < 0]["hours_variance_job"].sum()
        overrun_hours = job_df[job_df["hours_variance_job"] > 0]["hours_variance_job"].sum()
    else:
        if quoted_work_variance < 0:
            underrun_hours = quoted_work_variance
            overrun_hours = 0
        else:
            underrun_hours = 0
            overrun_hours = quoted_work_variance

    revenue_capture_pct = (
        total_actual_revenue / total_quoted_amount * 100 if total_quoted_amount > 0 else 100
    )
    revenue_variance = total_actual_revenue - total_quoted_amount

    if job_df is not None and "revenue_job" in job_df.columns and "quoted_amount_job" in job_df.columns:
        job_copy = job_df.copy()
        job_copy["revenue_variance_job"] = job_copy["revenue_job"] - job_copy["quoted_amount_job"]
        recovery_revenue = job_copy[job_copy["revenue_variance_job"] > 0]["revenue_variance_job"].sum()
        leakage_revenue = job_copy[job_copy["revenue_variance_job"] < 0]["revenue_variance_job"].sum()
    else:
        if revenue_variance > 0:
            recovery_revenue = revenue_variance
            leakage_revenue = 0
        else:
            recovery_revenue = 0
            leakage_revenue = revenue_variance

    hours_underrun_effect = 0
    if total_actual_hours > 0 and underrun_hours < 0:
        hours_underrun_effect = (-underrun_hours * quote_rate) / total_actual_hours

    hours_overrun_effect = 0
    if total_actual_hours > 0 and overrun_hours > 0:
        hours_overrun_effect = -(overrun_hours * quote_rate) / total_actual_hours

    scope_creep_effect = 0
    if total_actual_hours > 0 and unquoted_hours > 0:
        scope_creep_effect = -(unquoted_hours * quote_rate) / total_actual_hours

    delivery_effect = hours_underrun_effect + hours_overrun_effect + scope_creep_effect
    delivery_effect_pct = (delivery_effect / quote_rate * 100) if quote_rate > 0 else 0

    revenue_recovery_effect = 0
    if total_actual_hours > 0 and recovery_revenue > 0:
        revenue_recovery_effect = recovery_revenue / total_actual_hours

    revenue_leakage_effect = 0
    if total_actual_hours > 0 and leakage_revenue < 0:
        revenue_leakage_effect = leakage_revenue / total_actual_hours

    commercial_effect = revenue_recovery_effect + revenue_leakage_effect
    commercial_effect_pct = (commercial_effect / quote_rate * 100) if quote_rate > 0 else 0

    mix_effect = total_gap - delivery_effect - commercial_effect
    mix_effect_pct = (mix_effect / quote_rate * 100) if quote_rate > 0 else 0

    reconciliation_check = total_gap - (delivery_effect + commercial_effect + mix_effect)

    return RateBridgeResult(
        quote_rate=quote_rate,
        realised_rate=realised_rate,
        total_gap=total_gap,
        delivery_effect=delivery_effect,
        delivery_effect_pct=delivery_effect_pct,
        hours_underrun_effect=hours_underrun_effect,
        hours_overrun_effect=hours_overrun_effect,
        scope_creep_effect=scope_creep_effect,
        commercial_effect=commercial_effect,
        commercial_effect_pct=commercial_effect_pct,
        revenue_recovery_effect=revenue_recovery_effect,
        revenue_leakage_effect=revenue_leakage_effect,
        mix_effect=mix_effect,
        mix_effect_pct=mix_effect_pct,
        reconciliation_check=reconciliation_check,
        total_quoted_hours=total_quoted_hours,
        total_actual_hours=total_actual_hours,
        total_quoted_amount=total_quoted_amount,
        total_actual_revenue=total_actual_revenue,
        hours_variance=hours_variance,
        hours_variance_pct=hours_variance_pct,
        revenue_capture_pct=revenue_capture_pct,
    )


def compute_rate_bridge_by_group(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """
    Compute rate bridge for each group (e.g., department).
    """
    results = []
    for group_val in df[group_key].dropna().unique():
        group_df = df[df[group_key] == group_val]
        bridge = compute_rate_bridge(group_df)
        results.append(
            {
                group_key: group_val,
                "quote_rate": bridge.quote_rate,
                "realised_rate": bridge.realised_rate,
                "total_gap": bridge.total_gap,
                "delivery_effect": bridge.delivery_effect,
                "commercial_effect": bridge.commercial_effect,
                "mix_effect": bridge.mix_effect,
                "hours_variance_pct": bridge.hours_variance_pct,
                "revenue_capture_pct": bridge.revenue_capture_pct,
            }
        )
    return pd.DataFrame(results)


def get_dominant_force(bridge: RateBridgeResult) -> Tuple[str, float, str]:
    """
    Identify the dominant force driving the rate gap.
    """
    forces = [
        ("Delivery Efficiency", bridge.delivery_effect, "Delivery Team"),
        ("Commercial Capture", bridge.commercial_effect, "Commercial Team"),
        ("Portfolio Mix", bridge.mix_effect, "BD/Intake"),
    ]
    return max(forces, key=lambda x: abs(x[1]))


def generate_bridge_insight(bridge: RateBridgeResult) -> str:
    """
    Generate plain-English insight from bridge results.
    """
    insights = []
    if bridge.total_gap > 0:
        insights.append(f"Realised rate is **${bridge.total_gap:+.0f}/hr above quote**.")
    elif bridge.total_gap < 0:
        insights.append(f"Realised rate is **${bridge.total_gap:.0f}/hr below quote**.")
    else:
        insights.append("Realised rate matches quote rate exactly.")

    force_name, effect, owner = get_dominant_force(bridge)
    direction = "lifting" if effect > 0 else "dragging down"
    insights.append(
        f"The dominant force is **{force_name}** ({direction} rate by "
        f"${abs(effect):.0f}/hr) — owned by **{owner}**."
    )

    if bridge.delivery_effect < 0 and bridge.total_gap > 0:
        insights.append(
            "⚠️ **The Contradiction:** Delivery leaked (overruns), but rate still beat quote. "
            "Mix or commercial recovery masked the delivery problem — this may not be sustainable."
        )

    return "\n\n".join(insights)
