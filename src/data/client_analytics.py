"""
Client analytics: profitability, LTV, and driver diagnostics.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Tuple

from src.data.semantic import profitability_rollup, quote_delivery_metrics, get_category_col, safe_quote_rollup


def _as_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")
    return pd.to_datetime(df[col])


@st.cache_data(show_spinner=False)
def compute_client_rollup(df: pd.DataFrame) -> pd.DataFrame:
    if "client" not in df.columns or len(df) == 0:
        return pd.DataFrame()
    rollup = profitability_rollup(df, ("client",))
    jobs = df.groupby("client")["job_no"].nunique().rename("job_count").reset_index()
    rollup = rollup.merge(jobs, on="client", how="left")
    return rollup


@st.cache_data(show_spinner=False)
def compute_client_portfolio_summary(df: pd.DataFrame) -> Dict[str, float]:
    rollup = compute_client_rollup(df)
    if len(rollup) == 0:
        return {
            "total_clients": 0,
            "portfolio_revenue": 0.0,
            "portfolio_profit": 0.0,
            "median_margin_pct": np.nan,
            "unprofitable_share": np.nan,
            "top5_profit_share": np.nan,
        }

    total_profit = rollup["margin"].sum()
    unprofitable_share = (rollup["margin"] < 0).mean() * 100 if len(rollup) > 0 else np.nan

    top5 = rollup.sort_values("margin", ascending=False).head(5)
    top5_profit_share = (
        (top5["margin"].sum() / total_profit * 100) if total_profit != 0 else np.nan
    )

    return {
        "total_clients": rollup["client"].nunique(),
        "portfolio_revenue": rollup["revenue"].sum(),
        "portfolio_profit": total_profit,
        "median_margin_pct": rollup["margin_pct"].median(),
        "unprofitable_share": unprofitable_share,
        "top5_profit_share": top5_profit_share,
    }


@st.cache_data(show_spinner=False)
def compute_client_quadrants(
    df: pd.DataFrame,
    y_mode: str = "profit",
) -> Tuple[pd.DataFrame, float, float]:
    rollup = compute_client_rollup(df)
    if len(rollup) == 0:
        return pd.DataFrame(), np.nan, np.nan

    rollup = rollup.copy()
    rollup["x_revenue"] = rollup["revenue"]

    if y_mode == "margin_pct":
        rollup["y_value"] = rollup["margin_pct"]
    else:
        rollup["y_value"] = rollup["margin"]

    med_x = rollup["x_revenue"].median()
    med_y = rollup["y_value"].median()

    def _label(row: pd.Series) -> str:
        high_x = row["x_revenue"] >= med_x
        high_y = row["y_value"] >= med_y
        if high_x and high_y:
            return "Partners"
        if high_x and not high_y:
            return "Underperformers"
        if not high_x and high_y:
            return "Niche"
        return "Drain"

    rollup["quadrant"] = rollup.apply(_label, axis=1)
    return rollup, med_x, med_y


@st.cache_data(show_spinner=False)
def compute_client_hours_overrun(df: pd.DataFrame) -> pd.DataFrame:
    if "quoted_time_total" not in df.columns:
        return pd.DataFrame()
    return quote_delivery_metrics(df, ("client",))


@st.cache_data(show_spinner=False)
def compute_client_growth(df: pd.DataFrame, months: int = 3) -> pd.DataFrame:
    if "month_key" not in df.columns or "client" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["month_key"] = _as_datetime(df, "month_key")
    months_sorted = sorted(df["month_key"].dropna().unique())
    if len(months_sorted) == 0:
        return pd.DataFrame()

    recent = months_sorted[-months:]
    prior = months_sorted[-months * 2:-months] if len(months_sorted) >= months * 2 else []

    recent_df = df[df["month_key"].isin(recent)]
    prior_df = df[df["month_key"].isin(prior)] if prior else pd.DataFrame(columns=df.columns)

    recent_rev = recent_df.groupby("client")["rev_alloc"].sum().rename("recent_revenue")
    prior_rev = prior_df.groupby("client")["rev_alloc"].sum().rename("prior_revenue")

    growth = pd.concat([recent_rev, prior_rev], axis=1).fillna(0).reset_index()
    growth["revenue_growth_pct"] = np.where(
        growth["prior_revenue"] > 0,
        (growth["recent_revenue"] - growth["prior_revenue"]) / growth["prior_revenue"] * 100,
        np.nan,
    )
    return growth


@st.cache_data(show_spinner=False)
def compute_primary_driver(df: pd.DataFrame) -> pd.DataFrame:
    if "client" not in df.columns or "department_final" not in df.columns:
        return pd.DataFrame(columns=["client", "primary_driver"])

    dept = profitability_rollup(df, ("client", "department_final"))
    dept = dept.sort_values(["client", "margin_pct"], ascending=[True, True])
    worst = dept.groupby("client").head(1).copy()
    worst["primary_driver"] = np.where(
        worst["margin"] < 0,
        "Loss in Dept " + worst["department_final"].astype(str),
        "Margin drag in Dept " + worst["department_final"].astype(str),
    )

    overrun = pd.DataFrame()
    if "quoted_time_total" in df.columns:
        overrun = quote_delivery_metrics(df, ("client", "department_final"))
        if len(overrun) > 0:
            overrun = overrun.sort_values(["client", "hours_variance_pct"], ascending=[True, False])
            overrun = overrun.groupby("client").head(1).copy()
            overrun["primary_driver"] = np.where(
                overrun["hours_variance_pct"] > 10,
                "Overrun in Dept " + overrun["department_final"].astype(str),
                np.nan,
            )
            overrun = overrun[["client", "primary_driver"]]

    if len(overrun) > 0:
        worst = worst.merge(overrun, on="client", how="left", suffixes=("", "_overrun"))
        worst["primary_driver"] = worst["primary_driver_overrun"].fillna(worst["primary_driver"])
        worst = worst[["client", "primary_driver"]]

    return worst


@st.cache_data(show_spinner=False)
def compute_client_queue(df: pd.DataFrame, quadrant: Optional[str], mode: str, y_mode: str = "profit") -> pd.DataFrame:
    rollup = compute_client_rollup(df)
    if len(rollup) == 0:
        return pd.DataFrame()

    overrun = compute_client_hours_overrun(df)
    growth = compute_client_growth(df)
    drivers = compute_primary_driver(df)

    if len(overrun) > 0:
        rollup = rollup.merge(
            overrun[["client", "hours_variance_pct"]],
            on="client",
            how="left",
        )
    else:
        rollup["hours_variance_pct"] = np.nan

    if len(growth) > 0:
        rollup = rollup.merge(growth[["client", "revenue_growth_pct"]], on="client", how="left")
    else:
        rollup["revenue_growth_pct"] = np.nan

    if len(drivers) > 0:
        rollup = rollup.merge(drivers, on="client", how="left")
    else:
        rollup["primary_driver"] = "—"

    if quadrant:
        quadrant_df, _, _ = compute_client_quadrants(df, y_mode=y_mode)
        if len(quadrant_df) > 0:
            rollup = rollup.merge(quadrant_df[["client", "quadrant"]], on="client", how="left")
            rollup = rollup[rollup["quadrant"] == quadrant]

    if mode == "Growth":
        rollup = rollup.sort_values(
            ["revenue_growth_pct", "margin_pct"], ascending=[False, False]
        )
    else:
        rollup = rollup.sort_values(
            ["margin_pct", "hours_variance_pct"], ascending=[True, False]
        )

    return rollup


@st.cache_data(show_spinner=False)
def compute_client_job_ledger(df: pd.DataFrame) -> pd.DataFrame:
    if "job_no" not in df.columns:
        return pd.DataFrame()

    category_col = get_category_col(df)
    ledger = df.groupby(["job_no", "department_final", category_col]).agg(
        hours=("hours_raw", "sum"),
        cost=("base_cost", "sum"),
        revenue=("rev_alloc", "sum"),
    ).reset_index()

    # Safe quote rollup at job level
    quote = safe_quote_rollup(df, ("job_no",))
    if len(quote) > 0:
        ledger = ledger.merge(
            quote[["job_no", "quoted_hours", "quoted_amount"]],
            on="job_no",
            how="left",
        )
    else:
        ledger["quoted_hours"] = np.nan
        ledger["quoted_amount"] = np.nan

    ledger["margin"] = ledger["revenue"] - ledger["cost"]
    ledger["margin_pct"] = np.where(
        ledger["revenue"] > 0,
        ledger["margin"] / ledger["revenue"] * 100,
        np.nan,
    )
    ledger["quoted_cost"] = np.where(
        ledger["hours"] > 0,
        ledger["quoted_hours"] * (ledger["cost"] / ledger["hours"]),
        np.nan,
    )
    ledger["quoted_margin_pct"] = np.where(
        ledger["quoted_amount"] > 0,
        (ledger["quoted_amount"] - ledger["quoted_cost"]) / ledger["quoted_amount"] * 100,
        np.nan,
    )
    ledger["realised_rate"] = np.where(
        ledger["hours"] > 0,
        ledger["revenue"] / ledger["hours"],
        np.nan,
    )
    ledger.rename(columns={category_col: "job_category"}, inplace=True)
    return ledger


@st.cache_data(show_spinner=False)
def compute_client_department_profit(df: pd.DataFrame) -> pd.DataFrame:
    if "department_final" not in df.columns:
        return pd.DataFrame()
    dept = profitability_rollup(df, ("department_final",))
    return dept.sort_values("margin", ascending=True)


@st.cache_data(show_spinner=False)
def compute_client_task_mix(df: pd.DataFrame) -> pd.DataFrame:
    if "task_name" not in df.columns:
        return pd.DataFrame()
    task = df.groupby("task_name")["hours_raw"].sum().reset_index()
    total = task["hours_raw"].sum()
    task["share_pct"] = np.where(total > 0, task["hours_raw"] / total * 100, 0)
    return task


@st.cache_data(show_spinner=False)
def compute_global_task_median_mix(df: pd.DataFrame) -> pd.DataFrame:
    if "client" not in df.columns or "task_name" not in df.columns:
        return pd.DataFrame(columns=["task_name", "global_share_pct"])
    client_task = df.groupby(["client", "task_name"])["hours_raw"].sum().reset_index()
    client_total = client_task.groupby("client")["hours_raw"].sum().rename("client_hours")
    client_task = client_task.merge(client_total.reset_index(), on="client", how="left")
    client_task["share_pct"] = np.where(
        client_task["client_hours"] > 0,
        client_task["hours_raw"] / client_task["client_hours"] * 100,
        0,
    )
    global_median = client_task.groupby("task_name")["share_pct"].median().reset_index()
    global_median.rename(columns={"share_pct": "global_share_pct"}, inplace=True)
    return global_median


@st.cache_data(show_spinner=False)
def compute_company_cost_rate(df: pd.DataFrame) -> float:
    if "hours_raw" not in df.columns or "base_cost" not in df.columns:
        return np.nan
    total_hours = df["hours_raw"].sum()
    if total_hours == 0:
        return np.nan
    return df["base_cost"].sum() / total_hours


@st.cache_data(show_spinner=False)
def compute_client_ltv(df: pd.DataFrame, client_name: str) -> Dict[str, pd.DataFrame]:
    if "client" not in df.columns or client_name is None:
        return {"monthly": pd.DataFrame(), "cumulative": pd.DataFrame()}

    df_client = df[df["client"] == client_name].copy()
    if len(df_client) == 0:
        return {"monthly": pd.DataFrame(), "cumulative": pd.DataFrame()}

    date_col = "month_key" if "month_key" in df_client.columns else "work_date"
    df_client[date_col] = _as_datetime(df_client, date_col)

    monthly = df_client.groupby(date_col).agg(
        revenue=("rev_alloc", "sum"),
        cost=("base_cost", "sum"),
        hours=("hours_raw", "sum"),
    ).reset_index().sort_values(date_col)
    monthly["margin"] = monthly["revenue"] - monthly["cost"]
    monthly["margin_pct"] = np.where(
        monthly["revenue"] > 0,
        monthly["margin"] / monthly["revenue"] * 100,
        np.nan,
    )

    monthly["cumulative_profit"] = monthly["margin"].cumsum()
    monthly["months_since_start"] = (
        (monthly[date_col].dt.year - monthly[date_col].min().year) * 12
        + (monthly[date_col].dt.month - monthly[date_col].min().month)
    )

    return {"monthly": monthly, "cumulative": monthly[[date_col, "months_since_start", "cumulative_profit"]].copy()}


@st.cache_data(show_spinner=False)
def compute_client_tenure_months(df: pd.DataFrame, client_name: str) -> int:
    if "client" not in df.columns or client_name is None:
        return 0
    df_client = df[df["client"] == client_name].copy()
    if len(df_client) == 0:
        return 0
    date_col = "month_key" if "month_key" in df_client.columns else "work_date"
    dates = _as_datetime(df_client, date_col).dropna()
    if len(dates) == 0:
        return 0
    min_date = dates.min()
    max_date = dates.max()
    return int((max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1)
