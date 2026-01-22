"""
Data Quality & QA Page

Make numbers defensible with reconciliation checks and data quality metrics.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import init_state
from src.ui.layout import section_header
from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_count
from src.data.loader import (
    load_fact_timesheet, load_audit_reconciliation, 
    load_audit_unallocated, get_data_status
)
from src.data.schema import validate_schema, get_column_info
from src.data.semantic import get_category_col
from src.config import config, REQUIRED_COLUMNS, OPTIONAL_COLUMNS


st.set_page_config(page_title="Data Quality & QA", page_icon="✅", layout="wide")

init_state()


def calculate_coverage_stats(df: pd.DataFrame) -> dict:
    """Calculate field coverage statistics."""
    stats = {}
    
    for col in df.columns:
        non_null = df[col].notna().sum()
        total = len(df)
        stats[col] = {
            "non_null": non_null,
            "null": total - non_null,
            "coverage_pct": non_null / total * 100 if total > 0 else 0,
        }
    
    return stats


def check_quote_match_coverage(df: pd.DataFrame) -> dict:
    """Check quote match coverage."""
    if "quote_match_flag" not in df.columns:
        return {"error": "quote_match_flag column not found"}
    
    matched = df[df["quote_match_flag"] == "matched"]
    unmatched = df[df["quote_match_flag"] != "matched"]
    
    return {
        "matched_rows": len(matched),
        "unmatched_rows": len(unmatched),
        "matched_pct": len(matched) / len(df) * 100 if len(df) > 0 else 0,
        "matched_hours": matched["hours_raw"].sum() if "hours_raw" in matched.columns else 0,
        "unmatched_hours": unmatched["hours_raw"].sum() if "hours_raw" in unmatched.columns else 0,
        "hours_matched_pct": matched["hours_raw"].sum() / df["hours_raw"].sum() * 100 
            if "hours_raw" in df.columns and df["hours_raw"].sum() > 0 else 0,
    }


def check_allocation_modes(df: pd.DataFrame) -> pd.DataFrame:
    """Check distribution of allocation modes."""
    if "alloc_mode" not in df.columns:
        return pd.DataFrame()
    
    mode_dist = df.groupby("alloc_mode").agg(
        rows=("alloc_mode", "count"),
        hours=("hours_raw", "sum") if "hours_raw" in df.columns else ("alloc_mode", "count"),
        revenue=("rev_alloc", "sum") if "rev_alloc" in df.columns else ("alloc_mode", "count"),
    ).reset_index()
    
    total_rows = mode_dist["rows"].sum()
    mode_dist["pct"] = mode_dist["rows"] / total_rows * 100 if total_rows > 0 else 0
    
    return mode_dist


def check_department_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Check department source distribution."""
    if "department_final_source" not in df.columns:
        return pd.DataFrame()
    
    source_dist = df.groupby("department_final_source").agg(
        rows=("department_final_source", "count"),
    ).reset_index()
    
    total = source_dist["rows"].sum()
    source_dist["pct"] = source_dist["rows"] / total * 100 if total > 0 else 0
    
    return source_dist.sort_values("rows", ascending=False)


def identify_anomalies(df: pd.DataFrame) -> list:
    """Identify data anomalies."""
    anomalies = []
    
    # Negative hours
    if "hours_raw" in df.columns:
        neg_hours = df[df["hours_raw"] < 0]
        if len(neg_hours) > 0:
            anomalies.append({
                "type": "Negative Hours",
                "count": len(neg_hours),
                "impact": f"{neg_hours['hours_raw'].sum():.1f} hours",
                "severity": "warning"
            })
    
    # Missing department
    if "department_final" in df.columns:
        missing_dept = df[df["department_final"].isna()]
        if len(missing_dept) > 0:
            anomalies.append({
                "type": "Missing Department",
                "count": len(missing_dept),
                "impact": f"{len(missing_dept) / len(df) * 100:.1f}% of rows",
                "severity": "warning"
            })
    
    # Missing category (revenue-first)
    category_col = get_category_col(df)
    if category_col in df.columns:
        missing_cat = df[df[category_col].isna()]
        if len(missing_cat) > 0:
            anomalies.append({
                "type": "Missing Job Category",
                "count": len(missing_cat),
                "impact": f"{len(missing_cat) / len(df) * 100:.1f}% of rows",
                "severity": "warning"
            })
    
    # Revenue without hours
    if "rev_alloc" in df.columns and "hours_raw" in df.columns and "job_no" in df.columns:
        job_rev = df.groupby("job_no").agg(
            hours=("hours_raw", "sum"),
            revenue=("rev_alloc", "sum"),
        ).reset_index()
        
        rev_no_hours = job_rev[(job_rev["revenue"] > 0) & (job_rev["hours"] == 0)]
        if len(rev_no_hours) > 0:
            anomalies.append({
                "type": "Revenue Without Hours",
                "count": len(rev_no_hours),
                "impact": f"${rev_no_hours['revenue'].sum():,.0f} unallocated",
                "severity": "error"
            })
    
    # Very high realised rates (possible data issue)
    if "realised_rate_alloc" in df.columns:
        high_rate = df[df["realised_rate_alloc"] > 1000]
        if len(high_rate) > 0:
            anomalies.append({
                "type": "Unusually High Rates",
                "count": len(high_rate),
                "impact": f"Rates > $1,000/hr",
                "severity": "info"
            })
    
    return anomalies


def main():
    st.title("Data Quality & QA")
    st.caption("Make numbers defensible with data quality checks")
    
    # Load data
    df = load_fact_timesheet()
    audit_recon = load_audit_reconciliation()
    audit_unalloc = load_audit_unallocated()
    
    # =========================================================================
    # SECTION A: DATA STATUS
    # =========================================================================
    section_header("Data Status")
    
    status = get_data_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Processed Tables**")
        for key, info in status["processed"].items():
            if info["parquet_exists"]:
                st.success(f"✓ {key} (parquet)")
            elif info["csv_exists"]:
                st.success(f"✓ {key} (csv)")
            else:
                st.error(f"✗ {key} (missing)")
    
    with col2:
        st.markdown("**Precomputed Marts**")
        for key, info in status["marts"].items():
            if info["parquet_exists"]:
                st.success(f"✓ {key}")
            elif info["csv_exists"]:
                st.success(f"✓ {key}")
            else:
                st.warning(f"○ {key} (not built)")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION B: SCHEMA VALIDATION
    # =========================================================================
    section_header("Schema Validation")
    
    result = validate_schema(df, "fact_timesheet", strict=False)
    
    if result["is_valid"]:
        st.success(f"✓ All required columns present ({result['total_columns']} columns, {result['total_rows']:,} rows)")
    else:
        st.error(f"✗ Missing required columns: {result['missing_required']}")
    
    if result["missing_optional"]:
        st.warning(f"Missing optional columns: {result['missing_optional']}")
    
    # Field coverage
    with st.expander("Field Coverage Details"):
        coverage = calculate_coverage_stats(df)
        
        coverage_df = pd.DataFrame([
            {
                "Column": col,
                "Non-Null": stats["non_null"],
                "Null": stats["null"],
                "Coverage %": stats["coverage_pct"]
            }
            for col, stats in coverage.items()
        ])
        
        # Sort by coverage ascending (show worst first)
        coverage_df = coverage_df.sort_values("Coverage %")
        
        st.dataframe(
            coverage_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Non-Null": st.column_config.NumberColumn(format="%d"),
                "Null": st.column_config.NumberColumn(format="%d"),
                "Coverage %": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION C: REVENUE RECONCILIATION
    # =========================================================================
    section_header("Revenue Reconciliation")
    
    if len(audit_recon) > 0:
        # Summary stats
        total_revenue = audit_recon["rev_total_job_month"].sum() if "rev_total_job_month" in audit_recon.columns else 0
        total_allocated = audit_recon["allocated_rev_sum"].sum() if "allocated_rev_sum" in audit_recon.columns else 0
        total_delta = abs(audit_recon["delta"]).sum() if "delta" in audit_recon.columns else 0
        
        recon_cols = st.columns(4)
        
        with recon_cols[0]:
            st.metric("Revenue Pool", fmt_currency(total_revenue))
        
        with recon_cols[1]:
            st.metric("Allocated", fmt_currency(total_allocated))
        
        with recon_cols[2]:
            st.metric("Total Delta", fmt_currency(total_delta))
        
        with recon_cols[3]:
            delta_pct = total_delta / total_revenue * 100 if total_revenue > 0 else 0
            st.metric("Delta %", fmt_percent(delta_pct))
        
        # Show problem rows
        if "delta" in audit_recon.columns:
            problem_rows = audit_recon[abs(audit_recon["delta"]) > 1].sort_values("delta", ascending=False)
            
            if len(problem_rows) > 0:
                st.warning(f"{len(problem_rows)} job-months with reconciliation delta > $1")
                
                with st.expander("View Problem Rows"):
                    st.dataframe(problem_rows.head(20), use_container_width=True, hide_index=True)
            else:
                st.success("✓ All job-months reconcile within $1 tolerance")
    else:
        st.info("No reconciliation audit data available")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION D: UNALLOCATED REVENUE
    # =========================================================================
    section_header("Unallocated Revenue")
    
    if len(audit_unalloc) > 0:
        total_unalloc = audit_unalloc["rev_total_job_month"].sum() if "rev_total_job_month" in audit_unalloc.columns else 0
        
        st.metric("Total Unallocated Revenue", fmt_currency(total_unalloc))
        
        st.warning(f"{len(audit_unalloc)} job-months have revenue but no timesheet hours to allocate to")
        
        with st.expander("View Unallocated Revenue"):
            st.dataframe(
                audit_unalloc.sort_values("rev_total_job_month", ascending=False).head(50),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.success("✓ All revenue has been allocated")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION E: QUOTE MATCH COVERAGE
    # =========================================================================
    section_header("Quote Match Coverage")
    
    quote_coverage = check_quote_match_coverage(df)
    
    if "error" not in quote_coverage:
        qc_cols = st.columns(3)
        
        with qc_cols[0]:
            st.metric("Matched Rows", fmt_count(quote_coverage["matched_rows"]))
        
        with qc_cols[1]:
            st.metric("Unmatched Rows", fmt_count(quote_coverage["unmatched_rows"]))
        
        with qc_cols[2]:
            st.metric("Hours Matched %", fmt_percent(quote_coverage["hours_matched_pct"]))
        
        if quote_coverage["hours_matched_pct"] < 80:
            st.warning(f"Only {quote_coverage['hours_matched_pct']:.1f}% of hours have matching quotes. Consider improving quote data coverage.")
    else:
        st.info("Quote match data not available")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION F: ALLOCATION MODE DISTRIBUTION
    # =========================================================================
    section_header("Allocation Mode Distribution")
    
    alloc_modes = check_allocation_modes(df)
    
    if len(alloc_modes) > 0:
        st.dataframe(
            alloc_modes.rename(columns={
                "alloc_mode": "Mode",
                "rows": "Rows",
                "hours": "Hours",
                "revenue": "Revenue",
                "pct": "%"
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Hours": st.column_config.NumberColumn(format="%.0f"),
                "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                "%": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )
    else:
        st.info("Allocation mode data not available")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION G: DEPARTMENT PROVENANCE
    # =========================================================================
    section_header("Department Source")
    
    dept_source = check_department_provenance(df)
    
    if len(dept_source) > 0:
        st.dataframe(
            dept_source.rename(columns={
                "department_final_source": "Source",
                "rows": "Rows",
                "pct": "%"
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "%": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )
        
        # Check if too many from fallback
        ts_fallback = dept_source[dept_source["department_final_source"] == "timesheet_fallback"]
        if len(ts_fallback) > 0 and ts_fallback["pct"].iloc[0] > 20:
            st.warning("Over 20% of departments sourced from timesheet fallback. Consider improving revenue department data.")
    else:
        st.info("Department source tracking not available")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION H: ANOMALIES
    # =========================================================================
    section_header("Data Anomalies")
    
    anomalies = identify_anomalies(df)
    
    if anomalies:
        for a in anomalies:
            if a["severity"] == "error":
                st.error(f"**{a['type']}**: {a['count']} occurrences - {a['impact']}")
            elif a["severity"] == "warning":
                st.warning(f"**{a['type']}**: {a['count']} occurrences - {a['impact']}")
            else:
                st.info(f"**{a['type']}**: {a['count']} occurrences - {a['impact']}")
    else:
        st.success("✓ No significant anomalies detected")


if __name__ == "__main__":
    main()
