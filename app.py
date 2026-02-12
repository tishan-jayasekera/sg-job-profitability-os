"""
SG Job Profitability Operating System

Main entry point for Streamlit app.
"""
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Job Profitability OS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.state import init_state
from src.data.loader import load_fact_timesheet, get_data_status
from src.data.schema import validate_schema, display_validation_result
from src.config import config, TABLE_FILES


def main():
    """Main app entry point."""
    
    # Initialize session state
    init_state()
    
    # Header
    st.title("Job Profitability Operating System")
    st.caption("Company → Department → Category → Task/Staff")
    
    # Check data availability
    status = get_data_status()
    
    fact_available = (
        status["processed"]["fact_timesheet"]["parquet_exists"] or 
        status["processed"]["fact_timesheet"]["csv_exists"]
    )
    
    if not fact_available:
        st.error("No data found!")
        st.markdown(f"""
        ### Setup Required
        
        Please place your data files in: `{config.processed_dir}`
        
        Required files:
        - `fact_timesheet_day_enriched.parquet` (or .csv)
        - `fact_job_task_month.parquet` (or .csv)
        
        Optional files:
        - `audit_revenue_reconciliation_job_month.parquet`
        - `audit_unallocated_revenue.parquet`
        
        **To generate these files:**
        Run the parsing layer notebook: `01_parse_and_unify_job_profitability.ipynb`
        """)
        
        st.info("Once data is in place, refresh this page.")
        return

    with st.expander("Data fingerprint", expanded=False):
        rows = []
        for filename in TABLE_FILES.values():
            for ext in ("parquet", "csv"):
                path = config.processed_dir / f"{filename}.{ext}"
                if path.exists():
                    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                    rows.append({
                        "file": path.name,
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                        "modified_utc": mtime.strftime("%Y-%m-%d %H:%M"),
                    })
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info(f"No files found in {config.processed_dir}.")
    
    # Load and validate data
    with st.spinner("Loading data..."):
        try:
            df = load_fact_timesheet()
            result = validate_schema(df, "fact_timesheet", strict=False)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Show validation status
    if not result["is_valid"]:
        st.warning(f"Missing required columns: {result['missing_required']}")
        st.info("Some features may be limited.")
    
    # Navigation
    st.markdown("---")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("### Quick Links")
        st.page_link("pages/1_Executive_Summary.py", label="Executive Summary", icon="📈")
        st.page_link("pages/2_Quote_Builder.py", label="Quote Builder", icon="📝")
        st.page_link("pages/4_Active_Delivery.py", label="Active Delivery", icon="🎯")
        st.page_link("pages/5_Revenue_Reconciliation.py", label="Revenue Reconciliation", icon="💰")
        st.page_link("pages/6_Job_Mix_and_Demand.py", label="Job Mix & Demand", icon="📊")
        st.page_link("pages/7_Data_Quality_QA.py", label="Data Quality & QA", icon="✅")
        st.page_link("pages/8_Glossary_Method.py", label="Glossary & Method", icon="📖")
    
    with col2:
        st.markdown("### Data Overview")
        
        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.metric("Total Rows", f"{len(df):,}")
        
        with c2:
            if "job_no" in df.columns:
                st.metric("Jobs", f"{df['job_no'].nunique():,}")
        
        with c3:
            if "staff_name" in df.columns:
                st.metric("Staff", f"{df['staff_name'].nunique():,}")
        
        with c4:
            if "month_key" in df.columns:
                min_date = df["month_key"].min()
                max_date = df["month_key"].max()
                st.metric("Date Range", f"{min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}")
        
        # Key metrics
        st.markdown("#### Key Metrics")
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            if "rev_alloc" in df.columns:
                st.metric("Total Revenue", f"${df['rev_alloc'].sum():,.0f}")
        
        with m2:
            if "base_cost" in df.columns:
                st.metric("Total Cost", f"${df['base_cost'].sum():,.0f}")
        
        with m3:
            if "rev_alloc" in df.columns and "base_cost" in df.columns:
                margin = df["rev_alloc"].sum() - df["base_cost"].sum()
                margin_pct = margin / df["rev_alloc"].sum() * 100 if df["rev_alloc"].sum() > 0 else 0
                st.metric("Margin", f"${margin:,.0f} ({margin_pct:.1f}%)")
        
        with m4:
            if "hours_raw" in df.columns:
                st.metric("Total Hours", f"{df['hours_raw'].sum():,.0f}")
    
    # Data status
    st.markdown("---")
    with st.expander("Data Status"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Processed Tables**")
            for key, info in status["processed"].items():
                icon = "✅" if info["parquet_exists"] or info["csv_exists"] else "❌"
                format_used = "parquet" if info["parquet_exists"] else "csv" if info["csv_exists"] else "missing"
                st.markdown(f"{icon} `{key}` ({format_used})")
        
        with col2:
            st.markdown("**Precomputed Marts**")
            for key, info in status["marts"].items():
                icon = "✅" if info["parquet_exists"] or info["csv_exists"] else "⚪"
                format_used = "parquet" if info["parquet_exists"] else "csv" if info["csv_exists"] else "not built"
                st.markdown(f"{icon} `{key}` ({format_used})")
            
            if not any(info["parquet_exists"] or info["csv_exists"] for info in status["marts"].values()):
                st.info("Run `python scripts/build_marts.py` to build marts for faster loading.")


if __name__ == "__main__":
    main()
