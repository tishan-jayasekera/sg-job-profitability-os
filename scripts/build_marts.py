#!/usr/bin/env python
"""
Build precomputed marts for faster app loading.

Usage:
    python scripts/build_marts.py
    python scripts/build_marts.py --data-dir /path/to/data
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.loader import _load_file
from src.data.marts import build_all_marts


def main():
    parser = argparse.ArgumentParser(description="Build precomputed marts")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Set data dir if provided
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = config.data_dir
    
    processed_dir = data_dir / "processed"
    marts_dir = data_dir / "marts"
    
    print(f"Building marts...")
    print(f"  Source: {processed_dir}")
    print(f"  Output: {marts_dir}")
    print()
    
    # Load fact table
    fact_path = processed_dir / "fact_timesheet_day_enriched"
    
    df = _load_file(fact_path)
    
    if df is None:
        print(f"ERROR: Could not load fact_timesheet_day_enriched from {processed_dir}")
        print("Please ensure the file exists as .parquet or .csv")
        sys.exit(1)
    
    print(f"Loaded {len(df):,} rows from fact_timesheet_day_enriched")
    print()
    
    # Ensure date columns
    date_cols = ["month_key", "work_date", "job_start_date", "job_due_date", "job_completed_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Build marts
    try:
        marts = build_all_marts(df, output_dir=marts_dir)
        print()
        print("✓ All marts built successfully!")
        print()
        print("Summary:")
        for name, mart_df in marts.items():
            print(f"  {name}: {len(mart_df):,} rows")
    except Exception as e:
        print(f"ERROR building marts: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import pandas as pd
    main()
