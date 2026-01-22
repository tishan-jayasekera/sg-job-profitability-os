#!/usr/bin/env python
"""
Validate input data files against schema requirements.

Usage:
    python scripts/validate_inputs.py
    python scripts/validate_inputs.py --data-dir /path/to/data
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import config, REQUIRED_COLUMNS, OPTIONAL_COLUMNS, TABLE_FILES
from src.data.schema import validate_schema, check_optional_columns


def validate_file(filepath: Path, table_name: str) -> dict:
    """Validate a single file."""
    result = {
        "exists": False,
        "format": None,
        "rows": 0,
        "columns": 0,
        "valid": False,
        "missing_required": [],
        "missing_optional": [],
        "errors": []
    }
    
    # Check file existence
    parquet_path = filepath.with_suffix(".parquet")
    csv_path = filepath.with_suffix(".csv")
    
    if parquet_path.exists():
        result["exists"] = True
        result["format"] = "parquet"
        load_path = parquet_path
    elif csv_path.exists():
        result["exists"] = True
        result["format"] = "csv"
        load_path = csv_path
    else:
        result["errors"].append(f"File not found: {filepath}.(parquet|csv)")
        return result
    
    # Load file
    try:
        if result["format"] == "parquet":
            df = pd.read_parquet(load_path)
        else:
            df = pd.read_csv(load_path)
        
        result["rows"] = len(df)
        result["columns"] = len(df.columns)
    except Exception as e:
        result["errors"].append(f"Failed to load: {e}")
        return result
    
    # Validate schema
    try:
        schema_result = validate_schema(df, table_name, strict=False)
        result["valid"] = schema_result["is_valid"]
        result["missing_required"] = schema_result["missing_required"]
        result["missing_optional"] = schema_result["missing_optional"]
    except Exception as e:
        result["errors"].append(f"Schema validation failed: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate input data files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory"
    )
    
    args = parser.parse_args()
    
    # Set data dir
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = config.data_dir
    
    processed_dir = data_dir / "processed"
    
    print("=" * 60)
    print("Data Input Validation")
    print("=" * 60)
    print(f"Source directory: {processed_dir}")
    print()
    
    all_valid = True
    
    # Validate each table
    for table_key, filename in TABLE_FILES.items():
        filepath = processed_dir / filename
        
        print(f"Validating: {table_key}")
        print("-" * 40)
        
        result = validate_file(filepath, table_key)
        
        if result["exists"]:
            print(f"  ✓ Found: {filename}.{result['format']}")
            print(f"    Rows: {result['rows']:,}")
            print(f"    Columns: {result['columns']}")
            
            if result["valid"]:
                print(f"  ✓ Schema valid")
            else:
                print(f"  ✗ Schema invalid")
                print(f"    Missing required: {result['missing_required']}")
                all_valid = False
            
            if result["missing_optional"]:
                print(f"  ⚠ Missing optional: {result['missing_optional']}")
        else:
            print(f"  ✗ Not found: {filename}")
            if table_key in ["fact_timesheet", "fact_job_task_month"]:
                all_valid = False
                print(f"    (REQUIRED)")
            else:
                print(f"    (optional)")
        
        if result["errors"]:
            for err in result["errors"]:
                print(f"  ✗ Error: {err}")
            all_valid = False
        
        print()
    
    print("=" * 60)
    if all_valid:
        print("✓ All validations passed")
        sys.exit(0)
    else:
        print("✗ Validation failed - see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
