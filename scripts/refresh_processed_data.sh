#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: scripts/refresh_processed_data.sh /path/to/new/processed_files"
  exit 1
fi

src_dir="$1"
dest_dir="src/data/processed"

expected_files=(
  "audit_revenue_reconciliation_job_month.csv"
  "audit_revenue_reconciliation_job_month.parquet"
  "audit_unallocated_revenue.csv"
  "audit_unallocated_revenue.parquet"
  "fact_job_task_month.csv"
  "fact_job_task_month.parquet"
  "fact_timesheet_day_enriched.csv"
  "fact_timesheet_day_enriched.parquet"
)

for filename in "${expected_files[@]}"; do
  if [ ! -f "${src_dir}/${filename}" ]; then
    echo "Missing ${filename} in ${src_dir}"
    exit 1
  fi
done

mkdir -p "${dest_dir}"

for filename in "${expected_files[@]}"; do
  cp -f "${src_dir}/${filename}" "${dest_dir}/${filename}"
done

git add "${dest_dir}"/*.csv "${dest_dir}"/*.parquet

echo "Processed data replaced and staged."
echo "Next: git commit -m \"Refresh processed data\" && git push"
