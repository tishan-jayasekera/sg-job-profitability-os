# SG Job Profitability Operating System

A production-grade Streamlit web app for job profitability analysis, quoting discipline, delivery control, and capacity management.

## Canonical Hierarchy

All navigation follows:

**Company → `department_final` → `job_category` → (`task_name` | `staff_name`)**

Everything else (client, state, role, etc.) is filter-only.

## Features

### Pages

1. **Executive Summary** - KPIs, quote-delivery reconciliation, rate capture analysis, drill tables
2. **Quote Builder** - Historical benchmarks, task templates, economics preview
3. **Capacity & Staffing** - Capacity overview, staffing recommendations, staff scatter
4. **Active Delivery** - Risk monitoring, job breakdown, staff attribution
5. **Utilisation & Time Use** - Utilisation analysis, time breakdown, leakage identification
6. **Job Mix & Demand** - Intake trends, implied FTE demand, demand vs supply
7. **Data Quality & QA** - Schema validation, reconciliation, anomaly detection
8. **Glossary & Method** - Metric definitions, formulas, methodology

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Data Files

Put your data files in `./data/processed/`:

**Required:**
- `fact_timesheet_day_enriched.parquet` (or .csv)

**Optional:**
- `fact_job_task_month.parquet`
- `audit_revenue_reconciliation_job_month.parquet`
- `audit_unallocated_revenue.parquet`

### 3. Run the App

```bash
streamlit run app.py
```

### 4. (Optional) Build Marts

For faster loading with large datasets:

```bash
python scripts/build_marts.py
```

## Data Generation

Data files are produced by the parsing layer notebook:

```
01_parse_and_unify_job_profitability.ipynb
```

See the Data Dictionary for column specifications.

## Project Structure

```
job-profitability-os/
├── app.py                      # Main Streamlit entry point
├── pages/                      # Page modules
│   ├── 1_Executive_Summary.py
│   ├── 2_Quote_Builder.py
│   ├── 3_Capacity_Staffing.py
│   ├── 4_Active_Delivery.py
│   ├── 5_Utilisation_Time_Use.py
│   ├── 6_Job_Mix_and_Demand.py
│   ├── 7_Data_Quality_QA.py
│   └── 8_Glossary_Method.py
├── src/
│   ├── config.py               # Configuration
│   ├── exports.py              # Export utilities
│   ├── data/
│   │   ├── loader.py           # Data loading with caching
│   │   ├── schema.py           # Schema validation
│   │   ├── semantic.py         # Safe aggregations
│   │   ├── cohorts.py          # Time windows, recency
│   │   └── marts.py            # Mart builders
│   └── ui/
│       ├── state.py            # Session state management
│       ├── layout.py           # Layout components
│       ├── formatting.py       # Number formatting
│       └── charts.py           # Chart wrappers
├── scripts/
│   ├── build_marts.py          # Offline mart builder
│   └── validate_inputs.py      # Schema validation
├── tests/                      # Unit tests
├── .streamlit/
│   └── config.toml             # Streamlit config
├── requirements.txt
└── README.md
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Data directory path |
| `APP_ENV` | `dev` | Environment (dev/prod) |
| `CACHE_TTL_SECONDS` | `3600` | Cache time-to-live |
| `ACTIVE_JOB_RECENCY_DAYS` | `21` | Days for active job definition |
| `ACTIVE_STAFF_RECENCY_MONTHS` | `6` | Months for active staff definition |
| `RECENCY_HALF_LIFE_MONTHS` | `6` | Half-life for recency weighting |

## Aggregation Safety

### Quote Fields

Quote fields repeat on every fact row. Always dedupe at job-task level:

```python
# CORRECT
from src.data.semantic import safe_quote_rollup
quote_totals = safe_quote_rollup(df, ["department_final"])

# WRONG (inflates values!)
quote_total = df["quoted_time_total"].sum()
```

### Rate Calculations

Always compute rates as weighted averages:

```python
realised_rate = df["rev_alloc"].sum() / df["hours_raw"].sum()  # ✓
realised_rate = df["realised_rate_alloc"].mean()  # ✗
```

### Leave Exclusion

For utilisation calculations, exclude leave tasks:

```python
from src.data.semantic import exclude_leave
df_no_leave = exclude_leave(df)
```

## Deployment

### Streamlit Cloud

1. Push to GitHub
2. Connect repo to Streamlit Cloud
3. Set secrets in Streamlit Cloud dashboard
4. Configure data source (upload or connect to storage)

### Local/Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_aggregations.py

# Run with verbose output
pytest -v tests/
```

## Key Metrics Reference

| Metric | Formula |
|--------|---------|
| Revenue | `Σ rev_alloc` |
| Margin | `Revenue - Cost` |
| Margin % | `Margin / Revenue × 100` |
| Realised Rate | `Revenue / Hours` |
| Quote Rate | `Quoted Amount / Quoted Hours` (safe rollup) |
| Hours Variance | `Actual Hours - Quoted Hours` |
| Scope Creep % | `Unquoted Hours / Total Hours × 100` |
| Utilisation | `Billable Hours / Total Hours × 100` (excl. leave) |
| Implied FTE | `Quoted Hours / (weeks × 38 × util_target)` |

## License

Proprietary - SG Internal Use Only
