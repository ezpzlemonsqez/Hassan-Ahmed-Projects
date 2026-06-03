from pathlib import Path
import duckdb


# -----------------------------
# Project paths
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Telco-Customer-Churn.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
SQL_DIR = PROJECT_DIR / "sql"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# SQL cleaning query
# -----------------------------
cleaning_query = f"""
WITH raw_data AS (
    SELECT *
    FROM read_csv_auto('{DATA_PATH.as_posix()}')
),

cleaned AS (
    SELECT
        customerID AS customer_id,
        gender,
        SeniorCitizen AS senior_citizen,
        Partner AS partner,
        Dependents AS dependents,
        tenure,
        PhoneService AS phone_service,
        MultipleLines AS multiple_lines,
        InternetService AS internet_service,
        OnlineSecurity AS online_security,
        OnlineBackup AS online_backup,
        DeviceProtection AS device_protection,
        TechSupport AS tech_support,
        StreamingTV AS streaming_tv,
        StreamingMovies AS streaming_movies,
        Contract AS contract_type,
        PaperlessBilling AS paperless_billing,
        PaymentMethod AS payment_method,
        MonthlyCharges AS monthly_charges,
        CAST(NULLIF(TRIM(TotalCharges), '') AS DOUBLE) AS total_charges,
        Churn AS churn,
        CASE 
            WHEN Churn = 'Yes' THEN 1 
            ELSE 0 
        END AS churn_flag,

        CASE
            WHEN tenure < 12 THEN '0-12 months'
            WHEN tenure < 24 THEN '12-24 months'
            WHEN tenure < 48 THEN '24-48 months'
            ELSE '48+ months'
        END AS tenure_band,

        CASE
            WHEN MonthlyCharges < 35 THEN 'Low'
            WHEN MonthlyCharges < 70 THEN 'Medium'
            ELSE 'High'
        END AS monthly_charge_band,

        CASE
            WHEN Contract = 'Month-to-month' THEN 1
            ELSE 0
        END AS is_month_to_month,

        CASE
            WHEN Contract IN ('One year', 'Two year') THEN 1
            ELSE 0
        END AS is_long_term_contract,

        CASE
            WHEN InternetService = 'No' THEN 0
            ELSE 1
        END AS has_internet_service,

        CASE
            WHEN OnlineSecurity = 'Yes'
              OR OnlineBackup = 'Yes'
              OR DeviceProtection = 'Yes'
              OR TechSupport = 'Yes'
              OR StreamingTV = 'Yes'
              OR StreamingMovies = 'Yes'
            THEN 1
            ELSE 0
        END AS has_additional_services

    FROM raw_data
    WHERE NULLIF(TRIM(TotalCharges), '') IS NOT NULL
)

SELECT *
FROM cleaned;
"""


# -----------------------------
# Run SQL query
# -----------------------------
con = duckdb.connect()

cleaned_df = con.execute(cleaning_query).fetchdf()

# Save cleaned dataset
cleaned_path = TABLES_DIR / "telco_churn_cleaned.csv"
cleaned_df.to_csv(cleaned_path, index=False)

# Save SQL query for project evidence
sql_path = SQL_DIR / "01_clean_telco_data.sql"
sql_path.write_text(cleaning_query, encoding="utf-8")


# -----------------------------
# Create simple summary outputs
# -----------------------------
summary_query = """
SELECT
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2) AS avg_total_charges
FROM cleaned_df;
"""

contract_summary_query = """
SELECT
    contract_type,
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM cleaned_df
GROUP BY contract_type
ORDER BY churn_rate_pct DESC;
"""

tenure_summary_query = """
SELECT
    tenure_band,
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM cleaned_df
GROUP BY tenure_band
ORDER BY churn_rate_pct DESC;
"""

summary_df = con.execute(summary_query).fetchdf()
contract_summary_df = con.execute(contract_summary_query).fetchdf()
tenure_summary_df = con.execute(tenure_summary_query).fetchdf()

summary_df.to_csv(TABLES_DIR / "overall_churn_summary.csv", index=False)
contract_summary_df.to_csv(TABLES_DIR / "churn_by_contract_type.csv", index=False)
tenure_summary_df.to_csv(TABLES_DIR / "churn_by_tenure_band.csv", index=False)


# -----------------------------
# Console output
# -----------------------------
print("Data cleaning complete.")
print(f"Cleaned rows: {len(cleaned_df):,}")
print(f"Cleaned dataset saved to: {cleaned_path}")
print(f"SQL query saved to: {sql_path}")

print("\nOverall churn summary:")
print(summary_df)

print("\nChurn by contract type:")
print(contract_summary_df)

print("\nChurn by tenure band:")
print(tenure_summary_df)