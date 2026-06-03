from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
SQL_DIR = PROJECT_DIR / "sql"

CLEANED_PATH = TABLES_DIR / "telco_churn_cleaned.csv"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(CLEANED_PATH)

con = duckdb.connect()
con.register("telco", df)


eda_queries = {
    "churn_by_contract_type": """
        SELECT
            contract_type,
            COUNT(*) AS customers,
            SUM(churn_flag) AS churned_customers,
            ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
        FROM telco
        GROUP BY contract_type
        ORDER BY churn_rate_pct DESC;
    """,

    "churn_by_tenure_band": """
        SELECT
            tenure_band,
            COUNT(*) AS customers,
            SUM(churn_flag) AS churned_customers,
            ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
        FROM telco
        GROUP BY tenure_band
        ORDER BY churn_rate_pct DESC;
    """,

    "churn_by_payment_method": """
        SELECT
            payment_method,
            COUNT(*) AS customers,
            SUM(churn_flag) AS churned_customers,
            ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
        FROM telco
        GROUP BY payment_method
        ORDER BY churn_rate_pct DESC;
    """,

    "churn_by_monthly_charge_band": """
        SELECT
            monthly_charge_band,
            COUNT(*) AS customers,
            SUM(churn_flag) AS churned_customers,
            ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
        FROM telco
        GROUP BY monthly_charge_band
        ORDER BY churn_rate_pct DESC;
    """,

    "high_risk_segments": """
        SELECT
            contract_type,
            tenure_band,
            monthly_charge_band,
            payment_method,
            COUNT(*) AS customers,
            SUM(churn_flag) AS churned_customers,
            ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
        FROM telco
        GROUP BY
            contract_type,
            tenure_band,
            monthly_charge_band,
            payment_method
        HAVING COUNT(*) >= 50
        ORDER BY churn_rate_pct DESC, customers DESC
        LIMIT 10;
    """
}


results = {}

for name, query in eda_queries.items():
    output = con.execute(query).fetchdf()
    results[name] = output
    output.to_csv(TABLES_DIR / f"{name}.csv", index=False)

sql_text = "\n\n".join(
    f"-- {name}\n{query.strip()}" for name, query in eda_queries.items()
)
(SQL_DIR / "02_commercial_eda_queries.sql").write_text(sql_text, encoding="utf-8")


def save_churn_bar(data, x_col, title, filename, order=None):
    plot_data = data.copy()

    if order is not None:
        plot_data[x_col] = pd.Categorical(plot_data[x_col], categories=order, ordered=True)
        plot_data = plot_data.sort_values(x_col)
    else:
        plot_data = plot_data.sort_values("churn_rate_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_data[x_col].astype(str), plot_data["churn_rate_pct"])

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Churn rate (%)")
    plt.xticks(rotation=25, ha="right")

    for i, value in enumerate(plot_data["churn_rate_pct"]):
        ax.text(i, value + 0.8, f"{value:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


save_churn_bar(
    results["churn_by_contract_type"],
    "contract_type",
    "Churn Rate by Contract Type",
    "01_churn_by_contract_type.png",
    order=["Month-to-month", "One year", "Two year"],
)

save_churn_bar(
    results["churn_by_tenure_band"],
    "tenure_band",
    "Churn Rate by Tenure Band",
    "02_churn_by_tenure_band.png",
    order=["0-12 months", "12-24 months", "24-48 months", "48+ months"],
)

save_churn_bar(
    results["churn_by_payment_method"],
    "payment_method",
    "Churn Rate by Payment Method",
    "03_churn_by_payment_method.png",
)


print("EDA complete.")
print(f"Tables saved: {TABLES_DIR}")
print(f"Figures saved: {FIGURES_DIR}")
print(f"SQL saved: {SQL_DIR / '02_commercial_eda_queries.sql'}")