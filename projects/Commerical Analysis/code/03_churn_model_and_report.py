from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


PROJECT_DIR = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
REPORT_DIR = PROJECT_DIR / "outputs" / "report"

CLEANED_PATH = TABLES_DIR / "telco_churn_cleaned.csv"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(CLEANED_PATH)


target = "churn_flag"

numeric_features = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
    "has_internet_service",
    "has_additional_services",
]

categorical_features = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract_type",
    "paperless_billing",
    "payment_method",
]

X = df[numeric_features + categorical_features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


preprocess = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
    ]
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

metrics_df = pd.DataFrame(
    {
        "metric": ["accuracy", "precision", "recall", "roc_auc"],
        "value": [
            round(accuracy_score(y_test, y_pred), 4),
            round(precision_score(y_test, y_pred), 4),
            round(recall_score(y_test, y_pred), 4),
            round(roc_auc_score(y_test, y_prob), 4),
        ],
    }
)

metrics_df.to_csv(TABLES_DIR / "model_metrics.csv", index=False)

confusion_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=["Actual non-churn", "Actual churn"],
    columns=["Predicted non-churn", "Predicted churn"],
)

confusion_df.to_csv(TABLES_DIR / "confusion_matrix.csv")


scored_df = df.copy()
scored_df["predicted_churn_probability"] = pipeline.predict_proba(X)[:, 1]

scored_df["risk_band"] = pd.cut(
    scored_df["predicted_churn_probability"],
    bins=[0, 0.30, 0.60, 1.00],
    labels=["Low", "Medium", "High"],
    include_lowest=True,
)

risk_summary = (
    scored_df.groupby("risk_band", observed=False)
    .agg(
        customers=("customer_id", "count"),
        churned_customers=("churn_flag", "sum"),
        churn_rate_pct=("churn_flag", lambda x: round(x.mean() * 100, 2)),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
    )
    .reset_index()
)

risk_summary["avg_monthly_charges"] = risk_summary["avg_monthly_charges"].round(2)
risk_summary["avg_predicted_churn_probability"] = (
    risk_summary["avg_predicted_churn_probability"] * 100
).round(2)

risk_summary.to_csv(TABLES_DIR / "customer_risk_band_summary.csv", index=False)

scored_df[
    [
        "customer_id",
        "contract_type",
        "tenure",
        "tenure_band",
        "monthly_charge_band",
        "payment_method",
        "monthly_charges",
        "churn_flag",
        "predicted_churn_probability",
        "risk_band",
    ]
].to_csv(TABLES_DIR / "scored_customer_churn_risk.csv", index=False)


feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
coefficients = pipeline.named_steps["model"].coef_[0]

coef_df = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
    }
)

coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

top_positive_drivers = (
    coef_df.sort_values("coefficient", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

top_negative_drivers = (
    coef_df.sort_values("coefficient", ascending=True)
    .head(10)
    .reset_index(drop=True)
)

top_positive_drivers.to_csv(TABLES_DIR / "top_positive_churn_drivers.csv", index=False)
top_negative_drivers.to_csv(TABLES_DIR / "top_negative_churn_drivers.csv", index=False)


def clean_feature_name(value):
    value = value.replace("numeric__", "")
    value = value.replace("categorical__", "")
    value = value.replace("_", " ")
    return value


# Validation figure: ROC curve
roc_auc = metrics_df.loc[metrics_df["metric"] == "roc_auc", "value"].iloc[0]
fpr, tpr, _ = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_title("Churn Model ROC Curve")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_model_roc_curve.png", dpi=300)
plt.close()


# Business figure: risk bands
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(risk_summary["risk_band"].astype(str), risk_summary["churn_rate_pct"])
ax.set_title("Actual Churn Rate by Predicted Risk Band")
ax.set_xlabel("Predicted risk band")
ax.set_ylabel("Actual churn rate (%)")

for i, value in enumerate(risk_summary["churn_rate_pct"]):
    ax.text(i, value + 0.8, f"{value:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_churn_rate_by_risk_band.png", dpi=300)
plt.close()


# Interpretability figure: main positive drivers
driver_chart = top_positive_drivers.copy()
driver_chart["feature"] = driver_chart["feature"].apply(clean_feature_name)
driver_chart = driver_chart.sort_values("coefficient", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(driver_chart["feature"], driver_chart["coefficient"])
ax.set_title("Largest Positive Churn Model Coefficients")
ax.set_xlabel("Coefficient")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_positive_churn_coefficients.png", dpi=300)
plt.close()


overall_churn_rate = round(df["churn_flag"].mean() * 100, 2)
customers = len(df)
churned_customers = int(df["churn_flag"].sum())

contract_summary = pd.read_csv(TABLES_DIR / "churn_by_contract_type.csv")
tenure_summary = pd.read_csv(TABLES_DIR / "churn_by_tenure_band.csv")
payment_summary = pd.read_csv(TABLES_DIR / "churn_by_payment_method.csv")
charge_summary = pd.read_csv(TABLES_DIR / "churn_by_monthly_charge_band.csv")
high_risk_segments = pd.read_csv(TABLES_DIR / "high_risk_segments.csv")


html_report = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Commercial Analysis & Strategic Insight: IBM Telco Customer Churn</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 980px;
      margin: 40px auto;
      line-height: 1.55;
      color: #1f2937;
    }}
    h1, h2, h3 {{
      color: #111827;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 14px 0 26px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 8px;
      text-align: left;
    }}
    th {{
      background: #f3f4f6;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 20px 0 24px;
    }}
    .metric {{
      border: 1px solid #d1d5db;
      padding: 14px;
      border-radius: 8px;
      background: #f9fafb;
    }}
    .metric span {{
      display: block;
      font-size: 13px;
      color: #4b5563;
    }}
    .metric strong {{
      display: block;
      font-size: 22px;
      margin-top: 6px;
      color: #111827;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #e5e7eb;
      margin: 10px 0 24px;
    }}
    .note {{
      color: #4b5563;
      font-size: 14px;
    }}
  </style>
</head>
<body>

<h1>Commercial Analysis & Strategic Insight: IBM Telco Customer Churn</h1>

<h2>Executive summary</h2>
<p>
This analysis uses the IBM Telco customer churn dataset to identify the customer groups most associated with churn and to produce a simple churn-risk prioritisation model. The work combines SQL-based data preparation, commercial segmentation and an interpretable logistic regression model.
</p>

<div class="metric-grid">
  <div class="metric"><span>Customers analysed</span><strong>{customers:,}</strong></div>
  <div class="metric"><span>Churned customers</span><strong>{churned_customers:,}</strong></div>
  <div class="metric"><span>Overall churn rate</span><strong>{overall_churn_rate}%</strong></div>
  <div class="metric"><span>Model ROC AUC</span><strong>{roc_auc:.3f}</strong></div>
</div>

<p>
The strongest churn concentrations are among month-to-month customers, customers in their first year, electronic check users and customers with high monthly charges. These patterns point to retention opportunities around onboarding, contract conversion and targeted pricing or service interventions.
</p>

<h2>1. Churn patterns</h2>

<h3>Contract type</h3>
<p>
Month-to-month customers account for the highest churn rate, making contract structure a clear retention lever.
</p>
{contract_summary.to_html(index=False)}
<img src="../figures/01_churn_by_contract_type.png" alt="Churn rate by contract type">

<h3>Tenure</h3>
<p>
Churn is most concentrated in the first 12 months, suggesting that early customer experience is commercially important.
</p>
{tenure_summary.to_html(index=False)}
<img src="../figures/02_churn_by_tenure_band.png" alt="Churn rate by tenure band">

<h3>Payment method</h3>
<p>
Electronic check customers show the highest churn rate. This may reflect differences in customer profile, billing experience or payment friction.
</p>
{payment_summary.to_html(index=False)}
<img src="../figures/03_churn_by_payment_method.png" alt="Churn rate by payment method">

<h3>Monthly charge band</h3>
{charge_summary.to_html(index=False)}

<h2>2. High-risk segments</h2>
<p>
The highest-risk groups combine short tenure, month-to-month contracts, high monthly charges and specific payment methods.
</p>
{high_risk_segments.to_html(index=False)}

<h2>3. Churn-risk model</h2>
<p>
A logistic regression model was used to estimate churn probability. The model is intentionally simple and interpretable, which is appropriate for a commercial analysis setting.
</p>

{metrics_df.to_html(index=False)}
<img src="../figures/04_model_roc_curve.png" alt="ROC curve">

<p class="note">
The model is designed for prioritisation rather than exact forecasting. The ROC AUC indicates how well the model ranks customers by churn risk.
</p>

<h2>4. Customer risk bands</h2>
<p>
Predicted churn probabilities were grouped into low, medium and high-risk bands to support retention campaign prioritisation.
</p>

{risk_summary.to_html(index=False)}
<img src="../figures/05_churn_rate_by_risk_band.png" alt="Actual churn rate by predicted risk band">

<h2>5. Model interpretation</h2>
<p>
The chart below shows the largest positive logistic regression coefficients. These indicate variables associated with higher predicted churn, holding the other model inputs constant.
</p>

<img src="../figures/06_positive_churn_coefficients.png" alt="Largest positive churn coefficients">

<h2>6. Recommendations</h2>
<ol>
  <li>Prioritise first-year, month-to-month customers for onboarding and retention activity.</li>
  <li>Test contract-conversion offers for high-risk monthly customers.</li>
  <li>Review the payment and billing journey for electronic check customers.</li>
  <li>Use churn-risk bands to prioritise customer contact rather than applying broad retention campaigns.</li>
  <li>Track high-risk segment churn over time to assess whether interventions are reducing churn concentration.</li>
</ol>

<h2>7. Limitations</h2>
<p>
The dataset is static, so the model predicts churn probability rather than forecasting churn through time. The analysis should be interpreted as customer-risk scoring and commercial segmentation, not as a live operational churn system.
</p>

</body>
</html>
"""

html_path = REPORT_DIR / "commercial_analysis_telco_churn_report.html"
html_path.write_text(html_report, encoding="utf-8")


summary_text = f"""Commercial Analysis & Strategic Insight: IBM Telco Customer Churn

Customers analysed: {customers:,}
Churned customers: {churned_customers:,}
Overall churn rate: {overall_churn_rate}%
Model ROC AUC: {roc_auc:.3f}

Main findings:
- Month-to-month contracts have the highest churn rate.
- First-year customers show the strongest tenure-related churn concentration.
- Electronic check users have materially higher churn than automatic payment users.
- The high-risk model band captures customers with substantially higher observed churn.

Recommended actions:
- Prioritise first-year, month-to-month customers.
- Test targeted contract-conversion offers.
- Review payment and billing friction for electronic check customers.
- Use risk bands to prioritise retention activity.
"""

summary_path = REPORT_DIR / "commercial_analysis_telco_churn_summary.txt"
summary_path.write_text(summary_text, encoding="utf-8")


print("Model and report outputs updated.")
print(f"ROC AUC: {roc_auc:.3f}")
print(f"HTML report: {html_path}")
print(f"Summary: {summary_path}")