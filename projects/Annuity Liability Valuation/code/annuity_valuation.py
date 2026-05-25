from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\hassa\Documents\Code\Python\Annuity Liability Valuation")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
BOE_DIR = RAW_DIR / "boe"
HMD_DIR = RAW_DIR / "hmd"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Assumptions
# ============================================================

ASSUMPTIONS = {
    "valuation_age": 65,
    "maximum_age": 110,
    "annual_pension": 12_000,
    "annual_expense": 75,
    "inflation_rate": 0.025,
    "payment_timing": "Annual payments in arrears",
}


# ============================================================
# Data loading: Bank of England nominal spot curve
# ============================================================

def find_boe_nominal_file() -> Path:
    """
    Finds the BoE nominal daily data workbook inside data/raw/boe.
    Expected file name is similar to:
    GLC Nominal daily data current month.xlsx
    """

    matches = list(BOE_DIR.rglob("*Nominal*daily*.xlsx"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find the BoE nominal daily Excel file in {BOE_DIR}.\n"
            "Expected something like: GLC Nominal daily data current month.xlsx"
        )

    return matches[0]


def extract_latest_nominal_spot_curve(boe_file: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Reads sheet '4. spot curve' from the BoE nominal workbook.

    The workbook has:
    - maturities in row 4
    - dates in column A
    - spot rates across columns B onward
    - rates are in percent, so we divide by 100
    """

    sheet_name = "4. spot curve"

    raw = pd.read_excel(
        boe_file,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl"
    )

    # Row 4 in Excel is index 3 in pandas.
    maturities = raw.iloc[3, 1:].dropna().astype(float)

    # Data starts after metadata rows.
    data = raw.iloc[5:, :].copy()
    data = data.dropna(how="all")

    # First column should be dates.
    data_dates = pd.to_datetime(data.iloc[:, 0], errors="coerce")
    data = data.loc[data_dates.notna()].copy()
    data_dates = data_dates.loc[data_dates.notna()]

    # Find the latest row with usable numeric spot rates.
    latest_curve = None
    latest_date = None

    for idx in reversed(data.index):
        row_date = pd.to_datetime(data.loc[idx, 0], errors="coerce")
        rate_values = pd.to_numeric(data.loc[idx, 1:len(maturities)], errors="coerce")

        if rate_values.notna().sum() >= 10:
            latest_curve = rate_values
            latest_date = row_date
            break

    if latest_curve is None:
        raise ValueError("Could not find a usable spot curve row in the BoE workbook.")

    curve = pd.DataFrame(
        {
            "maturity": maturities.values,
            "spot_rate": latest_curve.values / 100.0,  # convert percent to decimal
        }
    )

    curve = curve.dropna()
    curve = curve[curve["maturity"] >= 1.0]
    curve = curve.sort_values("maturity").reset_index(drop=True)

    output_path = PROCESSED_DIR / "nominal_spot_curve.csv"
    curve.to_csv(output_path, index=False)

    print(f"Saved cleaned BoE nominal spot curve: {output_path}")
    print(f"Latest BoE curve date used: {latest_date.date()}")

    return curve, latest_date


# ============================================================
# Data loading: HMD life table
# ============================================================

def find_hmd_life_table_file() -> Path:
    """
    Finds the HMD both-sex period life table file.
    Expected file:
    bltper_1x1.txt
    """

    matches = list(HMD_DIR.rglob("bltper_1x1.txt"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find bltper_1x1.txt in {HMD_DIR}.\n"
            "Place the HMD total both-sex 1x1 period life table there."
        )

    return matches[0]


def read_hmd_life_table(hmd_file: Path) -> pd.DataFrame:
    """
    Reads HMD period life table text file.

    Expected columns:
    Year, Age, mx, qx, ax, lx, dx, Lx, Tx, ex
    """

    lines = hmd_file.read_text(encoding="utf-8", errors="ignore").splitlines()

    header_row = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Year") and "Age" in line and "qx" in line:
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not find the HMD table header row.")

    df = pd.read_csv(
        hmd_file,
        sep=r"\s+",
        skiprows=header_row,
        engine="python"
    )

    df["Age"] = (
        df["Age"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .astype(int)
    )

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["qx"] = pd.to_numeric(df["qx"], errors="coerce")
    df["lx"] = pd.to_numeric(df["lx"], errors="coerce")
    df["ex"] = pd.to_numeric(df["ex"], errors="coerce")

    df = df.dropna(subset=["Year", "Age", "qx"])

    return df


def extract_latest_mortality_table(hmd: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Uses the latest available HMD year and extracts age/qx.
    """

    latest_year = int(hmd["Year"].max())

    mortality = (
        hmd[hmd["Year"] == latest_year]
        .copy()
        .sort_values("Age")
    )

    mortality = mortality[["Age", "qx", "lx", "ex"]].rename(
        columns={
            "Age": "age",
            "qx": "qx",
            "lx": "lx",
            "ex": "life_expectancy",
        }
    )

    output_path = PROCESSED_DIR / "mortality_table.csv"
    mortality[["age", "qx"]].to_csv(output_path, index=False)

    detailed_output_path = PROCESSED_DIR / "mortality_table_detailed.csv"
    mortality.to_csv(detailed_output_path, index=False)

    print(f"Saved cleaned HMD mortality table: {output_path}")
    print(f"Latest HMD year used: {latest_year}")

    return mortality, latest_year


# ============================================================
# Annuity valuation model
# ============================================================

def interpolate_spot_rates(curve: pd.DataFrame, projection_years: np.ndarray) -> np.ndarray:
    """
    Interpolates BoE spot rates to annual projection years.
    For maturities beyond the curve, the final available rate is used.
    """

    maturities = curve["maturity"].to_numpy()
    rates = curve["spot_rate"].to_numpy()

    interpolated = np.interp(
        projection_years,
        maturities,
        rates,
        left=rates[0],
        right=rates[-1],
    )

    return interpolated


def build_annuity_cashflows(
    mortality: pd.DataFrame,
    spot_curve: pd.DataFrame,
    valuation_age: int = 65,
    maximum_age: int = 110,
    annual_pension: float = 12_000,
    annual_expense: float = 75,
    inflation_rate: float = 0.025,
    rate_shift: float = 0.0,
    mortality_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Builds annual survival-weighted annuity cashflows.

    Assumption:
    - Single-life annuity.
    - Annual payments in arrears.
    - Payment at year t is made if the policyholder survives to that payment date.
    - qx is adjusted by mortality_multiplier for longevity sensitivities.
    """

    table = mortality[
        (mortality["age"] >= valuation_age)
        & (mortality["age"] <= maximum_age)
    ].copy()

    table = table.sort_values("age").reset_index(drop=True)

    ages = table["age"].to_numpy()
    qx = table["qx"].to_numpy() * mortality_multiplier
    qx = np.clip(qx, 0.0, 1.0)

    projection_years = np.arange(1, len(ages) + 1)

    # Survival to each payment date.
    # For payment at t=1, survival probability is 1 - q_65.
    survival_probabilities = np.cumprod(1.0 - qx)

    spot_rates = interpolate_spot_rates(spot_curve, projection_years) + rate_shift
    spot_rates = np.maximum(spot_rates, -0.99)

    discount_factors = 1.0 / ((1.0 + spot_rates) ** projection_years)

    pension_payments = annual_pension * ((1.0 + inflation_rate) ** (projection_years - 1))
    expenses = annual_expense * ((1.0 + inflation_rate) ** (projection_years - 1))

    gross_cashflows = pension_payments + expenses
    expected_cashflows = survival_probabilities * gross_cashflows
    present_values = expected_cashflows * discount_factors

    cashflows = pd.DataFrame(
        {
            "projection_year": projection_years,
            "payment_age": ages + 1,
            "qx": qx,
            "survival_probability": survival_probabilities,
            "spot_rate": spot_rates,
            "discount_factor": discount_factors,
            "pension_payment": pension_payments,
            "expense": expenses,
            "gross_cashflow": gross_cashflows,
            "expected_cashflow": expected_cashflows,
            "present_value": present_values,
        }
    )

    return cashflows


def calculate_bel(cashflows: pd.DataFrame) -> float:
    """
    Best-estimate liability is the sum of discounted expected cashflows.
    """

    return float(cashflows["present_value"].sum())


def run_sensitivity_scenarios(mortality: pd.DataFrame, spot_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Runs core sensitivities:
    - interest rates
    - longevity
    - inflation
    - expenses
    """

    base_params = ASSUMPTIONS.copy()

    scenarios = {
        "Base": {},
        "Rates +100bps": {"rate_shift": 0.0100},
        "Rates -100bps": {"rate_shift": -0.0100},
        "Longevity improvement: qx -10%": {"mortality_multiplier": 0.90},
        "Mortality worsening: qx +10%": {"mortality_multiplier": 1.10},
        "Inflation +100bps": {"inflation_rate": base_params["inflation_rate"] + 0.0100},
        "Inflation -100bps": {"inflation_rate": max(base_params["inflation_rate"] - 0.0100, 0.0)},
        "Expenses +10%": {"annual_expense": base_params["annual_expense"] * 1.10},
    }

    rows = []
    base_bel = None

    for scenario_name, override in scenarios.items():
        params = {
            "valuation_age": base_params["valuation_age"],
            "maximum_age": base_params["maximum_age"],
            "annual_pension": base_params["annual_pension"],
            "annual_expense": base_params["annual_expense"],
            "inflation_rate": base_params["inflation_rate"],
            "rate_shift": 0.0,
            "mortality_multiplier": 1.0,
        }

        params.update(override)

        cashflows = build_annuity_cashflows(
            mortality=mortality,
            spot_curve=spot_curve,
            **params,
        )

        bel = calculate_bel(cashflows)

        if scenario_name == "Base":
            base_bel = bel

        rows.append(
            {
                "scenario": scenario_name,
                "BEL": bel,
                "impact_vs_base": bel - base_bel,
                "impact_vs_base_pct": (bel / base_bel - 1.0) if base_bel else 0.0,
            }
        )

    summary = pd.DataFrame(rows)

    return summary


# ============================================================
# Output charts
# ============================================================

def create_cashflow_chart(base_cashflows: pd.DataFrame) -> Path:
    chart_path = OUTPUT_DIR / "base_expected_cashflows.png"

    plt.figure(figsize=(9, 5))
    plt.plot(
        base_cashflows["projection_year"],
        base_cashflows["expected_cashflow"],
        marker="o",
        linewidth=1.5,
    )
    plt.title("Base Scenario Expected Annuity Cashflows")
    plt.xlabel("Projection Year")
    plt.ylabel("Expected Cashflow (£)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

    return chart_path


def create_sensitivity_chart(sensitivity_summary: pd.DataFrame) -> Path:
    chart_path = OUTPUT_DIR / "bel_sensitivity_summary.png"

    chart_data = sensitivity_summary.copy()
    chart_data = chart_data[chart_data["scenario"] != "Base"]

    plt.figure(figsize=(10, 5))
    plt.bar(
        chart_data["scenario"],
        chart_data["impact_vs_base"],
    )
    plt.title("BEL Sensitivity Impact vs Base")
    plt.xlabel("Scenario")
    plt.ylabel("Impact on BEL (£)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

    return chart_path

def create_spot_curve_chart(spot_curve: pd.DataFrame) -> Path:
    chart_path = OUTPUT_DIR / "nominal_spot_curve.png"

    plt.figure(figsize=(9, 5))
    plt.plot(
        spot_curve["maturity"],
        spot_curve["spot_rate"] * 100,
        marker="o",
        linewidth=1.5,
    )
    plt.title("Bank of England Nominal Gilt Spot Curve")
    plt.xlabel("Maturity")
    plt.ylabel("Spot Rate (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

    return chart_path


def create_survival_probability_chart(base_cashflows: pd.DataFrame) -> Path:
    chart_path = OUTPUT_DIR / "survival_probability_curve.png"

    plt.figure(figsize=(9, 5))
    plt.plot(
        base_cashflows["payment_age"],
        base_cashflows["survival_probability"],
        marker="o",
        linewidth=1.5,
    )
    plt.title("Survival Probability from Valuation Age")
    plt.xlabel("Payment Age")
    plt.ylabel("Survival Probability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

    return chart_path

def create_mortality_rate_chart(mortality: pd.DataFrame) -> Path:
    chart_path = OUTPUT_DIR / "mortality_rate_by_age.png"

    chart_data = mortality[
        (mortality["age"] >= ASSUMPTIONS["valuation_age"])
        & (mortality["age"] <= 100)
    ].copy()

    plt.figure(figsize=(9, 5))
    plt.plot(
        chart_data["age"],
        chart_data["qx"] * 100,
        marker="o",
        linewidth=1.5,
    )
    plt.title("Mortality Rate by Age")
    plt.xlabel("Age")
    plt.ylabel("One-Year Death Probability (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

    return chart_path


# ============================================================
# Excel export
# ============================================================

def export_outputs(
    spot_curve: pd.DataFrame,
    mortality: pd.DataFrame,
    base_cashflows: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    boe_curve_date: pd.Timestamp,
    hmd_year: int,
) -> Path:
    output_path = OUTPUT_DIR / "annuity_liability_valuation_outputs.xlsx"

    assumptions_table = pd.DataFrame(
        [
            {"item": "Project", "value": "Annuity Liability Valuation & Sensitivity"},
            {"item": "Valuation date source", "value": f"BoE curve date: {boe_curve_date.date()}"},
            {"item": "Mortality source", "value": f"HMD UK both-sex period life table, {hmd_year}"},
            {"item": "Valuation age", "value": ASSUMPTIONS["valuation_age"]},
            {"item": "Maximum age", "value": ASSUMPTIONS["maximum_age"]},
            {"item": "Annual pension", "value": ASSUMPTIONS["annual_pension"]},
            {"item": "Annual expense", "value": ASSUMPTIONS["annual_expense"]},
            {"item": "Inflation rate", "value": ASSUMPTIONS["inflation_rate"]},
            {"item": "Payment timing", "value": ASSUMPTIONS["payment_timing"]},
            {"item": "Discounting", "value": "Interpolated nominal gilt spot rates"},
        ]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        assumptions_table.to_excel(writer, sheet_name="Assumptions", index=False)
        sensitivity_summary.to_excel(writer, sheet_name="Sensitivity Summary", index=False)
        base_cashflows.to_excel(writer, sheet_name="Base Cashflows", index=False)
        spot_curve.to_excel(writer, sheet_name="Nominal Spot Curve", index=False)
        mortality.to_excel(writer, sheet_name="Mortality Table", index=False)

    return output_path


# ============================================================
# Main run
# ============================================================

def main() -> None:
    print("=" * 70)
    print("Annuity Liability Valuation & Sensitivity")
    print("=" * 70)

    # 1. BoE curve
    boe_file = find_boe_nominal_file()
    print(f"\nReading BoE file:\n{boe_file}")

    spot_curve, boe_curve_date = extract_latest_nominal_spot_curve(boe_file)

    # 2. HMD life table
    hmd_file = find_hmd_life_table_file()
    print(f"\nReading HMD file:\n{hmd_file}")

    hmd_raw = read_hmd_life_table(hmd_file)
    mortality, hmd_year = extract_latest_mortality_table(hmd_raw)

    # 3. Base valuation
    base_cashflows = build_annuity_cashflows(
        mortality=mortality,
        spot_curve=spot_curve,
        valuation_age=ASSUMPTIONS["valuation_age"],
        maximum_age=ASSUMPTIONS["maximum_age"],
        annual_pension=ASSUMPTIONS["annual_pension"],
        annual_expense=ASSUMPTIONS["annual_expense"],
        inflation_rate=ASSUMPTIONS["inflation_rate"],
    )

    base_bel = calculate_bel(base_cashflows)

    # 4. Sensitivities
    sensitivity_summary = run_sensitivity_scenarios(
        mortality=mortality,
        spot_curve=spot_curve,
    )

    # 5. Charts
    mortality_chart = create_mortality_rate_chart(mortality)
    spot_curve_chart = create_spot_curve_chart(spot_curve)
    survival_chart = create_survival_probability_chart(base_cashflows)
    cashflow_chart = create_cashflow_chart(base_cashflows)
    sensitivity_chart = create_sensitivity_chart(sensitivity_summary)

    # 6. Excel output
    excel_output = export_outputs(
        
        spot_curve=spot_curve,
        mortality=mortality,
        base_cashflows=base_cashflows,
        sensitivity_summary=sensitivity_summary,
        boe_curve_date=boe_curve_date,
        hmd_year=hmd_year,
    )

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Base BEL: £{base_bel:,.2f}")
    print("\nSensitivity summary:")
    print(sensitivity_summary.to_string(index=False))

    print("\nOutputs saved:")
    print(f"Excel workbook: {excel_output}")
    print(f"Cashflow chart: {cashflow_chart}")
    print(f"Sensitivity chart: {sensitivity_chart}")

    print("\nDone.")


if __name__ == "__main__":
    main()
