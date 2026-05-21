import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from returns import get_returns, SECTORS
from estimation import compute_estimates
from backtest import run_backtest, run_benchmarks
from analysis import run_analysis, max_drawdown, SUBPERIODS, STRESS, historical_var, historical_es

# ── Style ─────────────────────────────────────────────────────────────────────
ACCENT   = "#1A3C6B"
ACCENT2  = "#2E6DA4"
MIDBLUE  = "#AABDD4"
LIGHTBLUE= "#EEF3F9"
GREY     = "#444444"

COLORS = {
    "CAPM_MV"  : ACCENT,
    "Naive_MV" : ACCENT2,
    "EW"       : MIDBLUE,
    "SPY"      : "#888888",
}

OUTDIR = r"C:\Users\hassa\Documents\Python\Portfolio Optimization\plots"

import os
os.makedirs(OUTDIR, exist_ok=True)

def savefig(name):
    path = os.path.join(OUTDIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")

# ── 1. Cumulative return chart ────────────────────────────────────────────────
def plot_cumulative(all_returns):
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, r in all_returns.items():
        cum = (1 + r.dropna()).cumprod()
        ax.plot(cum.index, cum.values,
                label=name,
                color=COLORS[name],
                linewidth=1.8 if name == "CAPM_MV" else 1.2,
                linestyle="-" if name in ("CAPM_MV", "SPY") else "--")

    ax.axhline(1, color=GREY, linewidth=0.5, linestyle=":")
    ax.set_title("Cumulative Return — Mar 2010 to Dec 2024",
                 fontsize=12, color=ACCENT, fontweight="bold")
    ax.set_ylabel("Growth of $1", fontsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig("01_cumulative_returns.png")

# ── 2. Drawdown chart ─────────────────────────────────────────────────────────
def plot_drawdown(all_returns):
    fig, ax = plt.subplots(figsize=(10, 4))

    for name, r in all_returns.items():
        cum     = (1 + r.dropna()).cumprod()
        roll_max= cum.cummax()
        dd      = (cum - roll_max) / roll_max * 100
        ax.plot(dd.index, dd.values,
                label=name,
                color=COLORS[name],
                linewidth=1.8 if name == "CAPM_MV" else 1.2,
                linestyle="-" if name in ("CAPM_MV", "SPY") else "--")

    # Shade stress episodes
    for scenario, (start, end) in STRESS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.08, color=ACCENT, label="_nolegend_")

    ax.set_title("Drawdown Through Time",
                 fontsize=12, color=ACCENT, fontweight="bold")
    ax.set_ylabel("Drawdown (%)", fontsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig("02_drawdown.png")

# ── 3. Sector weight heatmap ──────────────────────────────────────────────────
def plot_weight_heatmap(weights_df):
    # Resample to quarterly for readability
    wq = weights_df.resample("QE").last().T * 100   # percent

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        wq,
        ax=ax,
        cmap="Blues",
        linewidths=0.3,
        linecolor="white",
        annot=False,
        cbar_kws={"label": "Weight (%)"},
        vmin=0, vmax=40
    )
    ax.set_title("CAPM-MV Sector Weights Over Time (quarterly)",
                 fontsize=12, color=ACCENT, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Thin out x-axis labels
    xlabels = [t.strftime("%Y") if t.month <= 3 else "" for t in wq.columns]
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    savefig("03_weight_heatmap.png")

# ── 4. Rolling 12-month Sharpe ────────────────────────────────────────────────
def plot_rolling_sharpe(all_returns, window=12):
    fig, ax = plt.subplots(figsize=(10, 4))

    for name, r in all_returns.items():
        roll_sr = (r.dropna().rolling(window).mean() * 12) / \
                  (r.dropna().rolling(window).std() * np.sqrt(12))
        ax.plot(roll_sr.index, roll_sr.values,
                label=name,
                color=COLORS[name],
                linewidth=1.8 if name == "CAPM_MV" else 1.2,
                linestyle="-" if name in ("CAPM_MV", "SPY") else "--")

    ax.axhline(0, color=GREY, linewidth=0.6, linestyle=":")
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio",
                 fontsize=12, color=ACCENT, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio", fontsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig("04_rolling_sharpe.png")

# ── 5. Sub-period Sharpe bar chart ────────────────────────────────────────────
def plot_subperiod_bars(all_returns):
    strategies = list(all_returns.keys())
    periods    = list(SUBPERIODS.keys())
    x          = np.arange(len(periods))
    width      = 0.18

    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (name, r) in enumerate(all_returns.items()):
        sharpes = []
        for period, (start, end) in SUBPERIODS.items():
            r_sub = r.loc[start:end].dropna()
            sr    = (r_sub.mean() * 12) / (r_sub.std() * np.sqrt(12)) \
                    if len(r_sub) > 2 else 0
            sharpes.append(sr)
        offset = (j - 1.5) * width
        bars = ax.bar(x + offset, sharpes, width,
                      label=name, color=COLORS[name],
                      edgecolor="white", linewidth=0.5)

    ax.axhline(0, color=GREY, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=10)
    ax.set_ylabel("Sharpe Ratio", fontsize=10)
    ax.set_title("Sub-Period Sharpe Ratios",
                 fontsize=12, color=ACCENT, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig("05_subperiod_sharpe.png")

# ── 6. Stress scenario bar chart ─────────────────────────────────────────────
def plot_stress_bars(all_returns):
    scenarios  = list(STRESS.keys())
    x          = np.arange(len(scenarios))
    width      = 0.18

    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (name, r) in enumerate(all_returns.items()):
        cum_rets = []
        for scenario, (start, end) in STRESS.items():
            r_sub = r.loc[start:end].dropna()
            cum_rets.append(r_sub.sum() * 100)
        offset = (j - 1.5) * width
        ax.bar(x + offset, cum_rets, width,
               label=name, color=COLORS[name],
               edgecolor="white", linewidth=0.5)

    ax.axhline(0, color=GREY, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax.set_title("Stress Scenario Returns",
                 fontsize=12, color=ACCENT, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig("06_stress_scenarios.png")

# ── 7. Tail risk: historical VaR and ES ──────────────────────────────────────
def plot_tail_risk_bars(all_returns, level=0.95):
    strategies = list(all_returns.keys())
    var_vals = [historical_var(all_returns[name].dropna(), level) * 100 for name in strategies]
    es_vals  = [historical_es(all_returns[name].dropna(), level) * 100 for name in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, var_vals, width, label=f"1M VaR {int(level * 100)}%",
           color=ACCENT2, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, es_vals, width, label=f"1M ES {int(level * 100)}%",
           color=MIDBLUE, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel("Positive Loss (%)", fontsize=10)
    ax.set_title("Historical One-Month Tail Risk",
                 fontsize=12, color=ACCENT, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    savefig("08_tail_risk_var_es.png")

# ── 8. Sensitivity: Sharpe vs lambda ─────────────────────────────────────────
def plot_sensitivity(log_ret, excess, rf):
    from optimiser import solve_weights, N, C_COST
    from estimation import compute_estimates
    from backtest import run_backtest
    import optimiser as opt_module

    lambdas  = [0.5, 1.0, 1.5, 2.0, 3.0]
    lookbacks = [24, 36, 48, 60]

    sr_lambda = []
    for lam in lambdas:
        opt_module.LAMBDA = lam
        est = compute_estimates(excess, rf)
        res = run_backtest(est, log_ret, rf)
        r   = res["returns"]
        sr_lambda.append((r.mean() * 12) / (r.std() * np.sqrt(12)))
    opt_module.LAMBDA = 1.5

    sr_lookback = []
    for lb in lookbacks:
        est = compute_estimates(excess, rf, lookback=lb)
        res = run_backtest(est, log_ret, rf)
        r   = res["returns"]
        sr_lookback.append((r.mean() * 12) / (r.std() * np.sqrt(12)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(lambdas, sr_lambda, "o-", color=ACCENT, linewidth=2, markersize=6)
    ax1.axhline(0.881, color=GREY, linewidth=0.8, linestyle="--", label="SPY")
    ax1.set_xlabel("Risk Aversion λ", fontsize=10)
    ax1.set_ylabel("Sharpe Ratio", fontsize=10)
    ax1.set_title("Sharpe vs λ", fontsize=11, color=ACCENT, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    ax2.plot(lookbacks, sr_lookback, "o-", color=ACCENT2, linewidth=2, markersize=6)
    ax2.axhline(0.881, color=GREY, linewidth=0.8, linestyle="--", label="SPY")
    ax2.set_xlabel("Lookback Window (months)", fontsize=10)
    ax2.set_ylabel("Sharpe Ratio", fontsize=10)
    ax2.set_title("Sharpe vs Lookback", fontsize=11, color=ACCENT, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    plt.suptitle("Sensitivity Analysis", fontsize=12,
                 color=ACCENT, fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig("07_sensitivity.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data and running backtest...")
    prices, log_ret, excess, rf = get_returns()
    estimates  = compute_estimates(excess, rf)
    capm_res   = run_backtest(estimates, log_ret, rf)
    benchmarks = run_benchmarks(log_ret, rf)

    capm_r  = capm_res["returns"]
    all_returns = {
        "CAPM_MV"  : capm_r,
        "Naive_MV" : benchmarks["Naive_MV"].reindex(capm_r.index),
        "EW"       : benchmarks["EW"].reindex(capm_r.index),
        "SPY"      : benchmarks["SPY"].reindex(capm_r.index),
    }

    print("\nGenerating plots...")
    plot_cumulative(all_returns)
    plot_drawdown(all_returns)
    plot_weight_heatmap(capm_res["weights"])
    plot_rolling_sharpe(all_returns)
    plot_subperiod_bars(all_returns)
    plot_stress_bars(all_returns)
    plot_tail_risk_bars(all_returns, level=0.95)

    print("\nGenerating sensitivity plots (takes ~60s)...")
    plot_sensitivity(log_ret, excess, rf)

    print(f"\nAll plots saved to: {OUTDIR}")