import numpy as np
import pandas as pd
from returns import get_returns, SECTORS
from estimation import compute_estimates
from backtest import run_backtest, run_benchmarks, BACKTEST_START

# ── 1. Core metrics ───────────────────────────────────────────────────────────

def sharpe(r):
    return (r.mean() * 12) / (r.std() * np.sqrt(12))

def max_drawdown(r):
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return dd.min()

def calmar(r):
    ann_ret = r.mean() * 12
    mdd     = abs(max_drawdown(r))
    return ann_ret / mdd if mdd > 0 else np.nan

def tracking_error(r, benchmark):
    diff = r - benchmark.reindex(r.index)
    return diff.std() * np.sqrt(12)

def information_ratio(r, benchmark):
    diff   = r - benchmark.reindex(r.index)
    te     = diff.std() * np.sqrt(12)
    excess = diff.mean() * 12
    return excess / te if te > 0 else np.nan

def historical_var(r, level=0.95):
    """
    Historical one-period Value at Risk using the positive-loss convention.

    The input series contains monthly portfolio returns. Because the rest of
    this project works with monthly log returns, this is a one-month historical
    VaR on log returns. A reported value of 8.0 means the 5% left-tail cutoff
    is approximately a loss of 8% or worse.
    """
    r = pd.Series(r).dropna()
    if r.empty:
        return np.nan
    return -np.quantile(r, 1 - level)

def historical_es(r, level=0.95):
    """
    Historical one-period Expected Shortfall using the positive-loss convention.

    ES is the average return conditional on being in the VaR tail. A reported
    value of 10.0 means the average monthly loss in the worst 5% of months is
    approximately 10%.
    """
    r = pd.Series(r).dropna()
    if r.empty:
        return np.nan
    cutoff = np.quantile(r, 1 - level)
    tail = r[r <= cutoff]
    return -tail.mean() if len(tail) > 0 else np.nan

def turnover_ann(to_series):
    return to_series.mean() * 12

def summary_metrics(r, spy, label="", var_level=0.95):
    ann_ret = r.mean() * 12 * 100
    ann_vol = r.std() * np.sqrt(12) * 100
    sr      = sharpe(r)
    mdd     = max_drawdown(r) * 100
    cal     = calmar(r)
    te      = tracking_error(r, spy) * 100
    ir      = information_ratio(r, spy)
    var     = historical_var(r, level=var_level) * 100
    es      = historical_es(r, level=var_level) * 100
    var_label = f"1M VaR {int(var_level * 100)} (%)"
    es_label  = f"1M ES {int(var_level * 100)} (%)"
    return {
        "label"      : label,
        "Ann. Return": round(ann_ret, 2),
        "Ann. Vol"   : round(ann_vol, 2),
        "Sharpe"     : round(sr, 3),
        "Max DD"     : round(mdd, 2),
        var_label     : round(var, 2),
        es_label      : round(es, 2),
        "Calmar"     : round(cal, 3),
        "TE vs SPY"  : round(te, 2),
        "IR vs SPY"  : round(ir, 3),
    }

# ── 2. Sub-period breakdown ───────────────────────────────────────────────────

SUBPERIODS = {
    "2010-2014": ("2010-01-01", "2014-12-31"),
    "2015-2019": ("2015-01-01", "2019-12-31"),
    "2020-2024": ("2020-01-01", "2024-12-31"),
}

def subperiod_sharpes(all_returns_dict):
    """
    Returns a DataFrame: rows = sub-periods, cols = strategies.
    """
    rows = []
    for period, (start, end) in SUBPERIODS.items():
        row = {"Period": period}
        for name, r in all_returns_dict.items():
            r_sub = r.loc[start:end].dropna()
            row[name] = round(sharpe(r_sub), 3) if len(r_sub) > 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Period")

# ── 3. Stress scenario extraction ────────────────────────────────────────────

STRESS = {
    "COVID Crash"     : ("2020-02-01", "2020-03-31"),
    "2022 Rate Shock" : ("2022-01-01", "2022-12-31"),
    "2011 Debt Crisis": ("2011-08-01", "2011-10-31"),
}

def stress_returns(all_returns_dict):
    """
    Returns a DataFrame: rows = stress periods, cols = strategies.
    Values are cumulative log returns over the episode (%).
    """
    rows = []
    for scenario, (start, end) in STRESS.items():
        row = {"Scenario": scenario, "Period": f"{start[:7]} to {end[:7]}"}
        for name, r in all_returns_dict.items():
            r_sub = r.loc[start:end].dropna()
            cum   = r_sub.sum() * 100   # log returns sum to approx cumulative
            row[name] = round(cum, 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Scenario")

def tail_risk_table(all_returns_dict, level=0.95):
    """
    Returns one-month historical VaR and ES for each strategy.
    Values are positive loss percentages.
    """
    rows = []
    for name, r in all_returns_dict.items():
        r = r.dropna()
        rows.append({
            "Strategy": name,
            f"1M VaR {int(level * 100)} (%)": round(historical_var(r, level) * 100, 2),
            f"1M ES {int(level * 100)} (%)": round(historical_es(r, level) * 100, 2),
            "Worst Month (%)": round(r.min() * 100, 2),
        })
    return pd.DataFrame(rows).set_index("Strategy")

# ── 4. Sensitivity analysis ───────────────────────────────────────────────────

def sensitivity_analysis(log_ret, excess, rf):
    """
    Varies lambda and lookback window one-at-a-time.
    Returns a DataFrame of Sharpe ratios.
    """
    from optimiser import solve_weights, drift_weights, N, C_COST, TAU, W_MIN, W_MAX
    from estimation import compute_estimates, ledoit_wolf_cov, capm_mu
    import optimiser as opt_module

    results = []

    # ── Lambda sensitivity ────────────────────────────────────────────────
    for lam in [0.5, 1.0, 1.5, 2.0, 3.0]:
        opt_module.LAMBDA = lam
        estimates = compute_estimates(excess, rf)
        res       = run_backtest(estimates, log_ret, rf)
        r         = res["returns"]
        sr        = sharpe(r)
        mdd       = max_drawdown(r) * 100
        var95     = historical_var(r, 0.95) * 100
        es95      = historical_es(r, 0.95) * 100
        results.append({
            "Parameter"  : f"lambda={lam}",
            "Sharpe"     : round(sr, 3),
            "Max DD (%)" : round(mdd, 2),
            "1M VaR 95 (%)": round(var95, 2),
            "1M ES 95 (%)" : round(es95, 2),
        })

    # Reset lambda
    opt_module.LAMBDA = 1.5

    # ── Lookback window sensitivity ───────────────────────────────────────
    from estimation import LOOKBACK as DEFAULT_LOOKBACK
    import estimation as est_module

    for lb in [24, 36, 48, 60]:
        estimates = compute_estimates(excess, rf, lookback=lb)
        res       = run_backtest(estimates, log_ret, rf)
        r         = res["returns"]
        sr        = sharpe(r)
        mdd       = max_drawdown(r) * 100
        var95     = historical_var(r, 0.95) * 100
        es95      = historical_es(r, 0.95) * 100
        results.append({
            "Parameter"  : f"lookback={lb}m",
            "Sharpe"     : round(sr, 3),
            "Max DD (%)" : round(mdd, 2),
            "1M VaR 95 (%)": round(var95, 2),
            "1M ES 95 (%)" : round(es95, 2),
        })

    return pd.DataFrame(results).set_index("Parameter")


# ── 5. Master function ────────────────────────────────────────────────────────

def run_analysis():
    prices, log_ret, excess, rf = get_returns()
    estimates = compute_estimates(excess, rf)

    capm_results = run_backtest(estimates, log_ret, rf)
    capm_r       = capm_results["returns"]

    benchmarks   = run_benchmarks(log_ret, rf)
    spy_r        = benchmarks["SPY"].reindex(capm_r.index).dropna()
    ew_r         = benchmarks["EW"].reindex(capm_r.index).dropna()
    naive_r      = benchmarks["Naive_MV"].reindex(capm_r.index).dropna()

    all_returns = {
        "CAPM_MV"  : capm_r,
        "Naive_MV" : naive_r,
        "EW"       : ew_r,
        "SPY"      : spy_r,
    }

    # ── Full-period metrics table ─────────────────────────────────────────
    print("\n══ FULL-PERIOD PERFORMANCE (Jan 2010 – Dec 2024) ════════════════════")
    metrics = []
    for name, r in all_returns.items():
        m = summary_metrics(r.dropna(), spy_r, label=name)
        metrics.append(m)
    metrics_df = pd.DataFrame(metrics).set_index("label")
    print(metrics_df.to_string())
    print("\n  VaR/ES convention: one-month historical 95% positive loss, computed on monthly log returns.")

    # ── Standalone tail-risk table ────────────────────────────────────────
    print("\n══ TAIL RISK: HISTORICAL VaR / EXPECTED SHORTFALL ═══════════════════")
    tail_df = tail_risk_table(all_returns, level=0.95)
    print(tail_df.to_string())

    # ── Annualised turnover ───────────────────────────────────────────────
    ann_to = turnover_ann(capm_results["turnover"]) * 100
    print(f"\n  CAPM-MV Ann. Turnover : {ann_to:.1f}%")

    # ── Sub-period Sharpe ratios ──────────────────────────────────────────
    print("\n══ SUB-PERIOD SHARPE RATIOS ══════════════════════════════════════════")
    sp_df = subperiod_sharpes(all_returns)
    print(sp_df.to_string())

    # ── Stress scenarios ──────────────────────────────────────────────────
    print("\n══ STRESS SCENARIO RETURNS (%) ═══════════════════════════════════════")
    stress_df = stress_returns(all_returns)
    print(stress_df.to_string())

    # ── Sensitivity analysis ──────────────────────────────────────────────
    print("\n══ SENSITIVITY ANALYSIS ══════════════════════════════════════════════")
    sens_df = sensitivity_analysis(log_ret, excess, rf)
    print(sens_df.to_string())

    return {
        "metrics"    : metrics_df,
        "subperiods" : sp_df,
        "stress"     : stress_df,
        "sensitivity": sens_df,
        "tail_risk"  : tail_df,
        "capm_results": capm_results,
        "all_returns" : all_returns,
    }


if __name__ == "__main__":
    results = run_analysis()