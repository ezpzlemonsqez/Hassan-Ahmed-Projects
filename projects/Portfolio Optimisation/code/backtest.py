import numpy as np
import pandas as pd
from returns import get_returns, SECTORS
from estimation import compute_estimates
from optimiser import solve_weights, drift_weights, N, C_COST

BACKTEST_START = "2010-01-31"

def run_backtest(estimates, log_ret, rf):
    """
    Walk-forward backtest over all rebalance dates.

    At each month-end t:
      1. Drift prior weights by realised returns
      2. Solve constrained QP for new weights
      3. Deduct transaction costs on net trades
      4. Record portfolio return for month t+1

    Returns:
        results : dict with keys:
            'returns'      : Series, monthly net portfolio returns
            'weights'      : DataFrame, weights at each rebalance date
            'turnover'     : Series, one-way turnover each month
            'costs'        : Series, transaction cost deducted each month
            'solver_flags' : Series, True if solver succeeded
    """
    dates         = sorted(estimates.keys())
    sector_ret    = log_ret[SECTORS]

    # ── Initialise at equal weight ─────────────────────────────────────────
    w_prev = np.ones(N) / N

    # Storage
    port_returns  = {}
    weight_hist   = {}
    turnover_hist = {}
    cost_hist     = {}
    flag_hist     = {}

    for i, t in enumerate(dates):
        e = estimates[t]

        # ── 1. Drift prior weights by this month's returns ─────────────────
        # We drift using the return of month t (which was earned while holding
        # w_prev, set at the end of month t-1)
        if t in sector_ret.index:
            ret_t = np.exp(sector_ret.loc[t].values) - 1   # simple returns
        else:
            ret_t = np.zeros(N)

        w_drifted = drift_weights(w_prev, ret_t)

        # ── 2. Solve for new weights ───────────────────────────────────────
        w_opt, success = solve_weights(e["mu"], e["sigma"], w_drifted)

        # ── 3. Compute turnover and transaction costs ──────────────────────
        turnover = np.sum(np.abs(w_opt - w_drifted))
        cost     = C_COST * turnover

        # ── 4. Record weights and flags at rebalance date t ───────────────
        weight_hist[t]   = w_opt
        turnover_hist[t] = turnover
        cost_hist[t]     = cost
        flag_hist[t]     = success

        # ── 5. Compute portfolio return for the NEXT month ─────────────────
        # Find next date in the return series after t
        future_dates = sector_ret.index[sector_ret.index > t]
        if len(future_dates) == 0:
            break
        t_next = future_dates[0]

        ret_next       = sector_ret.loc[t_next].values          # log returns
        port_ret_gross = w_opt @ ret_next                        # log portfolio return
        port_ret_net   = port_ret_gross - cost                   # deduct costs
        port_returns[t_next] = port_ret_net

        # ── 6. Carry weights forward ───────────────────────────────────────
        w_prev = w_opt

    # ── Package results ────────────────────────────────────────────────────
    returns_s  = pd.Series(port_returns,  name="CAPM_MV")
    weights_df = pd.DataFrame(weight_hist, index=SECTORS).T
    turnover_s = pd.Series(turnover_hist, name="turnover")
    cost_s     = pd.Series(cost_hist,     name="cost")
    flags_s    = pd.Series(flag_hist,     name="solver_ok")

    return {
        "returns"      : returns_s,
        "weights"      : weights_df,
        "turnover"     : turnover_s,
        "costs"        : cost_s,
        "solver_flags" : flags_s,
    }


def run_benchmarks(log_ret, rf):
    """
    Computes monthly returns for three benchmarks over the backtest window:
        SPY        : buy-and-hold S&P 500
        EW         : equal-weight sector ETF portfolio, monthly rebalanced
        Naive MV   : MV with historical mean returns, same constraints

    Returns:
        dict of Series, each monthly log returns
    """
    from optimiser import solve_weights, drift_weights, LAMBDA, W_MIN, W_MAX, TAU
    from scipy.optimize import minimize

    sector_ret = log_ret[SECTORS]
    spy_ret    = log_ret["SPY"]

    # Restrict to backtest window
    mask       = sector_ret.index >= BACKTEST_START
    dates      = sector_ret.index[mask]

    # ── SPY ────────────────────────────────────────────────────────────────
    spy_returns = spy_ret.loc[dates]

    # ── Equal Weight ───────────────────────────────────────────────────────
    ew_returns = sector_ret.loc[dates].mean(axis=1)
    ew_returns.name = "EW"

    # ── Naive MV (historical mean returns, same LW covariance) ────────────
    from estimation import ledoit_wolf_cov
    from optimiser  import N, C_COST

    LOOKBACK   = 36
    naive_rets = {}
    w_prev_nmv = np.ones(N) / N

    all_dates  = sector_ret.index
    excess_all = log_ret[SECTORS].subtract(
        rf["rf_monthly"].reindex(log_ret.index, method="ffill"), axis=0
    )

    for i, t in enumerate(all_dates):
        if t < pd.Timestamp(BACKTEST_START):
            continue
        idx = all_dates.get_loc(t)
        if idx < LOOKBACK:
            continue

        window     = sector_ret.iloc[idx - LOOKBACK : idx]
        excess_window = excess_all.iloc[idx - LOOKBACK : idx]
        mu_hist = excess_window.mean().values + rf["rf_monthly"].reindex(sector_ret.index, method="ffill").loc[t]
        sigma, _   = ledoit_wolf_cov(excess_all.iloc[idx - LOOKBACK : idx])

        # Drift
        ret_t     = np.exp(sector_ret.loc[t].values) - 1
        w_drifted = drift_weights(w_prev_nmv, ret_t)

        w_opt, _  = solve_weights(mu_hist, sigma, w_drifted)
        turnover  = np.sum(np.abs(w_opt - w_drifted))
        cost      = C_COST * turnover

        future = sector_ret.index[sector_ret.index > t]
        if len(future) == 0:
            break
        t_next = future[0]

        ret_next = sector_ret.loc[t_next].values
        naive_rets[t_next] = w_opt @ ret_next - cost
        w_prev_nmv = w_opt

    naive_s = pd.Series(naive_rets, name="Naive_MV")

    return {
        "SPY"      : spy_returns,
        "EW"       : ew_returns,
        "Naive_MV" : naive_s,
    }


# ── Sanity check when run directly ───────────────────────────────────────────
if __name__ == "__main__":
    prices, log_ret, excess, rf = get_returns()
    estimates = compute_estimates(excess, rf)

    results = run_backtest(estimates, log_ret, rf)

    r  = results["returns"]
    wt = results["weights"]
    to = results["turnover"]

    print(f"\n── Backtest Summary ───────────────────────────────────────────────")
    print(f"  Periods          : {len(r)}")
    print(f"  Date range       : {r.index[0].date()} to {r.index[-1].date()}")
    print(f"  Solver failures  : {(~results['solver_flags']).sum()}")
    print(f"  Mean monthly ret : {r.mean()*100:.4f}%")
    print(f"  Ann. return      : {r.mean()*12*100:.2f}%")
    print(f"  Ann. volatility  : {r.std()*np.sqrt(12)*100:.2f}%")
    sharpe = (r.mean() * 12) / (r.std() * np.sqrt(12))
    print(f"  Sharpe ratio     : {sharpe:.3f}")
    print(f"  Mean turnover    : {to.mean()*100:.1f}% per month")
    print(f"  Total costs      : {results['costs'].sum()*100:.2f}%")

    print(f"\n── Average Sector Weights ─────────────────────────────────────────")
    avg_w = wt.mean()
    for s, w in avg_w.items():
        bar = "█" * int(w * 100)
        print(f"  {s:5s}  {w:.4f}  {bar}")

    benchmarks = run_benchmarks(log_ret, rf)

    # Align all to common dates
    common = r.index
    for name, bret in benchmarks.items():
        bret_aligned = bret.reindex(common)
        ann_ret = bret_aligned.mean() * 12 * 100
        ann_vol = bret_aligned.std() * np.sqrt(12) * 100
        sr      = (bret_aligned.mean() * 12) / (bret_aligned.std() * np.sqrt(12))
        print(f"  {name:10s}  Ann.Ret={ann_ret:.2f}%  Ann.Vol={ann_vol:.2f}%  Sharpe={sr:.3f}")

    # CAPM-MV line for comparison
    ann_ret = r.mean() * 12 * 100
    ann_vol = r.std() * np.sqrt(12) * 100
    sr      = (r.mean() * 12) / (r.std() * np.sqrt(12))
    print(f"  {'CAPM_MV':10s}  Ann.Ret={ann_ret:.2f}%  Ann.Vol={ann_vol:.2f}%  Sharpe={sr:.3f}")

    print(f"\n── Weight trajectory (selected dates) ─────────────────────────────")
    check_dates = ["2010-03-31", "2012-12-31", "2015-12-31", "2018-12-31", "2021-12-31", "2024-12-31"]
    print(f"  {'Sector':6s}", end="")
    for cd in check_dates:
        print(f"  {cd[:7]:>8s}", end="")
    print()
    for s in SECTORS:
        print(f"  {s:6s}", end="")
        for cd in check_dates:
            ts = pd.Timestamp(cd)
            closest = wt.index[wt.index <= ts]
            if len(closest):
                print(f"  {wt.loc[closest[-1], s]:>8.3f}", end="")
        print()