import pandas as pd
import numpy as np
import os

DATA_DIR = r"C:\Users\hassa\Documents\Python\Portfolio Optimization\data"

SECTORS = ["XLK","XLF","XLV","XLY","XLI","XLC","XLP","XLE","XLU","XLRE","XLB"]
ALL     = SECTORS + ["SPY"]

# ── 1. Load raw data ──────────────────────────────────────────────────────────
def load_raw():
    prices = pd.read_csv(
        os.path.join(DATA_DIR, "prices_monthly.csv"),
        index_col=0, parse_dates=True
    )
    rf = pd.read_csv(
        os.path.join(DATA_DIR, "rf_monthly.csv"),
        index_col=0, parse_dates=True
    )
    # Forward-fill RF to cover any trailing gaps
    rf = rf.reindex(prices.index, method="ffill")
    return prices, rf

# ── 2. Backfill XLC and XLRE ──────────────────────────────────────────────────
def backfill_etfs(prices):
    """
    XLC (launched Jun 2018) and XLRE (launched Oct 2015) have NaNs before
    their launch dates. We backfill using SPY-scaled synthetic prices:
    the ratio of the ETF's first valid price to SPY at that date is used
    to scale SPY backwards. This is a simple, transparent approximation
    that fills the burn-in window without introducing look-ahead.
    """
    prices = prices.copy()

    for ticker in ["XLC", "XLRE"]:
        first_idx = prices[ticker].first_valid_index()
        mask      = prices.index < first_idx

        # Ratio at launch date
        etf_launch = prices.loc[first_idx, ticker]
        spy_launch = prices.loc[first_idx, "SPY"]
        ratio      = etf_launch / spy_launch

        # Fill pre-launch with SPY * ratio
        prices.loc[mask, ticker] = prices.loc[mask, "SPY"] * ratio

    return prices

# ── 3. Compute log returns ────────────────────────────────────────────────────
def compute_log_returns(prices):
    """
    r_{i,t} = ln(P_{i,t} / P_{i,t-1})
    Returns a DataFrame of the same shape minus the first row.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    return log_ret

# ── 4. Compute excess returns ─────────────────────────────────────────────────
def compute_excess_returns(log_ret, rf):
    """
    Excess return = log return - monthly RF rate.
    Aligns on index before subtracting.
    """
    rf_aligned = rf["rf_monthly"].reindex(log_ret.index, method="ffill")
    excess = log_ret.subtract(rf_aligned, axis=0)
    return excess

# ── 5. Master function ────────────────────────────────────────────────────────
def get_returns():
    """
    Returns:
        prices   : raw adjusted close prices (after backfill)
        log_ret  : log returns for all tickers
        excess   : excess log returns for all tickers
        rf       : risk-free rate DataFrame (rf_annual_pct, rf_monthly)
    """
    prices_raw, rf = load_raw()
    prices         = backfill_etfs(prices_raw)
    log_ret        = compute_log_returns(prices)
    excess         = compute_excess_returns(log_ret, rf)
    return prices, log_ret, excess, rf

# ── 6. Sanity check when run directly ────────────────────────────────────────
if __name__ == "__main__":
    prices, log_ret, excess, rf = get_returns()

    print("── Prices ─────────────────────────────────────────────────────────")
    print(f"  Shape : {prices.shape}")
    print(f"  NaNs  : {prices.isnull().sum().sum()} (should be 0 after backfill)")

    print("\n── Log Returns ────────────────────────────────────────────────────")
    print(f"  Shape : {log_ret.shape}")
    print(f"  Dates : {log_ret.index[0].date()} to {log_ret.index[-1].date()}")

    print("\n── Excess Returns (annualised mean × 12) ──────────────────────────")
    print((excess[SECTORS] * 12).mean().round(4).to_string())

    print("\n── XLC / XLRE backfill check ──────────────────────────────────────")
    print(f"  XLC  Jan 2010 price : {prices.loc['2010-01-31', 'XLC']:.4f}")
    print(f"  XLRE Jan 2010 price : {prices.loc['2010-01-31', 'XLRE']:.4f}")
    print(f"  SPY  Jan 2010 price : {prices.loc['2010-01-31', 'SPY']:.4f}")

    print("\n── RF rate sample ─────────────────────────────────────────────────")
    print(rf[["rf_annual_pct","rf_monthly"]].iloc[::24].round(5).to_string())
    