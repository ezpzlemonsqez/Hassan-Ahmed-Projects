import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from returns import get_returns, SECTORS

LOOKBACK = 36      # months of rolling estimation window
ERP      = 0.055  # equity risk premium, annualised (5.5%)
WINSOR   = (0.05, 0.95)  # percentile bounds for beta winsorisation

# ── 1. Rolling CAPM beta for a single asset ───────────────────────────────────
def estimate_beta(excess_asset, excess_market, lookback=LOOKBACK):
    """
    OLS beta from a window of excess returns.
    Returns scalar beta.
    """
    x = excess_market.values
    y = excess_asset.values
    # Stack into design matrix [1, x]
    X = np.column_stack([np.ones(len(x)), x])
    # OLS: beta = (X'X)^{-1} X'y
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs[1]   # slope = beta

# ── 2. CAPM expected return vector at date t ──────────────────────────────────
def capm_mu(excess_window, rf_t, lookback=LOOKBACK, erp=ERP, winsor=WINSOR):
    """
    excess_window : DataFrame shape (lookback, n_sectors+1)
                    columns = SECTORS + ['SPY']
    rf_t          : scalar, monthly risk-free rate at rebalance date

    Returns:
        mu  : array shape (n_sectors,), monthly expected returns
        betas: array shape (n_sectors,), raw (winsorised) betas
    """
    mkt = excess_window["SPY"].values
    betas = np.array([
        estimate_beta(excess_window[s], excess_window["SPY"])
        for s in SECTORS
    ])

    # Winsorise cross-sectionally
    lo, hi = np.nanpercentile(betas, [w * 100 for w in winsor])
    betas_w = np.clip(betas, lo, hi)

    # Monthly expected return: rf + beta * (ERP / 12)
    mu = rf_t + betas_w * (erp / 12)
    return mu, betas_w

# ── 3. Ledoit-Wolf covariance ─────────────────────────────────────────────────
def ledoit_wolf_cov(excess_window):
    """
    Fits Ledoit-Wolf shrinkage estimator on sector excess returns.
    Returns:
        sigma : ndarray shape (n_sectors, n_sectors)
        delta : float, shrinkage intensity
    """
    X = excess_window[SECTORS].values   # shape (lookback, n_sectors)
    lw = LedoitWolf().fit(X)
    return lw.covariance_, lw.shrinkage_

# ── 4. Compute estimates for every rebalance date ────────────────────────────
def compute_estimates(excess, rf, lookback=LOOKBACK):
    """
    Walks through every valid rebalance date and computes:
        mu, betas, sigma, shrinkage intensity

    Returns a dict keyed by date with values:
        {
          'mu'       : array (n_sectors,),
          'betas'    : array (n_sectors,),
          'sigma'    : array (n_sectors, n_sectors),
          'delta'    : float
        }

    First valid date is index[lookback] (first date with a full window).
    """
    estimates = {}
    dates     = excess.index

    for i in range(lookback, len(dates)):
        t       = dates[i]
        window  = excess.iloc[i - lookback : i]   # [t-36, t) exclusive of t
        rf_t    = rf.loc[t, "rf_monthly"]

        mu, betas       = capm_mu(window, rf_t)
        sigma, delta    = ledoit_wolf_cov(window)

        estimates[t] = {
            "mu"    : mu,
            "betas" : betas,
            "sigma" : sigma,
            "delta" : delta,
        }

    return estimates

if __name__ == "__main__":
    _, _, excess, rf = get_returns()
    estimates = compute_estimates(excess, rf)
    dates = sorted(estimates.keys())
    print(f"First rebalance: {dates[0].date()}")
    print(f"Last rebalance : {dates[-1].date()}")
    print(f"Total periods  : {len(dates)}")