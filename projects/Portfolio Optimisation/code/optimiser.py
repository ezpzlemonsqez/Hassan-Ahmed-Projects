import numpy as np
from scipy.optimize import minimize
from returns import SECTORS

N = len(SECTORS)   # 11

# ── Constraints and bounds ────────────────────────────────────────────────────
LAMBDA = 1.5    # risk aversion
C_COST   = 0.0010  # 10 bps transaction cost per unit turnover
TAU      = 0.30    # max one-way turnover per rebalance
W_MIN    = 0.02    # min weight per sector
W_MAX    = 0.40    # max weight per sector

def drift_weights(w_prev, returns_t):
    """
    Adjust prior weights by realised returns to get drifted weights.
    w_prev     : array (N,), weights at start of period
    returns_t  : array (N,), simple returns over the period
    Returns drifted weights normalised to sum to 1.
    """
    w_drifted = w_prev * (1 + returns_t)
    w_drifted = w_drifted / w_drifted.sum()
    return w_drifted

def objective(w, mu, sigma, w_prev):
    """
    Negative of: w'mu - (lambda/2) w'Sigma w - c * ||w - w_prev||_1
    (negative because scipy minimises)
    """
    ret      = w @ mu
    risk     = 0.5 * LAMBDA * w @ sigma @ w
    turnover = C_COST * np.sum(np.abs(w - w_prev))
    return -(ret - risk - turnover)

def solve_weights(mu, sigma, w_prev):
    """
    Solves the constrained QP.

    mu     : array (N,), monthly expected returns
    sigma  : array (N, N), covariance matrix
    w_prev : array (N,), drifted prior weights

    Returns:
        w_opt  : array (N,), optimal weights
        success: bool
    """
    # Use midpoint between equal weight and w_prev as starting point
    w_equal = np.ones(N) / N
    w0      = 0.5 * w_prev + 0.5 * w_equal
    w0      = np.clip(w0, W_MIN, W_MAX)
    w0      = w0 / w0.sum()

    # Bounds: each weight in [W_MIN, W_MAX]
    bounds = [(W_MIN, W_MAX)] * N

    # Constraints
    constraints = [
        # Full investment
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0
        },
        # Turnover cap
        {
            "type": "ineq",
            "fun": lambda w: TAU - np.sum(np.abs(w - w_prev))
        },
    ]

    result = minimize(
        objective,
        w0,
        args       = (mu, sigma, w_prev),
        method     = "SLSQP",
        bounds     = bounds,
        constraints= constraints,
        options    = {"ftol": 1e-9, "maxiter": 1000}
    )

    if result.success or result.status == 9:
        # status 9 = iteration limit but still feasible solution
        w_opt = result.x
        w_opt = np.clip(w_opt, W_MIN, W_MAX)
        w_opt = w_opt / w_opt.sum()   # renormalise after clip
        return w_opt, True
    else:
        # Fallback to equal weight with a warning
        print(f"  [WARNING] Optimiser failed: {result.message}. Falling back to equal weight.")
        return np.ones(N) / N, False

# ── Sanity check when run directly ───────────────────────────────────────────
if __name__ == "__main__":
    from estimation import compute_estimates
    from returns import get_returns

    _, log_ret, excess, rf = get_returns()
    estimates = compute_estimates(excess, rf)
    dates     = sorted(estimates.keys())

    w_equal = np.ones(N) / N

    print("── Independent solves from equal weight ────────────────────────────")
    for d in [dates[0], dates[len(dates)//2], dates[-1]]:
        e = estimates[d]
        w_opt, success = solve_weights(e["mu"], e["sigma"], w_equal)

        print(f"\n── {d.date()} ───────────────────────────────────────────────")
        print(f"  Solver success  : {success}")
        print(f"  Weights sum     : {w_opt.sum():.6f}")
        print(f"  Turnover vs EW  : {np.sum(np.abs(w_opt - w_equal)):.4f}  (cap = {TAU})")
        print(f"  Weights:")
        for s, ww in zip(SECTORS, w_opt):
            bar = "█" * int(ww * 100)
            print(f"    {s:5s}  {ww:.4f}  {bar}")

        ret  = w_opt @ e["mu"] * 12
        risk = 0.5 * LAMBDA * w_opt @ e["sigma"] @ w_opt * 12
        print(f"  Return term (ann) : {ret:.5f}")
        print(f"  Risk term   (ann) : {risk:.5f}")

    print("\n── Unconstrained min-variance (last date, no bounds) ───────────────")
    e         = estimates[dates[-1]]
    sigma_inv = np.linalg.inv(e["sigma"])
    ones      = np.ones(N)
    w_mv      = sigma_inv @ ones / (ones @ sigma_inv @ ones)
    for s, ww in zip(SECTORS, w_mv):
        print(f"    {s:5s}  {ww:.4f}")
    print(f"  Sum: {w_mv.sum():.4f}")