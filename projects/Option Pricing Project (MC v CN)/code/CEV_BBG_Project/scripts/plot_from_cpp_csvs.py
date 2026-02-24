import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

OUT_PREFIX = "BBG"
CSV_IN = f"{OUT_PREFIX}_enriched_chain.csv"

K_STRIKE = 140.0 

def _ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

def _save(fig, fname, dpi=200):
    fig.tight_layout()
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _fit_powerlaw_const(x, y, power, n_fit=10):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = min(n_fit, len(x))
    if n == 0:
        return 1.0
    lx = np.log(x[:n]); ly = np.log(y[:n])
    logc = np.mean(ly - power * lx)
    return float(np.exp(logc))


def partA_cn_delta_t0():
    path = "PARTA_cn_delta_t0.csv"
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return
    d = pd.read_csv(path)
    _ensure_cols(d, ["S", "delta"])

    fig = plt.figure(figsize=(10, 6))
    plt.plot(d["S"].values, d["delta"].values)
    plt.axvline(K_STRIKE, linestyle="--", alpha=0.6)
    plt.xlabel("Stock price (S)")
    plt.ylabel("Delta at t=0")
    plt.title("Crank–Nicolson delta profile at t=0")
    plt.grid(True, alpha=0.25)
    _save(fig, "CNDelta_t0.png")


def partA_cn_surface():
    path = "PARTA_cn_surface.csv"
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return

    df = pd.read_csv(path)
    if "S" not in df.columns:
        raise ValueError("PARTA_cn_surface.csv must have column 'S'")

    S = df["S"].values
    t_cols = [c for c in df.columns if c.startswith("t=")]
    if not t_cols:
        raise ValueError("PARTA_cn_surface.csv must have columns like 't=0.0', 't=...'")

    t = np.array([float(c.split("=")[1]) for c in t_cols], dtype=float)
    V = df[t_cols].values 

    t_mesh, S_mesh = np.meshgrid(t, S)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(t_mesh, S_mesh, V, alpha=0.35, edgecolor="k", linewidth=0.15)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Stock price (S)")
    ax.set_zlabel("Option price")
    ax.set_title("Crank–Nicolson option value surface")
    _save(fig, "CrankNicolsonPlot.png")


def partA_mc_paths_band(conf=0.95):
    path = "PARTA_mc_paths.csv"
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return

    df = pd.read_csv(path)
    if "t" not in df.columns:
        raise ValueError("PARTA_mc_paths.csv must have column 't'")

    time_grid = df["t"].values.astype(float)

    stock_paths = df.drop(columns=["t"]).values.T

    a = (1.0 - conf) / 2.0
    lower = np.percentile(stock_paths, 100.0 * a, axis=0)
    upper = np.percentile(stock_paths, 100.0 * (1.0 - a), axis=0)
    mean_path = stock_paths.mean(axis=0)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(time_grid, stock_paths[:100].T, color="gray", alpha=0.3, linewidth=0.5)
    plt.plot(time_grid, lower, color="red", label=f"Lower {conf:.0%} band")
    plt.plot(time_grid, upper, color="red", label=f"Upper {conf:.0%} band")
    plt.plot(time_grid, mean_path, color="blue", label="Mean path")
    plt.fill_between(time_grid, lower, upper, color="green", alpha=0.2, label=f"{conf:.0%} band")
    plt.xlabel("Time (years)")
    plt.ylabel("Stock price")
    plt.title(f"Monte Carlo simulated paths with {conf:.0%} band")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("Montecarlo95CIBand.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved Montecarlo95CIBand.png")

def partA_mc_std_error_line():
    path = "PARTA_mc_se_line.csv"
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return
    d = pd.read_csv(path)
    _ensure_cols(d, ["N", "se"])

    fig = plt.figure(figsize=(10, 6))
    plt.loglog(d["N"].values.astype(float), d["se"].values.astype(float),
               marker="o", label=r"$\hat{\sigma}/\sqrt{N}$")
    plt.xlabel("N (number of simulations)")
    plt.ylabel("Estimated standard error")
    plt.title(r"Monte Carlo standard error: $\hat{\sigma}/\sqrt{N}$")
    plt.grid(True, which="both", ls="--", alpha=0.25)
    plt.legend()
    _save(fig, "MCStdErrorLine.png")


def _mc_err_on_effort_grid(mc_df: pd.DataFrame, effort: np.ndarray) -> np.ndarray:
    """
    Original Python called mc_error_curve(effort, ...), i.e. MC error evaluated at the same x points.
    We don't have raw payoffs here, so closest match is log-log interpolation of rel_err(N).
    """
    mcN = mc_df["N"].values.astype(float)
    mcE = mc_df["rel_err"].values.astype(float)

    mcN = np.maximum(mcN, 1.0)
    mcE = np.maximum(mcE, 1e-300)

    eff = np.asarray(effort, float)
    eff_clip = np.clip(eff, mcN.min(), mcN.max())

    return np.exp(np.interp(np.log(eff_clip), np.log(mcN), np.log(mcE)))


def partA_convergence_spatial():
    mc_path = "PARTA_mc_error_curve.csv"
    cn_path = "PARTA_cn_spatial_errors.csv"
    if not (os.path.exists(mc_path) and os.path.exists(cn_path)):
        print(f"  [skip] ConvergenceSpatial (missing {mc_path} or {cn_path})")
        return

    mc = pd.read_csv(mc_path)
    cn = pd.read_csv(cn_path)
    _ensure_cols(mc, ["N", "rel_err"])
    _ensure_cols(cn, ["effort", "rel_err"])

    effort = cn["effort"].values.astype(float)
    cn_err = cn["rel_err"].values.astype(float)

    mc_err = _mc_err_on_effort_grid(mc, effort)

    x = effort.astype(np.float64)

    c_mc = _fit_powerlaw_const(x, mc_err, power=-0.5, n_fit=10)
    mc_ref_line = c_mc * (x ** -0.5)

    c_cn = _fit_powerlaw_const(x, cn_err, power=-2.0, n_fit=10)
    cn_ref_line = c_cn * (x ** -2.0)

    fig = plt.figure(figsize=(10, 6))
    plt.loglog(effort, mc_err, marker="o", label="Monte Carlo")
    plt.loglog(effort, cn_err, marker="o", label="Crank–Nicolson (spatial)")
    plt.loglog(effort, mc_ref_line, "--", alpha=0.7, label=r"$\mathcal{O}(N^{-1/2})$")
    plt.loglog(effort, cn_ref_line, "--", alpha=0.7, label=r"$\mathcal{O}(M^{-2})$")
    plt.xlabel(r"Computational effort: $N$ (MC paths) or $M\times P$ (CN grid size)")
    plt.ylabel("Relative error")
    plt.title("Monte Carlo vs Crank–Nicolson spatial convergence")
    plt.grid(True, which="both", ls="--", alpha=0.25)
    plt.legend()
    _save(fig, "ConvergenceSpatial.png")


def partA_convergence_temporal():
    mc_path = "PARTA_mc_error_curve.csv"
    cn_path = "PARTA_cn_temporal_errors.csv"
    if not (os.path.exists(mc_path) and os.path.exists(cn_path)):
        print(f"  [skip] ConvergenceTemporal (missing {mc_path} or {cn_path})")
        return

    mc = pd.read_csv(mc_path)
    cn = pd.read_csv(cn_path)
    _ensure_cols(mc, ["N", "rel_err"])
    _ensure_cols(cn, ["effort", "rel_err"])

    effort = cn["effort"].values.astype(float)
    cn_err = cn["rel_err"].values.astype(float)

    mc_err = _mc_err_on_effort_grid(mc, effort)

    x = effort.astype(np.float64)

    c_mc = _fit_powerlaw_const(x, mc_err, power=-0.5, n_fit=10)
    mc_ref_line = c_mc * (x ** -0.5)

    c_cn = _fit_powerlaw_const(x, cn_err, power=-1.0, n_fit=10)
    cn_ref_line = c_cn * (x ** -1.0)

    fig = plt.figure(figsize=(10, 6))
    plt.loglog(effort, mc_err, marker="o", label="Monte Carlo")
    plt.loglog(effort, cn_err, marker="o", label="Crank–Nicolson (temporal)")
    plt.loglog(effort, mc_ref_line, "--", alpha=0.7, label=r"$\mathcal{O}(N^{-1/2})$")
    plt.loglog(effort, cn_ref_line, "--", alpha=0.7, label=r"$\mathcal{O}(P^{-1})$")
    plt.xlabel(r"Computational effort: $N$ (MC paths) or $M\times P$ (CN grid size)")
    plt.ylabel("Relative error")
    plt.title("Monte Carlo vs Crank–Nicolson temporal convergence")
    plt.grid(True, which="both", ls="--", alpha=0.25)
    plt.legend()
    _save(fig, "ConvergenceTemporal.png")


def run_partA_plots():
    print("\n=== PART A PLOTS ===")
    partA_cn_delta_t0()
    partA_cn_surface()
    partA_mc_paths_band(conf=0.95)
    partA_convergence_spatial()
    partA_convergence_temporal()
    partA_mc_std_error_line()


def plot_vol_smile(df):
    d = df.dropna(subset=["iv", "strike", "expiry"]).copy()
    expiries = d["expiry"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(expiries))))

    fig, ax = plt.subplots(figsize=(12, 6))
    for col, exp in zip(colors, expiries):
        sub = d[d["expiry"] == exp].sort_values("strike")
        if len(sub) < 2:
            continue
        ax.plot(sub["strike"], sub["iv"] * 100.0, marker="o", markersize=4, linewidth=1.5, color=col, label=str(exp))

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title("Implied Volatility Smile / Skew by Expiry")
    ax.legend(title="Expiry", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_VolSmile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT_PREFIX}_VolSmile.png")



def plot_greeks_profile(df):
    d = df.dropna(subset=["expiry", "strike", "type", "delta", "gamma", "vega"]).copy()
    expiries = d["expiry"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(expiries))))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    specs = [("delta", "Delta (Δ)"), ("gamma", "Gamma (Γ)"), ("vega", "Vega (per 1 vol pt)")]

    for ax, (col, lab) in zip(axes, specs):
        for c, exp in zip(colors, expiries):
            sub = d[d["expiry"] == exp]
            for opt_type, ls in (("put", "-"), ("call", "--")):
                s2 = sub[sub["type"] == opt_type].sort_values("strike")
                if len(s2) < 2:
                    continue
                ax.plot(s2["strike"], s2[col], linestyle=ls, marker=".", linewidth=1.5, color=c)
        ax.set_ylabel(lab)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Strike")
    axes[0].set_title("Greeks Profile by Expiry (calls dashed)")
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_GreeksProfile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT_PREFIX}_GreeksProfile.png")


def plot_mispricing(df):
    d = df.dropna(subset=["bs_vs_mkt", "mc_vs_mkt", "cn_vs_mkt"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (col, title, color) in zip(
        axes,
        [("bs_vs_mkt", "BS − Mid", "tab:blue"),
         ("mc_vs_mkt", "MC − Mid", "tab:orange"),
         ("cn_vs_mkt", "CN − Mid", "tab:green")]
    ):
        vals = d[col].dropna().values
        ax.hist(vals, bins=30, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax.axvline(vals.mean(), color="red", linewidth=1.5, label=f"Mean: {vals.mean():.2f}")
        ax.set_title(title)
        ax.set_xlabel("Mispricing (points)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_Mispricing.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT_PREFIX}_Mispricing.png")


def plot_scenario_heatmap(df):
    d = df.dropna(subset=["iv", "ttm_years", "spot", "strike", "rfr", "type", "bs_price", "div_yield"]).copy()
    d["moneyness"] = abs(d["strike"] / d["spot"] - 1.0)
    sample = d.nsmallest(6, "moneyness")

    spot_shocks = (-0.10, -0.05, 0.0, 0.05, 0.10)
    vol_shocks = (-0.05, -0.02, 0.0, 0.02, 0.05)

    def bs_price(S, K, T, r, sigma, flag="call", q=0.0):
        if T <= 0.0 or sigma <= 0.0:
            intrinsic = max(S - K, 0.0) if flag == "call" else max(K - S, 0.0)
            return float(intrinsic)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        disc_q = math.exp(-q * T)
        disc_r = math.exp(-r * T)
        if flag == "call":
            return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
        return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)

    rows = []
    for _, r0 in sample.iterrows():
        for dS in spot_shocks:
            for dv in vol_shocks:
                p = bs_price(
                    r0["spot"] * (1.0 + dS),
                    r0["strike"],
                    r0["ttm_years"],
                    r0["rfr"],
                    max(r0["iv"] + dv, 1e-6),
                    r0["type"],
                    q=r0["div_yield"],
                )
                rows.append((dS, dv, p - r0["bs_price"]))

    scen = pd.DataFrame(rows, columns=["spot_shock", "vol_shock", "pnl"])
    pivot = scen.groupby(["spot_shock", "vol_shock"])["pnl"].sum().unstack()

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x*100:+.0f}pp" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x*100:+.0f}%" for x in pivot.index])
    ax.set_xlabel("Vol Shock")
    ax.set_ylabel("Spot Shock")
    ax.set_title("Scenario PnL Heatmap — ATM-ish sample")
    fig.colorbar(im, ax=ax, label="Aggregate PnL (points)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_ScenarioHeatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT_PREFIX}_ScenarioHeatmap.png")


def plot_mc_confidence(df, max_options=40, tick_every=2):
    d = df.dropna(subset=["mc_price", "mc_ci_lo", "mc_ci_hi", "mid_mkt"]).copy()
    d["_atm"] = abs(d["strike"] / d["spot"] - 1.0)
    d = d.nsmallest(max_options, "_atm").sort_values(["expiry", "strike"]).reset_index(drop=True)

    x = np.arange(len(d))
    lo_err = d["mc_price"].values - d["mc_ci_lo"].values
    hi_err = d["mc_ci_hi"].values - d["mc_price"].values

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.errorbar(
        x, d["mc_price"], yerr=[lo_err, hi_err],
        fmt="o", color="tab:purple", ecolor="gray",
        elinewidth=1.2, capsize=3, markersize=4, label="MC ± 95% CI"
    )
    ax.scatter(x, d["mid_mkt"], color="tab:blue", s=20, zorder=5, label="Market Mid")

    # --- FIX: fewer ticks + compact labels ---
    tick_idx = x[::tick_every]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [f"{d.loc[i,'expiry']} {d.loc[i,'type'][0].upper()}{d.loc[i,'strike']:.0f}" for i in tick_idx],
        fontsize=8, rotation=45, ha="right"
    )

    ax.set_ylabel("Option Price (points)")
    ax.set_title("Monte Carlo Prices with 95% CI vs Market Mid (ATM-ish sample)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_MC_CI.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT_PREFIX}_MC_CI.png")



def run_partB_plots():
    csv_path = os.path.join(PROJECT_ROOT, CSV_IN)
    if not os.path.exists(csv_path):
        print(f"  [skip] {CSV_IN} not found at project root ({csv_path})")
        return
    df = pd.read_csv(csv_path)
    print(f"\n=== PART B (BBG) PLOTS ===")
    print(f"[OK] Loaded {CSV_IN}: {len(df)} rows")
    print("Generating BBG plots...")

    plot_vol_smile(df)
    plot_greeks_profile(df)
    plot_mispricing(df)
    plot_scenario_heatmap(df)
    plot_mc_confidence(df)

if __name__ == "__main__":
    run_partA_plots()
    run_partB_plots()
    print("\nDone. PNGs saved in project root.\n")