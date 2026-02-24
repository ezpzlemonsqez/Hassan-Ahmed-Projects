#include "bs.hpp"
#include "stats.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

double bs_price_bbg(double S, double K, double T, double r, double sigma, const std::string& flag, double q) {
    if (T <= 0.0 || sigma <= 0.0) {
        double intrinsic = (flag == "call") ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
        return intrinsic;
    }
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double disc_q = std::exp(-q * T);
    const double disc_r = std::exp(-r * T);

    if (flag == "call") return S * disc_q * stats::norm_cdf(d1) - K * disc_r * stats::norm_cdf(d2);
    return K * disc_r * stats::norm_cdf(-d2) - S * disc_q * stats::norm_cdf(-d1);
}

BSGreeks bs_greeks_bbg(double S, double K, double T, double r, double sigma, const std::string& flag, double q) {
    BSGreeks g;
    if (T <= 0.0 || sigma <= 0.0) return g;

    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double disc_q = std::exp(-q * T);
    const double disc_r = std::exp(-r * T);
    const double pdf1   = stats::norm_pdf(d1);

    g.gamma = disc_q * pdf1 / (S * sigma * sqrtT);
    g.vega  = S * disc_q * pdf1 * sqrtT / 100.0;

    if (flag == "call") {
        g.delta = disc_q * stats::norm_cdf(d1);
        g.theta = (-(S * disc_q * pdf1 * sigma) / (2.0 * sqrtT)
                   - r * K * disc_r * stats::norm_cdf(d2)
                   + q * S * disc_q * stats::norm_cdf(d1)) / 365.0;
        g.rho = K * T * disc_r * stats::norm_cdf(d2) / 100.0;
    } else {
        g.delta = disc_q * (stats::norm_cdf(d1) - 1.0);
        g.theta = (-(S * disc_q * pdf1 * sigma) / (2.0 * sqrtT)
                   + r * K * disc_r * stats::norm_cdf(-d2)
                   - q * S * disc_q * stats::norm_cdf(-d1)) / 365.0;
        g.rho = -K * T * disc_r * stats::norm_cdf(-d2) / 100.0;
    }
    return g;
}

double implied_vol_newton_bbg(double S, double K, double T, double r, double price_target,
                             const std::string& flag, double q, double tol, int max_iter) {
    double sigma = 0.20;
    for (int it = 0; it < max_iter; ++it) {
        double p = bs_price_bbg(S, K, T, r, sigma, flag, q);
        double v = bs_greeks_bbg(S, K, T, r, sigma, flag, q).vega * 100.0;
        if (!std::isfinite(v) || std::abs(v) < 1e-12) break;

        sigma -= (p - price_target) / v;
        sigma = std::max(sigma, 1e-6);

        double p2 = bs_price_bbg(S, K, T, r, sigma, flag, q);
        if (std::abs(p2 - price_target) < tol) break;
    }
    if (sigma > 0.0 && sigma < 5.0) return sigma;
    return std::numeric_limits<double>::quiet_NaN();
}