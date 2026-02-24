#pragma once
#include <string>

struct BSGreeks {
    double delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0;
};

double bs_price_bbg(double S, double K, double T, double r, double sigma, const std::string& flag, double q);
BSGreeks bs_greeks_bbg(double S, double K, double T, double r, double sigma, const std::string& flag, double q);
double implied_vol_newton_bbg(double S, double K, double T, double r, double price_target,
                             const std::string& flag, double q,
                             double tol=1e-6, int max_iter=80);