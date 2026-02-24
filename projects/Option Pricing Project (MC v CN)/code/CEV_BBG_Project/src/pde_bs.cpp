#include "pde_bs.hpp"
#include "tri.hpp"
#include "utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

double cn_price_bs_bbg(double S, double K, double T, double r, double sigma, const std::string& flag, double q,
                       int M, int P, double S_max_mult) {
    if (T <= 0.0 || sigma <= 0.0) {
        double intrinsic = (flag=="call")? std::max(S-K,0.0) : std::max(K-S,0.0);
        return intrinsic;
    }

    const double S_max = S_max_mult * std::max(S, K);
    const double dS = S_max / (double)M;
    const double dt = T / (double)P;

    std::vector<double> Sv(M + 1);
    for (int j = 0; j <= M; ++j) Sv[j] = j * dS;

    const int m = M - 1;
    std::vector<double> Sj(m);
    for (int j = 0; j < m; ++j) Sj[j] = Sv[j + 1];

    const double sig2 = sigma * sigma;
    const double rq = r - q;

    std::vector<double> pj(m), qj(m), uj(m);
    for (int j = 0; j < m; ++j) {
        double Sjj = Sj[j];
        pj[j] = 0.5 * sig2 * Sjj * Sjj / (dS * dS) - 0.5 * rq * Sjj / dS;
        qj[j] = -(sig2 * Sjj * Sjj / (dS * dS) + r);
        uj[j] = 0.5 * sig2 * Sjj * Sjj / (dS * dS) + 0.5 * rq * Sjj / dS;
    }

    std::vector<double> lhs_sub(m,0.0), lhs_diag(m,0.0), lhs_sup(m,0.0);
    std::vector<double> rhs_sub(m,0.0), rhs_diag(m,0.0), rhs_sup(m,0.0);
    for (int j = 0; j < m; ++j) {
        lhs_sub[j] = -0.5 * dt * pj[j];
        lhs_diag[j] = 1.0 - 0.5 * dt * qj[j];
        lhs_sup[j] = -0.5 * dt * uj[j];

        rhs_sub[j] = 0.5 * dt * pj[j];
        rhs_diag[j] = 1.0 + 0.5 * dt * qj[j];
        rhs_sup[j] = 0.5 * dt * uj[j];
    }

    std::vector<double> a(m,0.0), b(m,0.0), c(m,0.0);
    for (int j = 0; j < m; ++j) b[j] = lhs_diag[j];
    for (int j = 1; j < m; ++j) a[j] = lhs_sub[j];
    for (int j = 0; j < m - 1; ++j) c[j] = lhs_sup[j];

    std::vector<double> V(M + 1, 0.0);
    for (int j = 0; j <= M; ++j) {
        V[j] = (flag=="call")? std::max(Sv[j] - K, 0.0) : std::max(K - Sv[j], 0.0);
    }

    for (int step = 0; step < P; ++step) {
        double tau_now = (P - step) * dt;
        double tau_next = (P - step - 1) * dt;

        double bc_lo_now=0.0, bc_lo_next=0.0, bc_hi_now=0.0, bc_hi_next=0.0;
        if (flag == "call") {
            bc_lo_now = 0.0;
            bc_lo_next = 0.0;
            bc_hi_now = S_max * std::exp(-q * tau_now) - K * std::exp(-r * tau_now);
            bc_hi_next = S_max * std::exp(-q * tau_next) - K * std::exp(-r * tau_next);
        } else {
            bc_lo_now = K * std::exp(-r * tau_now);
            bc_lo_next = K * std::exp(-r * tau_next);
            bc_hi_now = 0.0;
            bc_hi_next = 0.0;
        }

        std::vector<double> Vint(m);
        for (int j = 0; j < m; ++j) Vint[j] = V[j + 1];

        std::vector<double> rhs(m, 0.0);
        for (int j = 0; j < m; ++j) rhs[j] = rhs_diag[j] * Vint[j];
        for (int j = 1; j < m; ++j) rhs[j] += rhs_sub[j] * Vint[j - 1];
        for (int j = 0; j < m - 1; ++j) rhs[j] += rhs_sup[j] * Vint[j + 1];

        rhs[0] += rhs_sub[0] * bc_lo_now;
        rhs[m-1] += rhs_sup[m-1] * bc_hi_now;
        rhs[0] -= lhs_sub[0] * bc_lo_next;
        rhs[m-1] -= lhs_sup[m-1] * bc_hi_next;

        auto sol = solve_tridiagonal(a, b, c, rhs);

        V[0] = bc_lo_next;
        V[M] = bc_hi_next;
        for (int j = 0; j < m; ++j) V[j + 1] = sol[j];
    }

    return lerp1d(Sv, V, S);
}