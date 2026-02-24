#include "cev.hpp"
#include "config.hpp"
#include "stats.hpp"
#include "rng.hpp"
#include "tri.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>

double sigma_function(double t) {
    return 0.18 + 0.07 * (t / cfg::T0);
}

double bs_put_price(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return std::max(K - S, 0.0);
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    return K * std::exp(-r * T) * stats::norm_cdf(-d2) - S * stats::norm_cdf(-d1);
}

double sigma_eff_for_cev(double S0_) {
    const double a = 0.18;
    const double b = 0.07;
    const double integral = cfg::T0 * (a * a + a * b + (b * b) / 3.0);
    return std::pow(S0_, cfg::gamma_cev - 1.0) * std::sqrt(integral / cfg::T0);
}

MCPriceResult price_put_by_mc(double S0_, double r, double K, double T, int steps, int paths,
                              uint64_t seed, double conf, bool store_paths) {
    MCPriceResult res;
    RNG rng(seed);
    const double dt = T / (double)steps;
    const double sdt = std::sqrt(dt);

    std::vector<double> sig(steps);
    for (int i = 0; i < steps; ++i) sig[i] = sigma_function((double)i * dt);

    std::vector<double> S(paths, S0_);
    if (store_paths) {
        res.paths.assign(paths, std::vector<double>(steps + 1, 0.0));
        for (int p = 0; p < paths; ++p) res.paths[p][0] = S0_;
    }

    for (int i = 0; i < steps; ++i) {
        for (int p = 0; p < paths; ++p) {
            double z = rng.normal01();
            double Si = S[p];
            double diff = sig[i] * std::pow(std::max(Si, 0.0), cfg::gamma_cev) * sdt * z;
            double drift = r * Si * dt;
            S[p] = Si + drift + diff;
            if (store_paths) res.paths[p][i + 1] = S[p];
        }
    }

    res.disc_payoffs.resize(paths);
    const double disc = std::exp(-r * T);
    for (int p = 0; p < paths; ++p) res.disc_payoffs[p] = disc * std::max(K - S[p], 0.0);

    res.price = mean(res.disc_payoffs);
    double alpha = stats::norm_ppf((1.0 - conf) / 2.0);
    double half = (-alpha) * stdev_sample(res.disc_payoffs) / std::sqrt((double)paths);
    res.ci_lo = res.price - half;
    res.ci_hi = res.price + half;
    return res;
}

MCGreeks greeks_by_mc(double S0_, double r, double K, double T, int steps, int paths, uint64_t seed, double conf) {
    MCGreeks g;
    const double dS = 0.01 * S0_;

    auto p0  = price_put_by_mc(S0_, r, K, T, steps, paths, seed, conf, false);
    auto pup = price_put_by_mc(S0_ + dS, r, K, T, steps, paths, seed, conf, false);
    auto pdn = price_put_by_mc(S0_ - dS, r, K, T, steps, paths, seed, conf, false);

    g.price = p0.price;

    std::vector<double> delta_samples(paths);
    for (int i = 0; i < paths; ++i) delta_samples[i] = (pup.disc_payoffs[i] - pdn.disc_payoffs[i]) / (2.0 * dS);

    g.delta = mean(delta_samples);
    double alpha = stats::norm_ppf((1.0 - conf) / 2.0);
    double half = (-alpha) * stdev_sample(delta_samples) / std::sqrt((double)paths);
    g.delta_ci_lo = g.delta - half;
    g.delta_ci_hi = g.delta + half;

    g.gamma = (pup.price - 2.0 * p0.price + pdn.price) / (dS * dS);

    const double dv = 0.05;
    auto mc_price_sigma_bump = [&](double bump) {
        RNG rr(seed);
        const double dt = T / (double)steps;
        const double sdt = std::sqrt(dt);
        std::vector<double> sig(steps);
        for (int i = 0; i < steps; ++i) sig[i] = sigma_function((double)i * dt) + bump;

        std::vector<double> S(paths, S0_);
        for (int i = 0; i < steps; ++i) {
            for (int p = 0; p < paths; ++p) {
                double z = rr.normal01();
                double Si = S[p];
                S[p] = Si + r * Si * dt + sig[i] * std::pow(std::max(Si, 0.0), cfg::gamma_cev) * sdt * z;
            }
        }
        const double disc = std::exp(-r * T);
        long double sum = 0.0L;
        for (int p = 0; p < paths; ++p) sum += disc * std::max(K - S[p], 0.0);
        return (double)(sum / (long double)paths);
    };
    g.vega = (mc_price_sigma_bump(+dv) - mc_price_sigma_bump(-dv)) / (2.0 * dv);

    const double dT = 1.0 / 252.0;
    auto pTdn = price_put_by_mc(S0_, r, K, T - dT, steps, paths, seed, conf, false);
    g.theta = (pTdn.price - p0.price) / dT;

    const double dr = 0.01;
    auto prup = price_put_by_mc(S0_, r + dr, K, T, steps, paths, seed, conf, false);
    auto prdn = price_put_by_mc(S0_, r - dr, K, T, steps, paths, seed, conf, false);
    g.rho = (prup.price - prdn.price) / (2.0 * dr);

    return g;
}

CNSurface crank_nicolson_put_surface(double K, double T, double r,
                                     int P_time, int M_space,
                                     double S_max, int rannacher_steps,
                                     bool store_surface) {
    CNSurface out;
    const double dt = T / (double)P_time;
    out.t.resize(P_time + 1);
    for (int i = 0; i <= P_time; ++i) out.t[i] = i * dt;

    out.S.resize(M_space + 1);
    const double dS = S_max / (double)M_space;
    for (int j = 0; j <= M_space; ++j) out.S[j] = j * dS;

    const int m = M_space - 1;
    std::vector<double> Sj(m);
    for (int j = 0; j < m; ++j) Sj[j] = out.S[j + 1];

    auto V_left = [&](double tau) { return K * std::exp(-r * (T - tau)); };

    std::vector<double> V_next(M_space + 1, 0.0);
    for (int j = 0; j <= M_space; ++j) V_next[j] = std::max(K - out.S[j], 0.0);
    V_next[0] = V_left(T);
    V_next[M_space] = 0.0;

    if (store_surface) {
        out.V.assign(P_time + 1, std::vector<double>(M_space + 1, 0.0));
        out.V[P_time] = V_next;
        for (int i = 0; i <= P_time; ++i) {
            out.V[i][0] = V_left(out.t[i]);
            out.V[i][M_space] = 0.0;
        }
    }

    for (int i = P_time - 1; i >= 0; --i) {
        const double sig = sigma_function(out.t[i]);
        std::vector<double> var_term(m);
        for (int j = 0; j < m; ++j) var_term[j] = (sig * sig) * std::pow(Sj[j], 2.0 * cfg::gamma_cev);

        std::vector<double> a(m), b(m), c(m);
        for (int j = 0; j < m; ++j) {
            const double Sjv = Sj[j];
            a[j] = -var_term[j] / (2.0 * dS * dS) + (r * Sjv) / (2.0 * dS);
            b[j] = r + var_term[j] / (dS * dS);
            c[j] = -var_term[j] / (2.0 * dS * dS) - (r * Sjv) / (2.0 * dS);
        }

        std::vector<double> V_i(M_space + 1, 0.0);

        if (i >= P_time - rannacher_steps) {
            double tau = out.t[i + 1];
            std::vector<double> V_tmp = V_next;
            for (int rep = 0; rep < 2; ++rep) {
                const double hdt = dt / 2.0;
                const double tau_new = tau - hdt;
                const double V0 = V_left(tau_new);

                std::vector<double> diag(m), lower(m, 0.0), upper(m, 0.0), rhs(m);
                for (int j = 0; j < m; ++j) diag[j] = 1.0 + hdt * b[j];
                for (int j = 1; j < m; ++j) lower[j] = hdt * a[j];
                for (int j = 0; j < m - 1; ++j) upper[j] = hdt * c[j];

                for (int j = 0; j < m; ++j) rhs[j] = V_tmp[j + 1];
                rhs[0] -= hdt * a[0] * V0;

                auto sol = solve_tridiagonal(lower, diag, upper, rhs);

                V_tmp[0] = V0;
                V_tmp[M_space] = 0.0;
                for (int j = 0; j < m; ++j) V_tmp[j + 1] = sol[j];

                tau = tau_new;
            }
            V_i = V_tmp;
        } else {
            const double V0_i = V_left(out.t[i]);
            const double V0_ip1 = V_left(out.t[i + 1]);

            std::vector<double> Vint(m);
            for (int j = 0; j < m; ++j) Vint[j] = V_next[j + 1];

            std::vector<double> diagA(m), diagB(m), lowerA(m, 0.0), upperA(m, 0.0), lowerB(m, 0.0), upperB(m, 0.0);
            for (int j = 0; j < m; ++j) {
                diagA[j] = 1.0 + 0.5 * dt * b[j];
                diagB[j] = 1.0 - 0.5 * dt * b[j];
            }
            for (int j = 1; j < m; ++j) {
                lowerA[j] = 0.5 * dt * a[j];
                lowerB[j] = -0.5 * dt * a[j];
            }
            for (int j = 0; j < m - 1; ++j) {
                upperA[j] = 0.5 * dt * c[j];
                upperB[j] = -0.5 * dt * c[j];
            }

            std::vector<double> rhs(m, 0.0);
            for (int j = 0; j < m; ++j) rhs[j] = diagB[j] * Vint[j];
            for (int j = 1; j < m; ++j) rhs[j] += lowerB[j] * Vint[j - 1];
            for (int j = 0; j < m - 1; ++j) rhs[j] += upperB[j] * Vint[j + 1];
            rhs[0] -= 0.5 * dt * a[0] * (V0_ip1 + V0_i);

            auto sol = solve_tridiagonal(lowerA, diagA, upperA, rhs);

            V_i[0] = V0_i;
            V_i[M_space] = 0.0;
            for (int j = 0; j < m; ++j) V_i[j + 1] = sol[j];
        }

        V_next = V_i;
        if (store_surface) out.V[i] = V_i;
        if (i == 0) out.V0 = V_i;
    }

    if (out.V0.empty()) out.V0 = V_next;
    return out;
}

std::vector<double> cn_delta_t0_from_V0(const std::vector<double>& V0, const std::vector<double>& S) {
    const size_t n = S.size();
    std::vector<double> delta(n, 0.0);
    if (n < 2) return delta;
    const double dS = S[1] - S[0];
    if (n >= 3) {
        for (size_t j = 1; j + 1 < n; ++j) delta[j] = (V0[j + 1] - V0[j - 1]) / (2.0 * dS);
    }
    delta[0] = (V0[1] - V0[0]) / dS;
    delta[n - 1] = (V0[n - 1] - V0[n - 2]) / dS;
    return delta;
}

double cn_price_t0_cev(double K, double T, double r, int P, int M) {
    auto surf = crank_nicolson_put_surface(K, T, r, P, M, cfg::S_MAX, cfg::RANNACHER_STEPS, false);
    return lerp1d(surf.S, surf.V0, K);
}

std::vector<int64_t> make_work_grid() {
    auto round_ll = [](double x) -> int64_t { return (int64_t)std::llround(x); };

    std::vector<int64_t> base;
    base.reserve(cfg::WORK_BASE_POINTS);
    double lo = std::log10((double)cfg::WORK_MIN);
    double hi = std::log10((double)cfg::WORK_MAX);
    for (int i = 0; i < cfg::WORK_BASE_POINTS; ++i) {
        double t = (cfg::WORK_BASE_POINTS == 1) ? 0.0 : (double)i / (double)(cfg::WORK_BASE_POINTS - 1);
        base.push_back(round_ll(std::pow(10.0, lo + t * (hi - lo))));
    }

    int64_t lo_end  = std::min<int64_t>(cfg::WORK_MAX, (int64_t)cfg::WORK_MIN * 10);
    int64_t lo_step = std::max<int64_t>(1, (lo_end - cfg::WORK_MIN) / std::max<int64_t>(1, cfg::WORK_TAIL_POINTS - 1));
    std::vector<int64_t> lo_lin;
    for (int64_t v = cfg::WORK_MIN; v <= lo_end; v += lo_step) lo_lin.push_back(v);

    int64_t hi_start = std::max<int64_t>(cfg::WORK_MIN, ((int64_t)cfg::WORK_MAX * 8) / 10);
    int64_t hi_step  = std::max<int64_t>(1, ((int64_t)cfg::WORK_MAX - hi_start) / std::max<int64_t>(1, cfg::WORK_TAIL_POINTS - 1));
    std::vector<int64_t> hi_lin;
    for (int64_t v = hi_start; v <= cfg::WORK_MAX; v += hi_step) hi_lin.push_back(v);

    std::vector<int64_t> x;
    x.reserve(base.size() + lo_lin.size() + hi_lin.size());
    x.insert(x.end(), base.begin(), base.end());
    x.insert(x.end(), lo_lin.begin(), lo_lin.end());
    x.insert(x.end(), hi_lin.begin(), hi_lin.end());

    std::sort(x.begin(), x.end());
    x.erase(std::unique(x.begin(), x.end()), x.end());
    x.erase(std::remove_if(x.begin(), x.end(), [](int64_t v) { return v < cfg::WORK_MIN || v > cfg::WORK_MAX; }), x.end());
    return x;
}

std::vector<double> mc_discounted_payoffs_put(int N, uint64_t seed) {
    RNG rng(seed);
    const double dt = cfg::T0 / (double)cfg::n_steps;
    const double sdt = std::sqrt(dt);

    std::vector<double> sig(cfg::n_steps);
    for (int i = 0; i < cfg::n_steps; ++i) sig[i] = sigma_function(i * dt);

    std::vector<double> S(N, cfg::S0);
    for (int i = 0; i < cfg::n_steps; ++i) {
        for (int p = 0; p < N; ++p) {
            double z = rng.normal01();
            double Si = S[p];
            S[p] = Si + cfg::r0 * Si * dt + sig[i] * std::pow(std::max(Si, 0.0), cfg::gamma_cev) * sdt * z;
        }
    }

    const double disc = std::exp(-cfg::r0 * cfg::T0);
    std::vector<double> out(N);
    for (int p = 0; p < N; ++p) out[p] = disc * std::max(cfg::K0 - S[p], 0.0);
    return out;
}

double mc_sigma_hat(int N_ref, uint64_t seed_ref) {
    auto pay = mc_discounted_payoffs_put(N_ref, seed_ref);
    return stdev_sample(pay);
}

void print_final_price_comparison(double bs_put) {
    double mc_ref = mean(mc_discounted_payoffs_put(cfg::MC_BASELINE_N, 1));
    double mc_at_max = mean(mc_discounted_payoffs_put(cfg::WORK_MAX, 0));

    double cn_spatial_ref  = cn_price_t0_cev(cfg::K0, cfg::T0, cfg::r0, cfg::CN_SPATIAL_P, cfg::CN_SPATIAL_M_REF);
    double cn_temporal_ref = cn_price_t0_cev(cfg::K0, cfg::T0, cfg::r0, cfg::CN_TEMPORAL_P_REF, cfg::CN_TEMPORAL_M);
    double cn_avg = 0.5 * (cn_spatial_ref + cn_temporal_ref);

    auto line = [&](const std::string& name, double val) {
        double diff = val - bs_put;
        double rel  = diff / bs_put * 100.0;
        std::cout << std::left << std::setw(22) << name << ": "
                  << std::fixed << std::setprecision(6) << val
                  << "   diff=" << std::showpos << std::fixed << std::setprecision(6) << diff << std::noshowpos
                  << "   rel="  << std::showpos << std::fixed << std::setprecision(3) << rel  << "%" << std::noshowpos
                  << "\n";
    };

    std::cout << "===== FINAL PRICE COMPARISON (vs Black–Scholes baseline) =====\n";
    std::cout << "BS (sigma_eff)         : " << std::fixed << std::setprecision(6) << bs_put << "   (baseline)\n";
    line("MC ref (N=1,000,000)", mc_ref);
    line("MC at max (N=WORK_MAX)", mc_at_max);
    line("CN spatial ref", cn_spatial_ref);
    line("CN temporal ref", cn_temporal_ref);
    line("CN ref (avg)", cn_avg);
}

struct ErrorCurve {
    double p_ref = 0.0;
    std::vector<double> prices;
    std::vector<double> rel_err;
};

static ErrorCurve mc_error_curve(const std::vector<int64_t>& N_list, int N_ref, uint64_t seed_curve, uint64_t seed_ref) {
    ErrorCurve ec;
    auto pay_ref = mc_discounted_payoffs_put(N_ref, seed_ref);
    ec.p_ref = mean(pay_ref);

    int64_t N_max = *std::max_element(N_list.begin(), N_list.end());
    auto pay = mc_discounted_payoffs_put((int)N_max, seed_curve);

    std::vector<long double> csum((size_t)N_max);
    long double s = 0.0L;
    for (int64_t i = 0; i < N_max; ++i) {
        s += (long double)pay[(size_t)i];
        csum[(size_t)i] = s;
    }

    ec.prices.resize(N_list.size());
    ec.rel_err.resize(N_list.size());
    for (size_t k = 0; k < N_list.size(); ++k) {
        int64_t N = N_list[k];
        double p = (double)(csum[(size_t)N - 1] / (long double)N);
        ec.prices[k] = p;
        ec.rel_err[k] = std::abs(p - ec.p_ref) / (std::abs(ec.p_ref) + 1e-16);
    }
    return ec;
}

static void write_csv_2col(const std::string& path,
                           const std::string& h1, const std::string& h2,
                           const std::vector<double>& x,
                           const std::vector<double>& y) {
    std::ofstream f(path);
    f << h1 << "," << h2 << "\n";
    for (size_t i = 0; i < x.size(); ++i) f << std::setprecision(16) << x[i] << "," << y[i] << "\n";
}

static void write_csv_3col_ll_dd(const std::string& path,
                                 const std::string& h1, const std::string& h2, const std::string& h3,
                                 const std::vector<int64_t>& a,
                                 const std::vector<int64_t>& b,
                                 const std::vector<double>& c) {
    std::ofstream f(path);
    f << h1 << "," << h2 << "," << h3 << "\n";
    for (size_t i = 0; i < c.size(); ++i) f << a[i] << "," << b[i] << "," << std::setprecision(16) << c[i] << "\n";
}

static void write_cn_surface_csv(const std::string& path,
                                 const std::vector<double>& t,
                                 const std::vector<double>& S,
                                 const std::vector<std::vector<double>>& V) {

    std::ofstream f(path);
    f << "S";
    for (double tt : t) f << ",t=" << std::setprecision(16) << tt;
    f << "\n";

    for (size_t j = 0; j < S.size(); ++j) {
        f << std::setprecision(16) << S[j];
        for (size_t i = 0; i < t.size(); ++i) f << "," << std::setprecision(16) << V[i][j];
        f << "\n";
    }
}

void write_partA_csvs(const std::string& out_dir) {
    auto mc = price_put_by_mc(cfg::S0, cfg::r0, cfg::K0, cfg::T0, cfg::n_steps, cfg::n_paths, 0, cfg::CONF_LEVEL, true);

    std::vector<double> times(cfg::n_steps + 1);
    for (int i = 0; i <= cfg::n_steps; ++i) times[i] = cfg::T0 * (double)i / (double)cfg::n_steps;

    {
        std::ofstream f(out_dir + "/PARTA_mc_paths.csv");
        f << "t";
        int keep = std::min<int>(1000, (int)mc.paths.size());
        for (int p = 0; p < keep; ++p) f << ",path" << p;
        f << "\n";
        for (int k = 0; k <= cfg::n_steps; ++k) {
            f << std::setprecision(16) << times[k];
            for (int p = 0; p < keep; ++p) f << "," << std::setprecision(16) << mc.paths[p][k];
            f << "\n";
        }
    }

    int P_plot = 100, M_plot = 300;
    auto surf = crank_nicolson_put_surface(cfg::K0, cfg::T0, cfg::r0, P_plot, M_plot, cfg::S_MAX, cfg::RANNACHER_STEPS, true);
    auto delta = cn_delta_t0_from_V0(surf.V0, surf.S);

    write_cn_surface_csv(out_dir + "/PARTA_cn_surface.csv", surf.t, surf.S, surf.V);

    {
        std::ofstream f(out_dir + "/PARTA_cn_delta_t0.csv");
        f << "S,delta\n";
        for (size_t i = 0; i < surf.S.size(); ++i)
            f << std::setprecision(16) << surf.S[i] << "," << std::setprecision(16) << delta[i] << "\n";
    }

    auto work = make_work_grid();

    auto mc_ec = mc_error_curve(work, cfg::MC_BASELINE_N, 0, 1);
    {
        std::ofstream f(out_dir + "/PARTA_mc_error_curve.csv");
        f << "N,price_est,rel_err,p_ref\n";
        for (size_t i = 0; i < work.size(); ++i) {
            f << work[i] << "," << std::setprecision(16) << mc_ec.prices[i] << "," << std::setprecision(16) << mc_ec.rel_err[i]
              << "," << std::setprecision(16) << mc_ec.p_ref << "\n";
        }
    }

    {
        const int P = cfg::CN_SPATIAL_P;
        double p_ref = cn_price_t0_cev(cfg::K0, cfg::T0, cfg::r0, P, cfg::CN_SPATIAL_M_REF);

        std::vector<int64_t> Ms;
        for (auto w : work) {
            int64_t M = (int64_t)std::llround((double)w / (double)P);
            if (M < 5) M = 5;
            Ms.push_back(M);
        }
        std::sort(Ms.begin(), Ms.end());
        Ms.erase(std::unique(Ms.begin(), Ms.end()), Ms.end());

        std::vector<int64_t> effort, M_list;
        std::vector<double> err;
        for (auto M : Ms) {
            double p = cn_price_t0_cev(cfg::K0, cfg::T0, cfg::r0, P, (int)M);
            effort.push_back(M * (int64_t)P);
            M_list.push_back(M);
            err.push_back(std::abs(p - p_ref) / (std::abs(p_ref) + 1e-16));
        }
        write_csv_3col_ll_dd(out_dir + "/PARTA_cn_spatial_errors.csv", "effort", "M", "rel_err", effort, M_list, err);
    }

    {
        const int M = cfg::CN_TEMPORAL_M;
        double p_ref = cn_price_t0_cev(cfg::K0, cfg::T0, cfg::r0, cfg::CN_TEMPORAL_P_REF, M);

        std::vector<int64_t> Ps;
        for (auto w : work) {
            int64_t P = (int64_t)std::llround((double)w / (double)M);
            if (P < 5) P = 5;
            Ps.push_back(P);
        }
        std::sort(Ps.begin(), Ps.end());
        Ps.erase(std::unique(Ps.begin(), Ps.end()), Ps.end());

        std::vector<int64_t> effort, P_list;
        std::vector<double> err;
        for (auto P : Ps) {
            double p = cn_price_t0_cev(cfg::K0, cfg::T0, cfg::r0, (int)P, M);
            effort.push_back((int64_t)M * P);
            P_list.push_back(P);
            err.push_back(std::abs(p - p_ref) / (std::abs(p_ref) + 1e-16));
        }
        write_csv_3col_ll_dd(out_dir + "/PARTA_cn_temporal_errors.csv", "effort", "P", "rel_err", effort, P_list, err);
    }

    {
        double sigma_hat = mc_sigma_hat(cfg::MC_BASELINE_N, 123);
        std::ofstream f(out_dir + "/PARTA_mc_se_line.csv");
        f << "N,se\n";
        for (auto N : work) {
            double se = sigma_hat / std::sqrt((double)N);
            f << N << "," << std::setprecision(16) << se << "\n";
        }
    }
}