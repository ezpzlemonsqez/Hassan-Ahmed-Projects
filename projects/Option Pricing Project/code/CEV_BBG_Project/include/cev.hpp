#pragma once
#include <vector>
#include <cstdint>
#include <string>

struct MCPriceResult {
    double price = 0.0;
    double ci_lo = 0.0;
    double ci_hi = 0.0;
    std::vector<std::vector<double>> paths;
    std::vector<double> disc_payoffs;
};

struct MCGreeks {
    double price = 0.0;
    double delta = 0.0;
    double delta_ci_lo = 0.0;
    double delta_ci_hi = 0.0;
    double gamma = 0.0;
    double vega = 0.0;
    double theta = 0.0;
    double rho = 0.0;
};

struct CNSurface {
    std::vector<double> t;
    std::vector<double> S;
    std::vector<std::vector<double>> V;
    std::vector<double> V0;
};

double sigma_function(double t);
double bs_put_price(double S, double K, double T, double r, double sigma);
double sigma_eff_for_cev(double S0);

MCPriceResult price_put_by_mc(double S0, double r, double K, double T, int steps, int paths,
                              uint64_t seed, double conf, bool store_paths);

MCGreeks greeks_by_mc(double S0, double r, double K, double T, int steps, int paths,
                      uint64_t seed, double conf);

CNSurface crank_nicolson_put_surface(double K, double T, double r,
                                     int P_time, int M_space,
                                     double S_max, int rannacher_steps,
                                     bool store_surface);

std::vector<double> cn_delta_t0_from_V0(const std::vector<double>& V0, const std::vector<double>& S);

double cn_price_t0_cev(double K, double T, double r, int P, int M);

std::vector<int64_t> make_work_grid();

std::vector<double> mc_discounted_payoffs_put(int N, uint64_t seed);

double mc_sigma_hat(int N_ref, uint64_t seed_ref);

void print_final_price_comparison(double bs_put);

void write_partA_csvs(const std::string& out_dir);