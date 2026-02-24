#pragma once
#include <string>

double cn_price_bs_bbg(double S, double K, double T, double r, double sigma, const std::string& flag, double q,
                       int M, int P, double S_max_mult);