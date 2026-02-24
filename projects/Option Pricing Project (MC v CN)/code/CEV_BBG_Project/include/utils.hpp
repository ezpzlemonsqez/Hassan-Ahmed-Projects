#pragma once
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>
#include <cctype>

inline double mean(const std::vector<double>& x) {
    if (x.empty()) return std::numeric_limits<double>::quiet_NaN();
    long double s = 0.0L;
    for (double v : x) s += v;
    return (double)(s / (long double)x.size());
}

inline double stdev_sample(const std::vector<double>& x) {
    const size_t n = x.size();
    if (n < 2) return 0.0;
    const double m = mean(x);
    long double ss = 0.0L;
    for (double v : x) {
        long double d = (long double)v - (long double)m;
        ss += d * d;
    }
    return std::sqrt((double)(ss / (long double)(n - 1)));
}

inline double lerp1d(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
    if (xs.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (x <= xs.front()) return ys.front();
    if (x >= xs.back())  return ys.back();
    auto it = std::upper_bound(xs.begin(), xs.end(), x);
    size_t i1 = (size_t)(it - xs.begin());
    size_t i0 = i1 - 1;
    double x0 = xs[i0], x1 = xs[i1];
    double y0 = ys[i0], y1 = ys[i1];
    double w = (x - x0) / (x1 - x0);
    return y0 + w * (y1 - y0);
}

inline bool is_finite_pos(double x) {
    return std::isfinite(x) && x > 0.0;
}

inline bool to_double(const std::string& s, double& out) {
    try {
        size_t idx = 0;
        std::string ss = s;
        ss.erase(ss.begin(), std::find_if(ss.begin(), ss.end(), [](unsigned char c){return !std::isspace(c);}));
        ss.erase(std::find_if(ss.rbegin(), ss.rend(), [](unsigned char c){return !std::isspace(c);}).base(), ss.end());
        if (ss.empty()) return false;
        out = std::stod(ss, &idx);
        return idx > 0;
    } catch (...) {
        return false;
    }
}

inline std::string upper_trim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c){return !std::isspace(c);}));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c){return !std::isspace(c);}).base(), s.end());
    for (char &c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}