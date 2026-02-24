#pragma once
#include <vector>
#include <cstddef>

inline std::vector<double> solve_tridiagonal(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d
) {
    const size_t n = d.size();
    std::vector<double> cp(n, 0.0), dp(n, 0.0), x(n, 0.0);
    if (n == 0) return x;

    double denom = b[0];
    cp[0] = (n > 1) ? (c[0] / denom) : 0.0;
    dp[0] = d[0] / denom;

    for (size_t i = 1; i < n; ++i) {
        denom = b[i] - a[i] * cp[i - 1];
        cp[i] = (i + 1 < n) ? (c[i] / denom) : 0.0;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
    }

    x[n - 1] = dp[n - 1];
    for (size_t i = n - 1; i-- > 0;) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    return x;
}