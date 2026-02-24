#pragma once
#include <random>
#include <cmath>
#include <cstdint>
#include <vector>

struct RNG {
    std::mt19937_64 eng;
    std::uniform_real_distribution<double> uni;
    bool has_spare = false;
    double spare = 0.0;

    explicit RNG(uint64_t seed) : eng(seed), uni(0.0, 1.0) {}

    double normal01() {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        double u1 = 0.0;
        do { u1 = uni(eng); } while (u1 <= 0.0);
        double u2 = uni(eng);
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * std::acos(-1.0) * u2; 
        spare = r * std::sin(theta);
        has_spare = true;
        return r * std::cos(theta);
    }

    void normals01(std::vector<double>& out) {
        for (double &v : out) v = normal01();
    }
};