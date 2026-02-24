#pragma once
#include <string>

namespace cfg {

inline constexpr double S0 = 140.0;
inline constexpr double K0 = 140.0;
inline constexpr double T0 = 1.4;
inline constexpr double r0 = 0.02;
inline constexpr double gamma_cev = 0.95;

inline constexpr int n_steps = 100;
inline constexpr int n_paths = 10'000;

inline constexpr int MC_BASELINE_N = 1'000'000;

inline constexpr int CN_SPATIAL_P = 500;
inline constexpr int CN_SPATIAL_M_REF = 100'000;

inline constexpr int CN_TEMPORAL_M = 500;
inline constexpr int CN_TEMPORAL_P_REF = 100'000;

inline constexpr double S_MAX = 250.0;
inline constexpr int RANNACHER_STEPS = 2;
inline constexpr double CONF_LEVEL = 0.95;

inline constexpr int WORK_MIN = 1'000;
inline constexpr int WORK_MAX = MC_BASELINE_N;
inline constexpr int WORK_BASE_POINTS = 35;
inline constexpr int WORK_TAIL_POINTS = 25;

inline const std::string SNAPSHOT_PATH = "data/snapshot.csv";
inline const std::string CHAIN_PATH    = "data/options_chain_clean.csv";
inline const std::string OUT_PREFIX    = "BBG";

inline constexpr int    BBG_MC_PATHS   = 30'000;
inline constexpr int    BBG_CN_M       = 200;
inline constexpr int    BBG_CN_P       = 200;
inline constexpr double BBG_S_MAX_MULT = 2.0;
inline constexpr double BBG_CONF_LEVEL = 0.95;

inline constexpr bool RUN_PART_A = true;
inline constexpr bool RUN_PART_B = true;

} 