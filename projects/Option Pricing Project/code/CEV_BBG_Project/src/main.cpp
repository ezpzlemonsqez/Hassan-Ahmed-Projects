#include "config.hpp"
#include "cev.hpp"
#include "utils.hpp"
#include "bbg.hpp"

#include <iostream>
#include <iomanip>
#include <string>

int main() {
    try {
        if (cfg::RUN_PART_A) {
            const double sigma_eff = sigma_eff_for_cev(cfg::S0);
            const double bs_put    = bs_put_price(cfg::S0, cfg::K0, cfg::T0, cfg::r0, sigma_eff);

            auto mc = price_put_by_mc(cfg::S0, cfg::r0, cfg::K0, cfg::T0, cfg::n_steps, cfg::n_paths, 0, cfg::CONF_LEVEL, true);
            auto g  = greeks_by_mc(cfg::S0, cfg::r0, cfg::K0, cfg::T0, cfg::n_steps, cfg::n_paths, 0, cfg::CONF_LEVEL);

            int P_plot = 100;
            int M_plot = 300;
            auto surf = crank_nicolson_put_surface(cfg::K0, cfg::T0, cfg::r0, P_plot, M_plot, cfg::S_MAX, cfg::RANNACHER_STEPS, true);
            auto delta_grid = cn_delta_t0_from_V0(surf.V0, surf.S);

            double cn_price = lerp1d(surf.S, surf.V0, cfg::K0);
            double cn_delta = lerp1d(surf.S, delta_grid, cfg::K0);

            std::cout << "===== BASELINES =====\n";
            std::cout << "BS put price : " << std::fixed << std::setprecision(10) << bs_put << "\n";
            std::cout << "BS sigma_eff : " << std::fixed << std::setprecision(10) << sigma_eff << "\n\n";

            std::cout << "===== MONTE CARLO =====\n";
            std::cout << "MC put price : " << std::fixed << std::setprecision(10) << mc.price << "\n";
            std::cout << "MC put 95% CI: [ " << std::fixed << std::setprecision(10) << mc.ci_lo
                      << " , " << mc.ci_hi << " ]\n";
            std::cout << "MC delta     : " << std::fixed << std::setprecision(10) << g.delta << "\n";
            std::cout << "MC delta 95% : [ " << std::fixed << std::setprecision(10) << g.delta_ci_lo
                      << " , " << g.delta_ci_hi << " ]\n";
            std::cout << "MC gamma     : " << std::fixed << std::setprecision(10) << g.gamma << "\n";
            std::cout << "MC vega      : " << std::fixed << std::setprecision(10) << g.vega << "\n";
            std::cout << "MC theta     : " << std::fixed << std::setprecision(10) << g.theta << "\n";
            std::cout << "MC rho       : " << std::fixed << std::setprecision(10) << g.rho << "\n\n";

            std::cout << "===== CRANK–NICOLSON =====\n";
            std::cout << "CN put price : " << std::fixed << std::setprecision(10) << cn_price
                      << " (P=" << P_plot << ", M=" << M_plot << ", S_max=" << cfg::S_MAX << ")\n";
            std::cout << "CN delta     : " << std::fixed << std::setprecision(10) << cn_delta << "\n\n";

            std::cout << "[INFO] Running final comparison (may be heavy with default reference sizes)...\n";
            print_final_price_comparison(bs_put);
            write_partA_csvs(".");
        }

        if (cfg::RUN_PART_B) {
            run_bloomberg_analysis(cfg::SNAPSHOT_PATH, cfg::CHAIN_PATH, cfg::OUT_PREFIX);
        }

        std::cout << "\nDone.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}