#include "bbg.hpp"
#include "csv.hpp"
#include "bs.hpp"
#include "pde_bs.hpp"
#include "rng.hpp"
#include "stats.hpp"
#include "utils.hpp"
#include "config.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <limits>
#include <algorithm>

struct MCPriceCI { double price=0.0, ci_lo=0.0, ci_hi=0.0; };

static MCPriceCI mc_price_bs_exact_bbg(double S, double K, double T, double r, double sigma, const std::string& flag, double q,
                                       int n_paths, uint64_t seed, double conf) {
    MCPriceCI out;
    if (T <= 0.0) {
        double iv = (flag=="call")? std::max(S-K,0.0) : std::max(K-S,0.0);
        out.price = out.ci_lo = out.ci_hi = iv;
        return out;
    }
    n_paths = std::max(2, n_paths);
    if (n_paths % 2 == 1) ++n_paths;

    RNG rng(seed);
    int half = n_paths / 2;
    std::vector<double> Z(half);
    rng.normals01(Z);

    const double drift = (r - q - 0.5 * sigma * sigma) * T;
    const double vol = sigma * std::sqrt(T);

    std::vector<double> disc(n_paths);
    const double disc_r = std::exp(-r * T);

    for (int i = 0; i < half; ++i) {
        double z = Z[i];
        double st1 = S * std::exp(drift + vol * z);
        double st2 = S * std::exp(drift + vol * (-z));
        double payoff1 = (flag=="call")? std::max(st1 - K, 0.0) : std::max(K - st1, 0.0);
        double payoff2 = (flag=="call")? std::max(st2 - K, 0.0) : std::max(K - st2, 0.0);
        disc[i] = disc_r * payoff1;
        disc[i + half] = disc_r * payoff2;
    }

    out.price = mean(disc);
    double alpha = stats::norm_ppf((1.0 - conf) / 2.0);
    double halfw = (-alpha) * stdev_sample(disc) / std::sqrt((double)n_paths);
    out.ci_lo = out.price - halfw;
    out.ci_hi = out.price + halfw;
    return out;
}

struct EnrichedRow {
    std::string expiry;
    double ttm_years = std::numeric_limits<double>::quiet_NaN();
    std::string type;
    double spot=std::numeric_limits<double>::quiet_NaN();
    double strike=std::numeric_limits<double>::quiet_NaN();
    double rfr=std::numeric_limits<double>::quiet_NaN();
    double div_yield=std::numeric_limits<double>::quiet_NaN();
    double iv=std::numeric_limits<double>::quiet_NaN();
    double bid=std::numeric_limits<double>::quiet_NaN();
    double ask=std::numeric_limits<double>::quiet_NaN();
    double mid_mkt=std::numeric_limits<double>::quiet_NaN();

    double bs_price=std::numeric_limits<double>::quiet_NaN();
    double mc_price=std::numeric_limits<double>::quiet_NaN();
    double mc_ci_lo=std::numeric_limits<double>::quiet_NaN();
    double mc_ci_hi=std::numeric_limits<double>::quiet_NaN();
    double cn_price=std::numeric_limits<double>::quiet_NaN();

    double bs_vs_mkt=std::numeric_limits<double>::quiet_NaN();
    double mc_vs_mkt=std::numeric_limits<double>::quiet_NaN();
    double cn_vs_mkt=std::numeric_limits<double>::quiet_NaN();

    double delta=std::numeric_limits<double>::quiet_NaN();
    double gamma=std::numeric_limits<double>::quiet_NaN();
    double vega=std::numeric_limits<double>::quiet_NaN();
    double theta=std::numeric_limits<double>::quiet_NaN();
    double rho=std::numeric_limits<double>::quiet_NaN();

    double oi=std::numeric_limits<double>::quiet_NaN();
    double volume=std::numeric_limits<double>::quiet_NaN();
};

static void write_enriched_csv(const std::string& path, const std::vector<EnrichedRow>& rows) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write CSV: " + path);

    f << "expiry,ttm_years,type,spot,strike,rfr,div_yield,iv,bid,ask,mid_mkt,"
         "bs_price,mc_price,mc_ci_lo,mc_ci_hi,cn_price,"
         "bs_vs_mkt,mc_vs_mkt,cn_vs_mkt,delta,gamma,vega,theta,rho,oi,volume\n";

    auto w = [&](double x) {
        if (!std::isfinite(x)) return std::string("");
        std::ostringstream oss;
        oss << std::setprecision(12) << x;
        return oss.str();
    };

    for (const auto& r : rows) {
        f << r.expiry << ','
          << w(r.ttm_years) << ','
          << r.type << ','
          << w(r.spot) << ','
          << w(r.strike) << ','
          << w(r.rfr) << ','
          << w(r.div_yield) << ','
          << w(r.iv) << ','
          << w(r.bid) << ','
          << w(r.ask) << ','
          << w(r.mid_mkt) << ','
          << w(r.bs_price) << ','
          << w(r.mc_price) << ','
          << w(r.mc_ci_lo) << ','
          << w(r.mc_ci_hi) << ','
          << w(r.cn_price) << ','
          << w(r.bs_vs_mkt) << ','
          << w(r.mc_vs_mkt) << ','
          << w(r.cn_vs_mkt) << ','
          << w(r.delta) << ','
          << w(r.gamma) << ','
          << w(r.vega) << ','
          << w(r.theta) << ','
          << w(r.rho) << ','
          << w(r.oi) << ','
          << w(r.volume) << '\n';
    }
}

static void print_summary_table_bbg(const std::vector<EnrichedRow>& rows) {
    auto is_call = [](const EnrichedRow& r){ return r.type=="call"; };
    auto is_put  = [](const EnrichedRow& r){ return r.type=="put"; };

    size_t calls = std::count_if(rows.begin(), rows.end(), is_call);
    size_t puts  = std::count_if(rows.begin(), rows.end(), is_put);

    std::vector<double> strikes;
    std::vector<double> ivs;
    strikes.reserve(rows.size());
    ivs.reserve(rows.size());
    for (const auto& r : rows) {
        if (std::isfinite(r.strike)) strikes.push_back(r.strike);
        if (std::isfinite(r.iv)) ivs.push_back(r.iv);
    }

    auto minmax = [](const std::vector<double>& v){
        if (v.empty()) return std::pair<double,double>(NAN,NAN);
        auto mm = std::minmax_element(v.begin(), v.end());
        return std::pair<double,double>(*mm.first, *mm.second);
    };

    auto strike_mm = minmax(strikes);
    auto iv_mm = minmax(ivs);

    double q = rows.empty() ? NAN : rows[0].div_yield;

    auto collect = [&](auto getter){
        std::vector<double> v;
        v.reserve(rows.size());
        for (auto &r : rows) {
            double x = getter(r);
            if (std::isfinite(x)) v.push_back(x);
        }
        return v;
    };

    auto print_stats = [&](const std::string& label, const std::vector<double>& v) {
        if (v.empty()) return;
        auto mm = std::minmax_element(v.begin(), v.end());
        std::cout << "  " << std::left << std::setw(14) << label
                  << ": mean=" << std::right << std::setw(9) << std::fixed << std::setprecision(4) << mean(v)
                  << "  std=" << std::setw(8) << std::fixed << std::setprecision(4) << stdev_sample(v)
                  << "  [" << std::fixed << std::setprecision(2) << *mm.first
                  << ", " << std::fixed << std::setprecision(2) << *mm.second << "]\n";
    };

    auto print_mis = [&](const std::string& label, const std::vector<double>& v) {
        if (v.empty()) return;
        double m = mean(v);
        double mae = 0.0;
        double mx = 0.0;
        for (double x : v) {
            mae += std::abs(x);
            mx = std::max(mx, std::abs(x));
        }
        mae /= (double)v.size();
        std::cout << "  " << std::left << std::setw(10) << label
                  << ": mean=" << std::showpos << std::fixed << std::setprecision(4) << m << std::noshowpos
                  << "  MAE=" << std::fixed << std::setprecision(4) << mae
                  << "  MaxAbs=" << std::fixed << std::setprecision(4) << mx << "\n";
    };

    const std::string SEP(70, '=');
    std::cout << "\n" << SEP << "\n  SPX OPTIONS CHAIN — MODEL PRICING & GREEKS SUMMARY\n" << SEP << "\n";
    std::cout << "  Options priced     : " << rows.size() << "\n";
    std::cout << "  Calls / Puts       : " << calls << " / " << puts << "\n";
    std::cout << "  Strike range       : " << std::fixed << std::setprecision(0) << strike_mm.first << " – " << strike_mm.second << "\n";
    std::cout << "  IV range           : " << std::fixed << std::setprecision(1) << iv_mm.first*100.0 << "% – " << iv_mm.second*100.0
              << "% (mean " << std::fixed << std::setprecision(2) << mean(ivs)*100.0 << "%)\n";
    std::cout << "  Dividend yield (q) : " << std::fixed << std::setprecision(3) << q*100.0 << "%\n";

    std::cout << "\n  " << std::string(30,'-') << " Price Comparison " << std::string(22,'-') << "\n";
    print_stats("Market Mid", collect([](const EnrichedRow&r){return r.mid_mkt;}));
    print_stats("BS",         collect([](const EnrichedRow&r){return r.bs_price;}));
    print_stats("MC",         collect([](const EnrichedRow&r){return r.mc_price;}));
    print_stats("CN",         collect([](const EnrichedRow&r){return r.cn_price;}));

    std::cout << "\n  " << std::string(30,'-') << " Mispricing (model - mid) " << std::string(14,'-') << "\n";
    print_mis("BS−Mid", collect([](const EnrichedRow&r){return r.bs_vs_mkt;}));
    print_mis("MC−Mid", collect([](const EnrichedRow&r){return r.mc_vs_mkt;}));
    print_mis("CN−Mid", collect([](const EnrichedRow&r){return r.cn_vs_mkt;}));

    std::cout << SEP << "\n";
}

int run_bloomberg_analysis(const std::string& snapshot_path,
                           const std::string& chain_path,
                           const std::string& out_prefix) {
    std::cout << "\n" << std::string(70,'=') << "\n  BLOOMBERG OPTIONS — DATA ANALYSIS + ENRICHMENT\n" << std::string(70,'=') << "\n";

    CSVTable snap = read_csv(snapshot_path);
    CSVTable chain = read_csv(chain_path);

    std::cout << "  [OK]  snapshot: " << snap.rows.size() << " rows | cols: [";
    for (size_t i=0;i<snap.cols.size();++i){ std::cout<<snap.cols[i]<<(i+1<snap.cols.size()?", ":""); }
    std::cout << "]\n";

    std::cout << "  [OK]  clean chain: " << chain.rows.size() << " rows | cols: [";
    for (size_t i=0;i<chain.cols.size();++i){ std::cout<<chain.cols[i]<<(i+1<chain.cols.size()?", ":""); }
    std::cout << "]\n";

    if (snap.rows.empty()) throw std::runtime_error("snapshot.csv is empty");

    double spot=0.0, q=0.0;
    {
        auto &r = snap.rows[0];
        if (!to_double(r["spot"], spot)) throw std::runtime_error("snapshot missing/invalid spot");
        if (!to_double(r["dividend_yield"], q)) q = 0.0;
        if (q > 1.0) q /= 100.0;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  [INFO] Spot       : " << spot << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  [INFO] Div yield  : " << q*100.0 << "%\n";

    const size_t total = chain.rows.size();
    std::cout << "\n  Pricing " << total << " options — this may take a minute...\n";

    std::vector<EnrichedRow> out;
    out.reserve(total);

    for (size_t idx = 0; idx < total; ++idx) {
        if (idx % std::max<size_t>(1, total/10) == 0) {
            std::cout << "    " << idx << "/" << total << " options processed...\n";
        }
        auto &row = chain.rows[idx];

        std::string right = row.count("right") ? row["right"] : (row.count("type")? row["type"] : "");
        std::string flagU = upper_trim(right);
        std::string flag;
        if (flagU == "C" || flagU == "CALL") flag = "call";
        else if (flagU == "P" || flagU == "PUT") flag = "put";
        else {
            std::string t0 = upper_trim(row.count("type")? row["type"] : "");
            if (t0=="CALL") flag="call";
            else if (t0=="PUT") flag="put";
            else continue;
        }

        double T_=NAN, K_=NAN, r_=NAN, iv=NAN, bid=NAN, ask=NAN, mid=NAN, oi=NAN, vol=NAN;
        to_double(row.count("ttm_years")? row["ttm_years"] : row["T"], T_);
        to_double(row.count("strike")? row["strike"] : row["K"], K_);
        to_double(row.count("r")? row["r"] : (row.count("rfr")? row["rfr"] : ""), r_);
        to_double(row.count("implied_vol")? row["implied_vol"] : (row.count("iv")? row["iv"] : ""), iv);
        to_double(row.count("bid")? row["bid"] : "", bid);
        to_double(row.count("ask")? row["ask"] : "", ask);
        to_double(row.count("mid")? row["mid"] : "", mid);
        if (row.count("open_interest")) to_double(row["open_interest"], oi);
        else if (row.count("oi")) to_double(row["oi"], oi);
        if (row.count("volume")) to_double(row["volume"], vol);

        if (std::isfinite(r_) && r_ > 1.0) r_ /= 100.0;
        if (std::isfinite(iv) && iv > 3.0) iv /= 100.0;

        if (!(is_finite_pos(mid)) && std::isfinite(bid) && std::isfinite(ask)) mid = 0.5*(bid+ask);
        if (!(is_finite_pos(iv)) && is_finite_pos(mid)) iv = implied_vol_newton_bbg(spot, K_, T_, r_, mid, flag, q);
        if (!is_finite_pos(iv)) continue;

        double bs_p = bs_price_bbg(spot, K_, T_, r_, iv, flag, q);
        BSGreeks gk = bs_greeks_bbg(spot, K_, T_, r_, iv, flag, q);
        MCPriceCI mc = mc_price_bs_exact_bbg(spot, K_, T_, r_, iv, flag, q, cfg::BBG_MC_PATHS, 42, cfg::BBG_CONF_LEVEL);
        double cn_p = cn_price_bs_bbg(spot, K_, T_, r_, iv, flag, q, cfg::BBG_CN_M, cfg::BBG_CN_P, cfg::BBG_S_MAX_MULT);

        EnrichedRow er;
        er.expiry = row.count("expiry") ? row["expiry"] : "";
        er.ttm_years = T_;
        er.type = flag;
        er.spot = spot;
        er.strike = K_;
        er.rfr = r_;
        er.div_yield = q;
        er.iv = iv;
        er.bid = bid;
        er.ask = ask;
        er.mid_mkt = mid;
        er.bs_price = bs_p;
        er.mc_price = mc.price;
        er.mc_ci_lo = mc.ci_lo;
        er.mc_ci_hi = mc.ci_hi;
        er.cn_price = cn_p;
        er.bs_vs_mkt = bs_p - mid;
        er.mc_vs_mkt = mc.price - mid;
        er.cn_vs_mkt = cn_p - mid;
        er.delta = gk.delta;
        er.gamma = gk.gamma;
        er.vega = gk.vega;
        er.theta = gk.theta;
        er.rho = gk.rho;
        er.oi = oi;
        er.volume = vol;
        out.push_back(std::move(er));
    }

    std::cout << "    " << total << "/" << total << " options processed...\n";
    std::cout << "    " << out.size() << "/" << total << " options priced successfully.\n";

    std::string csv_out = out_prefix + std::string("_enriched_chain.csv");
    write_enriched_csv(csv_out, out);
    std::cout << "  Saved " << csv_out << "\n";

    print_summary_table_bbg(out);

    std::cout << "\n  (Plots omitted in C++ version; use the enriched CSV for plotting in Python/R if needed.)\n";
    std::cout << "\n  Bloomberg analysis complete.\n";

    return 0;
}