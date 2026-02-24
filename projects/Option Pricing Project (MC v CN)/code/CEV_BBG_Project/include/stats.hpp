#pragma once
#include <boost/math/distributions/normal.hpp>

namespace stats {

inline double norm_cdf(double x) {
    static const boost::math::normal_distribution<double> nd(0.0, 1.0);
    return boost::math::cdf(nd, x);
}

inline double norm_pdf(double x) {
    static const boost::math::normal_distribution<double> nd(0.0, 1.0);
    return boost::math::pdf(nd, x);
}

inline double norm_ppf(double p) {
    static const boost::math::normal_distribution<double> nd(0.0, 1.0);
    return boost::math::quantile(nd, p);
}

} 