#include "ComponentAnalysis.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>

namespace trspv {
namespace {

inline double robust_sigma(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::vector<double> u = v;
    std::nth_element(u.begin(), u.begin() + u.size() / 2, u.end());
    double med = u[u.size() / 2];
    for (auto& x : u) x = std::abs(x - med);
    std::nth_element(u.begin(), u.begin() + u.size() / 2, u.end());
    double mad = u[u.size() / 2];
    return 1.4826 * mad;
}

inline double Ebeta_negx(double x, double beta) {
    if (beta <= 0) beta = 1.0;
    if (beta == 1.0) return std::exp(-x);
    if (x <= 30.0) {
        const int Kmax = 200;
        double term = 1.0;
        double sum = term;
        for (int k = 1; k <= Kmax; ++k) {
            term *= (-x) / k;
            double gamma_ratio = std::tgamma(k + 1) / std::tgamma(beta * k + 1.0);
            double add = term * gamma_ratio;
            sum += add;
            if (std::abs(add) < 1e-16 * std::abs(sum)) break;
        }
        return sum;
    }
    double s = 0.0;
    for (int n = 1; n <= 4; ++n) {
        s -= std::pow(x, -n) / std::tgamma(1.0 - beta * n);
    }
    return s;
}

}  // namespace

std::vector<Component> extract_components(
    const Eigen::VectorXcd& x2d,
    const std::vector<double>& taus,
    const std::vector<double>& betas,
    double k_sigma,
    double alpha_of_max,
    int min_pixels,
    double merge_dlogtau,
    double merge_dbeta) {
    const int Nt = static_cast<int>(taus.size());
    const int Nb = static_cast<int>(betas.size());
    auto idx = [&](int it, int jb) { return it + jb * Nt; };

    std::vector<double> absv;
    absv.reserve(static_cast<size_t>(x2d.size()));
    double vmax = 0.0;
    for (int jb = 0; jb < Nb; ++jb) {
        for (int it = 0; it < Nt; ++it) {
            double a = std::abs(x2d[idx(it, jb)]);
            absv.push_back(a);
            vmax = std::max(vmax, a);
        }
    }
    double sig = robust_sigma(absv);
    double thr = std::max(k_sigma * sig, alpha_of_max * vmax);
    if (thr <= 0) thr = 0.1 * vmax;

    std::vector<char> mask(static_cast<size_t>(x2d.size()), 0),
        vis(static_cast<size_t>(x2d.size()), 0);
    for (int jb = 0; jb < Nb; ++jb) {
        for (int it = 0; it < Nt; ++it) {
            if (std::abs(x2d[idx(it, jb)]) >= thr) mask[static_cast<size_t>(idx(it, jb))] = 1;
        }
    }

    static const int d8[8][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1} };
    std::vector<Component> comps;

    for (int jb = 0; jb < Nb; ++jb) {
        for (int it = 0; it < Nt; ++it) {
            int start = idx(it, jb);
            if (!mask[static_cast<size_t>(start)] || vis[static_cast<size_t>(start)]) continue;

            std::vector<int> pixels;
            std::deque<int> dq;
            dq.push_back(start);
            vis[static_cast<size_t>(start)] = 1;
            while (!dq.empty()) {
                int id = dq.front();
                dq.pop_front();
                pixels.push_back(id);
                int it0 = id % Nt, jb0 = id / Nt;
                for (auto& d : d8) {
                    int it1 = it0 + d[0], jb1 = jb0 + d[1];
                    if (it1 < 0 || it1 >= Nt || jb1 < 0 || jb1 >= Nb) continue;
                    int id1 = idx(it1, jb1);
                    if (mask[static_cast<size_t>(id1)] && !vis[static_cast<size_t>(id1)]) {
                        vis[static_cast<size_t>(id1)] = 1;
                        dq.push_back(id1);
                    }
                }
            }
            if (static_cast<int>(pixels.size()) < min_pixels) continue;

            double wsum = 0.0, sum_logt = 0.0, sum_beta = 0.0;
            double amp_sum = 0.0, peak_abs = 0.0;
            int it_peak = it, jb_peak = jb;

            for (int id : pixels) {
                std::complex<double> v = x2d[id];
                double w = std::abs(v);
                int it0 = id % Nt, jb0 = id / Nt;
                wsum += w;
                sum_logt += w * std::log10(taus[static_cast<size_t>(it0)]);
                sum_beta += w * betas[static_cast<size_t>(jb0)];
                amp_sum += v.real();
                if (w > peak_abs) {
                    peak_abs = w;
                    it_peak = it0;
                    jb_peak = jb0;
                }
            }

            Component c{};
            c.tau = (wsum > 0 ? std::pow(10.0, sum_logt / wsum) : taus[static_cast<size_t>(it_peak)]);
            c.beta = (wsum > 0 ? sum_beta / wsum : betas[static_cast<size_t>(jb_peak)]);
            c.amp = amp_sum;
            c.prominence = peak_abs / std::max(1e-16, sig);
            c.it = it_peak;
            c.jb = jb_peak;
            comps.push_back(c);
        }
    }

    auto dlog = [](double t1, double t2) { return std::abs(std::log10(t1 / t2)); };
    std::sort(comps.begin(), comps.end(),
        [](const Component& a, const Component& b) { return std::abs(a.amp) > std::abs(b.amp); });
    std::vector<Component> merged;
    for (auto& c : comps) {
        bool keep = true;
        for (auto& m : merged) {
            if (dlog(c.tau, m.tau) < merge_dlogtau && std::abs(c.beta - m.beta) < merge_dbeta) {
                double A = std::abs(m.amp) + std::abs(c.amp);
                if (A == 0) {
                    keep = false;
                    break;
                }
                m.tau = std::pow(10.0,
                    (std::abs(m.amp) * std::log10(m.tau) + std::abs(c.amp) * std::log10(c.tau)) / A);
                m.beta = (std::abs(m.amp) * m.beta + std::abs(c.amp) * c.beta) / A;
                m.amp += c.amp;
                m.prominence = std::max(m.prominence, c.prominence);
                keep = false;
                break;
            }
        }
        if (keep) merged.push_back(c);
    }
    return merged;
}

double h_on(double t, double tau, double beta) {
    double x = std::pow(t / std::max(1e-300, tau), beta);
    return 1.0 - Ebeta_negx(x, beta);
}

double h_off(double t, double tau, double beta) {
    double x = std::pow(t / std::max(1e-300, tau), beta);
    return Ebeta_negx(x, beta);
}

}  // namespace trspv

