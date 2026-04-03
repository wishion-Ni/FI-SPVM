#include "SpectrumCompletion.h"
#include "Logger.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace trspv {

static void computeDifferences(const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    Eigen::VectorXd& d) {
    const int n = x.size();
    d.resize(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        d[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }
}

static void computePCHIP(const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    Eigen::VectorXd& m) {
    const int n = x.size();
    Eigen::VectorXd d;
    computeDifferences(x, y, d);
    m.resize(n);

    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for (int i = 1; i < n - 1; ++i) {
        if (d[i - 1] * d[i] > 0) {
            const double w1 = 2 * (x[i] - x[i - 1]) + (x[i + 1] - x[i]);
            const double w2 = (x[i] - x[i - 1]) + 2 * (x[i + 1] - x[i]);
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
        } else {
            m[i] = 0.0;
        }
    }
}

static void computeAkima(const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    Eigen::VectorXd& m) {
    const int n = x.size();
    Eigen::VectorXd d;
    computeDifferences(x, y, d);

    Eigen::VectorXd dd(n + 3);
    dd.segment(2, n - 1) = d;
    dd[1] = d[0];
    dd[0] = d[1];
    dd[n + 1] = d[n - 2];
    dd[n + 2] = d[n - 3];

    m.resize(n);
    for (int i = 0; i < n; ++i) {
        const double w1 = std::abs(dd[i + 3] - dd[i + 2]);
        const double w2 = std::abs(dd[i + 1] - dd[i]);
        if (w1 + w2 > 0) {
            m[i] = (w1 * dd[i + 1] + w2 * dd[i + 2]) / (w1 + w2);
        } else {
            m[i] = (dd[i + 2] + dd[i + 1]) * 0.5;
        }
    }
}

static double evalHermite(double xi,
    double x0, double x1,
    double y0, double y1,
    double m0, double m1) {
    const double dx = x1 - x0;
    const double t = (xi - x0) / dx;
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double h00 = 2 * t3 - 3 * t2 + 1;
    const double h10 = t3 - 2 * t2 + t;
    const double h01 = -2 * t3 + 3 * t2;
    const double h11 = t3 - t2;
    return h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1;
}

SpectrumData SpectrumCompletion::complete(const SpectrumData& data,
    const SpectrumCompletionConfig& config) {
    if (!config.interpolate || config.method == CompletionMethod::None) {
        Logger::info("SpectrumCompletion: No interpolation applied");
        return data;
    }

    const auto t_start = std::chrono::high_resolution_clock::now();

    const int N = static_cast<int>(data.freq.size());
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return data.freq[a] < data.freq[b];
    });

    std::vector<double> freq_sorted(N), weights_sorted(N);
    Eigen::VectorXd yr(N), yi(N);
    for (int i = 0; i < N; ++i) {
        const size_t k = idx[i];
        freq_sorted[i] = data.freq[k];
        yr[i] = data.values[k].real();
        yi[i] = data.values[k].imag();
        if (!data.weights.empty() && data.weights.size() == static_cast<size_t>(N)) {
            weights_sorted[i] = data.weights[k];
        } else {
            weights_sorted[i] = 1.0;
        }
    }
    const Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(freq_sorted.data(), N);

    Eigen::VectorXd mr(N), mi(N);
    if (config.method == CompletionMethod::PCHIP) {
        computePCHIP(x, yr, mr);
        computePCHIP(x, yi, mi);
    } else {
        computeAkima(x, yr, mr);
        computeAkima(x, yi, mi);
    }

    const int M = config.num_points;
    const double fmin = x.minCoeff();
    const double fmax = x.maxCoeff();
    Eigen::VectorXd xi_all(M);
    if (config.log_space) {
        const Eigen::VectorXd logv = Eigen::VectorXd::LinSpaced(M, std::log10(fmin), std::log10(fmax));
        for (int i = 0; i < M; ++i) {
            xi_all[i] = std::pow(10.0, logv[i]);
        }
    } else {
        xi_all = Eigen::VectorXd::LinSpaced(M, fmin, fmax);
    }

    const double tol = 1e-8;
    std::vector<double> xi_sel;
    xi_sel.reserve(M);
    for (double f_orig : freq_sorted) {
        xi_sel.push_back(f_orig);
    }
    for (int i = 0; i < M && static_cast<int>(xi_sel.size()) < M; ++i) {
        const double xv = xi_all[i];
        bool is_orig = false;
        for (double f_orig : freq_sorted) {
            if (std::abs(xv - f_orig) < tol) {
                is_orig = true;
                break;
            }
        }
        if (!is_orig) {
            xi_sel.push_back(xv);
        }
    }
    std::sort(xi_sel.begin(), xi_sel.end());

    SpectrumData result;
    result.freq.reserve(M);
    result.values.reserve(M);
    result.weights.reserve(M);

    for (int i = 0; i < M; ++i) {
        const double xv = xi_sel[i];
        const int idx_seg = std::clamp(int((std::upper_bound(x.data(), x.data() + N, xv) - x.data())) - 1, 0, N - 2);
        const double vr = evalHermite(xv, x[idx_seg], x[idx_seg + 1], yr[idx_seg], yr[idx_seg + 1], mr[idx_seg], mr[idx_seg + 1]);
        const double vi = evalHermite(xv, x[idx_seg], x[idx_seg + 1], yi[idx_seg], yi[idx_seg + 1], mi[idx_seg], mi[idx_seg + 1]);
        result.freq.push_back(xv);
        result.values.emplace_back(vr, vi);

        double w = config.weight;
        for (int j = 0; j < N; ++j) {
            if (std::abs(xv - freq_sorted[j]) < tol) {
                w = weights_sorted[j];
                break;
            }
        }
        result.weights.push_back(w);
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
    Logger::info("SpectrumCompletion: Interpolation applied with {} points in {} ms", M, elapsed.count());

    return result;
}

} // namespace trspv
