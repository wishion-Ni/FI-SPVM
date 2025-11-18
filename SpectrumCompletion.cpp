#include "SpectrumCompletion.h"
#include "Logger.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>

namespace trspv {

    // PCHIP 及 Akima 均需用到差分
    static void computeDifferences(const Eigen::VectorXd& x,
        const Eigen::VectorXd& y,
        Eigen::VectorXd& d) {
        int n = x.size();
        d.resize(n - 1);
        for (int i = 0; i < n - 1; ++i) {
            d[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        }
    }

    // 计算 PCHIP 斜率 (Fritsch-Carlson 方法)
    static void computePCHIP(const Eigen::VectorXd& x,
        const Eigen::VectorXd& y,
        Eigen::VectorXd& m) {
        int n = x.size();
        Eigen::VectorXd d;
        computeDifferences(x, y, d);
        m.resize(n);
        // 边界
        m[0] = d[0];
        m[n - 1] = d[n - 2];
        for (int i = 1; i < n - 1; ++i) {
            if (d[i - 1] * d[i] > 0) {
                double w1 = 2 * (x[i] - x[i - 1]) + (x[i + 1] - x[i]);
                double w2 = (x[i] - x[i - 1]) + 2 * (x[i + 1] - x[i]);
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
            }
            else {
                m[i] = 0.0;
            }
        }
    }

    // 计算 Akima 斜率
    static void computeAkima(const Eigen::VectorXd& x,
        const Eigen::VectorXd& y,
        Eigen::VectorXd& m) {
        int n = x.size();
        Eigen::VectorXd d;
        computeDifferences(x, y, d);
        // 双端扩展
        Eigen::VectorXd dd(n + 3);
        dd.segment(2, n - 1) = d;
        // 首末 2 个按镜像填充
        dd[1] = d[0]; dd[0] = d[1];
        dd[n + 1] = d[n - 2]; dd[n + 2] = d[n - 3];

        m.resize(n);
        for (int i = 0; i < n; ++i) {
            double w1 = std::abs(dd[i + 3] - dd[i + 2]);
            double w2 = std::abs(dd[i + 1] - dd[i]);
            if (w1 + w2 > 0) {
                m[i] = (w1 * dd[i + 1] + w2 * dd[i + 2]) / (w1 + w2);
            }
            else {
                m[i] = (dd[i + 2] + dd[i + 1]) * 0.5;
            }
        }
    }

    // 评估 Hermite 多项式
    static double evalHermite(double xi,
        double x0, double x1,
        double y0, double y1,
        double m0, double m1) {
        double dx = x1 - x0;
        double t = (xi - x0) / dx;
        double t2 = t * t;
        double t3 = t2 * t;
        double h00 = 2 * t3 - 3 * t2 + 1;
        double h10 = t3 - 2 * t2 + t;
        double h01 = -2 * t3 + 3 * t2;
        double h11 = t3 - t2;
        return h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1;
    }

    SpectrumData SpectrumCompletion::complete(const SpectrumData& data,
        const SpectrumCompletionConfig& config) {
        if (!config.interpolate || config.method == CompletionMethod::None) {
            Logger::info("SpectrumCompletion: No interpolation applied");
            return data;
        }
        // Start timing
        auto t_start = std::chrono::high_resolution_clock::now();

        int N = static_cast<int>(data.freq.size());
        // Copy and sort input data by frequency
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            return data.freq[a] < data.freq[b];
            });
        std::vector<double> freq_sorted(N), weights_sorted(N);
        Eigen::VectorXd yr(N), yi(N);
        for (int i = 0; i < N; ++i) {
            size_t k = idx[i];
            freq_sorted[i] = data.freq[k];
            yr[i] = data.values[k].real();
            yi[i] = data.values[k].imag();
            if (!data.weights.empty() && data.weights.size() == static_cast<size_t>(N)) {
                weights_sorted[i] = data.weights[k];
            }
            else {
                weights_sorted[i] = 1.0;
            }
        }
        Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(freq_sorted.data(), N);

        // Compute slopes
        Eigen::VectorXd mr(N), mi(N);
        if (config.method == CompletionMethod::PCHIP) {
            computePCHIP(x, yr, mr);
            computePCHIP(x, yi, mi);
        }
        else {
            computeAkima(x, yr, mr);
            computeAkima(x, yi, mi);
        }

        // Generate full interpolation grid (log or linear)
        int M = config.num_points;
        double fmin = x.minCoeff(), fmax = x.maxCoeff();
        Eigen::VectorXd xi_all(M);
        if (config.log_space) {
            Eigen::VectorXd logv = Eigen::VectorXd::LinSpaced(M, std::log10(fmin), std::log10(fmax));
            for (int i = 0; i < M; ++i) xi_all[i] = std::pow(10.0, logv[i]);
        }
        else {
            xi_all = Eigen::VectorXd::LinSpaced(M, fmin, fmax);
        }

        // Select M points: include all original frequencies, then fill with grid points
        const double tol = 1e-8;
        std::vector<double> xi_sel;
        xi_sel.reserve(M);
        // 1) include originals
        for (double f_orig : freq_sorted) {
            xi_sel.push_back(f_orig);
        }
        // 2) add grid points not near originals until have M
        for (int i = 0; i < M && (int)xi_sel.size() < M; ++i) {
            double xv = xi_all[i];
            bool is_orig = false;
            for (double f_orig : freq_sorted) {
                if (std::abs(xv - f_orig) < tol) { is_orig = true; break; }
            }
            if (!is_orig) xi_sel.push_back(xv);
        }
        // 3) sort selection for monotonicity
        std::sort(xi_sel.begin(), xi_sel.end());

        SpectrumData result;
        result.freq.reserve(M);
        result.values.reserve(M);
        result.weights.reserve(M);

        // Interpolate and assign weights
        for (int i = 0; i < M; ++i) {
            double xv = xi_sel[i];
            int idx_seg = std::clamp(int((std::upper_bound(x.data(), x.data() + N, xv) - x.data())) - 1, 0, N - 2);
            double vr = evalHermite(xv, x[idx_seg], x[idx_seg + 1], yr[idx_seg], yr[idx_seg + 1], mr[idx_seg], mr[idx_seg + 1]);
            double vi = evalHermite(xv, x[idx_seg], x[idx_seg + 1], yi[idx_seg], yi[idx_seg + 1], mi[idx_seg], mi[idx_seg + 1]);
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

        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
        Logger::info("SpectrumCompletion: Interpolation applied with {} points in {} ms", M, elapsed.count());

        return result;
    }

} // namespace trspv
