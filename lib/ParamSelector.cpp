// ParamSelector.cpp
#define FMT_HEADER_ONLY

#include "ParamSelector.h"
#include "ADMMOptimizer.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#if defined(FI_SPVM_HAS_OPENMP)
#include <omp.h>
#endif

namespace trspv {

ParamSelector::ParamSelector(
    const Eigen::MatrixXcd& A,
    const Eigen::VectorXcd& b,
    const Eigen::SparseMatrix<double>& D,
    const ADMMConfig& baseCfg,
    const ParamSelectionConfig& psc)
    : A_(A), b_(b), D_(D), baseCfg_(baseCfg), psc_(psc) {}

ADMMConfig ParamSelector::select() const {
    ADMMConfig scanCfg = baseCfg_;
    scanCfg.max_iters = psc_.scan_max_iters;
    scanCfg.tol_primal = psc_.scan_tol;
    scanCfg.tol_dual = psc_.scan_tol;

    const int Nt = std::max(1, baseCfg_.Nt);
    const int Nb = std::max(1, baseCfg_.Nb);
    const int stride = (baseCfg_.gamma_stride > 0 ? baseCfg_.gamma_stride : Nt);
    const int col_offset = (A_.cols() == Nt * Nb ? 0 : 1);

    const int rowsTau = Nb * std::max(0, Nt - 1);
    const std::vector<double>& w = baseCfg_.l1_weights;
    const int Gt = std::max(1, baseCfg_.group_size_tau);
    const int Gb = std::max(1, baseCfg_.group_size_beta);

    auto safe_pos = [](double x, double eps = 1e-12) { return (x > eps ? x : eps); };

    double lambda1 = 0.5 * (psc_.lambda1_min + psc_.lambda1_max);
    double tau = 0.5 * (psc_.lambdat_min + psc_.lambdat_max);
    double beta = 0.5 * (psc_.lambdab_min + psc_.lambdab_max);

    auto compute_score = [&](const Eigen::VectorXcd& x, double lam1, double lamt, double lamb) {
        const double R2 = (A_ * x - b_).squaredNorm();

        double groupL1 = 0.0;
        for (int bStart = 0; bStart < Nb; bStart += Gb) {
            for (int tStart = 0; tStart < Nt; tStart += Gt) {
                const int bSpan = std::min(Gb, Nb - bStart);
                const int tSpan = std::min(Gt, Nt - tStart);
                double n2 = 0.0;
                double wsum = 0.0;
                int cnt = 0;
                for (int gb = 0; gb < bSpan; ++gb) {
                    for (int gt = 0; gt < tSpan; ++gt) {
                        const int j = col_offset + (bStart + gb) * stride + (tStart + gt);
                        n2 += std::norm(x[j]);
                        if (!w.empty() && (unsigned)j < (unsigned)w.size()) {
                            wsum += w[j];
                            ++cnt;
                        }
                    }
                }
                const double wavg = (cnt > 0 ? wsum / cnt : 1.0);
                groupL1 += wavg * std::sqrt(n2);
            }
        }

        const Eigen::VectorXcd Dx = D_ * x;
        const double tvTau = Dx.head(rowsTau).cwiseAbs().sum();
        const double tvBeta = Dx.tail(Dx.size() - rowsTau).cwiseAbs().sum();

        return 0.5 * R2 + lam1 * groupL1 + lamt * tvTau + lamb * tvBeta;
    };

    const int nL1 = psc_.num_lambda1;
    const int nLt = psc_.num_lambdat;
    const int nLb = psc_.num_lambdab;

    std::vector<double> grid1(nL1), gridTau(nLt), gridBeta(nLb);
    std::vector<double> scores1(nL1), scoresTau(nLt), scoresBeta(nLb);

    const double l1min = safe_pos(psc_.lambda1_min);
    const double l1max = safe_pos(psc_.lambda1_max);
    const double ltmin = safe_pos(psc_.lambdat_min);
    const double ltmax = safe_pos(psc_.lambdat_max);
    const double lbmin = safe_pos(psc_.lambdab_min);
    const double lbmax = safe_pos(psc_.lambdab_max);

    for (int i = 0; i < nL1; ++i) {
        double f = (nL1 > 1) ? double(i) / (nL1 - 1) : 0.0;
        grid1[i] = l1min * std::pow(l1max / l1min, f);
    }
    for (int i = 0; i < nLt; ++i) {
        double f = (nLt > 1) ? double(i) / (nLt - 1) : 0.0;
        gridTau[i] = ltmin * std::pow(ltmax / ltmin, f);
    }
    for (int i = 0; i < nLb; ++i) {
        double f = (nLb > 1) ? double(i) / (nLb - 1) : 0.0;
        gridBeta[i] = lbmin * std::pow(lbmax / lbmin, f);
    }

    std::cout << "[ParamSelector] Scanning lambda1 (" << nL1 << " points) in parallel\n";
#if defined(FI_SPVM_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < nL1; ++i) {
        const double cand = grid1[i];
        ADMMConfig cfg = scanCfg;
        cfg.lambda1 = cand;
        cfg.lambda_tv_tau = tau;
        cfg.lambda_tv_beta = beta;

        ADMMOptimizer solver(A_, b_, D_, cfg);
        const auto x = solver.solve();
        scores1[i] = compute_score(x, cand, tau, beta);

#if defined(FI_SPVM_HAS_OPENMP)
#pragma omp critical
#endif
        {
            std::cout << "  [lambda1 " << std::setw(3) << (i + 1) << "/" << nL1
                      << "]=" << cand << " score=" << scores1[i] << "\n";
        }
    }

    {
        int best_i = 0;
        double best = std::numeric_limits<double>::infinity();
        for (int i = 0; i < nL1; ++i) {
            double s = scores1[i];
            if (!std::isfinite(s)) s = 1e300;
            if (s < best) {
                best = s;
                best_i = i;
            }
        }
        lambda1 = grid1[best_i];
        std::cout << "[ParamSelector] Best lambda1 = " << lambda1
                  << " (idx=" << best_i << ", score=" << best << ")\n\n";
    }

    std::cout << "[ParamSelector] Scanning tau (" << nLt << " points) in parallel\n";
#if defined(FI_SPVM_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < nLt; ++i) {
        const double cand = gridTau[i];
        ADMMConfig cfg = scanCfg;
        cfg.lambda1 = lambda1;
        cfg.lambda_tv_tau = cand;
        cfg.lambda_tv_beta = beta;

        ADMMOptimizer solver(A_, b_, D_, cfg);
        const auto x = solver.solve();
        scoresTau[i] = compute_score(x, lambda1, cand, beta);

#if defined(FI_SPVM_HAS_OPENMP)
#pragma omp critical
#endif
        {
            std::cout << "  [tau    " << std::setw(3) << (i + 1) << "/" << nLt
                      << "]=" << cand << " score=" << scoresTau[i] << "\n";
        }
    }

    {
        int best_i = 0;
        double best = std::numeric_limits<double>::infinity();
        for (int i = 0; i < nLt; ++i) {
            double s = scoresTau[i];
            if (!std::isfinite(s)) s = 1e300;
            if (s < best) {
                best = s;
                best_i = i;
            }
        }
        tau = gridTau[best_i];
        std::cout << "[ParamSelector] Best tau = " << tau
                  << " (idx=" << best_i << ", score=" << best << ")\n\n";
    }

    std::cout << "[ParamSelector] Scanning beta (" << nLb << " points) in parallel\n";
#if defined(FI_SPVM_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < nLb; ++i) {
        const double cand = gridBeta[i];
        ADMMConfig cfg = scanCfg;
        cfg.lambda1 = lambda1;
        cfg.lambda_tv_tau = tau;
        cfg.lambda_tv_beta = cand;

        ADMMOptimizer solver(A_, b_, D_, cfg);
        const auto x = solver.solve();
        scoresBeta[i] = compute_score(x, lambda1, tau, cand);

#if defined(FI_SPVM_HAS_OPENMP)
#pragma omp critical
#endif
        {
            std::cout << "  [beta   " << std::setw(3) << (i + 1) << "/" << nLb
                      << "]=" << cand << " score=" << scoresBeta[i] << "\n";
        }
    }

    {
        int best_i = 0;
        double best = std::numeric_limits<double>::infinity();
        for (int i = 0; i < nLb; ++i) {
            double s = scoresBeta[i];
            if (!std::isfinite(s)) s = 1e300;
            if (s < best) {
                best = s;
                best_i = i;
            }
        }
        beta = gridBeta[best_i];
        std::cout << "[ParamSelector] Best beta = " << beta
                  << " (idx=" << best_i << ", score=" << best << ")\n\n";
    }

    ADMMConfig out = baseCfg_;
    out.lambda1 = lambda1;
    out.lambda_tv_tau = tau;
    out.lambda_tv_beta = beta;

    std::cout << "[ParamSelector] Coordinate descent done. Final (lambda1,tau,beta) = ("
              << lambda1 << ", " << tau << ", " << beta << ")\n";

    return out;
}

}  // namespace trspv
