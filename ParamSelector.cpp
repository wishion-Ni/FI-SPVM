// ParamSelector.cpp
#define FMT_HEADER_ONLY

#include "ParamSelector.h"
#include "ADMMOptimizer.h"
#include <filesystem>
#include <fstream>
#include <cmath>
#include <limits>
#include <vector>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <vector>

namespace trspv {
    ParamSelector::ParamSelector(
        const Eigen::MatrixXcd& A,
        const Eigen::VectorXcd& b,
        const Eigen::SparseMatrix<double>& D,
        const ADMMConfig& baseCfg,
        const ParamSelectionConfig& psc
    ) : A_(A), b_(b), D_(D), baseCfg_(baseCfg), psc_(psc)
    {}



    ADMMConfig ParamSelector::select() const {
        // 1) Prepare loose-scan ADMM parameters
        ADMMConfig scanCfg = baseCfg_;
        scanCfg.max_iters = psc_.scan_max_iters;
        scanCfg.tol_primal = psc_.scan_tol;
        scanCfg.tol_dual = psc_.scan_tol;

        // 从 baseCfg_ 取网格与分组信息（你之前已在 ADMMConfig 里设置过）
        const int Nt = std::max(1, baseCfg_.Nt);
        const int Nb = std::max(1, baseCfg_.Nb);
        const int stride = (baseCfg_.gamma_stride > 0 ? baseCfg_.gamma_stride : Nt);

        // 常数列偏移：A 的列数可能 = 1 + Nt*Nb（含常数基）或 Nt*Nb（无）
        const int col_offset = (A_.cols() == Nt * Nb ? 0 : 1);

        // τ/β TV 的行分界
        const int rowsTau = Nb * std::max(0, Nt - 1);
        const int rowsBeta = Nt * std::max(0, Nb - 1);

        // L1 权重（与列顺序 β外、τ内一致）
        const std::vector<double>& w = baseCfg_.l1_weights;
        const int Gt = std::max(1, baseCfg_.group_size_tau);
        const int Gb = std::max(1, baseCfg_.group_size_beta);

        // 保护 logspace（避免 min=0）
        auto safe_pos = [](double x, double eps = 1e-12) { return (x > eps ? x : eps); };


        // 2) Initial guesses at midpoints
        double lambda1 = 0.5 * (psc_.lambda1_min + psc_.lambda1_max);
        double tau = 0.5 * (psc_.lambdat_min + psc_.lambdat_max);
        double beta = 0.5 * (psc_.lambdab_min + psc_.lambdab_max);

        auto compute_score = [&](const Eigen::VectorXcd& x,
            double lam1, double lamt, double lamb)
        {
            // 数据项（可用 0.5因子，也可不乘，选择不影响相对比较）
            const double R2 = (A_ * x - b_).squaredNorm();

            // 组-L1：按 (Gb×Gt) 小块，组均权 wavg，与 z1 完全一致
            double groupL1 = 0.0;
            for (int bStart = 0; bStart < Nb; bStart += Gb) {
                for (int tStart = 0; tStart < Nt; tStart += Gt) {
                    const int bSpan = std::min(Gb, Nb - bStart);
                    const int tSpan = std::min(Gt, Nt - tStart);
                    double n2 = 0.0, wsum = 0.0; int cnt = 0;
                    for (int gb = 0; gb < bSpan; ++gb)
                        for (int gt = 0; gt < tSpan; ++gt) {
                            const int j = col_offset + (bStart + gb) * stride + (tStart + gt);
                            n2 += std::norm(x[j]);
                            if (!w.empty() && (unsigned)j < (unsigned)w.size()) { wsum += w[j]; ++cnt; }
                        }
                    const double wavg = (cnt > 0 ? wsum / cnt : 1.0);
                    groupL1 += wavg * std::sqrt(n2);
                }
            }

            // 分向 TV：先算 Dx，再拆 τ/β 两块分别求 L1
            const Eigen::VectorXcd Dx = D_ * x;
            const double tvTau = Dx.head(rowsTau).cwiseAbs().sum();
            const double tvBeta = Dx.tail(Dx.size() - rowsTau).cwiseAbs().sum();

            // 目标值
            return 0.5 * R2 + lam1 * groupL1 + lamt * tvTau + lamb * tvBeta;
        };


        // 3) Build 1D log‐spaced grids
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

        // --- Parallel scan lambda1 ---
        std::cout << "[ParamSelector] Scanning lambda1 (" << nL1 << " points) in parallel\n";
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nL1; ++i) {
            double cand = grid1[i];
            ADMMConfig cfg = scanCfg;
            cfg.lambda1 = cand; cfg.lambda_tv_tau = tau; cfg.lambda_tv_beta = beta;
            ADMMOptimizer solver(A_, b_, D_, cfg);   // ★ 用 cfg 构造
            auto x = solver.solve();
            scores1[i] = compute_score(x, cand, tau, beta);
#pragma omp critical
            std::cout << "  [lambda1 " << std::setw(3) << (i + 1) << "/" << nL1
                << "]=" << cand << " score=" << scores1[i] << "\n";
        }
        // --- after scanning lambda1 ---
        {
            int best_i = 0;
            double best = std::numeric_limits<double>::infinity();
            for (int i = 0; i < nL1; ++i) {
                double s = scores1[i];
                if (!std::isfinite(s)) s = 1e300;      // 防 NaN/Inf
                if (s < best) { best = s; best_i = i; }
            }
            lambda1 = grid1[best_i];
            std::cout << "[ParamSelector] Best lambda1 = " << lambda1
                << " (idx=" << best_i << ", score=" << best << ")\n\n";
        }


        // --- Parallel scan tau ---
        std::cout << "[ParamSelector] Scanning tau (" << nLt << " points) in parallel\n";
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nLt; ++i) {
            double cand = gridTau[i];
            ADMMConfig cfg = scanCfg;
            cfg.lambda1 = lambda1; cfg.lambda_tv_tau = cand; cfg.lambda_tv_beta = beta;
            ADMMOptimizer solver(A_, b_, D_, cfg);   // ★
            auto x = solver.solve();
            scoresTau[i] = compute_score(x, lambda1, cand, beta);
#pragma omp critical
            std::cout << "  [tau    " << std::setw(3) << (i + 1) << "/" << nLt
                << "]=" << cand << " score=" << scoresTau[i] << "\n";
        }
        // --- after scanning tau ---
        {
            int best_i = 0;
            double best = std::numeric_limits<double>::infinity();
            for (int i = 0; i < nLt; ++i) {
                double s = scoresTau[i];
                if (!std::isfinite(s)) s = 1e300;
                if (s < best) { best = s; best_i = i; }
            }
            tau = gridTau[best_i];
            std::cout << "[ParamSelector] Best tau = " << tau
                << " (idx=" << best_i << ", score=" << best << ")\n\n";
        }


        // --- Parallel scan beta ---
        std::cout << "[ParamSelector] Scanning beta (" << nLb << " points) in parallel\n";
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nLb; ++i) {
            double cand = gridBeta[i];
            ADMMConfig cfg = scanCfg;
            cfg.lambda1 = lambda1; cfg.lambda_tv_tau = tau; cfg.lambda_tv_beta = cand;
            ADMMOptimizer solver(A_, b_, D_, cfg);   // ★
            auto x = solver.solve();
            scoresBeta[i] = compute_score(x, lambda1, tau, cand);
#pragma omp critical
            std::cout << "  [beta   " << std::setw(3) << (i + 1) << "/" << nLb
                << "]=" << cand << " score=" << scoresBeta[i] << "\n";
        }
        // --- after scanning beta ---
        {
            int best_i = 0;
            double best = std::numeric_limits<double>::infinity();
            for (int i = 0; i < nLb; ++i) {
                double s = scoresBeta[i];
                if (!std::isfinite(s)) s = 1e300;
                if (s < best) { best = s; best_i = i; }
            }
            beta = gridBeta[best_i];
            std::cout << "[ParamSelector] Best beta = " << beta
                << " (idx=" << best_i << ", score=" << best << ")\n\n";
        }


        // 6) Return final strict config
        ADMMConfig out = baseCfg_;
        out.lambda1 = lambda1;
        out.lambda_tv_tau = tau;
        out.lambda_tv_beta = beta;
        std::cout << "[ParamSelector] Coordinate descent done. Final (lambda1,tau,beta) = ("
            << lambda1 << ", " << tau << ", " << beta << ")\n";

        return out;
    }


} // namespace trspv