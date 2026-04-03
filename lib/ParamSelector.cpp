#include "ParamSelector.h"

#include "ADMMOptimizer.h"
#include "Logger.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>

namespace trspv {

namespace {

using Clock = std::chrono::steady_clock;

std::vector<double> make_log_grid(double min_value, double max_value, int count) {
    std::vector<double> grid(static_cast<size_t>(count));
    if (count <= 1) {
        grid[0] = min_value;
        return grid;
    }

    for (int i = 0; i < count; ++i) {
        const double f = static_cast<double>(i) / static_cast<double>(count - 1);
        grid[static_cast<size_t>(i)] = min_value * std::pow(max_value / min_value, f);
    }
    return grid;
}

}  // namespace

ParamSelector::ParamSelector(const Eigen::MatrixXcd& A,
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

    const auto compute_score = [&](const Eigen::VectorXcd& x, double lam1, double lamt, double lamb) {
        const double residual = (A_ * x - b_).squaredNorm();

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
                        if (!w.empty() && static_cast<unsigned>(j) < static_cast<unsigned>(w.size())) {
                            wsum += w[static_cast<size_t>(j)];
                            ++cnt;
                        }
                    }
                }
                const double wavg = (cnt > 0 ? wsum / cnt : 1.0);
                groupL1 += wavg * std::sqrt(n2);
            }
        }

        const Eigen::VectorXcd Dx = D_ * x;
        const Eigen::Index tauRows = std::clamp<Eigen::Index>(
            static_cast<Eigen::Index>(rowsTau), 0, Dx.size());
        const Eigen::Index betaRows = Dx.size() - tauRows;
        const double tvTau = (tauRows > 0 ? Dx.head(tauRows).cwiseAbs().sum() : 0.0);
        const double tvBeta = (betaRows > 0 ? Dx.tail(betaRows).cwiseAbs().sum() : 0.0);

        return 0.5 * residual + lam1 * groupL1 + lamt * tvTau + lamb * tvBeta;
    };

    const auto safe_pos = [](double x, double eps = 1e-12) {
        return (x > eps ? x : eps);
    };

    const std::vector<double> grid1 = make_log_grid(
        safe_pos(psc_.lambda1_min), safe_pos(psc_.lambda1_max), psc_.num_lambda1);
    const std::vector<double> gridTau = make_log_grid(
        safe_pos(psc_.lambdat_min), safe_pos(psc_.lambdat_max), psc_.num_lambdat);
    const std::vector<double> gridBeta = make_log_grid(
        safe_pos(psc_.lambdab_min), safe_pos(psc_.lambdab_max), psc_.num_lambdab);

    std::vector<double> scores1(grid1.size(), std::numeric_limits<double>::infinity());
    std::vector<double> scoresTau(gridTau.size(), std::numeric_limits<double>::infinity());
    std::vector<double> scoresBeta(gridBeta.size(), std::numeric_limits<double>::infinity());
    std::vector<int> iters1(grid1.size(), 0);
    std::vector<int> itersTau(gridTau.size(), 0);
    std::vector<int> itersBeta(gridBeta.size(), 0);

    const auto start_total = Clock::now();

    double lambda1 = 0.5 * (psc_.lambda1_min + psc_.lambda1_max);
    double tau = 0.5 * (psc_.lambdat_min + psc_.lambdat_max);
    double beta = 0.5 * (psc_.lambdab_min + psc_.lambdab_max);

    auto scan_axis = [&](const std::vector<double>& grid,
                         std::vector<double>& scores,
                         std::vector<int>& iterations,
                         const char* axis_name,
                         auto apply_candidate) {
        Logger::info("ParamSelector: scanning {} ({} points)", axis_name, grid.size());
        auto axis_start = Clock::now();

        ADMMOptimizer solver(A_, b_, D_, scanCfg);
        Eigen::VectorXcd x_seed = Eigen::VectorXcd::Zero(A_.cols());
        Eigen::VectorXcd z1_seed = Eigen::VectorXcd::Zero(A_.cols());
        Eigen::VectorXcd z2_seed = Eigen::VectorXcd::Zero(D_.rows());
        Eigen::VectorXcd u1_seed = Eigen::VectorXcd::Zero(A_.cols());
        Eigen::VectorXcd u2_seed = Eigen::VectorXcd::Zero(D_.rows());

        for (size_t i = 0; i < grid.size(); ++i) {
            ADMMConfig cfg = scanCfg;
            apply_candidate(cfg, grid[i]);
            solver.updateParams(cfg);
            auto result = solver.solveWarmStart(x_seed, z1_seed, z2_seed, u1_seed, u2_seed);
            x_seed = std::get<0>(result);
            z1_seed = std::get<1>(result);
            z2_seed = std::get<2>(result);
            u1_seed = std::get<3>(result);
            u2_seed = std::get<4>(result);

            const double score = compute_score(x_seed, cfg.lambda1, cfg.lambda_tv_tau, cfg.lambda_tv_beta);
            scores[i] = score;
            iterations[i] = solver.last_iterations();
            Logger::info("ParamSelector: {}[{}/{}]={:.12g} score={:.12g} iters={}",
                         axis_name,
                         i + 1,
                         grid.size(),
                         grid[i],
                         score,
                         iterations[i]);
        }

        const auto axis_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - axis_start).count();
        Logger::info("ParamSelector: completed {} scan in {} ms", axis_name, axis_ms);
    };

    scan_axis(grid1, scores1, iters1, "lambda1", [&](ADMMConfig& cfg, double cand) {
        cfg.lambda1 = cand;
        cfg.lambda_tv_tau = tau;
        cfg.lambda_tv_beta = beta;
    });
    {
        size_t best_i = 0;
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < scores1.size(); ++i) {
            const double score = std::isfinite(scores1[i]) ? scores1[i] : 1e300;
            if (score < best) {
                best = score;
                best_i = i;
            }
        }
        lambda1 = grid1[best_i];
    }

    scan_axis(gridTau, scoresTau, itersTau, "lambda_tv_tau", [&](ADMMConfig& cfg, double cand) {
        cfg.lambda1 = lambda1;
        cfg.lambda_tv_tau = cand;
        cfg.lambda_tv_beta = beta;
    });
    {
        size_t best_i = 0;
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < scoresTau.size(); ++i) {
            const double score = std::isfinite(scoresTau[i]) ? scoresTau[i] : 1e300;
            if (score < best) {
                best = score;
                best_i = i;
            }
        }
        tau = gridTau[best_i];
    }

    scan_axis(gridBeta, scoresBeta, itersBeta, "lambda_tv_beta", [&](ADMMConfig& cfg, double cand) {
        cfg.lambda1 = lambda1;
        cfg.lambda_tv_tau = tau;
        cfg.lambda_tv_beta = cand;
    });
    {
        size_t best_i = 0;
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < scoresBeta.size(); ++i) {
            const double score = std::isfinite(scoresBeta[i]) ? scoresBeta[i] : 1e300;
            if (score < best) {
                best = score;
                best_i = i;
            }
        }
        beta = gridBeta[best_i];
    }

    ADMMConfig out = baseCfg_;
    out.lambda1 = lambda1;
    out.lambda_tv_tau = tau;
    out.lambda_tv_beta = beta;

    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_total).count();

    namespace fs = std::filesystem;
    fs::create_directories(psc_.outputDir);
    nlohmann::json report;
    report["selected"] = {
        {"lambda1", lambda1},
        {"lambda_tv_tau", tau},
        {"lambda_tv_beta", beta}
    };
    report["timings_ms"] = {
        {"total", total_ms}
    };

    auto add_axis = [&](const char* name,
                        const std::vector<double>& grid,
                        const std::vector<double>& scores,
                        const std::vector<int>& iterations) {
        auto& items = report["axes"][name] = nlohmann::json::array();
        for (size_t i = 0; i < grid.size(); ++i) {
            items.push_back({
                {"value", grid[i]},
                {"score", scores[i]},
                {"iterations", iterations[i]}
            });
        }
    };

    add_axis("lambda1", grid1, scores1, iters1);
    add_axis("lambda_tv_tau", gridTau, scoresTau, itersTau);
    add_axis("lambda_tv_beta", gridBeta, scoresBeta, itersBeta);

    const fs::path report_path = fs::path(psc_.outputDir) / "param_selection_report.json";
    std::ofstream ofs(report_path);
    ofs << report.dump(2) << '\n';

    Logger::info(
        "ParamSelector: final (lambda1, lambda_tv_tau, lambda_tv_beta) = ({:.12g}, {:.12g}, {:.12g})",
        lambda1,
        tau,
        beta);
    Logger::info("ParamSelector: report written to {}", report_path.string());

    return out;
}

}  // namespace trspv
