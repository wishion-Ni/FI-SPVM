#include "Solver2D.h"

#include <sstream>

#include "ADMMOptimizer.h"
#include "ParamSelector.h"

namespace trspv {

Solver2D::Solver2D(
    const Eigen::MatrixXcd& A,
    const Eigen::VectorXcd& b,
    const ADMMConfig& solve_cfg,
    const Eigen::SparseMatrix<double>& D)
    : A_(A), b_(b), D_(D), solve_cfg_(solve_cfg), scan_cfg_(solve_cfg) {}

void Solver2D::set_scan_config(const ADMMConfig& scan_cfg, const ParamSelectionConfig& param_cfg) {
    scan_cfg_ = scan_cfg;
    param_cfg_ = param_cfg;
    has_scan_cfg_ = true;
}

void Solver2D::solve() {
    ADMMConfig selected_cfg = solve_cfg_;
    std::ostringstream summary;

    if (has_scan_cfg_ && param_cfg_.enable) {
        ParamSelector selector(A_, b_, D_, scan_cfg_, param_cfg_);
        selected_cfg = selector.select();
        summary << "Param scan enabled\n";
    } else {
        summary << "Param scan disabled\n";
    }

    best_result_.lambda1 = selected_cfg.lambda1;
    best_result_.lambda_tv_tau = selected_cfg.lambda_tv_tau;
    best_result_.lambda_tv_beta = selected_cfg.lambda_tv_beta;

    ADMMConfig final_cfg = solve_cfg_;
    final_cfg.lambda1 = selected_cfg.lambda1;
    final_cfg.lambda_tv_tau = selected_cfg.lambda_tv_tau;
    final_cfg.lambda_tv_beta = selected_cfg.lambda_tv_beta;
    if (!selected_cfg.l1_weights.empty()) {
        final_cfg.l1_weights = selected_cfg.l1_weights;
    }

    ADMMOptimizer optimizer(A_, b_, D_, final_cfg);
    best_solution_ = optimizer.solve();

    summary << "Final solve config:\n";
    summary << "  lambda1=" << final_cfg.lambda1 << '\n';
    summary << "  lambda_tv_tau=" << final_cfg.lambda_tv_tau << '\n';
    summary << "  lambda_tv_beta=" << final_cfg.lambda_tv_beta << '\n';
    summary << "  rho=" << final_cfg.rho << '\n';
    summary << "  max_iters=" << final_cfg.max_iters << '\n';
    summary << "  tol_primal=" << final_cfg.tol_primal << '\n';
    summary << "  tol_dual=" << final_cfg.tol_dual << '\n';
    debug_summary_ = summary.str();
}

const Eigen::VectorXcd& Solver2D::best_solution() const {
    return best_solution_;
}

ParamSelectionResult Solver2D::best_result() const {
    return best_result_;
}

const std::string& Solver2D::debug_summary() const {
    return debug_summary_;
}

}  // namespace trspv
