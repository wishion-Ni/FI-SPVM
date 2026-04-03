#pragma once

#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Config.h"

namespace trspv {

struct ParamSelectionResult {
    double lambda1 = 0.0;
    double lambda_tv_tau = 0.0;
    double lambda_tv_beta = 0.0;
};

class Solver2D {
public:
    Solver2D(
        const Eigen::MatrixXcd& A,
        const Eigen::VectorXcd& b,
        const ADMMConfig& solve_cfg,
        const Eigen::SparseMatrix<double>& D);

    void set_scan_config(const ADMMConfig& scan_cfg, const ParamSelectionConfig& param_cfg);
    void solve();

    const Eigen::VectorXcd& best_solution() const;
    ParamSelectionResult best_result() const;
    const std::string& debug_summary() const;

private:
    Eigen::MatrixXcd A_;
    Eigen::VectorXcd b_;
    Eigen::SparseMatrix<double> D_;
    ADMMConfig solve_cfg_;
    ADMMConfig scan_cfg_;
    ParamSelectionConfig param_cfg_;
    bool has_scan_cfg_ = false;
    Eigen::VectorXcd best_solution_;
    ParamSelectionResult best_result_;
    std::string debug_summary_;
};

}  // namespace trspv
