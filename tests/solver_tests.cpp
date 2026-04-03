#include <filesystem>

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../lib/ADMMOptimizer.h"
#include "../lib/ParamSelector.h"

TEST(ADMMOptimizerTest, StopsEarlyWhenResidualsReachTolerance) {
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Identity(2, 2);
    Eigen::VectorXcd b = Eigen::VectorXcd::Zero(2);
    Eigen::SparseMatrix<double> D(0, 2);

    trspv::ADMMConfig cfg;
    cfg.lambda1 = 1e-4;
    cfg.lambda_tv_tau = 1e-4;
    cfg.lambda_tv_beta = 1e-4;
    cfg.rho = 1.0;
    cfg.max_iters = 50;
    cfg.tol_primal = 1e-12;
    cfg.tol_dual = 1e-12;
    cfg.group_size_tau = 1;
    cfg.group_size_beta = 1;
    cfg.gamma_stride = 2;
    cfg.Nt = 2;
    cfg.Nb = 1;
    cfg.l1_weights = {1.0, 1.0};

    trspv::ADMMOptimizer optimizer(A, b, D, cfg);
    const auto x = optimizer.solve();

    EXPECT_TRUE(optimizer.converged());
    EXPECT_LT(optimizer.last_iterations(), cfg.max_iters);
    EXPECT_NEAR(x.norm(), 0.0, 1e-12);
}

TEST(ParamSelectorTest, SelectsFirstCandidateDeterministicallyOnEqualScores) {
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Identity(2, 2);
    Eigen::VectorXcd b = Eigen::VectorXcd::Zero(2);
    Eigen::SparseMatrix<double> D(0, 2);

    trspv::ADMMConfig baseCfg;
    baseCfg.lambda1 = 1e-4;
    baseCfg.lambda_tv_tau = 1e-5;
    baseCfg.lambda_tv_beta = 1e-5;
    baseCfg.rho = 1.0;
    baseCfg.max_iters = 20;
    baseCfg.tol_primal = 1e-12;
    baseCfg.tol_dual = 1e-12;
    baseCfg.group_size_tau = 1;
    baseCfg.group_size_beta = 1;
    baseCfg.gamma_stride = 2;
    baseCfg.Nt = 2;
    baseCfg.Nb = 1;
    baseCfg.l1_weights = {1.0, 1.0};

    trspv::ParamSelectionConfig psc;
    psc.enable = true;
    psc.num_lambda1 = 2;
    psc.lambda1_min = 1e-5;
    psc.lambda1_max = 1e-3;
    psc.num_lambdat = 2;
    psc.lambdat_min = 1e-6;
    psc.lambdat_max = 1e-4;
    psc.num_lambdab = 2;
    psc.lambdab_min = 1e-6;
    psc.lambdab_max = 1e-4;
    psc.scan_max_iters = 10;
    psc.scan_tol = 1e-12;
    psc.outputDir = (std::filesystem::temp_directory_path() / "fi_spvm_param_selector_test").string();

    trspv::ParamSelector selector(A, b, D, baseCfg, psc);
    const auto selected = selector.select();

    EXPECT_DOUBLE_EQ(selected.lambda1, psc.lambda1_min);
    EXPECT_DOUBLE_EQ(selected.lambda_tv_tau, psc.lambdat_min);
    EXPECT_DOUBLE_EQ(selected.lambda_tv_beta, psc.lambdab_min);

    std::filesystem::remove_all(psc.outputDir);
}
