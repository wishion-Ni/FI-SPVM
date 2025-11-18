// Solver2D.cpp
#include "Solver2D.h"
#include "DictionaryGenerator.h"
#include "Utils.h"
#include "ADMMOptimizer.h"

namespace trspv {

    Solver2D::Solver2D(const std::vector<double>& omega,
        const Eigen::VectorXcd& b,
        const std::vector<double>& taus,
        const std::vector<double>& betas,
        const ADMMConfig& acfg)
        : omega_(omega), b_(b), taus_(taus), betas_(betas), acfg_(acfg) {}

    Eigen::VectorXcd Solver2D::solve() {
        // 1. 生成二维字典矩阵 A (M × (Nτ·Nβ))
        DictionaryConfig dcfg;
        dcfg.tau_list = taus_;
        dcfg.gamma_list = betas_;
        dcfg.enable_cache = false;
        Eigen::MatrixXcd A = DictionaryGenerator(dcfg).generate(omega_);

        // 2. 构造二维 TV 差分算子 D2D
        int N_tau = static_cast<int>(taus_.size());
        int N_beta = static_cast<int>(betas_.size());
        Eigen::SparseMatrix<double> D2D = trspv::build2DTV(N_tau, N_beta);
        double dlogt = (N_tau > 1 ? std::log(taus_[1] / taus_[0]) : 1.0);
        double dbeta = (N_beta > 1 ? (betas_[1] - betas_[0]) : 1.0);
       // trspv::scaleTVBySteps(D2D, N_tau, N_beta, dlogt, dbeta);


        

        // 3. 调用 ADMM 优化
        ADMMOptimizer opt(A, b_, D2D, acfg_);
        Eigen::VectorXcd x = opt.solve();

        return x;
    }

} // namespace trspv
