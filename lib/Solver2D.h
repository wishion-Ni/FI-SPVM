// Solver2D.h
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Config.h"

namespace trspv {

    class Solver2D {
    public:
        // omega: M 个角频率； b: M 维复观测向量
        // taus: Nτ 个松弛时间； betas: Nβ 个拉伸指数
        // acfg: ADMM 配置（含 lambda1, lambda_tv_tau, lambda_tv_beta, rho 等）
        Solver2D(const std::vector<double>& omega,
            const Eigen::VectorXcd& b,
            const std::vector<double>& taus,
            const std::vector<double>& betas,
            const ADMMConfig& acfg);

        // 运行 2D 反演，返回长度 Nτ·Nβ 的系数向量 x
        Eigen::VectorXcd solve();

    private:
        std::vector<double>    omega_;
        Eigen::VectorXcd       b_;
        std::vector<double>    taus_, betas_;
        ADMMConfig             acfg_;
    };

} // namespace trspv

