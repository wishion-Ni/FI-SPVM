// ParamSelector.h
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Config.h"

namespace trspv {

    /// 自动参数选择：扫描 λ1、λτ、λβ 并用 L-curve 拐点法选最优 λ1
    class ParamSelector {
    public:
        /**
         * @param A       M×N 字典矩阵
         * @param b       M 维观测向量
         * @param D       TV 差分算子矩阵
         * @param baseCfg 基础 ADMM 配置（保留 λτ、λβ、ρ 等）
         * @param psc     自动参数选择配置
         */
        ParamSelector(const Eigen::MatrixXcd& A,
            const Eigen::VectorXcd& b,
            const Eigen::SparseMatrix<double>& D,
            const ADMMConfig& baseCfg,
            const ParamSelectionConfig& psc);
        /// 扫描 3D 网格，输出 lcurve.csv，返回最优 ADMMConfig
        ADMMConfig select() const;

    private:
        

        Eigen::MatrixXcd            A_;
        Eigen::VectorXcd            b_;
        Eigen::SparseMatrix<double> D_;
        ADMMConfig                  baseCfg_;
        ParamSelectionConfig        psc_;
    };

} // namespace trspv
