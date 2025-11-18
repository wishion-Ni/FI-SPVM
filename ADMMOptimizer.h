#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Config.h"
#include <tuple>



namespace trspv {

    using std::tuple;

    /// @brief ADMM 优化器，用于求解 1/2||Ax-b||^2 + lambda1||x||_1 + lambda_tv||D x||_1
    class ADMMOptimizer {
    public:
        /**
         * @param A     MxN 复数设计矩阵
         * @param b     长度 M 的复数观测向量
         * @param D     (N-1)xN TV 差分稀疏矩阵
         * @param cfg   ADMM 配置参数
         */
        ADMMOptimizer(const Eigen::MatrixXcd& A,
            const Eigen::VectorXcd& b,
            const Eigen::SparseMatrix<double>& D,
            const ADMMConfig& cfg);

        /// 执行 ADMM, 返回 N 维复数解 x
        Eigen::VectorXcd solve();

        /// 更新正则参数，但不重复分解矩阵
        void updateParams(const ADMMConfig& cfg);

        /**
     * 带 Warm Start 的求解
     * @param x0  上一次 x
     * @param z1_0 上一次 z1
     * @param z2_0 上一次 z2
     * @param u1_0 上一次 u1
     * @param u2_0 上一次 u2
     * @return tuple< x, z1, z2, u1, u2 >
     */
        tuple<Eigen::VectorXcd, Eigen::VectorXcd,
            Eigen::VectorXcd, Eigen::VectorXcd,
            Eigen::VectorXcd>
            solveWarmStart(const Eigen::VectorXcd& x0,
                const Eigen::VectorXcd& z1_0,
                const Eigen::VectorXcd& z2_0,
                const Eigen::VectorXcd& u1_0,
                const Eigen::VectorXcd& u2_0);

    private:
  

        Eigen::MatrixXcd           A_;       ///< 设计矩阵
        Eigen::VectorXcd           b_;       ///< 观测向量
        Eigen::SparseMatrix<double> D_;      ///< TV 差分矩阵
        std::vector<double> w_;   // 权重副本
        ADMMConfig                 cfg_;     ///< 配置
        int groupSize_;
        int gsTau_, gsBeta_, gammaStride_;
        // 预分解矩阵: (A^H A + rho I + rho D^T D)
        Eigen::LDLT<Eigen::MatrixXcd> solver_;

        // 原始 solve() 步骤改为接收/不接收初值
        tuple<Eigen::VectorXcd, Eigen::VectorXcd,
            Eigen::VectorXcd, Eigen::VectorXcd,
            Eigen::VectorXcd>
            solveInternal(const Eigen::VectorXcd* x0,
                const Eigen::VectorXcd* z1_0,
                const Eigen::VectorXcd* z2_0,
                const Eigen::VectorXcd* u1_0,
                const Eigen::VectorXcd* u2_0);
    };

} // namespace trspv