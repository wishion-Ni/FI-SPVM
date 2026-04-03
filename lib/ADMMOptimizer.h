#pragma once

#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Config.h"

namespace trspv {

using std::tuple;
using ComplexVec = Eigen::VectorXcd;

/// @brief ADMM solver for 1/2||Ax-b||^2 + lambda1||x||_1 + lambda_tv||D x||_1
class ADMMOptimizer {
public:
    ADMMOptimizer(const Eigen::MatrixXcd& A,
                  const ComplexVec& b,
                  const Eigen::SparseMatrix<double>& D,
                  const ADMMConfig& cfg);

    /// Execute ADMM and return the optimized N-dimensional x
    ComplexVec solve();

    /// Update regularization parameters without recomputing matrix factors
    void updateParams(const ADMMConfig& cfg);

    int last_iterations() const;
    bool converged() const;

    tuple<ComplexVec, ComplexVec, ComplexVec, ComplexVec, ComplexVec>
    solveWarmStart(const ComplexVec& x0,
                   const ComplexVec& z1_0,
                   const ComplexVec& z2_0,
                   const ComplexVec& u1_0,
                   const ComplexVec& u2_0);

private:
    tuple<ComplexVec, ComplexVec, ComplexVec, ComplexVec, ComplexVec>
    solveInternal(const ComplexVec* x0,
                  const ComplexVec* z1_0,
                  const ComplexVec* z2_0,
                  const ComplexVec* u1_0,
                  const ComplexVec* u2_0);

    ComplexVec initializeVector(const ComplexVec* seed, int size) const;
    ComplexVec computeRhs(const ComplexVec& z1,
                          const ComplexVec& u1,
                          const ComplexVec& z2,
                          const ComplexVec& u2) const;
    void updateZ1(const ComplexVec& x, const ComplexVec& u1, ComplexVec& z1) const;
    void updateZ2(const ComplexVec& x, const ComplexVec& u2, ComplexVec& z2) const;
    void refactorize();
    double compute_primal_residual(
        const ComplexVec& x,
        const ComplexVec& z1,
        const ComplexVec& z2) const;
    double compute_dual_residual(
        const ComplexVec& z1,
        const ComplexVec& z1_prev,
        const ComplexVec& z2,
        const ComplexVec& z2_prev) const;

    Eigen::MatrixXcd A_; ///< Design matrix
    ComplexVec b_;       ///< Observations
    Eigen::SparseMatrix<double> D_; ///< TV operator
    std::vector<double> w_;
    ADMMConfig cfg_;

    int gsTau_;
    int gsBeta_;
    int gammaStride_;

    Eigen::LDLT<Eigen::MatrixXcd> solver_;
    int last_iterations_ = 0;
    bool converged_ = false;
};

} // namespace trspv
