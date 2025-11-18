#include "ADMMOptimizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace trspv {

ADMMOptimizer::ADMMOptimizer(const Eigen::MatrixXcd& A,
                             const ComplexVec& b,
                             const Eigen::SparseMatrix<double>& D,
                             const ADMMConfig& cfg)
    : A_(A), b_(b), D_(D), cfg_(cfg),
      gsTau_(cfg.group_size_tau),
      gsBeta_(cfg.group_size_beta),
      gammaStride_(cfg.gamma_stride) {
    const int N = static_cast<int>(A_.cols());
    w_ = cfg.l1_weights;
    if (static_cast<int>(w_.size()) != N || w_.empty()) {
        w_.assign(N, 1.0);
    }

    Eigen::MatrixXcd H = A_.adjoint() * A_;
    H.diagonal().array() += cfg_.rho;
    const Eigen::MatrixXd DtD = (D_.transpose() * D_).toDense();
    H += cfg_.rho * DtD.cast<std::complex<double>>();

    solver_.compute(H);
    assert(solver_.info() == Eigen::Success);
}

void ADMMOptimizer::updateParams(const ADMMConfig& cfg) {
    cfg_ = cfg;
}

tuple<ComplexVec, ComplexVec, ComplexVec, ComplexVec, ComplexVec>
ADMMOptimizer::solveWarmStart(const ComplexVec& x0,
                              const ComplexVec& z1_0,
                              const ComplexVec& z2_0,
                              const ComplexVec& u1_0,
                              const ComplexVec& u2_0) {
    return solveInternal(&x0, &z1_0, &z2_0, &u1_0, &u2_0);
}

ComplexVec ADMMOptimizer::solve() {
    auto tup = solveInternal(nullptr, nullptr, nullptr, nullptr, nullptr);
    return std::get<0>(tup);
}

ComplexVec ADMMOptimizer::initializeVector(const ComplexVec* seed, int size) const {
    return seed ? *seed : ComplexVec::Zero(size);
}

ComplexVec ADMMOptimizer::computeRhs(const ComplexVec& z1,
                                     const ComplexVec& u1,
                                     const ComplexVec& z2,
                                     const ComplexVec& u2) const {
    return A_.adjoint() * b_ + cfg_.rho * (z1 - u1)
        + cfg_.rho * (D_.transpose() * (z2 - u2));
}

void ADMMOptimizer::updateZ1(const ComplexVec& x, const ComplexVec& u1, ComplexVec& z1) const {
    const int N = static_cast<int>(x.size());
    const int G_tau = std::max(1, gsTau_);
    const int G_beta = std::max(1, gsBeta_);
    const int stride = std::max(1, gammaStride_);
    const int betaCount = (stride > 0 ? N / stride : 1);

    z1.setZero(N);
    ComplexVec v = x + u1;

    for (int b0 = 0; b0 < betaCount * stride; b0 += G_beta * stride) {
        for (int t0 = 0; t0 < stride; t0 += G_tau) {
            std::vector<int> idx;
            for (int gb = 0; gb < G_beta; ++gb) {
                for (int gt = 0; gt < G_tau; ++gt) {
                    int col = b0 + gb * stride + (t0 + gt);
                    if (col < N) idx.push_back(col);
                }
            }
            if (idx.empty()) continue;

            double nrm = 0.0;
            for (int id : idx) nrm += std::norm(v[id]);
            nrm = std::sqrt(nrm);

            double wsum = 0.0;
            int cnt = 0;
            for (int id : idx) {
                if (static_cast<unsigned>(id) < static_cast<unsigned>(w_.size())) {
                    wsum += w_[id];
                    ++cnt;
                }
            }
            const double wavg = (cnt > 0 ? wsum / cnt : 1.0);
            const double k = (cfg_.lambda1 * wavg) / cfg_.rho;
            const double sf = (nrm > 0.0 ? std::max(0.0, 1.0 - k / nrm) : 0.0);

            for (int id : idx) z1[id] = sf * v[id];
        }
    }
}

void ADMMOptimizer::updateZ2(const ComplexVec& x, const ComplexVec& u2, ComplexVec& z2) const {
    const int Nt = std::max(1, cfg_.Nt);
    const int Nb = std::max(1, cfg_.Nb);
    const int rowsTau = Nb * std::max(0, Nt - 1);
    const int rowsBeta = Nt * std::max(0, Nb - 1);
    assert(rowsTau + rowsBeta == D_.rows());

    ComplexVec v = D_ * x + u2;
    for (int i = 0; i < v.size(); ++i) {
        double k = (i < rowsTau) ? (cfg_.lambda_tv_tau / cfg_.rho)
                                 : (cfg_.lambda_tv_beta / cfg_.rho);
        double mag = std::abs(v[i]);
        double s = std::max(0.0, mag - k);
        z2[i] = (mag > 0 ? v[i] * (s / mag) : std::complex<double>(0, 0));
    }
}

tuple<ComplexVec, ComplexVec, ComplexVec, ComplexVec, ComplexVec>
ADMMOptimizer::solveInternal(const ComplexVec* x0,
                             const ComplexVec* z1_0,
                             const ComplexVec* z2_0,
                             const ComplexVec* u1_0,
                             const ComplexVec* u2_0) {
    const int N = static_cast<int>(A_.cols());
    const int M2 = static_cast<int>(D_.rows());

    ComplexVec x = initializeVector(x0, N);
    ComplexVec z1 = initializeVector(z1_0, N);
    ComplexVec z2 = initializeVector(z2_0, M2);
    ComplexVec u1 = initializeVector(u1_0, N);
    ComplexVec u2 = initializeVector(u2_0, M2);

    for (int iter = 0; iter < cfg_.max_iters; ++iter) {
        x = solver_.solve(computeRhs(z1, u1, z2, u2));
        updateZ1(x, u1, z1);
        updateZ2(x, u2, z2);

        u1 += x - z1;
        u2 += D_ * x - z2;
        // TODO: convergence check
    }

    return {x, z1, z2, u1, u2};
}

} // namespace trspv
