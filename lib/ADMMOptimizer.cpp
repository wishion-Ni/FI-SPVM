#include "ADMMOptimizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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
    refactorize();
}

void ADMMOptimizer::refactorize() {
    Eigen::MatrixXcd H = A_.adjoint() * A_;
    H.diagonal().array() += cfg_.rho;

    const Eigen::SparseMatrix<double> DtD = D_.transpose() * D_;
    for (int outer = 0; outer < DtD.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(DtD, outer); it; ++it) {
            H(it.row(), it.col()) += std::complex<double>(cfg_.rho * it.value(), 0.0);
        }
    }

    solver_.compute(H);
    if (solver_.info() != Eigen::Success) {
        throw std::runtime_error("ADMM factorization failed during normal-equation setup");
    }
}

void ADMMOptimizer::updateParams(const ADMMConfig& cfg) {
    const bool rho_changed = std::abs(cfg.rho - cfg_.rho) > 1e-12;
    cfg_ = cfg;
    gsTau_ = cfg.group_size_tau;
    gsBeta_ = cfg.group_size_beta;
    gammaStride_ = cfg.gamma_stride;

    const int N = static_cast<int>(A_.cols());
    w_ = cfg.l1_weights;
    if (static_cast<int>(w_.size()) != N || w_.empty()) {
        w_.assign(N, 1.0);
    }

    if (rho_changed) {
        refactorize();
    }
}

int ADMMOptimizer::last_iterations() const {
    return last_iterations_;
}

bool ADMMOptimizer::converged() const {
    return converged_;
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
    const ComplexVec v = x + u1;

    for (int b0 = 0; b0 < betaCount * stride; b0 += G_beta * stride) {
        for (int t0 = 0; t0 < stride; t0 += G_tau) {
            std::vector<int> idx;
            for (int gb = 0; gb < G_beta; ++gb) {
                for (int gt = 0; gt < G_tau; ++gt) {
                    const int col = b0 + gb * stride + (t0 + gt);
                    if (col < N) {
                        idx.push_back(col);
                    }
                }
            }
            if (idx.empty()) {
                continue;
            }

            double nrm = 0.0;
            for (int id : idx) {
                nrm += std::norm(v[id]);
            }
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

            for (int id : idx) {
                z1[id] = sf * v[id];
            }
        }
    }
}

void ADMMOptimizer::updateZ2(const ComplexVec& x, const ComplexVec& u2, ComplexVec& z2) const {
    const int Nt = std::max(1, cfg_.Nt);
    const int Nb = std::max(1, cfg_.Nb);
    const int rowsTau = Nb * std::max(0, Nt - 1);
    const int rowsBeta = Nt * std::max(0, Nb - 1);

    ComplexVec v = D_ * x + u2;
    z2.resize(v.size());
    for (int i = 0; i < v.size(); ++i) {
        const double k = (i < rowsTau) ? (cfg_.lambda_tv_tau / cfg_.rho)
                                       : (cfg_.lambda_tv_beta / cfg_.rho);
        const double mag = std::abs(v[i]);
        const double s = std::max(0.0, mag - k);
        z2[i] = (mag > 0.0 ? v[i] * (s / mag) : std::complex<double>(0.0, 0.0));
    }
}

double ADMMOptimizer::compute_primal_residual(
    const ComplexVec& x,
    const ComplexVec& z1,
    const ComplexVec& z2) const {
    const double r1 = (x - z1).norm();
    const double r2 = (D_ * x - z2).norm();
    return std::sqrt(r1 * r1 + r2 * r2);
}

double ADMMOptimizer::compute_dual_residual(
    const ComplexVec& z1,
    const ComplexVec& z1_prev,
    const ComplexVec& z2,
    const ComplexVec& z2_prev) const {
    const double s1 = (cfg_.rho * (z1 - z1_prev)).norm();
    const double s2 = (cfg_.rho * (D_.transpose() * (z2 - z2_prev))).norm();
    return std::sqrt(s1 * s1 + s2 * s2);
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

    last_iterations_ = 0;
    converged_ = false;

    for (int iter = 0; iter < cfg_.max_iters; ++iter) {
        const ComplexVec z1_prev = z1;
        const ComplexVec z2_prev = z2;

        x = solver_.solve(computeRhs(z1, u1, z2, u2));
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("ADMM linear solve failed during iteration");
        }

        updateZ1(x, u1, z1);
        updateZ2(x, u2, z2);

        u1 += x - z1;
        u2 += D_ * x - z2;

        last_iterations_ = iter + 1;
        const double primal = compute_primal_residual(x, z1, z2);
        const double dual = compute_dual_residual(z1, z1_prev, z2, z2_prev);
        if (primal <= cfg_.tol_primal && dual <= cfg_.tol_dual) {
            converged_ = true;
            break;
        }
    }

    return {x, z1, z2, u1, u2};
}

}  // namespace trspv
