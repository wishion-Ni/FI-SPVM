// ADMMOptimizer.cpp
#include "ADMMOptimizer.h"
#include <cassert>

namespace trspv {

    ADMMOptimizer::ADMMOptimizer(const Eigen::MatrixXcd& A,
        const Eigen::VectorXcd& b,
        const Eigen::SparseMatrix<double>& D,
        const ADMMConfig& cfg)
        : A_(A), b_(b), D_(D), cfg_(cfg), 
        gsTau_(cfg.group_size_tau),
        gsBeta_(cfg.group_size_beta),
        gammaStride_(cfg.gamma_stride)
    {
        const int N = (int)A_.cols();
        w_ = cfg.l1_weights;
        if ((int)w_.size() != N) {
            w_.assign(N, 1.0);
            // 可加日志：std::cerr << "[ADMM] l1_weights size mismatch. Fallback to all-ones.\n";
        }
        if (w_.empty()) w_.assign(A.cols(), 1.0);

        // 1) 构造 H = A^* A + rho I + rho D^T D
        Eigen::MatrixXcd H = A_.adjoint() * A_;
        H.diagonal().array() += cfg_.rho;
        // + rho * D^T D （稀疏转密集）
        Eigen::MatrixXd DtD = (D_.transpose() * D_).toDense();
        for (int i = 0; i < H.rows(); ++i)
            for (int j = 0; j < H.cols(); ++j)
                H(i, j) += cfg_.rho * DtD(i, j);

        // 2) LDLT 分解
        solver_.compute(H);
        assert(solver_.info() == Eigen::Success);
    }

    void ADMMOptimizer::updateParams(const ADMMConfig& cfg) {
        // 只更新正则参数，不重分解
        cfg_ = cfg;
        // 注意：如果你在 updateZ1/updateZ2 中用 cfg_.lambda*，
        // 就会生效；solver_ 保留旧的 H 分解。
    }

    auto ADMMOptimizer::solveWarmStart(const Eigen::VectorXcd& x0,
        const Eigen::VectorXcd& z1_0,
        const Eigen::VectorXcd& z2_0,
        const Eigen::VectorXcd& u1_0,
        const Eigen::VectorXcd& u2_0)
        -> tuple<Eigen::VectorXcd, Eigen::VectorXcd,
        Eigen::VectorXcd, Eigen::VectorXcd,
        Eigen::VectorXcd>
    {
        return solveInternal(&x0, &z1_0, &z2_0, &u1_0, &u2_0);
    }

    auto ADMMOptimizer::solveInternal(const Eigen::VectorXcd* x0,
        const Eigen::VectorXcd* z1_0,
        const Eigen::VectorXcd* z2_0,
        const Eigen::VectorXcd* u1_0,
        const Eigen::VectorXcd* u2_0)
        -> tuple<Eigen::VectorXcd, Eigen::VectorXcd,
        Eigen::VectorXcd, Eigen::VectorXcd,
        Eigen::VectorXcd>
    {
        int N = A_.cols();
        int M2 = D_.rows();
        // 初始化
        Eigen::VectorXcd x = x0 ? *x0 : Eigen::VectorXcd::Zero(N);
        Eigen::VectorXcd z1 = z1_0 ? *z1_0 : Eigen::VectorXcd::Zero(N);
        Eigen::VectorXcd z2 = z2_0 ? *z2_0 : Eigen::VectorXcd::Zero(M2);
        Eigen::VectorXcd u1 = u1_0 ? *u1_0 : Eigen::VectorXcd::Zero(N);
        Eigen::VectorXcd u2 = u2_0 ? *u2_0 : Eigen::VectorXcd::Zero(M2);

        for (int iter = 0; iter < cfg_.max_iters; ++iter) {
            // --- x-update: solve (H x = A^H b + rho*(z1-u1) + rho*D^T*(z2-u2)) ---
            Eigen::VectorXcd rhs = A_.adjoint() * b_
                + cfg_.rho * (z1 - u1)
                + cfg_.rho * (D_.transpose() * (z2 - u2));
            x = solver_.solve(rhs);

            // --- z1-update: soft-threshold
            {
                Eigen::VectorXcd v = x + u1;
                z1.setZero(N);

                const int G_tau = std::max(1, gsTau_);
                const int G_beta = std::max(1, gsBeta_);
                const int stride = std::max(1, gammaStride_);          // = τ_list.size()

                int betaCount = (stride > 0 ? N / stride : 1);          // 常数列已禁用，整除 OK
                for (int b0 = 0; b0 < betaCount * stride; b0 += G_beta * stride)
                {
                    for (int t0 = 0; t0 < stride; t0 += G_tau) {
                        // 收集一个 (Gβ × Gτ) 块
                        std::vector<int> idx;
                        for (int gb = 0; gb < G_beta; ++gb)
                            for (int gt = 0; gt < G_tau; ++gt) {
                                int col = b0 + gb * stride + (t0 + gt);
                                if (col < N) idx.push_back(col);
                            }
                        if (idx.empty()) continue;
                        // 计算 L2 范数
                        double nrm = 0.0;
                        for (int id : idx) nrm += std::norm(v[id]);
                        nrm = std::sqrt(nrm);

                        // 组平均权重（越界保护）
                        double wsum = 0.0; int cnt = 0;
                        for (int id : idx) {
                            if ((unsigned)id < (unsigned)w_.size()) { wsum += w_[id]; ++cnt; }
                        }
                        double wavg = (cnt > 0 ? wsum / cnt : 1.0);

                        // block soft-threshold
                        double k = (cfg_.lambda1 * wavg) / cfg_.rho;
                        double sf = (nrm > 0.0 ? std::max(0.0, 1.0 - k / nrm) : 0.0);

                        for (int id : idx) z1[id] = sf * v[id];
                    }
                }
            }

            // --- z2-update: TV thresholds per direction -----------------
            {
                const int Nt = std::max(1, cfg_.Nt);
                const int Nb = std::max(1, cfg_.Nb);
                const int rowsTau = Nb * std::max(0, Nt - 1);
                const int rowsBeta = Nt * std::max(0, Nb - 1);
                assert(rowsTau + rowsBeta == D_.rows());

                Eigen::VectorXcd v = D_ * x + u2;
                for (int i = 0; i < v.size(); ++i) {
                    double k = (i < rowsTau) ? (cfg_.lambda_tv_tau / cfg_.rho)
                        : (cfg_.lambda_tv_beta / cfg_.rho);
                    double mag = std::abs(v[i]);
                    double s = std::max(0.0, mag - k);
                    z2[i] = (mag > 0 ? v[i] * (s / mag) : std::complex<double>(0, 0));
                }
            }

            // --- u-update ---
            u1 += x - z1;
            u2 += D_ * x - z2;

            // TODO: 收敛判据判断，可提前 break
        }

        return { x, z1, z2, u1, u2 };
    }

    Eigen::VectorXcd ADMMOptimizer::solve() {
        // call solveInternal with null ptrs → uses zero initial conditions
        auto tup = solveInternal(nullptr, nullptr, nullptr, nullptr, nullptr);
        return std::get<0>(tup);  // unpack and return x
    }

} // namespace trspv
