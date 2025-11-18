#pragma once
#include <complex>

namespace trspv {

    /// @brief 核函数 H(ω; τ, γ) = 1 / (1 + (i·ω·τ)^γ)
    class KernelFunction {
    public:
        /// @param omega 角频率 ω (rad/s)
        /// @param tau   弛豫时间 τ (s)
        /// @param gamma 幂指数 γ (无量纲)
        /// @return      复数值 H(ω; τ, γ)
        static std::complex<double> evaluate(double omega,
            double tau_row,
            double tau_center,
            double gamma,
            double sigma_dec = 0.0);
    };

} // namespace trspv
