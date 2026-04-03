#pragma once
#include <complex>

namespace trspv {

    /// Kernel H(omega; tau, gamma) = 1 / (1 + (i * omega * tau)^gamma).
    class KernelFunction {
    public:
        /// omega: angular frequency in rad/s.
        /// tau_center: relaxation time center in seconds.
        /// gamma: fractional exponent.
        /// Returns the complex kernel response.
        static std::complex<double> evaluate(double omega,
            double tau_row,
            double tau_center,
            double gamma,
            double sigma_dec = 0.0);
    };

} // namespace trspv
