#include "KernelFunction.h"

#include <cmath>
#include <complex>

namespace trspv {

std::complex<double> KernelFunction::evaluate(double omega,
    double tau_row,
    double tau_center,
    double gamma,
    double sigma_dec) {
    using namespace std::complex_literals;

    std::complex<double> arg = 1i * (omega * tau_center);
    std::complex<double> z = std::exp(gamma * std::log(arg));
    std::complex<double> H = 1.0 / (1.0 + z);

    // Optional Gaussian broadening in log10(tau).
    if (sigma_dec > 0.0) {
        const double d_dec = std::log10(tau_row / tau_center);
        const double weight = std::exp(-0.5 * d_dec * d_dec /
            (sigma_dec * sigma_dec));
        H *= weight;
    }
    return H;
}

} // namespace trspv
