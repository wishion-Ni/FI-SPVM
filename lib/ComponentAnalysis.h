#pragma once

#include <complex>
#include <vector>

#include <Eigen/Dense>

namespace trspv {

struct Component {
    double tau;
    double beta;
    double amp;
    double prominence;
    int it;
    int jb;
};

std::vector<Component> extract_components(
    const Eigen::VectorXcd& x2d,
    const std::vector<double>& taus,
    const std::vector<double>& betas,
    double k_sigma = 3.0,
    double alpha_of_max = 0.05,
    int min_pixels = 4,
    double merge_dlogtau = 0.20,
    double merge_dbeta = 0.12);

// 单组分阶跃响应

double h_on(double t, double tau, double beta);
double h_off(double t, double tau, double beta);

}  // namespace trspv

