#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../lib/Config.h"
#include "../lib/ComponentAnalysis.h"
#include "../lib/SpectrumData.h"

class SolverApp {
public:
    int run(int argc, char** argv) const;

private:
    struct CliOptions {
        std::string configPath = "config.json";
        std::string overrideInput;
        std::string overrideOut;
    };

    CliOptions parse_arguments(int argc, char** argv) const;
    trspv::Config load_config(const CliOptions& opts) const;
    void ensure_output_dir(const std::string& dir) const;

    trspv::SpectrumData load_and_maybe_complete(
        const trspv::Config& cfg,
        trspv::SpectrumData& rawData) const;

    void write_interpolation_outputs(
        const trspv::Config& cfg,
        const trspv::SpectrumData& rawData,
        const trspv::SpectrumData& interpData) const;

    std::vector<double> build_tau_grid(
        const trspv::Config& cfg,
        const trspv::SpectrumData& data,
        std::vector<double>& l1Weights) const;

    std::vector<double> build_beta_grid(const trspv::Config& cfg) const;

    Eigen::VectorXcd build_rhs(
        const trspv::SpectrumData& data,
        std::vector<double>& omega) const;

    Eigen::SparseMatrix<double> build_tv_operator(
        int Nt, int Nb,
        const std::vector<double>& taus,
        const std::vector<double>& betas,
        double& dlogt,
        double& dbeta) const;

    trspv::ADMMConfig make_scan_config(
        const trspv::Config& cfg,
        const std::vector<double>& l1w2d) const;

    double compute_lambda1_max(
        const Eigen::MatrixXcd& A,
        const Eigen::VectorXcd& b,
        const std::vector<double>& l1w2d,
        int Nt,
        int Nb,
        const trspv::DictionaryConfig& dcfg,
        const trspv::Config& cfg) const;

    void write_transient_outputs(
        const trspv::Config& cfg,
        const std::vector<trspv::Component>& comps) const;

    void write_metrics(
        const trspv::Config& cfg,
        const trspv::SpectrumData& data,
        const std::vector<double>& taus,
        const std::vector<double>& betas,
        const Eigen::VectorXcd& x2d,
        const Eigen::MatrixXcd& A,
        const std::vector<trspv::Component>& comps) const;
};

