#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "ComponentAnalysis.h"
#include "Config.h"
#include "Solver2D.h"
#include "SpectrumData.h"

namespace trspv {

class ResultWriter {
public:
    static void write_admm_summary(
        const std::string& output_dir,
        const std::string& debug_summary,
        const ParamSelectionResult& best);

    static void write_components(
        const std::string& output_dir,
        const std::vector<Component>& comps);

    static void write_peak_seeds(
        const std::string& output_dir,
        const std::vector<double>& tau_seed);

    static void write_interpolation_outputs(
        const Config& cfg,
        const SpectrumData& raw_data,
        const SpectrumData& interp_data);

    static void write_transient_outputs(
        const Config& cfg,
        const std::vector<Component>& comps);

    static void write_metrics(
        const Config& cfg,
        const SpectrumData& data,
        const std::vector<double>& taus,
        const std::vector<double>& betas,
        const Eigen::VectorXcd& x2d,
        const Eigen::MatrixXcd& A,
        const std::vector<Component>& comps);
};

}  // namespace trspv
