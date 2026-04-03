#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "DictionaryGenerator.h"
#include "KernelFunction.h"

namespace trspv {

struct KernelConfig {
    double tau_min = 1e-4;
    double tau_max = 1.0;
    int num_tau = 50;
    double gamma_min = 0.2;
    double gamma_max = 1.8;
    int num_gamma = 20;
    std::string gamma_scale = "linear";
};

struct ADMMConfig {
    double lambda1 = 1e-3;
    double lambda_tv_tau = 1e-3;
    double lambda_tv_beta = 1e-3;
    double rho = 1.0;
    int max_iters = 500;
    double tol_primal = 1e-6;
    double tol_dual = 1e-6;

    std::vector<double> l1_weights;
    int group_size_tau = 5;
    int group_size_beta = 3;
    int gamma_stride = 0;

    int Nt = 0;
    int Nb = 0;
};

struct NormalizationConfig {
    bool enabled = true;
};

enum class CompletionMethod {
    None,
    PCHIP,
    Akima
};

struct SpectrumCompletionConfig {
    CompletionMethod method = CompletionMethod::None;
    bool interpolate = false;
    int num_points = 100;
    bool log_space = true;
    double weight = 0.2;
};

struct FindPeaksConfig {
    bool enable = true;
    std::string interp_type = "akima";
    int smooth_window = 5;
    double peak_prominence = 0.02;
    double peak_dist_dec = 0.15;
    int interp_factor = 4;
    double weight_factor = 0.5;
};

struct PreprocessConfig {
    FindPeaksConfig find_peaks;
};

enum class BetaMethod {
    Newton,
    GridSearch
};

struct PriorsConfig {
    double beta_center = 1.0;
    double beta_sigma = 0.25;
    double beta_strength = 2.0;
};

struct ParamSelectionConfig {
    bool enable = false;
    int num_lambda1 = 20;
    double lambda1_min = 1e-4;
    double lambda1_max = 1e-1;
    int num_lambdat = 20;
    double lambdat_min = 1e-4;
    double lambdat_max = 1e-1;
    int num_lambdab = 20;
    double lambdab_min = 1e-4;
    double lambdab_max = 1e-1;
    std::string outputDir;
    int scan_max_iters = 50;
    double scan_tol = 1e-3;
    bool refine_after = true;

    // Explicitly force derived ranges even when min/max are provided.
    bool auto_lambda1_range = false;
    bool auto_lambdat_range = false;
    bool auto_lambdab_range = false;

    // Filled during parsing to preserve "user provided" precedence.
    bool has_lambda1_range = false;
    bool has_lambdat_range = false;
    bool has_lambdab_range = false;
};

struct JointConfig {
    double lambda1 = 1e-3;
    double lambda_tv = 1e-3;
    double rho = 1.0;
    int max_iter_admm = 1000;
    double tol_admm_primal = 1e-6;
    double tol_admm_dual = 1e-6;

    double beta_init = 1.0;
    double beta_min = 0.1;
    double beta_max = 2.0;
    int max_iter_beta = 50;
    double tol_beta = 1e-4;
    enum class BetaMethod { Newton, GridSearch } beta_method = BetaMethod::GridSearch;
    int beta_grid_points = 20;
};

struct LoggingConfig {
    std::string file = "logs/run.log";
    std::string level = "INFO";
    int snapshotInterval = 100;
};

struct VisualizationConfig {
    bool enabled = false;
    std::string outputDir = "results";
    double transient_tmax = 1.0;
    int transient_samples = 300;
};

struct Config {
    std::string inputFile = "examples/basic_run/input.csv";
    bool noiseWeighted = true;
    std::string spectrum_input_type = "freq";

    KernelConfig kernel;
    ADMMConfig admm;
    NormalizationConfig normalization;
    SpectrumCompletionConfig completion;
    LoggingConfig logging;
    VisualizationConfig visualization;
    ParamSelectionConfig param_selection;
    PreprocessConfig preprocess;
    PriorsConfig priors;

    std::string source_path;
    std::string source_dir;
};

class ConfigLoader {
public:
    static Config from_file(const std::string& path);
};

void validate_config(const Config& cfg);

}  // namespace trspv
