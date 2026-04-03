#include "Config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string as_string(double value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

std::string as_string(int value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

trspv::CompletionMethod parse_completion_method(const std::string& raw) {
    const std::string value = to_lower(raw);
    if (value == "pchip") return trspv::CompletionMethod::PCHIP;
    if (value == "akima") return trspv::CompletionMethod::Akima;
    if (value == "none") return trspv::CompletionMethod::None;
    throw std::runtime_error("invalid spectrum_completion.method: " + raw);
}

void require_positive(double value, const char* field) {
    if (value <= 0.0) {
        throw std::runtime_error(std::string(field) + " must be > 0, got " + as_string(value));
    }
}

void require_non_negative(double value, const char* field) {
    if (value < 0.0) {
        throw std::runtime_error(std::string(field) + " must be >= 0, got " + as_string(value));
    }
}

void require_at_least(int value, int minimum, const char* field) {
    if (value < minimum) {
        throw std::runtime_error(
            std::string(field) + " must be >= " + as_string(minimum) + ", got " + as_string(value));
    }
}

void require_less(double lhs, double rhs, const char* lhs_name, const char* rhs_name) {
    if (!(lhs < rhs)) {
        throw std::runtime_error(
            std::string(lhs_name) + " must be < " + rhs_name + ", got " +
            as_string(lhs) + " and " + as_string(rhs));
    }
}

}  // namespace

namespace trspv {

void from_json(const json& j, KernelConfig& cfg) {
    cfg.tau_min = j.value("tau_min", cfg.tau_min);
    cfg.tau_max = j.value("tau_max", cfg.tau_max);
    cfg.num_tau = j.value("num_tau", cfg.num_tau);
    cfg.gamma_min = j.value("gamma_min", cfg.gamma_min);
    cfg.gamma_max = j.value("gamma_max", cfg.gamma_max);
    cfg.num_gamma = j.value("num_gamma", cfg.num_gamma);
    cfg.gamma_scale = j.value("gamma_scale", cfg.gamma_scale);
}

void from_json(const json& j, ADMMConfig& cfg) {
    cfg.lambda1 = j.value("lambda1", cfg.lambda1);
    cfg.lambda_tv_tau = j.value("lambda_tv_tau", cfg.lambda_tv_tau);
    cfg.lambda_tv_beta = j.value("lambda_tv_beta", cfg.lambda_tv_beta);
    cfg.rho = j.value("rho", cfg.rho);
    cfg.max_iters = j.value("max_iters", cfg.max_iters);
    cfg.tol_primal = j.value("tol_primal", cfg.tol_primal);
    cfg.tol_dual = j.value("tol_dual", cfg.tol_dual);
    cfg.group_size_tau = j.value("group_size_tau", cfg.group_size_tau);
    cfg.group_size_beta = j.value("group_size_beta", cfg.group_size_beta);
    cfg.gamma_stride = j.value("gamma_stride", cfg.gamma_stride);
    cfg.Nt = j.value("Nt", cfg.Nt);
    cfg.Nb = j.value("Nb", cfg.Nb);
}

void from_json(const json& j, NormalizationConfig& cfg) {
    cfg.enabled = j.value("enabled", cfg.enabled);
}

void from_json(const json& j, SpectrumCompletionConfig& cfg) {
    if (j.contains("method")) {
        cfg.method = parse_completion_method(j.at("method").get<std::string>());
    }
    cfg.interpolate = j.value("interpolate", cfg.interpolate);
    cfg.num_points = j.value("num_points", cfg.num_points);
    cfg.log_space = j.value("log_space", cfg.log_space);
    cfg.weight = j.value("weight", cfg.weight);
}

void from_json(const json& j, FindPeaksConfig& cfg) {
    cfg.enable = j.value("enable", cfg.enable);
    cfg.interp_type = j.value("interp_type", cfg.interp_type);
    cfg.smooth_window = j.value("smooth_window", cfg.smooth_window);
    cfg.peak_prominence = j.value("peak_prominence", cfg.peak_prominence);
    cfg.peak_dist_dec = j.value("peak_dist_dec", cfg.peak_dist_dec);
    cfg.interp_factor = j.value("interp_factor", cfg.interp_factor);
    cfg.weight_factor = j.value("weight_factor", cfg.weight_factor);
}

void from_json(const json& j, PreprocessConfig& cfg) {
    if (j.contains("find_peaks")) {
        j.at("find_peaks").get_to(cfg.find_peaks);
    }
}

void from_json(const json& j, PriorsConfig& cfg) {
    cfg.beta_center = j.value("beta_center", cfg.beta_center);
    cfg.beta_sigma = j.value("beta_sigma", cfg.beta_sigma);
    cfg.beta_strength = j.value("beta_strength", cfg.beta_strength);
}

void from_json(const json& j, LoggingConfig& cfg) {
    cfg.file = j.value("file", cfg.file);
    cfg.level = j.value("level", cfg.level);
    cfg.snapshotInterval = j.value("snapshotInterval", cfg.snapshotInterval);
}

void from_json(const json& j, VisualizationConfig& cfg) {
    cfg.enabled = j.value("enabled", cfg.enabled);
    cfg.outputDir = j.value("outputDir", cfg.outputDir);
    cfg.transient_tmax = j.value("transient_tmax", cfg.transient_tmax);
    cfg.transient_samples = j.value("transient_samples", cfg.transient_samples);
}

void from_json(const json& j, ParamSelectionConfig& cfg) {
    cfg.enable = j.value("enable", cfg.enable);
    cfg.num_lambda1 = j.value("num_lambda1", cfg.num_lambda1);
    cfg.lambda1_min = j.value("lambda1_min", cfg.lambda1_min);
    cfg.lambda1_max = j.value("lambda1_max", cfg.lambda1_max);
    cfg.num_lambdat = j.value("num_lambdat", cfg.num_lambdat);
    cfg.lambdat_min = j.value("lambdat_min", cfg.lambdat_min);
    cfg.lambdat_max = j.value("lambdat_max", cfg.lambdat_max);
    cfg.num_lambdab = j.value("num_lambdab", cfg.num_lambdab);
    cfg.lambdab_min = j.value("lambdab_min", cfg.lambdab_min);
    cfg.lambdab_max = j.value("lambdab_max", cfg.lambdab_max);
    cfg.outputDir = j.value("outputDir", cfg.outputDir);
    cfg.scan_max_iters = j.value("scan_max_iters", cfg.scan_max_iters);
    cfg.scan_tol = j.value("scan_tol", cfg.scan_tol);
    cfg.refine_after = j.value("refine_after", cfg.refine_after);
}

void from_json(const json& j, Config& cfg) {
    if (j.contains("data")) {
        const auto& data = j.at("data");
        cfg.inputFile = data.value("input_file", cfg.inputFile);
        cfg.noiseWeighted = data.value("noise_weighted", cfg.noiseWeighted);
        cfg.spectrum_input_type = data.value("input_type", cfg.spectrum_input_type);
    }
    if (j.contains("kernel")) j.at("kernel").get_to(cfg.kernel);
    if (j.contains("admm")) j.at("admm").get_to(cfg.admm);
    if (j.contains("normalization")) j.at("normalization").get_to(cfg.normalization);
    if (j.contains("spectrum_completion")) j.at("spectrum_completion").get_to(cfg.completion);
    if (j.contains("logging")) j.at("logging").get_to(cfg.logging);
    if (j.contains("visualization")) j.at("visualization").get_to(cfg.visualization);
    if (j.contains("param_selection")) j.at("param_selection").get_to(cfg.param_selection);
    if (j.contains("preprocess")) j.at("preprocess").get_to(cfg.preprocess);
    if (j.contains("priors")) j.at("priors").get_to(cfg.priors);

    if (cfg.param_selection.outputDir.empty()) {
        cfg.param_selection.outputDir = cfg.visualization.outputDir;
    }
}

void validate_config(const Config& cfg) {
    if (cfg.inputFile.empty()) {
        throw std::runtime_error("data.input_file must not be empty");
    }

    require_positive(cfg.kernel.tau_min, "kernel.tau_min");
    require_positive(cfg.kernel.tau_max, "kernel.tau_max");
    require_less(cfg.kernel.tau_min, cfg.kernel.tau_max, "kernel.tau_min", "kernel.tau_max");
    require_at_least(cfg.kernel.num_tau, 2, "kernel.num_tau");
    require_positive(cfg.kernel.gamma_min, "kernel.gamma_min");
    require_positive(cfg.kernel.gamma_max, "kernel.gamma_max");
    require_less(cfg.kernel.gamma_min, cfg.kernel.gamma_max, "kernel.gamma_min", "kernel.gamma_max");
    require_at_least(cfg.kernel.num_gamma, 2, "kernel.num_gamma");

    const std::string gamma_scale = to_lower(cfg.kernel.gamma_scale);
    if (gamma_scale != "linear" && gamma_scale != "log") {
        throw std::runtime_error("kernel.gamma_scale must be one of: linear, log; got " + cfg.kernel.gamma_scale);
    }

    require_non_negative(cfg.admm.lambda1, "admm.lambda1");
    require_non_negative(cfg.admm.lambda_tv_tau, "admm.lambda_tv_tau");
    require_non_negative(cfg.admm.lambda_tv_beta, "admm.lambda_tv_beta");
    require_positive(cfg.admm.rho, "admm.rho");
    require_at_least(cfg.admm.max_iters, 1, "admm.max_iters");
    require_non_negative(cfg.admm.tol_primal, "admm.tol_primal");
    require_non_negative(cfg.admm.tol_dual, "admm.tol_dual");
    require_at_least(cfg.admm.group_size_tau, 1, "admm.group_size_tau");
    require_at_least(cfg.admm.group_size_beta, 1, "admm.group_size_beta");

    require_at_least(cfg.completion.num_points, 2, "spectrum_completion.num_points");
    require_non_negative(cfg.completion.weight, "spectrum_completion.weight");

    const std::string interp_type = to_lower(cfg.preprocess.find_peaks.interp_type);
    if (interp_type != "akima" && interp_type != "pchip") {
        throw std::runtime_error(
            "preprocess.find_peaks.interp_type must be one of: akima, pchip; got " +
            cfg.preprocess.find_peaks.interp_type);
    }
    require_non_negative(cfg.preprocess.find_peaks.peak_prominence, "preprocess.find_peaks.peak_prominence");
    require_non_negative(cfg.preprocess.find_peaks.peak_dist_dec, "preprocess.find_peaks.peak_dist_dec");
    require_at_least(cfg.preprocess.find_peaks.interp_factor, 1, "preprocess.find_peaks.interp_factor");
    require_at_least(cfg.preprocess.find_peaks.smooth_window, 0, "preprocess.find_peaks.smooth_window");
    require_positive(cfg.preprocess.find_peaks.weight_factor, "preprocess.find_peaks.weight_factor");

    require_positive(cfg.priors.beta_sigma, "priors.beta_sigma");
    require_non_negative(cfg.priors.beta_strength, "priors.beta_strength");

    if (cfg.visualization.outputDir.empty()) {
        throw std::runtime_error("visualization.outputDir must not be empty");
    }
    require_positive(cfg.visualization.transient_tmax, "visualization.transient_tmax");
    require_at_least(cfg.visualization.transient_samples, 2, "visualization.transient_samples");

    if (cfg.param_selection.enable) {
        require_at_least(cfg.param_selection.num_lambda1, 1, "param_selection.num_lambda1");
        require_at_least(cfg.param_selection.num_lambdat, 1, "param_selection.num_lambdat");
        require_at_least(cfg.param_selection.num_lambdab, 1, "param_selection.num_lambdab");
        require_non_negative(cfg.param_selection.lambda1_min, "param_selection.lambda1_min");
        require_non_negative(cfg.param_selection.lambda1_max, "param_selection.lambda1_max");
        require_non_negative(cfg.param_selection.lambdat_min, "param_selection.lambdat_min");
        require_non_negative(cfg.param_selection.lambdat_max, "param_selection.lambdat_max");
        require_non_negative(cfg.param_selection.lambdab_min, "param_selection.lambdab_min");
        require_non_negative(cfg.param_selection.lambdab_max, "param_selection.lambdab_max");
        require_less(
            cfg.param_selection.lambda1_min,
            cfg.param_selection.lambda1_max,
            "param_selection.lambda1_min",
            "param_selection.lambda1_max");
        require_less(
            cfg.param_selection.lambdat_min,
            cfg.param_selection.lambdat_max,
            "param_selection.lambdat_min",
            "param_selection.lambdat_max");
        require_less(
            cfg.param_selection.lambdab_min,
            cfg.param_selection.lambdab_max,
            "param_selection.lambdab_min",
            "param_selection.lambdab_max");
        require_at_least(cfg.param_selection.scan_max_iters, 1, "param_selection.scan_max_iters");
        require_non_negative(cfg.param_selection.scan_tol, "param_selection.scan_tol");
    }
}

Config ConfigLoader::from_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("unable to open config file: " + path);
    }

    json root;
    try {
        ifs >> root;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("failed to parse config JSON: ") + e.what());
    }

    json normalized = root;
    bool used_legacy_mapping = false;

    if (root.contains("inputFile")) {
        normalized["data"]["input_file"] = root.at("inputFile");
        used_legacy_mapping = true;
    }
    if (root.contains("noiseWeighted")) {
        normalized["data"]["noise_weighted"] = root.at("noiseWeighted");
        used_legacy_mapping = true;
    }
    if (root.contains("spectrum_input_type")) {
        normalized["data"]["input_type"] = root.at("spectrum_input_type");
        used_legacy_mapping = true;
    }

    Config cfg;
    try {
        cfg = normalized.get<Config>();
    } catch (const json::exception& e) {
        throw std::runtime_error(std::string("failed to load config fields: ") + e.what());
    }

    if (used_legacy_mapping) {
        std::cerr << "[Config] warning: using legacy top-level keys; prefer the nested data.* layout.\n";
    }

    validate_config(cfg);
    return cfg;
}

}  // namespace trspv
