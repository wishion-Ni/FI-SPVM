#include "SolverApp.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "../lib/ADMMOptimizer.h"
#include "../lib/DictionaryGenerator.h"
#include "../lib/KernelFunction.h"
#include "../lib/Logger.h"
#include "../lib/ParamSelector.h"
#include "../lib/PeakSeedDetector.h"
#include "../lib/ResultWriter.h"
#include "../lib/Solver2D.h"
#include "../lib/SpectrumCompletion.h"
#include "../lib/Utils.h"

using namespace trspv;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

std::string format_usage() {
    return
        "Usage:\n"
        "  trspv --conf config.json [--input data.csv] [--out results/runX]\n"
        "  trspv config.json  # compatible legacy form\n";
}

LogLevel parse_log_level(const std::string& raw) {
    std::string value = raw;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "debug") return LogLevel::Debug;
    if (value == "warn" || value == "warning") return LogLevel::Warn;
    if (value == "error") return LogLevel::Error;
    return LogLevel::Info;
}

std::string normalize_path_string(const std::filesystem::path& path) {
    return path.lexically_normal().make_preferred().string();
}

bool should_derive_range(bool auto_flag, bool has_explicit_range) {
    return auto_flag || !has_explicit_range;
}

}  // namespace

int SolverApp::run(int argc, char** argv) const {
    const auto run_start = std::chrono::steady_clock::now();
    auto opts = parse_arguments(argc, argv);
    Config cfg = load_config(opts);

    ensure_output_dir(cfg.visualization.outputDir);
    ensure_output_dir(cfg.param_selection.outputDir);
    initialize_logging(cfg);

    Logger::info("SolverApp: config loaded from {}", cfg.source_path);
    Logger::info("SolverApp: input file {}", cfg.inputFile);
    Logger::info("SolverApp: output dir {}", cfg.visualization.outputDir);

    SpectrumData rawData = SpectrumDataLoader::load_csv(
        cfg.inputFile, cfg.noiseWeighted, cfg.spectrum_input_type
    );
    SpectrumData data = load_and_maybe_complete(cfg, rawData);

    std::vector<double> omega;
    Eigen::VectorXcd b = build_rhs(data, omega);

    std::vector<double> l1Weights;
    std::vector<double> taus = build_tau_grid(cfg, data, l1Weights);
    std::vector<double> betas = build_beta_grid(cfg);

    const int Nt = static_cast<int>(taus.size());
    const int Nb = static_cast<int>(betas.size());
    std::vector<double> l1w2d;
    l1w2d.reserve(static_cast<size_t>(Nt * Nb));
    auto beta_weight = [&](double beta) {
        const double z = (beta - cfg.priors.beta_center) / std::max(1e-12, cfg.priors.beta_sigma);
        const double g = std::exp(-0.5 * z * z);
        return 1.0 + cfg.priors.beta_strength * (1.0 - g);
    };
    for (int j = 0; j < Nb; ++j) {
        const double wb = beta_weight(betas[static_cast<size_t>(j)]);
        for (int i = 0; i < Nt; ++i) {
            const double wt = (i < static_cast<int>(l1Weights.size()) ? l1Weights[static_cast<size_t>(i)] : 1.0);
            l1w2d.push_back(wt * wb);
        }
    }

    DictionaryConfig dcfg = make_dictionary_config(cfg, taus, betas);
    const auto dictionary_start = std::chrono::steady_clock::now();
    Eigen::MatrixXcd A = DictionaryGenerator(dcfg).generate(omega);
    const auto dictionary_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - dictionary_start).count();

    double dlogt = 0.0;
    double dbeta = 0.0;
    Eigen::SparseMatrix<double> D2D = build_tv_operator(Nt, Nb, taus, betas, dlogt, dbeta);

    double gs = std::max(1, cfg.admm.group_size_tau) * std::max(1, cfg.admm.group_size_beta);
    if (gs > 1) {
        cfg.admm.lambda1 *= std::sqrt(gs);
    }
    cfg.admm.gamma_stride = Nt;
    cfg.admm.Nt = Nt;
    cfg.admm.Nb = Nb;

    derive_param_selection_ranges(cfg, A, b, l1w2d, Nt, Nb, dlogt, dbeta, dcfg);

    ADMMConfig strictCfg = cfg.admm;
    ADMMConfig scanCfg = make_scan_config(cfg, l1w2d);

    Solver2D solver(A, b, strictCfg, D2D);
    solver.set_scan_config(scanCfg, cfg.param_selection);
    const auto solve_start = std::chrono::steady_clock::now();
    solver.solve();
    const auto solve_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - solve_start).count();

    Eigen::VectorXcd x2d = solver.best_solution();
    auto comps = extract_components(x2d, taus, betas);
    write_outputs(cfg, data, taus, betas, x2d, A, comps, solver);

    const auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - run_start).count();
    Logger::info(
        "SolverApp: timings_ms dictionary={} solve={} total={}",
        dictionary_ms,
        solve_ms,
        total_ms);
    std::cout << "All results written to " << cfg.visualization.outputDir << std::endl;
    return 0;
}

SolverApp::CliOptions SolverApp::parse_arguments(int argc, char** argv) const {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](int& idx) -> const char* {
            if (idx + 1 < argc) return argv[++idx];
            throw std::runtime_error("missing value for argument: " + arg);
        };

        if (arg == "--conf" || arg == "-c") {
            opts.configPath = next(i);
        } else if (arg == "--input" || arg == "-i") {
            opts.overrideInput = next(i);
        } else if (arg == "--out" || arg == "-o") {
            opts.overrideOut = next(i);
        } else if (arg.size() > 5 && arg.substr(arg.size() - 5) == ".json" && i == 1) {
            opts.configPath = arg;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << format_usage();
            std::exit(0);
        }
    }
    return opts;
}

Config SolverApp::load_config(const CliOptions& opts) const {
    Config cfg = ConfigLoader::from_file(opts.configPath);
    if (!opts.overrideInput.empty()) {
        cfg.inputFile = resolve_cli_path(opts.overrideInput);
    }
    if (!opts.overrideOut.empty()) {
        const std::string previous_output = cfg.visualization.outputDir;
        cfg.visualization.outputDir = resolve_cli_path(opts.overrideOut);
        if (cfg.param_selection.outputDir == previous_output) {
            cfg.param_selection.outputDir = cfg.visualization.outputDir;
        }
    }
    return cfg;
}

void SolverApp::initialize_logging(const Config& cfg) const {
    const std::filesystem::path log_path(cfg.logging.file);
    if (log_path.has_parent_path()) {
        std::filesystem::create_directories(log_path.parent_path());
    }
    Logger::init(cfg.logging.file, parse_log_level(cfg.logging.level));
}

void SolverApp::ensure_output_dir(const std::string& dir) const {
    std::filesystem::create_directories(dir);
}

std::string SolverApp::resolve_cli_path(const std::string& raw) const {
    const std::filesystem::path path(raw);
    if (path.is_absolute()) {
        return normalize_path_string(path);
    }
    return normalize_path_string(std::filesystem::absolute(path));
}

SpectrumData SolverApp::load_and_maybe_complete(const Config& cfg, SpectrumData& rawData) const {
    SpectrumData data = rawData;
    if (!cfg.completion.interpolate) {
        return data;
    }

    SpectrumCompletion completion;
    SpectrumData interpData = completion.complete(rawData, cfg.completion);
    ResultWriter::write_interpolation_outputs(cfg, rawData, interpData);
    return interpData;
}

std::vector<double> SolverApp::build_tau_grid(
    const Config& cfg,
    const SpectrumData& data,
    std::vector<double>& l1Weights) const {
    std::vector<double> taus;
    taus.reserve(cfg.kernel.num_tau);
    const double logMin = std::log10(cfg.kernel.tau_min);
    const double logMax = std::log10(cfg.kernel.tau_max);
    for (int i = 0; i < cfg.kernel.num_tau; ++i) {
        const double t = logMin + (logMax - logMin) * i / (cfg.kernel.num_tau - 1.0);
        taus.push_back(std::pow(10.0, t));
    }

    auto pCfg = cfg.preprocess.find_peaks;
    if (pCfg.enable) {
        PeakSeedDetector::Options psOpt;
        psOpt.interp_type = pCfg.interp_type;
        psOpt.smooth_window = pCfg.smooth_window;
        psOpt.peak_prominence = pCfg.peak_prominence;
        psOpt.peak_dist_dec = pCfg.peak_dist_dec;
        psOpt.interp_factor = pCfg.interp_factor;
        PeakSeedDetector detector(psOpt);

        std::vector<double> tauSeed = detector(data.freq, data.values);

        ResultWriter::write_peak_seeds(cfg.visualization.outputDir, tauSeed);
        Logger::info("PeakDetect: found {} peaks", tauSeed.size());

        taus.insert(taus.end(), tauSeed.begin(), tauSeed.end());
        std::sort(taus.begin(), taus.end());
        taus.erase(std::unique(taus.begin(), taus.end(),
            [](double a, double b) { return std::fabs(std::log10(a / b)) < 1e-6; }), taus.end());

        l1Weights.assign(taus.size(), 1.0);
        const double wSeed = std::clamp(pCfg.weight_factor, 0.05, 1.0);
        const double sigma = 0.07;
        for (size_t i = 0; i < taus.size(); ++i) {
            double w = 1.0;
            for (double tSeed : tauSeed) {
                const double d = std::log10(tSeed / taus[i]);
                const double g = std::exp(-0.5 * (d * d) / (sigma * sigma));
                w = std::min(w, 1.0 - (1.0 - wSeed) * g);
            }
            l1Weights[i] = w;
        }
    } else {
        l1Weights.assign(taus.size(), 1.0);
    }
    return taus;
}

std::vector<double> SolverApp::build_beta_grid(const Config& cfg) const {
    std::vector<double> betas;
    betas.reserve(cfg.kernel.num_gamma);
    if (cfg.kernel.gamma_scale == "log") {
        const double logmin = std::log10(cfg.kernel.gamma_min);
        const double logmax = std::log10(cfg.kernel.gamma_max);
        for (int j = 0; j < cfg.kernel.num_gamma; ++j) {
            const double f = double(j) / (cfg.kernel.num_gamma - 1);
            betas.push_back(std::pow(10.0, logmin + f * (logmax - logmin)));
        }
    } else {
        const double gmin = cfg.kernel.gamma_min;
        const double gmax = cfg.kernel.gamma_max;
        for (int j = 0; j < cfg.kernel.num_gamma; ++j) {
            const double f = double(j) / (cfg.kernel.num_gamma - 1);
            betas.push_back(gmin + f * (gmax - gmin));
        }
    }
    return betas;
}

Eigen::VectorXcd SolverApp::build_rhs(
    const SpectrumData& data,
    std::vector<double>& omega) const {
    omega.resize(data.freq.size());
    Eigen::VectorXcd b(data.values.size());
    for (size_t i = 0; i < data.freq.size(); ++i) {
        omega[i] = 2.0 * M_PI * data.freq[i];
        b[static_cast<int>(i)] = data.values[i];
    }
    return b;
}

Eigen::SparseMatrix<double> SolverApp::build_tv_operator(
    int Nt, int Nb,
    const std::vector<double>& taus,
    const std::vector<double>& betas,
    double& dlogt,
    double& dbeta) const {
    Eigen::SparseMatrix<double> D2D = trspv::build2DTV(Nt, Nb);
    dlogt = (Nt > 1 ? std::log(taus[1] / taus[0]) : 1.0);
    dbeta = (Nb > 1 ? (betas[1] - betas[0]) : 1.0);
    return D2D;
}

ADMMConfig SolverApp::make_scan_config(
    const Config& cfg,
    const std::vector<double>& l1w2d) const {
    ADMMConfig scanCfg = cfg.admm;
    scanCfg.max_iters = cfg.param_selection.scan_max_iters;
    scanCfg.tol_primal = cfg.param_selection.scan_tol;
    scanCfg.tol_dual = cfg.param_selection.scan_tol;
    scanCfg.l1_weights = l1w2d;
    return scanCfg;
}

DictionaryConfig SolverApp::make_dictionary_config(
    const Config& cfg,
    const std::vector<double>& taus,
    const std::vector<double>& betas) const {
    DictionaryConfig dcfg;
    dcfg.tau_list = taus;
    dcfg.gamma_list = betas;
    dcfg.enable_cache = false;
    dcfg.include_constant_basis = false;
    dcfg.cache_path = normalize_path_string(std::filesystem::path(cfg.visualization.outputDir) / "cache.csv");
    return dcfg;
}

void SolverApp::derive_param_selection_ranges(
    Config& cfg,
    const Eigen::MatrixXcd& A,
    const Eigen::VectorXcd& b,
    const std::vector<double>& l1w2d,
    int Nt,
    int Nb,
    double dlogt,
    double dbeta,
    const DictionaryConfig& dcfg) const {
    if (!cfg.param_selection.enable) {
        return;
    }

    const double Lmax = compute_lambda1_max(A, b, l1w2d, Nt, Nb, dcfg, cfg);
    const double alph_min = 1e-5;
    const double alph_max = 1e-2;

    if (should_derive_range(cfg.param_selection.auto_lambda1_range, cfg.param_selection.has_lambda1_range)) {
        cfg.param_selection.lambda1_min = alph_min * Lmax;
        cfg.param_selection.lambda1_max = alph_max * Lmax;
    }
    if (should_derive_range(cfg.param_selection.auto_lambdat_range, cfg.param_selection.has_lambdat_range)) {
        cfg.param_selection.lambdat_min = 0.02 * cfg.param_selection.lambda1_min * dlogt;
        cfg.param_selection.lambdat_max = 0.08 * cfg.param_selection.lambda1_max * dlogt;
    }
    if (should_derive_range(cfg.param_selection.auto_lambdab_range, cfg.param_selection.has_lambdab_range)) {
        cfg.param_selection.lambdab_min = 0.05 * cfg.param_selection.lambda1_min * dbeta;
        cfg.param_selection.lambdab_max = 0.2 * cfg.param_selection.lambda1_max * dbeta;
    }
}

void SolverApp::write_outputs(
    const Config& cfg,
    const SpectrumData& data,
    const std::vector<double>& taus,
    const std::vector<double>& betas,
    const Eigen::VectorXcd& x2d,
    const Eigen::MatrixXcd& A,
    const std::vector<Component>& comps,
    const Solver2D& solver) const {
    ResultWriter::write_admm_summary(cfg.visualization.outputDir, solver.debug_summary(), solver.best_result());
    ResultWriter::write_components(cfg.visualization.outputDir, comps);
    ResultWriter::write_transient_outputs(cfg, comps);
    ResultWriter::write_metrics(cfg, data, taus, betas, x2d, A, comps);
}

double SolverApp::compute_lambda1_max(
    const Eigen::MatrixXcd& A,
    const Eigen::VectorXcd& b,
    const std::vector<double>& l1w2d,
    int Nt,
    int Nb,
    const DictionaryConfig& dcfg,
    const Config& cfg) const {
    const Eigen::VectorXcd c = A.adjoint() * b;
    const int Gt = std::max(1, cfg.admm.group_size_tau);
    const int Gb = std::max(1, cfg.admm.group_size_beta);
    const int stride = Nt;
    const int betaCount = Nb;
    const int Ncols = static_cast<int>(A.cols());
    const int col_offset = dcfg.include_constant_basis ? 1 : 0;

    auto group_wavg = [&](int bStart, int tStart, int bSpan, int tSpan) {
        if (l1w2d.empty()) return 1.0;
        double s = 0.0;
        int cnt = 0;
        for (int gb = 0; gb < bSpan; ++gb) {
            for (int gt = 0; gt < tSpan; ++gt) {
                int col = col_offset + (bStart + gb) * stride + (tStart + gt);
                if (static_cast<unsigned>(col) < static_cast<unsigned>(l1w2d.size())) {
                    s += l1w2d[static_cast<size_t>(col)];
                    ++cnt;
                }
            }
        }
        return cnt ? (s / cnt) : 1.0;
    };

    double Lmax = 0.0;
    for (int bStart = 0; bStart < betaCount; bStart += Gb) {
        for (int tStart = 0; tStart < stride; tStart += Gt) {
            const int bSpan = std::min(Gb, betaCount - bStart);
            const int tSpan = std::min(Gt, stride - tStart);

            double n2 = 0.0;
            for (int gb = 0; gb < bSpan; ++gb) {
                for (int gt = 0; gt < tSpan; ++gt) {
                    int col = col_offset + (bStart + gb) * stride + (tStart + gt);
                    if (col >= 0 && col < Ncols) n2 += std::norm(c[col]);
                }
            }

            const double num = std::sqrt(n2);
            const double wavg = group_wavg(bStart, tStart, bSpan, tSpan);
            const double den = std::max(1e-12, wavg);
            Lmax = std::max(Lmax, num / den);
        }
    }
    return Lmax;
}
