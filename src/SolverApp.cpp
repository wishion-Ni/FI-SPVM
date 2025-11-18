#include "SolverApp.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "../lib/ADMMOptimizer.h"
#include "../lib/DictionaryGenerator.h"
#include "../lib/KernelFunction.h"
#include "../lib/Logger.h"
#include "../lib/ParamSelector.h"
#include "../lib/PeakSeedDetector.h"
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
        "  trspv config.json  # 兼容旧用法\n";
}

}  // namespace

int SolverApp::run(int argc, char** argv) const {
    auto opts = parse_arguments(argc, argv);
    Config cfg = load_config(opts);

    ensure_output_dir(cfg.visualization.outputDir);

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
        double z = (beta - cfg.priors.beta_center) / std::max(1e-12, cfg.priors.beta_sigma);
        double g = std::exp(-0.5 * z * z);
        return 1.0 + cfg.priors.beta_strength * (1.0 - g);
    };
    for (int j = 0; j < Nb; ++j) {
        double wb = beta_weight(betas[static_cast<size_t>(j)]);
        for (int i = 0; i < Nt; ++i) {
            double wt = (i < static_cast<int>(l1Weights.size()) ? l1Weights[static_cast<size_t>(i)] : 1.0);
            l1w2d.push_back(wt * wb);
        }
    }

    DictionaryConfig dcfg;
    dcfg.tau_list = taus;
    dcfg.gamma_list = betas;
    dcfg.enable_cache = false;
    dcfg.include_constant_basis = false;
    dcfg.cache_path = "results/cache.csv";
    Eigen::MatrixXcd A = DictionaryGenerator(dcfg).generate(omega);

    double dlogt = 0.0, dbeta = 0.0;
    Eigen::SparseMatrix<double> D2D = build_tv_operator(Nt, Nb, taus, betas, dlogt, dbeta);

    double gs = std::max(1, cfg.admm.group_size_tau) * std::max(1, cfg.admm.group_size_beta);
    if (gs > 1) cfg.admm.lambda1 *= std::sqrt(gs);
    cfg.admm.gamma_stride = Nt;
    cfg.admm.Nt = Nt;
    cfg.admm.Nb = Nb;

    ADMMConfig strictCfg = cfg.admm;
    ADMMConfig scanCfg = make_scan_config(cfg, l1w2d);

    double Lmax = compute_lambda1_max(A, b, l1w2d, Nt, Nb, dcfg, cfg);
    double alph_min = 1e-5, alph_max = 1e-2;
    cfg.param_selection.lambda1_min = alph_min * Lmax;
    cfg.param_selection.lambda1_max = alph_max * Lmax;
    cfg.param_selection.lambdat_min = 0.02 * cfg.param_selection.lambda1_min * dlogt;
    cfg.param_selection.lambdat_max = 0.08 * cfg.param_selection.lambda1_max * dlogt;
    cfg.param_selection.lambdab_min = 0.05 * cfg.param_selection.lambda1_min * dbeta;
    cfg.param_selection.lambdab_max = 0.2 * cfg.param_selection.lambda1_max * dbeta;

    Solver2D solver(A, b, strictCfg, D2D);
    solver.set_scan_config(scanCfg, cfg.param_selection);
    solver.solve();

    Eigen::VectorXcd x2d = solver.best_solution();
    ParamSelectionResult best = solver.best_result();

    std::ofstream admmOut(cfg.visualization.outputDir + "/admm_summary.txt");
    admmOut << solver.debug_summary();
    admmOut << "\nBest parameters: lambda1=" << best.lambda1
            << ", lambda_tv_tau=" << best.lambda_tv_tau
            << ", lambda_tv_beta=" << best.lambda_tv_beta << '\n';

    auto comps = extract_components(x2d, taus, betas);

    {
        std::ofstream ofs(cfg.visualization.outputDir + "/components.txt");
        ofs << "# tau(s), beta, amp, prominence, it, jb\n";
        for (size_t i = 0; i < comps.size(); ++i) {
            const auto& c = comps[i];
            ofs << std::setprecision(12)
                << c.tau << ','.
                << c.beta << ','.
                << c.amp << ','.
                << c.prominence << ','
                << c.it << ','
                << c.jb << '\n';
        }
    }

    write_transient_outputs(cfg, comps);
    write_metrics(cfg, data, taus, betas, x2d, A, comps);

    std::cout << "All results written to " << cfg.visualization.outputDir << std::endl;
    return 0;
}

SolverApp::CliOptions SolverApp::parse_arguments(int argc, char** argv) const {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](int& idx) -> const char* {
            if (idx + 1 < argc) return argv[++idx];
            throw std::runtime_error("缺少参数值: " + arg);
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
    if (!opts.overrideInput.empty()) cfg.inputFile = opts.overrideInput;
    if (!opts.overrideOut.empty()) cfg.visualization.outputDir = opts.overrideOut;
    return cfg;
}

void SolverApp::ensure_output_dir(const std::string& dir) const {
    std::filesystem::create_directories(dir);
}

SpectrumData SolverApp::load_and_maybe_complete(const Config& cfg, SpectrumData& rawData) const {
    SpectrumData data = rawData;
    if (!cfg.completion.interpolate) return data;

    SpectrumCompletion completion;
    SpectrumData interpData = completion.complete(rawData, cfg.completion);
    write_interpolation_outputs(cfg, rawData, interpData);
    return interpData;
}

void SolverApp::write_interpolation_outputs(
    const Config& cfg,
    const SpectrumData& rawData,
    const SpectrumData& interpData) const {
    {
        std::ofstream ofs(cfg.visualization.outputDir + "/original_data.csv");
        ofs << "freq,real,imag,weight\n";
        for (size_t i = 0; i < rawData.freq.size(); ++i) {
            ofs << std::setprecision(12) << rawData.freq[i] << ','
                << rawData.values[i].real() << ','
                << rawData.values[i].imag() << ','
                << (i < rawData.weights.size() ? rawData.weights[i] : 1.0) << '\n';
        }
    }
    {
        std::ofstream ofs(cfg.visualization.outputDir + "/interpolated_data.csv");
        ofs << "freq,real,imag\n";
        for (size_t i = 0; i < interpData.freq.size(); ++i) {
            ofs << std::setprecision(12) << interpData.freq[i] << ','
                << interpData.values[i].real() << ','
                << interpData.values[i].imag() << '\n';
        }
    }
    {
        auto nearest = [&](double f) {
            size_t jbest = 0;
            double dmin = std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < interpData.freq.size(); ++j) {
                double d = std::abs(interpData.freq[j] - f);
                if (d < dmin) {
                    dmin = d;
                    jbest = j;
                }
            }
            return jbest;
        };
        std::ofstream ofs(cfg.visualization.outputDir + "/interpolation_vs_original.csv");
        ofs << "freq_raw,raw_real,raw_imag,freq_interp,interp_real,interp_imag,abs_error\n";
        for (size_t i = 0; i < rawData.freq.size(); ++i) {
            size_t j = nearest(rawData.freq[i]);
            std::complex<double> e = interpData.values[j] - rawData.values[i];
            ofs << std::setprecision(12)
                << rawData.freq[i] << ','
                << rawData.values[i].real() << ','
                << rawData.values[i].imag() << ','
                << interpData.freq[j] << ','
                << interpData.values[j].real() << ','
                << interpData.values[j].imag() << ','
                << std::abs(e) << '\n';
        }
    }
}

std::vector<double> SolverApp::build_tau_grid(
    const Config& cfg,
    const SpectrumData& data,
    std::vector<double>& l1Weights) const {
    std::vector<double> taus;
    taus.reserve(cfg.kernel.num_tau);
    double logMin = std::log10(cfg.kernel.tau_min);
    double logMax = std::log10(cfg.kernel.tau_max);
    for (int i = 0; i < cfg.kernel.num_tau; ++i) {
        double t = logMin + (logMax - logMin) * i / (cfg.kernel.num_tau - 1.0);
        taus.push_back(std::pow(10.0, t));
    }

    std::vector<int> peakIdx;
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

        std::ofstream peakOut(cfg.visualization.outputDir + "/detected_peaks.txt");
        peakOut << "# Detected " << tauSeed.size() << " peaks (tau in s) \n";
        for (double t : tauSeed) peakOut << std::setprecision(12) << t << '\n';
        std::cout << "[PeakDetect] found " << tauSeed.size() << " peak(s)." << std::endl;

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
                double d = std::log10(tSeed / taus[i]);
                double g = std::exp(-0.5 * (d * d) / (sigma * sigma));
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
        double logmin = std::log10(cfg.kernel.gamma_min);
        double logmax = std::log10(cfg.kernel.gamma_max);
        for (int j = 0; j < cfg.kernel.num_gamma; ++j) {
            double f = double(j) / (cfg.kernel.num_gamma - 1);
            betas.push_back(std::pow(10.0, logmin + f * (logmax - logmin)));
        }
    } else {
        double gmin = cfg.kernel.gamma_min;
        double gmax = cfg.kernel.gamma_max;
        for (int j = 0; j < cfg.kernel.num_gamma; ++j) {
            double f = double(j) / (cfg.kernel.num_gamma - 1);
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
                if ((unsigned)col < (unsigned)l1w2d.size()) {
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

void SolverApp::write_transient_outputs(
    const Config& cfg,
    const std::vector<Component>& comps) const {
    const double tmax = cfg.visualization.transient_tmax;
    const int    num_samples = cfg.visualization.transient_samples;
    std::vector<double> ts;
    ts.reserve(static_cast<size_t>(num_samples));
    for (int i = 0; i < num_samples; ++i) {
        ts.push_back(tmax * i / double(num_samples - 1));
    }

    {
        std::ofstream onTot(cfg.visualization.outputDir + "/transient_on_total.csv");
        std::ofstream offTot(cfg.visualization.outputDir + "/transient_off_total.csv");
        onTot << "t,SPV\n";
        offTot << "t,SPV\n";
        for (double t : ts) {
            double y_on = 0, y_off = 0;
            for (auto& c : comps) {
                y_on += c.amp * h_on(t, c.tau, c.beta);
                y_off += c.amp * h_off(t, c.tau, c.beta);
            }
            onTot << std::setprecision(12) << t << ',' << y_on << '\n';
            offTot << std::setprecision(12) << t << ',' << y_off << '\n';
        }
    }

    for (size_t k = 0; k < comps.size(); ++k) {
        std::ofstream onK(cfg.visualization.outputDir + "/transient_on_comp_" + std::to_string(k + 1) + ".csv");
        std::ofstream offK(cfg.visualization.outputDir + "/transient_off_comp_" + std::to_string(k + 1) + ".csv");
        onK << "t,SPV\n";
        offK << "t,SPV\n";
        for (double t : ts) {
            double y_on = comps[k].amp * h_on(t, comps[k].tau, comps[k].beta);
            double y_off = comps[k].amp * h_off(t, comps[k].tau, comps[k].beta);
            onK << std::setprecision(12) << t << ',' << y_on << '\n';
            offK << std::setprecision(12) << t << ',' << y_off << '\n';
        }
    }
}

void SolverApp::write_metrics(
    const Config& cfg,
    const SpectrumData& data,
    const std::vector<double>& taus,
    const std::vector<double>& betas,
    const Eigen::VectorXcd& x2d,
    const Eigen::MatrixXcd& A,
    const std::vector<Component>& comps) const {
    Eigen::VectorXcd b_pred = A * x2d;
    const size_t N = data.freq.size();
    auto w_of = [&](size_t i) { return (i < data.weights.size() ? data.weights[i] : 1.0); };

    double rss_w = 0.0, wsum = 0.0;
    double rss_r = 0.0, rss_i = 0.0, tss_r = 0.0, tss_i = 0.0;
    double mean_r = 0.0, mean_i = 0.0, wsum_mean = 0.0;

    for (size_t i = 0; i < N; ++i) {
        double w = w_of(i);
        mean_r += w * data.values[i].real();
        mean_i += w * data.values[i].imag();
        wsum_mean += w;
    }
    mean_r /= std::max(1e-16, wsum_mean);
    mean_i /= std::max(1e-16, wsum_mean);

    for (size_t i = 0; i < N; ++i) {
        double w = w_of(i);
        auto e = b_pred[static_cast<int>(i)] - data.values[i];
        rss_w += w * std::norm(e);
        wsum += w;

        rss_r += w * std::pow(b_pred[static_cast<int>(i)].real() - data.values[i].real(), 2);
        rss_i += w * std::pow(b_pred[static_cast<int>(i)].imag() - data.values[i].imag(), 2);
        tss_r += w * std::pow(data.values[i].real() - mean_r, 2);
        tss_i += w * std::pow(data.values[i].imag() - mean_i, 2);
    }
    double wrmse = std::sqrt(rss_w / std::max(1e-16, wsum));
    double R2_real = 1.0 - rss_r / std::max(1e-16, tss_r);
    double R2_imag = 1.0 - rss_i / std::max(1e-16, tss_i);

    int nz = 0;
    double thr_nz = 1e-6 * x2d.cwiseAbs().maxCoeff();
    for (int k = 0; k < x2d.size(); ++k) if (std::abs(x2d[k]) > thr_nz) ++nz;
    int nobs = int(2 * N);
    int dof = std::max(1, nobs - nz);

    double chi2_red = (rss_w / std::max(1e-16, wsum)) * (nobs / double(dof));
    double RSS = 0.0; for (size_t i = 0; i < N; ++i) RSS += std::norm(b_pred[static_cast<int>(i)] - data.values[i]);
    double AIC = 2.0 * nz + nobs * std::log(std::max(1e-300, RSS / nobs));
    double BIC = nz * std::log(std::max(1, nobs)) + nobs * std::log(std::max(1e-300, RSS / nobs));

    using json = nlohmann::json;
    json mj;
    mj["weighted_rmse"] = wrmse;
    mj["R2_real"] = R2_real;
    mj["R2_imag"] = R2_imag;
    mj["chi2_reduced"] = chi2_red;
    mj["nz_coeffs"] = nz;
    mj["AIC"] = AIC;
    mj["BIC"] = BIC;
    std::ofstream jm(cfg.visualization.outputDir + "/metrics.json");
    jm << mj.dump(2) << std::endl;

    json sj;
    sj["num_points"] = N;
    sj["tau_range"] = { *std::min_element(taus.begin(),taus.end()),
                        *std::max_element(taus.begin(),taus.end()) };
    sj["beta_range"] = { *std::min_element(betas.begin(),betas.end()),
                        *std::max_element(betas.begin(),betas.end()) };
    sj["metrics"] = mj;
    auto& arr = sj["components"] = json::array();
    for (size_t k = 0; k < comps.size(); ++k) {
        json c;
        c["id"] = int(k + 1);
        c["tau"] = comps[k].tau;
        c["log10_tau"] = std::log10(comps[k].tau);
        c["beta"] = comps[k].beta;
        c["amp"] = comps[k].amp;
        c["prominence"] = comps[k].prominence;
        c["it"] = comps[k].it;
        c["jb"] = comps[k].jb;
        arr.push_back(c);
    }
    std::ofstream js(cfg.visualization.outputDir + "/summary.json");
    js << sj.dump(2) << std::endl;
}

