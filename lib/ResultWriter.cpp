#include "ResultWriter.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

#include <nlohmann/json.hpp>

namespace trspv {

namespace {

std::filesystem::path join_output(const std::string& output_dir, const char* filename) {
    return std::filesystem::path(output_dir) / filename;
}

}  // namespace

void ResultWriter::write_admm_summary(
    const std::string& output_dir,
    const std::string& debug_summary,
    const ParamSelectionResult& best) {
    std::ofstream admm_out(join_output(output_dir, "admm_summary.txt"));
    admm_out << debug_summary;
    admm_out << "\nBest parameters: lambda1=" << best.lambda1
             << ", lambda_tv_tau=" << best.lambda_tv_tau
             << ", lambda_tv_beta=" << best.lambda_tv_beta << '\n';
}

void ResultWriter::write_components(
    const std::string& output_dir,
    const std::vector<Component>& comps) {
    std::ofstream ofs(join_output(output_dir, "components.txt"));
    ofs << "# tau(s), beta, amp, prominence, it, jb\n";
    for (const auto& c : comps) {
        ofs << std::setprecision(12)
            << c.tau << ','
            << c.beta << ','
            << c.amp << ','
            << c.prominence << ','
            << c.it << ','
            << c.jb << '\n';
    }
}

void ResultWriter::write_peak_seeds(
    const std::string& output_dir,
    const std::vector<double>& tau_seed) {
    std::ofstream peak_out(join_output(output_dir, "detected_peaks.txt"));
    peak_out << "# Detected " << tau_seed.size() << " peaks (tau in s)\n";
    for (double t : tau_seed) {
        peak_out << std::setprecision(12) << t << '\n';
    }
}

void ResultWriter::write_interpolation_outputs(
    const Config& cfg,
    const SpectrumData& raw_data,
    const SpectrumData& interp_data) {
    {
        std::ofstream ofs(join_output(cfg.visualization.outputDir, "original_data.csv"));
        ofs << "freq,real,imag,weight\n";
        for (size_t i = 0; i < raw_data.freq.size(); ++i) {
            ofs << std::setprecision(12) << raw_data.freq[i] << ','
                << raw_data.values[i].real() << ','
                << raw_data.values[i].imag() << ','
                << (i < raw_data.weights.size() ? raw_data.weights[i] : 1.0) << '\n';
        }
    }
    {
        std::ofstream ofs(join_output(cfg.visualization.outputDir, "interpolated_data.csv"));
        ofs << "freq,real,imag\n";
        for (size_t i = 0; i < interp_data.freq.size(); ++i) {
            ofs << std::setprecision(12) << interp_data.freq[i] << ','
                << interp_data.values[i].real() << ','
                << interp_data.values[i].imag() << '\n';
        }
    }
    {
        auto nearest = [&](double f) {
            size_t jbest = 0;
            double dmin = std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < interp_data.freq.size(); ++j) {
                const double d = std::abs(interp_data.freq[j] - f);
                if (d < dmin) {
                    dmin = d;
                    jbest = j;
                }
            }
            return jbest;
        };

        std::ofstream ofs(join_output(cfg.visualization.outputDir, "interpolation_vs_original.csv"));
        ofs << "freq_raw,raw_real,raw_imag,freq_interp,interp_real,interp_imag,abs_error\n";
        for (size_t i = 0; i < raw_data.freq.size(); ++i) {
            const size_t j = nearest(raw_data.freq[i]);
            const std::complex<double> e = interp_data.values[j] - raw_data.values[i];
            ofs << std::setprecision(12)
                << raw_data.freq[i] << ','
                << raw_data.values[i].real() << ','
                << raw_data.values[i].imag() << ','
                << interp_data.freq[j] << ','
                << interp_data.values[j].real() << ','
                << interp_data.values[j].imag() << ','
                << std::abs(e) << '\n';
        }
    }
}

void ResultWriter::write_transient_outputs(
    const Config& cfg,
    const std::vector<Component>& comps) {
    const double tmax = cfg.visualization.transient_tmax;
    const int num_samples = cfg.visualization.transient_samples;
    std::vector<double> ts;
    ts.reserve(static_cast<size_t>(num_samples));
    for (int i = 0; i < num_samples; ++i) {
        ts.push_back(tmax * i / double(num_samples - 1));
    }

    {
        std::ofstream on_tot(join_output(cfg.visualization.outputDir, "transient_on_total.csv"));
        std::ofstream off_tot(join_output(cfg.visualization.outputDir, "transient_off_total.csv"));
        on_tot << "t,SPV\n";
        off_tot << "t,SPV\n";
        for (double t : ts) {
            double y_on = 0.0;
            double y_off = 0.0;
            for (const auto& c : comps) {
                y_on += c.amp * h_on(t, c.tau, c.beta);
                y_off += c.amp * h_off(t, c.tau, c.beta);
            }
            on_tot << std::setprecision(12) << t << ',' << y_on << '\n';
            off_tot << std::setprecision(12) << t << ',' << y_off << '\n';
        }
    }

    for (size_t k = 0; k < comps.size(); ++k) {
        std::ofstream on_k(std::filesystem::path(cfg.visualization.outputDir) /
                           ("transient_on_comp_" + std::to_string(k + 1) + ".csv"));
        std::ofstream off_k(std::filesystem::path(cfg.visualization.outputDir) /
                            ("transient_off_comp_" + std::to_string(k + 1) + ".csv"));
        on_k << "t,SPV\n";
        off_k << "t,SPV\n";
        for (double t : ts) {
            const double y_on = comps[k].amp * h_on(t, comps[k].tau, comps[k].beta);
            const double y_off = comps[k].amp * h_off(t, comps[k].tau, comps[k].beta);
            on_k << std::setprecision(12) << t << ',' << y_on << '\n';
            off_k << std::setprecision(12) << t << ',' << y_off << '\n';
        }
    }
}

void ResultWriter::write_metrics(
    const Config& cfg,
    const SpectrumData& data,
    const std::vector<double>& taus,
    const std::vector<double>& betas,
    const Eigen::VectorXcd& x2d,
    const Eigen::MatrixXcd& A,
    const std::vector<Component>& comps) {
    Eigen::VectorXcd b_pred = A * x2d;
    const size_t N = data.freq.size();
    auto w_of = [&](size_t i) { return (i < data.weights.size() ? data.weights[i] : 1.0); };

    double rss_w = 0.0;
    double wsum = 0.0;
    double rss_r = 0.0;
    double rss_i = 0.0;
    double tss_r = 0.0;
    double tss_i = 0.0;
    double mean_r = 0.0;
    double mean_i = 0.0;
    double wsum_mean = 0.0;

    for (size_t i = 0; i < N; ++i) {
        const double w = w_of(i);
        mean_r += w * data.values[i].real();
        mean_i += w * data.values[i].imag();
        wsum_mean += w;
    }
    mean_r /= std::max(1e-16, wsum_mean);
    mean_i /= std::max(1e-16, wsum_mean);

    for (size_t i = 0; i < N; ++i) {
        const double w = w_of(i);
        const auto e = b_pred[static_cast<int>(i)] - data.values[i];
        rss_w += w * std::norm(e);
        wsum += w;
        rss_r += w * std::pow(b_pred[static_cast<int>(i)].real() - data.values[i].real(), 2);
        rss_i += w * std::pow(b_pred[static_cast<int>(i)].imag() - data.values[i].imag(), 2);
        tss_r += w * std::pow(data.values[i].real() - mean_r, 2);
        tss_i += w * std::pow(data.values[i].imag() - mean_i, 2);
    }

    const double wrmse = std::sqrt(rss_w / std::max(1e-16, wsum));
    const double r2_real = 1.0 - rss_r / std::max(1e-16, tss_r);
    const double r2_imag = 1.0 - rss_i / std::max(1e-16, tss_i);

    int nz = 0;
    const double thr_nz = 1e-6 * x2d.cwiseAbs().maxCoeff();
    for (int k = 0; k < x2d.size(); ++k) {
        if (std::abs(x2d[k]) > thr_nz) {
            ++nz;
        }
    }
    const int nobs = int(2 * N);
    const int dof = std::max(1, nobs - nz);

    const double chi2_red = (rss_w / std::max(1e-16, wsum)) * (nobs / double(dof));
    double rss = 0.0;
    for (size_t i = 0; i < N; ++i) {
        rss += std::norm(b_pred[static_cast<int>(i)] - data.values[i]);
    }
    const double aic = 2.0 * nz + nobs * std::log(std::max(1e-300, rss / nobs));
    const double bic = nz * std::log(std::max(1, nobs)) + nobs * std::log(std::max(1e-300, rss / nobs));

    using json = nlohmann::json;
    json metrics;
    metrics["weighted_rmse"] = wrmse;
    metrics["R2_real"] = r2_real;
    metrics["R2_imag"] = r2_imag;
    metrics["chi2_reduced"] = chi2_red;
    metrics["nz_coeffs"] = nz;
    metrics["AIC"] = aic;
    metrics["BIC"] = bic;

    std::ofstream metrics_out(join_output(cfg.visualization.outputDir, "metrics.json"));
    metrics_out << metrics.dump(2) << std::endl;

    json summary;
    summary["num_points"] = N;
    summary["tau_range"] = {*std::min_element(taus.begin(), taus.end()), *std::max_element(taus.begin(), taus.end())};
    summary["beta_range"] = {*std::min_element(betas.begin(), betas.end()), *std::max_element(betas.begin(), betas.end())};
    summary["metrics"] = metrics;
    auto& arr = summary["components"] = json::array();
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

    std::ofstream summary_out(join_output(cfg.visualization.outputDir, "summary.json"));
    summary_out << summary.dump(2) << std::endl;
}

}  // namespace trspv
