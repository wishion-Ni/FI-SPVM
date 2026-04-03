#include <array>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../src/SolverApp.h"

namespace {

bool exists_and_nonempty(const std::filesystem::path& path) {
    return std::filesystem::exists(path) && std::filesystem::file_size(path) > 0;
}

nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    nlohmann::json j;
    ifs >> j;
    return j;
}

int run_with_config(const std::filesystem::path& source_dir,
                    const std::filesystem::path& config_path,
                    const std::filesystem::path& out_dir,
                    const std::filesystem::path& cwd) {
    std::vector<std::string> args_store = {
        "trspv_cli",
        "--conf",
        config_path.string(),
        "--out",
        out_dir.string()
    };
    std::vector<char*> argv;
    argv.reserve(args_store.size());
    for (auto& arg : args_store) {
        argv.push_back(arg.data());
    }

    SolverApp app;
    const std::filesystem::path old_cwd = std::filesystem::current_path();
    std::error_code ec;
    std::filesystem::current_path(cwd, ec);
    if (ec) {
        std::cerr << "Failed to switch working directory to " << cwd << "\n";
        return 1;
    }

    int rc = 1;
    try {
        rc = app.run(static_cast<int>(argv.size()), argv.data());
    } catch (const std::exception& e) {
        std::filesystem::current_path(old_cwd, ec);
        std::cerr << "SolverApp threw exception: " << e.what() << "\n";
        return 1;
    }

    std::filesystem::current_path(old_cwd, ec);
    return rc;
}

}  // namespace

int main() {
    const std::filesystem::path source_dir(FI_SPVM_SOURCE_DIR);
    const std::filesystem::path config_path = source_dir / "tests" / "fixtures" / "smoke_config.json";
    const std::filesystem::path out_dir = std::filesystem::temp_directory_path() / "fi_spvm_cli_smoke_out";
    const std::filesystem::path alt_cwd = std::filesystem::temp_directory_path() / "fi_spvm_cli_smoke_cwd";

    std::error_code ec;
    std::filesystem::remove_all(out_dir, ec);
    std::filesystem::remove_all(alt_cwd, ec);
    std::filesystem::create_directories(out_dir, ec);
    std::filesystem::create_directories(alt_cwd, ec);
    if (ec) {
        std::cerr << "Failed to create temp directories\n";
        return 1;
    }

    if (run_with_config(source_dir, config_path, out_dir, alt_cwd) != 0) {
        std::cerr << "Smoke run returned non-zero status\n";
        return 1;
    }

    const std::array<std::string, 6> expected_files = {
        "admm_summary.txt",
        "components.txt",
        "metrics.json",
        "summary.json",
        "transient_on_total.csv",
        "transient_off_total.csv"
    };

    for (const auto& rel : expected_files) {
        const auto full = out_dir / rel;
        if (!exists_and_nonempty(full)) {
            std::cerr << "Missing or empty expected output: " << full << "\n";
            return 1;
        }
    }

    const auto metrics = read_json(out_dir / "metrics.json");
    if (!metrics.contains("weighted_rmse") || !metrics.contains("R2_real") || !metrics.contains("R2_imag")) {
        std::cerr << "metrics.json missing expected keys\n";
        return 1;
    }

    const auto summary = read_json(out_dir / "summary.json");
    if (!summary.contains("components") || !summary["components"].is_array()) {
        std::cerr << "summary.json missing components array\n";
        return 1;
    }
    if (!summary.contains("tau_range") || summary["tau_range"].size() != 2 ||
        !summary.contains("beta_range") || summary["beta_range"].size() != 2) {
        std::cerr << "summary.json missing expected range keys\n";
        return 1;
    }

    const double tau_min = summary["tau_range"][0].get<double>();
    const double tau_max = summary["tau_range"][1].get<double>();
    const double beta_min = summary["beta_range"][0].get<double>();
    const double beta_max = summary["beta_range"][1].get<double>();
    if (!(tau_min > 0.0 && tau_max >= tau_min && beta_max >= beta_min)) {
        std::cerr << "summary.json range values are invalid\n";
        return 1;
    }

    std::filesystem::remove_all(out_dir, ec);
    std::filesystem::remove_all(alt_cwd, ec);
    return 0;
}
