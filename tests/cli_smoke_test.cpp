#include <array>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "../src/SolverApp.h"

namespace {

bool exists_and_nonempty(const std::filesystem::path& path) {
    return std::filesystem::exists(path) && std::filesystem::file_size(path) > 0;
}

}  // namespace

int main() {
    const std::filesystem::path source_dir(FI_SPVM_SOURCE_DIR);
    const std::filesystem::path config_path = source_dir / "tests" / "fixtures" / "smoke_config.json";
    const std::filesystem::path out_dir = std::filesystem::temp_directory_path() / "fi_spvm_cli_smoke_out";
    const std::filesystem::path old_cwd = std::filesystem::current_path();

    std::error_code ec;
    std::filesystem::remove_all(out_dir, ec);
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "Failed to create output directory: " << out_dir << "\n";
        return 1;
    }

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
    std::filesystem::current_path(source_dir, ec);
    if (ec) {
        std::cerr << "Failed to switch working directory to source root\n";
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
    if (rc != 0) {
        std::cerr << "SolverApp returned non-zero status: " << rc << "\n";
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

    std::filesystem::remove_all(out_dir, ec);
    return 0;
}
