#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "../lib/Config.h"

namespace {

std::filesystem::path write_temp_config(const std::string& body, const std::string& name) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream ofs(path);
    ofs << body;
    return path;
}

}  // namespace

TEST(ConfigLoaderTest, LoadsCurrentNestedConfig) {
    const auto path = write_temp_config(R"json(
{
  "data": {
    "input_file": "examples/sample.csv",
    "noise_weighted": false,
    "input_type": "period"
  },
  "kernel": {
    "tau_min": 1e-5,
    "tau_max": 1.0,
    "num_tau": 8,
    "gamma_min": 0.2,
    "gamma_max": 2.0,
    "num_gamma": 6,
    "gamma_scale": "linear"
  },
  "visualization": {
    "outputDir": "results/",
    "transient_tmax": 2.0,
    "transient_samples": 20
  },
  "param_selection": {
    "enable": true,
    "num_lambda1": 3,
    "lambda1_min": 1e-5,
    "lambda1_max": 1e-2,
    "num_lambdat": 3,
    "lambdat_min": 1e-5,
    "lambdat_max": 1e-2,
    "num_lambdab": 3,
    "lambdab_min": 1e-5,
    "lambdab_max": 1e-2,
    "scan_max_iters": 5,
    "scan_tol": 1e-3
  }
}
)json", "fi_spvm_config_nested_test.json");

    const auto cfg = trspv::ConfigLoader::from_file(path.string());

    EXPECT_EQ(cfg.inputFile, "examples/sample.csv");
    EXPECT_FALSE(cfg.noiseWeighted);
    EXPECT_EQ(cfg.spectrum_input_type, "period");
    EXPECT_DOUBLE_EQ(cfg.kernel.tau_min, 1e-5);
    EXPECT_DOUBLE_EQ(cfg.visualization.transient_tmax, 2.0);
    EXPECT_EQ(cfg.visualization.transient_samples, 20);
    EXPECT_EQ(cfg.param_selection.outputDir, "results/");

    std::filesystem::remove(path);
}

TEST(ConfigLoaderTest, AcceptsLegacyTopLevelDataKeys) {
    const auto path = write_temp_config(R"json(
{
  "inputFile": "legacy.csv",
  "noiseWeighted": true,
  "spectrum_input_type": "freq",
  "kernel": {
    "tau_min": 1e-4,
    "tau_max": 1.0,
    "num_tau": 4,
    "gamma_min": 0.2,
    "gamma_max": 1.2,
    "num_gamma": 4,
    "gamma_scale": "log"
  },
  "visualization": {
    "outputDir": "results/"
  }
}
)json", "fi_spvm_config_legacy_test.json");

    const auto cfg = trspv::ConfigLoader::from_file(path.string());

    EXPECT_EQ(cfg.inputFile, "legacy.csv");
    EXPECT_TRUE(cfg.noiseWeighted);
    EXPECT_EQ(cfg.spectrum_input_type, "freq");

    std::filesystem::remove(path);
}

TEST(ConfigLoaderTest, RejectsInvalidKernelRange) {
    const auto path = write_temp_config(R"json(
{
  "data": { "input_file": "bad.csv" },
  "kernel": {
    "tau_min": 1.0,
    "tau_max": 1.0,
    "num_tau": 4,
    "gamma_min": 0.2,
    "gamma_max": 1.2,
    "num_gamma": 4
  },
  "visualization": {
    "outputDir": "results/"
  }
}
)json", "fi_spvm_config_invalid_range_test.json");

    EXPECT_THROW(
        {
            try {
                (void)trspv::ConfigLoader::from_file(path.string());
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("kernel.tau_min"), std::string::npos);
                throw;
            }
        },
        std::runtime_error);

    std::filesystem::remove(path);
}

TEST(ConfigLoaderTest, RejectsInvalidPeakInterpType) {
    const auto path = write_temp_config(R"json(
{
  "data": { "input_file": "bad.csv" },
  "preprocess": {
    "find_peaks": {
      "interp_type": "spline"
    }
  },
  "visualization": {
    "outputDir": "results/"
  }
}
)json", "fi_spvm_config_invalid_interp_test.json");

    EXPECT_THROW((void)trspv::ConfigLoader::from_file(path.string()), std::runtime_error);

    std::filesystem::remove(path);
}
