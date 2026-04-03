#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "../lib/Config.h"

namespace {

std::filesystem::path write_temp_config(
    const std::filesystem::path& dir,
    const std::string& body,
    const std::string& name) {
    std::filesystem::create_directories(dir);
    const auto path = dir / name;
    std::ofstream ofs(path);
    ofs << body;
    return path;
}

std::string normalize_path(const std::filesystem::path& path) {
    return std::filesystem::absolute(path).lexically_normal().make_preferred().string();
}

}  // namespace

TEST(ConfigLoaderTest, LoadsCurrentNestedConfig) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "fi_spvm_config_nested";
    const auto path = write_temp_config(temp_dir, R"json(
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
    "outputDir": "results",
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

    EXPECT_EQ(cfg.inputFile, normalize_path(temp_dir / "examples/sample.csv"));
    EXPECT_FALSE(cfg.noiseWeighted);
    EXPECT_EQ(cfg.spectrum_input_type, "period");
    EXPECT_DOUBLE_EQ(cfg.kernel.tau_min, 1e-5);
    EXPECT_DOUBLE_EQ(cfg.visualization.transient_tmax, 2.0);
    EXPECT_EQ(cfg.visualization.transient_samples, 20);
    EXPECT_EQ(cfg.visualization.outputDir, normalize_path(temp_dir / "results"));
    EXPECT_EQ(cfg.param_selection.outputDir, normalize_path(temp_dir / "results"));
    EXPECT_EQ(cfg.logging.file, normalize_path(temp_dir / "logs/run.log"));

    std::filesystem::remove_all(temp_dir);
}

TEST(ConfigLoaderTest, ResolvesRelativePathsAgainstConfigDirectory) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "fi_spvm_config_paths" / "nested";
    const auto path = write_temp_config(temp_dir, R"json(
{
  "data": {
    "input_file": "data/input.csv",
    "input_type": "freq"
  },
  "logging": {
    "file": "logs/case.log"
  },
  "visualization": {
    "outputDir": "outputs/case"
  },
  "param_selection": {
    "enable": false,
    "outputDir": "reports/case"
  }
}
)json", "config.json");

    const auto cfg = trspv::ConfigLoader::from_file(path.string());

    EXPECT_EQ(cfg.inputFile, normalize_path(temp_dir / "data/input.csv"));
    EXPECT_EQ(cfg.logging.file, normalize_path(temp_dir / "logs/case.log"));
    EXPECT_EQ(cfg.visualization.outputDir, normalize_path(temp_dir / "outputs/case"));
    EXPECT_EQ(cfg.param_selection.outputDir, normalize_path(temp_dir / "reports/case"));

    std::filesystem::remove_all(temp_dir.parent_path());
}

TEST(ConfigLoaderTest, AcceptsLegacyTopLevelDataKeys) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "fi_spvm_config_legacy";
    const auto path = write_temp_config(temp_dir, R"json(
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
    "outputDir": "results"
  }
}
)json", "fi_spvm_config_legacy_test.json");

    const auto cfg = trspv::ConfigLoader::from_file(path.string());

    EXPECT_EQ(cfg.inputFile, normalize_path(temp_dir / "legacy.csv"));
    EXPECT_TRUE(cfg.noiseWeighted);
    EXPECT_EQ(cfg.spectrum_input_type, "freq");

    std::filesystem::remove_all(temp_dir);
}

TEST(ConfigLoaderTest, RejectsInvalidKernelRange) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "fi_spvm_config_bad_range";
    const auto path = write_temp_config(temp_dir, R"json(
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
    "outputDir": "results"
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

    std::filesystem::remove_all(temp_dir);
}

TEST(ConfigLoaderTest, RejectsInvalidPeakInterpType) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "fi_spvm_config_bad_interp";
    const auto path = write_temp_config(temp_dir, R"json(
{
  "data": { "input_file": "bad.csv" },
  "preprocess": {
    "find_peaks": {
      "interp_type": "spline"
    }
  },
  "visualization": {
    "outputDir": "results"
  }
}
)json", "fi_spvm_config_invalid_interp_test.json");

    EXPECT_THROW((void)trspv::ConfigLoader::from_file(path.string()), std::runtime_error);

    std::filesystem::remove_all(temp_dir);
}

TEST(ConfigLoaderTest, RejectsInvalidInputType) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "fi_spvm_config_bad_input_type";
    const auto path = write_temp_config(temp_dir, R"json(
{
  "data": {
    "input_file": "bad.csv",
    "input_type": "phase"
  },
  "visualization": {
    "outputDir": "results"
  }
}
)json", "fi_spvm_config_invalid_input_type_test.json");

    EXPECT_THROW((void)trspv::ConfigLoader::from_file(path.string()), std::runtime_error);

    std::filesystem::remove_all(temp_dir);
}

TEST(ConfigLoaderTest, RejectsEmptyOutputAndLogPaths) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "fi_spvm_config_bad_paths";
    const auto path = write_temp_config(temp_dir, R"json(
{
  "data": {
    "input_file": "bad.csv",
    "input_type": "freq"
  },
  "logging": {
    "file": ""
  },
  "visualization": {
    "outputDir": ""
  }
}
)json", "fi_spvm_config_invalid_paths_test.json");

    EXPECT_THROW((void)trspv::ConfigLoader::from_file(path.string()), std::runtime_error);

    std::filesystem::remove_all(temp_dir);
}
