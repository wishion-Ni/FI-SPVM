#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "../lib/SpectrumData.h"

namespace {

std::filesystem::path write_temp_csv(const std::string& body, const std::string& name) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream ofs(path);
    ofs << body;
    return path;
}

}  // namespace

TEST(SpectrumDataLoaderTest, SupportsHeaderAndMixedDelimiters) {
    const auto path = write_temp_csv(
        "freq,real,imag\n"
        "1;0.1;0.2\n"
        "2\t0.2\t0.3\n"
        "3,0.3,0.4\n",
        "fi_spvm_loader_mixed.csv");

    const auto data = trspv::SpectrumDataLoader::load_csv(path.string(), false, "freq");

    ASSERT_EQ(data.freq.size(), 3u);
    EXPECT_DOUBLE_EQ(data.freq[0], 1.0);
    EXPECT_DOUBLE_EQ(data.values[2].real(), 0.3);
    EXPECT_DOUBLE_EQ(data.values[2].imag(), 0.4);

    std::filesystem::remove(path);
}

TEST(SpectrumDataLoaderTest, SkipsBadRowsButLoadsValidData) {
    const auto path = write_temp_csv(
        "freq,real,imag\n"
        "bad,row\n"
        "4,0.4,0.5\n"
        "5,0.5,0.6\n",
        "fi_spvm_loader_bad_rows.csv");

    const auto data = trspv::SpectrumDataLoader::load_csv(path.string(), false, "freq");

    ASSERT_EQ(data.freq.size(), 2u);
    EXPECT_DOUBLE_EQ(data.freq[0], 4.0);
    EXPECT_DOUBLE_EQ(data.freq[1], 5.0);

    std::filesystem::remove(path);
}

TEST(SpectrumDataLoaderTest, ConvertsPeriodToFrequency) {
    const auto path = write_temp_csv(
        "period,real,imag\n"
        "2,0.1,0.2\n"
        "4,0.2,0.3\n",
        "fi_spvm_loader_period.csv");

    const auto data = trspv::SpectrumDataLoader::load_csv(path.string(), false, "period");

    ASSERT_EQ(data.freq.size(), 2u);
    EXPECT_DOUBLE_EQ(data.freq[0], 0.5);
    EXPECT_DOUBLE_EQ(data.freq[1], 0.25);

    std::filesystem::remove(path);
}

TEST(SpectrumDataLoaderTest, LoadsWeightsWhenEnabled) {
    const auto path = write_temp_csv(
        "freq,real,imag,weight\n"
        "1,0.1,0.2,2.0\n"
        "2,0.2,0.3,3.0\n",
        "fi_spvm_loader_weights.csv");

    auto data = trspv::SpectrumDataLoader::load_csv(path.string(), true, "freq");

    ASSERT_EQ(data.weights.size(), 2u);
    EXPECT_DOUBLE_EQ(data.weights[0], 2.0);
    EXPECT_DOUBLE_EQ(data.weights[1], 3.0);

    trspv::SpectrumDataLoader::apply_weight(data);
    EXPECT_NEAR(data.values[0].real(), 0.1 * std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(data.values[1].imag(), 0.3 * std::sqrt(3.0), 1e-12);

    std::filesystem::remove(path);
}

TEST(SpectrumDataLoaderTest, RejectsInvalidInputType) {
    const auto path = write_temp_csv(
        "freq,real,imag\n"
        "1,0.1,0.2\n",
        "fi_spvm_loader_invalid_type.csv");

    EXPECT_THROW((void)trspv::SpectrumDataLoader::load_csv(path.string(), false, "phase"), std::runtime_error);

    std::filesystem::remove(path);
}
