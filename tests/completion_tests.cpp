#include <gtest/gtest.h>

#include "../lib/SpectrumCompletion.h"

TEST(SpectrumCompletionTest, ReturnsOriginalDataWhenInterpolationDisabled) {
    trspv::SpectrumData data;
    data.freq = {1.0, 2.0, 4.0};
    data.values = {{0.1, 0.01}, {0.2, 0.02}, {0.4, 0.04}};
    data.weights = {1.0, 1.5, 2.0};

    trspv::SpectrumCompletionConfig cfg;
    cfg.interpolate = false;
    cfg.method = trspv::CompletionMethod::None;

    const auto out = trspv::SpectrumCompletion::complete(data, cfg);

    ASSERT_EQ(out.freq.size(), data.freq.size());
    EXPECT_EQ(out.freq, data.freq);
    EXPECT_EQ(out.weights, data.weights);
}

TEST(SpectrumCompletionTest, ExpandsGridAndKeepsOriginalWeights) {
    trspv::SpectrumData data;
    data.freq = {1.0, 2.0, 4.0};
    data.values = {{0.1, 0.01}, {0.2, 0.02}, {0.4, 0.04}};
    data.weights = {1.0, 1.5, 2.0};

    trspv::SpectrumCompletionConfig cfg;
    cfg.interpolate = true;
    cfg.method = trspv::CompletionMethod::PCHIP;
    cfg.num_points = 5;
    cfg.log_space = false;
    cfg.weight = 0.25;

    const auto out = trspv::SpectrumCompletion::complete(data, cfg);

    ASSERT_EQ(out.freq.size(), 5u);
    ASSERT_EQ(out.values.size(), 5u);
    ASSERT_EQ(out.weights.size(), 5u);
    EXPECT_DOUBLE_EQ(out.freq.front(), 1.0);
    EXPECT_DOUBLE_EQ(out.freq.back(), 4.0);
    EXPECT_NEAR(out.weights.front(), 1.0, 1e-12);
    EXPECT_NEAR(out.weights.back(), 2.0, 1e-12);
}
