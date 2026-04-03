#pragma once

#include <complex>
#include <string>
#include <vector>

namespace trspv {

struct SpectrumData {
    std::vector<double> freq;
    std::vector<std::complex<double>> values;
    std::vector<double> weights;
};

class SpectrumDataLoader {
public:
    static SpectrumData load_csv(const std::string& path,
                                 bool noiseWeighted,
                                 const std::string& inputType = "freq");

    static void apply_weight(SpectrumData& data);
};

}  // namespace trspv
