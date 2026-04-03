#include "SpectrumData.h"

#include "Logger.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace trspv {

SpectrumData SpectrumDataLoader::load_csv(const std::string& path,
                                          bool noiseWeighted,
                                          const std::string& inputType) {
    const std::string normalized_input_type = [&]() {
        std::string value = inputType;
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value;
    }();

    if (normalized_input_type != "freq" && normalized_input_type != "period") {
        throw std::runtime_error("inputType must be one of: freq, period; got " + inputType);
    }

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open spectrum file: " + path);
    }

    SpectrumData data;
    std::string line;
    size_t lineNo = 0;
    while (std::getline(ifs, line)) {
        ++lineNo;
        if (line.empty()) {
            continue;
        }

        if (lineNo == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line.erase(0, 3);
        }

        auto notspace = [](unsigned char ch) { return !std::isspace(ch); };
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), notspace));
        line.erase(std::find_if(line.rbegin(), line.rend(), notspace).base(), line.end());
        if (line.empty()) {
            continue;
        }

        for (char& ch : line) {
            if (ch == ',' || ch == ';' || ch == '\t') {
                ch = ' ';
            }
        }

        std::istringstream iss(line);
        double x = 0.0;
        double re = 0.0;
        double im = 0.0;
        double w = 1.0;
        if (!(iss >> x >> re >> im)) {
            if (lineNo == 1) {
                continue;
            }
            Logger::warn("Line " + std::to_string(lineNo) + " parse fail, skip");
            continue;
        }

        double freq = x;
        if (normalized_input_type == "period") {
            if (x == 0.0) {
                Logger::warn("Line " + std::to_string(lineNo) + " period zero, skip");
                continue;
            }
            freq = 1.0 / x;
        }

        if (noiseWeighted && !(iss >> w)) {
            Logger::warn("Line " + std::to_string(lineNo) + " missing weight, default to 1.0");
            w = 1.0;
        }

        data.freq.push_back(freq);
        data.values.emplace_back(re, im);
        data.weights.push_back(w);
    }

    if (data.freq.empty()) {
        throw std::runtime_error("No data loaded from " + path);
    }

    Logger::info("Loaded {} points from {}", data.freq.size(), path);
    return data;
}

void SpectrumDataLoader::apply_weight(SpectrumData& data) {
    for (size_t i = 0; i < data.values.size(); ++i) {
        const double wt = std::sqrt(data.weights[i]);
        data.values[i] *= wt;
    }
    Logger::info("Applied weights to spectrum data");
}

}  // namespace trspv
