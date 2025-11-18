#include "SpectrumData.h"
#include "Logger.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace trspv {

    SpectrumData SpectrumDataLoader::load_csv(const std::string& path,
        bool noiseWeighted,
        const std::string& inputType) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            throw std::runtime_error("Cannot open spectrum file: " + path);
        }

        SpectrumData data;
        std::string line;
        size_t lineNo = 0;
        while (std::getline(ifs, line)) {
            ++lineNo;
            if (line.empty()) continue;

            // 去 BOM
            if (lineNo == 1 && line.size() >= 3 &&
                (unsigned char)line[0] == 0xEF &&
                (unsigned char)line[1] == 0xBB &&
                (unsigned char)line[2] == 0xBF) {
                line.erase(0, 3);
            }

            // 去首尾空白
            auto notspace = [](int ch) { return !std::isspace(ch); };
            line.erase(line.begin(), std::find_if(line.begin(), line.end(), notspace));
            line.erase(std::find_if(line.rbegin(), line.rend(), notspace).base(), line.end());
            if (line.empty()) continue;

            

            // 支持逗号/分号/制表符：统一替换成空格
            for (char& ch : line) {
                if (ch == ',' || ch == ';' || ch == '\t') ch = ' ';
            }

            std::istringstream iss(line);
            double x, re, im, w = 1.0;
            if (!(iss >> x >> re >> im)) {
                if (lineNo == 1) continue;
                Logger::warn("Line " + std::to_string(lineNo) + " parse fail, skip");
                continue;
                throw std::runtime_error("Parse error at line " + std::to_string(lineNo));
            }
            // 转换周期到频率
            double freq = x;
            if (inputType == "period") {
                if (x == 0) {
                    Logger::warn("Line " + std::to_string(lineNo) + " period zero, skip");
                    continue;
                }
                freq = 1.0 / x;
            }
            // 权重
            if (noiseWeighted) {
                if (!(iss >> w)) {
                    Logger::warn("Line " + std::to_string(lineNo) +
                        " missing weight, default to 1.0");
                    w = 1.0;
                }
            }
            data.freq.push_back(freq);
            data.values.emplace_back(re, im);
            data.weights.push_back(w);
        }
        if (data.freq.empty()) {
            throw std::runtime_error("No data loaded from " + path);
        }
        Logger::info("Loaded " + std::to_string(data.freq.size()) + " points from " + path);
        return data;
    }

    void SpectrumDataLoader::apply_weight(SpectrumData& data) {
        for (size_t i = 0; i < data.values.size(); ++i) {
            double wt = std::sqrt(data.weights[i]);
            data.values[i] *= wt;
        }
        Logger::info("Applied weights to spectrum data");
    }

} // namespace trspv
