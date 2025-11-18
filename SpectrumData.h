#pragma once

#include <string>
#include <vector>
#include <complex>

namespace trspv {

    /// 频谱数据结构：存储频率、复数信号及权重
    struct SpectrumData {
        std::vector<double>              freq;    // 频率 (Hz)
        std::vector<std::complex<double>> values;  // 复数响应
        std::vector<double>              weights; // 权重 (可基于噪声)
    };

    /// 频谱数据加载与预处理
    class SpectrumDataLoader {
    public:
        /**
         * 加载 CSV 频谱数据。
         * 文件格式：
         *  第1列：freq (Hz) 或 period (s)，由 inputType 指定
         *  第2列：实部
         *  第3列：虚部
         *  第4列（可选）：权重
         * @param path CSV 文件路径
         * @param noiseWeighted 是否加载第4列权重
         * @param inputType "freq" 或 "period"
         * @return SpectrumData 对象
         * @throws std::runtime_error 文件打不开或解析出错
         */
        static SpectrumData load_csv(const std::string& path,
            bool noiseWeighted,
            const std::string& inputType = "freq");

        /**
         * 基于 weights 对 values 加权：values[i] *= sqrt(weights[i])
         */
        static void apply_weight(SpectrumData& data);
    };

} // namespace trspv
