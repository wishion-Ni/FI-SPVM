#pragma once

#include "SpectrumData.h"
#include "Config.h"
#include <string>

namespace trspv {


    /// 插值/补点模块
    class SpectrumCompletion {
    public:
        /**
         * 根据配置对原始频谱数据进行插值补点
         * @param data 原始数据（已加载）
         * @param config 插值配置
         * @return 补点后的完整数据
         */
        static SpectrumData complete(const SpectrumData& data,
            const SpectrumCompletionConfig& config);
    };

} // namespace trspv
