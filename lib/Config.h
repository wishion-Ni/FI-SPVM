#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "DictionaryGenerator.h"
#include "KernelFunction.h"

namespace trspv {

    /// Kernel (τ, γ) 参数配置
    struct KernelConfig {
        double tau_min = 1e-4;
        double tau_max = 1.0;
        int num_tau = 50;
        double gamma_min = 0.2;
        double gamma_max = 1.8;
        int num_gamma = 20;
        std::string gamma_scale = "linear";
    };

    /// ADMM 优化参数配置
    struct ADMMConfig {
        double lambda1 = 1e-3;  // L1 稀疏
        double lambda_tv_tau = 1e-3;  // TV 在 τ 方向平滑
        double lambda_tv_beta = 1e-3;  // TV 在 β 方向平滑
        double rho = 1.0;   // 增广参数
        int    max_iters = 500;
        double tol_primal = 1e-6;
        double tol_dual = 1e-6;

        std::vector<double> l1_weights;
        int group_size_tau = 5;    // 相邻 τ  点数
        int group_size_beta = 3;    // 相邻 β  点数
        int gamma_stride = 0;    // = τ_list.size()，由字典回填

        int Nt = 0;  // τ 点数
        int Nb = 0;  // β 点数

    };

    /// 归一化设置
    struct NormalizationConfig {
        bool enabled = true;
    };

    /// 插值/补点方法枚举
    enum class CompletionMethod {
        None,
        PCHIP,
        Akima
    };

    /// 频谱插值/补点配置
    struct SpectrumCompletionConfig {
        CompletionMethod method = CompletionMethod::None;  // 插值方法
        bool             interpolate = false;                   // 是否启用插值
        int              num_points = 100;                     // 插值后采样点数
        bool             log_space = true;                    // 对数空间采样
        double           weight = 0.2;                     // 插值点权重
    };

    struct FindPeaksConfig {
        bool   enable = true;
        std::string interp_type = "akima"; // "akima" | "pchip"
        int    smooth_window = 5;       // odd ≥3; 0 = off
        double peak_prominence = 0.02;    // relative (0-1)
        double peak_dist_dec = 0.15;    // decades
        int    interp_factor = 4;       // up-sampling ×factor
        double      weight_factor = 0.5;
    };

    struct PreprocessConfig {
        FindPeaksConfig find_peaks;
    };

    /// Beta（拉伸指数）优化方法
    enum class BetaMethod { Newton, GridSearch };

    struct PriorsConfig {
        double beta_center = 1.0;   // 先验中心
        double beta_sigma = 0.25;  // 离 1 超过 ~0.25 就明显加罚
        double beta_strength = 2;   // 0 关闭；1~2 常用
    };


    struct ParamSelectionConfig {
        bool    enable = false;    ///< 是否启用自动调参
        int     num_lambda1 = 20;       ///< λ1 网格点数
        double  lambda1_min = 1e-4;
        double  lambda1_max = 1e-1;
        int     num_lambdat = 20;       ///< λτ TV 网格点数
        double  lambdat_min = 1e-4;
        double  lambdat_max = 1e-1;
        int     num_lambdab = 20;       ///< λβ TV 网格点数
        double  lambdab_min = 1e-4;
        double  lambdab_max = 1e-1;
        std::string outputDir = "";
        // ——— 扫描阶段 ADMM 参数 ———
        int     scan_max_iters = 50;   ///< 扫描时最大迭代数
        double  scan_tol = 1e-3; ///< 扫描时原始/对偶容忍度
        bool    refine_after = true; ///< 扫描后是否用严格 ADMM 再跑一次
    };

    struct JointConfig {
        // ADMM 相关
        double lambda1 = 1e-3;
        double lambda_tv = 1e-3;
        double rho = 1.0;
        int    max_iter_admm = 1000;
        double tol_admm_primal = 1e-6;
        double tol_admm_dual = 1e-6;

        // 拉伸指数优化相关
        double beta_init = 1.0;
        double beta_min = 0.1;
        double beta_max = 2.0;
        int    max_iter_beta = 50;
        double tol_beta = 1e-4;
        // 调用外部优化库或自写一维牛顿/网格
        enum class BetaMethod { Newton, GridSearch } beta_method = BetaMethod::GridSearch;
        int    beta_grid_points = 20; // 如果用网格搜索
    };

    /// 日志模块配置
    struct LoggingConfig {
        std::string file = "logs/run.log";
        std::string level = "INFO";
        int snapshotInterval = 100;
    };

    /// 可视化相关配置
    struct VisualizationConfig {
        bool enabled = false;
        std::string outputDir = "results/";
    };

    /// 完整配置结构
    struct Config {
        // 数据输入
        std::string inputFile = "examples/sample_spectrum.csv";
        bool noiseWeighted = true;

        std::string spectrum_input_type = "freq";

        // 各子模块配置
        KernelConfig kernel;
        ADMMConfig admm;
        NormalizationConfig normalization;
        SpectrumCompletionConfig completion;
        LoggingConfig logging;
        VisualizationConfig visualization;
        ParamSelectionConfig param_selection;
        PreprocessConfig preprocess;
        PriorsConfig priors;

    };

    /// 配置加载器
    class ConfigLoader {
    public:
        /**
         * 从 JSON 文件加载 Config。
         * @param path 配置文件路径
         * @return 配置对象
         * @throws std::runtime_error 如果文件打不开或格式不对
         */
        static Config from_file(const std::string& path);
    };

} // namespace trspv