#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "DictionaryGenerator.h"
#include "KernelFunction.h"

namespace trspv {

    /// Kernel (��, ��) ��������
    struct KernelConfig {
        double tau_min = 1e-4;
        double tau_max = 1.0;
        int num_tau = 50;
        double gamma_min = 0.2;
        double gamma_max = 1.8;
        int num_gamma = 20;
        std::string gamma_scale = "linear";
    };

    /// ADMM �Ż���������
    struct ADMMConfig {
        double lambda1 = 1e-3;  // L1 ϡ��
        double lambda_tv_tau = 1e-3;  // TV �� �� ����ƽ��
        double lambda_tv_beta = 1e-3;  // TV �� �� ����ƽ��
        double rho = 1.0;   // �������
        int    max_iters = 500;
        double tol_primal = 1e-6;
        double tol_dual = 1e-6;

        std::vector<double> l1_weights;
        int group_size_tau = 5;    // ���� ��  ����
        int group_size_beta = 3;    // ���� ��  ����
        int gamma_stride = 0;    // = ��_list.size()�����ֵ����

        int Nt = 0;  // �� ����
        int Nb = 0;  // �� ����

    };

    /// ��һ������
    struct NormalizationConfig {
        bool enabled = true;
    };

    /// ��ֵ/���㷽��ö��
    enum class CompletionMethod {
        None,
        PCHIP,
        Akima
    };

    /// Ƶ�ײ�ֵ/��������
    struct SpectrumCompletionConfig {
        CompletionMethod method = CompletionMethod::None;  // ��ֵ����
        bool             interpolate = false;                   // �Ƿ����ò�ֵ
        int              num_points = 100;                     // ��ֵ���������
        bool             log_space = true;                    // �����ռ����
        double           weight = 0.2;                     // ��ֵ��Ȩ��
    };

    struct FindPeaksConfig {
        bool   enable = true;
        std::string interp_type = "akima"; // "akima" | "pchip"
        int    smooth_window = 5;       // odd ��3; 0 = off
        double peak_prominence = 0.02;    // relative (0-1)
        double peak_dist_dec = 0.15;    // decades
        int    interp_factor = 4;       // up-sampling ��factor
        double      weight_factor = 0.5;
    };

    struct PreprocessConfig {
        FindPeaksConfig find_peaks;
    };

    /// Beta������ָ�����Ż�����
    enum class BetaMethod { Newton, GridSearch };

    struct PriorsConfig {
        double beta_center = 1.0;   // ��������
        double beta_sigma = 0.25;  // �� 1 ���� ~0.25 �����Լӷ�
        double beta_strength = 2;   // 0 �رգ�1~2 ����
    };


    struct ParamSelectionConfig {
        bool    enable = false;    ///< �Ƿ������Զ�����
        int     num_lambda1 = 20;       ///< ��1 �������
        double  lambda1_min = 1e-4;
        double  lambda1_max = 1e-1;
        int     num_lambdat = 20;       ///< �˦� TV �������
        double  lambdat_min = 1e-4;
        double  lambdat_max = 1e-1;
        int     num_lambdab = 20;       ///< �˦� TV �������
        double  lambdab_min = 1e-4;
        double  lambdab_max = 1e-1;
        std::string outputDir = "";
        // ������ ɨ��׶� ADMM ���� ������
        int     scan_max_iters = 50;   ///< ɨ��ʱ��������
        double  scan_tol = 1e-3; ///< ɨ��ʱԭʼ/��ż���̶�
        bool    refine_after = true; ///< ɨ����Ƿ����ϸ� ADMM ����һ��
    };

    struct JointConfig {
        // ADMM ���
        double lambda1 = 1e-3;
        double lambda_tv = 1e-3;
        double rho = 1.0;
        int    max_iter_admm = 1000;
        double tol_admm_primal = 1e-6;
        double tol_admm_dual = 1e-6;

        // ����ָ���Ż����
        double beta_init = 1.0;
        double beta_min = 0.1;
        double beta_max = 2.0;
        int    max_iter_beta = 50;
        double tol_beta = 1e-4;
        // �����ⲿ�Ż������дһάţ��/����
        enum class BetaMethod { Newton, GridSearch } beta_method = BetaMethod::GridSearch;
        int    beta_grid_points = 20; // �������������
    };

    /// ��־ģ������
    struct LoggingConfig {
        std::string file = "logs/run.log";
        std::string level = "INFO";
        int snapshotInterval = 100;
    };

    /// ���ӻ��������
    struct VisualizationConfig {
        bool enabled = false;
        std::string outputDir = "results/";
        double transient_tmax = 1.0;
        int transient_samples = 200;
    };

    /// �������ýṹ
    struct Config {
        // ��������
        std::string inputFile = "examples/sample_spectrum.csv";
        bool noiseWeighted = true;

        std::string spectrum_input_type = "freq";

        // ����ģ������
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

    /// ���ü�����
    class ConfigLoader {
    public:
        /**
         * �� JSON �ļ����� Config��
         * @param path �����ļ�·��
         * @return ���ö���
         * @throws std::runtime_error ����ļ��򲻿����ʽ����
         */
        static Config from_file(const std::string& path);
    };

} // namespace trspv