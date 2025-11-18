#define FMT_HEADER_ONLY

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <deque>


#include "Config.h"
#include "SpectrumData.h"
#include "SpectrumCompletion.h"
#include "DictionaryGenerator.h"
#include "Utils.h"
#include "Solver2D.h"
#include "ParamSelector.h"
#include "PeakSeedDetector.h"
#include <Eigen/Dense>


    using namespace trspv;

    // === 关键组分提取：阈值= max(3*MAD, 0.02*max)；局部极值→去重合并 ===
    struct Component {
        double tau, beta;   // 代表值
        double amp;         // 强度（含符号）：取 real(x)
        double prominence;  // 显著性
        int    it, jb;      // 网格索引（用于追溯）
    };

    static double robust_sigma(const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        std::vector<double> u = v;
        std::nth_element(u.begin(), u.begin() + u.size() / 2, u.end());
        double med = u[u.size() / 2];
        for (auto& x : u) x = std::abs(x - med);
        std::nth_element(u.begin(), u.begin() + u.size() / 2, u.end());
        double mad = u[u.size() / 2];
        return 1.4826 * mad;
    }

    static std::vector<Component> extract_components_cc(
        const Eigen::VectorXcd& x2d,
        const std::vector<double>& taus,
        const std::vector<double>& betas,
        double k_sigma = 3.0,
        double alpha_of_max = 0.05,   // 建议从 0.05~0.10 起
        int    min_pixels = 4,        // 过滤太小的碎片
        double merge_dlogtau = 0.20,  // 连通域之间再做一次合并（可选）
        double merge_dbeta = 0.12)
    {
        const int Nt = (int)taus.size(), Nb = (int)betas.size();
        auto IDX = [&](int it, int jb) { return it + jb * Nt; };

        // 1) 绝对值场与阈值
        std::vector<double> absv; absv.reserve(x2d.size());
        double vmax = 0.0;
        for (int jb = 0; jb < Nb; ++jb)
            for (int it = 0; it < Nt; ++it) {
                double a = std::abs(x2d[IDX(it, jb)]);
                absv.push_back(a); vmax = std::max(vmax, a);
            }
        double sig = robust_sigma(absv);
        double thr = std::max(k_sigma * sig, alpha_of_max * vmax);
        if (thr <= 0) thr = 0.1 * vmax; // 极端兜底

        // 2) 二值掩膜 + 连通域(BFS)
        std::vector<char> mask(x2d.size(), 0), vis(x2d.size(), 0);
        for (int jb = 0; jb < Nb; ++jb)
            for (int it = 0; it < Nt; ++it)
                if (std::abs(x2d[IDX(it, jb)]) >= thr) mask[IDX(it, jb)] = 1;

        static const int d8[8][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1} };
        std::vector<Component> comps;

        for (int jb = 0; jb < Nb; ++jb) {
            for (int it = 0; it < Nt; ++it) {
                int start = IDX(it, jb);
                if (!mask[start] || vis[start]) continue;

                // BFS 收集一个连通域
                std::vector<int> pixels;
                std::deque<int> dq; dq.push_back(start); vis[start] = 1;
                while (!dq.empty()) {
                    int id = dq.front(); dq.pop_front();
                    pixels.push_back(id);
                    int it0 = id % Nt, jb0 = id / Nt;
                    for (auto& d : d8) {
                        int it1 = it0 + d[0], jb1 = jb0 + d[1];
                        if (it1 < 0 || it1 >= Nt || jb1 < 0 || jb1 >= Nb) continue;
                        int id1 = IDX(it1, jb1);
                        if (mask[id1] && !vis[id1]) { vis[id1] = 1; dq.push_back(id1); }
                    }
                }
                if ((int)pixels.size() < min_pixels) continue;

                // 3) 该连通域聚合成一个组件：权重=|x|
                double wsum = 0.0, sum_logt = 0.0, sum_beta = 0.0;
                double amp_sum = 0.0, peak_abs = 0.0;
                int    it_peak = it, jb_peak = jb;

                for (int id : pixels) {
                    std::complex<double> v = x2d[id];
                    double w = std::abs(v);
                    int it0 = id % Nt, jb0 = id / Nt;
                    wsum += w;
                    sum_logt += w * std::log10(taus[it0]);
                    sum_beta += w * betas[jb0];
                    amp_sum += v.real(); // 强度带符号
                    if (w > peak_abs) { peak_abs = w; it_peak = it0; jb_peak = jb0; }
                }

                Component c;
                c.tau = (wsum > 0 ? std::pow(10.0, sum_logt / wsum) : taus[it_peak]);
                c.beta = (wsum > 0 ? sum_beta / wsum : betas[jb_peak]);
                c.amp = amp_sum;
                c.prominence = peak_abs / std::max(1e-16, sig);
                c.it = it_peak; c.jb = jb_peak;
                comps.push_back(c);
            }
        }

        // 4) 连通域之间的二次合并（防相邻团碎裂）
        auto dlog = [](double t1, double t2) { return std::abs(std::log10(t1 / t2)); };
        std::sort(comps.begin(), comps.end(),
            [](auto& a, auto& b) { return std::abs(a.amp) > std::abs(b.amp); });
        std::vector<Component> merged;
        for (auto& c : comps) {
            bool keep = true;
            for (auto& m : merged) {
                if (dlog(c.tau, m.tau) < merge_dlogtau && std::abs(c.beta - m.beta) < merge_dbeta) {
                    // 合并到已有的 m（简单求和&重心更新）
                    double A = std::abs(m.amp) + std::abs(c.amp);
                    if (A == 0) { keep = false; break; }
                    m.tau = std::pow(10.0, (std::abs(m.amp) * std::log10(m.tau) + std::abs(c.amp) * std::log10(c.tau)) / A);
                    m.beta = (std::abs(m.amp) * m.beta + std::abs(c.amp) * c.beta) / A;
                    m.amp += c.amp;
                    m.prominence = std::max(m.prominence, c.prominence);
                    keep = false; break;
                }
            }
            if (keep) merged.push_back(c);
        }
        return merged;
    }


    // === Mittag–Leffler E_beta(-x) 的简化近似 ===
    static double Ebeta_negx(double x, double beta) {
        if (beta <= 0) beta = 1.0;
        if (beta == 1.0) return std::exp(-x);
        // 小/中 x: 级数
        if (x <= 30.0) {
            const int Kmax = 200;
            double term = 1.0; // k=0
            double sum = term;
            for (int k = 1; k <= Kmax; ++k) {
                term *= (-x) / k; // 先按 k! 累乘
                // 用 Γ(βk+1) 修正（k!→Γ(k+1) 已在上式）
                double gamma_ratio = std::tgamma(k + 1) / std::tgamma(beta * k + 1.0);
                double add = term * gamma_ratio;
                sum += add;
                if (std::abs(add) < 1e-16 * std::abs(sum)) break;
            }
            return sum;
        }
        else {
            // 大 x: 渐近展开 (保留少量项)
            double s = 0.0;
            for (int n = 1; n <= 4; ++n) {
                s -= std::pow(x, -n) / std::tgamma(1.0 - beta * n);
            }
            return s;
        }
    }

    // 单组分阶跃响应
    static double h_on(double t, double tau, double beta) {
        double x = std::pow(t / std::max(1e-300, tau), beta);
        return 1.0 - Ebeta_negx(x, beta);
    }
    static double h_off(double t, double tau, double beta) {
        double x = std::pow(t / std::max(1e-300, tau), beta);
        return Ebeta_negx(x, beta);
    }



    int main(int argc, char** argv) {

        try {
            // 1. 读取配置文件路径（默认为 config.json）
            // --- CLI 解析：支持 --conf, --input, --out，并兼容“第一个参数是 *.json” ---
            std::string configPath = "config.json";
            std::string overrideInput, overrideOut;

            for (int i = 1; i < argc; ++i) {
                std::string arg = argv[i];
                auto next = [&](int& i) -> const char* {
                    if (i + 1 < argc) return argv[++i];
                    throw std::runtime_error("缺少参数值: " + arg);
                };

                if (arg == "--conf" || arg == "-c") {
                    configPath = next(i);
                }
                else if (arg == "--input" || arg == "-i") {
                    overrideInput = next(i);
                }
                else if (arg == "--out" || arg == "-o") {
                    overrideOut = next(i);
                }
                else if (arg.size() > 5 && arg.substr(arg.size() - 5) == ".json" && i == 1) {
                    // 兼容旧用法：第一个参数直接是配置文件
                    configPath = arg;
                }
                else if (arg == "--help" || arg == "-h") {
                    std::cout <<
                        "Usage:\n"
                        "  trspv --conf config.json [--input data.csv] [--out results/runX]\n"
                        "  trspv config.json  # 兼容旧用法\n";
                    return 0;
                }
            }

            Config cfg = ConfigLoader::from_file(configPath);

            // 覆盖少量字段，便于脚本批量化
            if (!overrideInput.empty()) cfg.inputFile = overrideInput;
            if (!overrideOut.empty())   cfg.visualization.outputDir = overrideOut;


            // 2. 创建输出目录
            std::filesystem::create_directories(cfg.visualization.outputDir);

            // 3. 加载并预处理数据
            SpectrumData rawData = SpectrumDataLoader::load_csv(
                cfg.inputFile, cfg.noiseWeighted, cfg.spectrum_input_type
            );
            SpectrumData data = rawData;

            if (cfg.completion.interpolate) {
                SpectrumData interpData = SpectrumCompletion{}.complete(rawData, cfg.completion);

                // a) 原始与插值分别落盘
                {
                    std::ofstream ofs(cfg.visualization.outputDir + "/original_data.csv");
                    ofs << "freq,real,imag,weight\n";
                    for (size_t i = 0; i < rawData.freq.size(); ++i) {
                        ofs << std::setprecision(12) << rawData.freq[i] << ','
                            << rawData.values[i].real() << ','
                            << rawData.values[i].imag() << ','
                            << (i < rawData.weights.size() ? rawData.weights[i] : 1.0) << '\n';
                    }
                }
                {
                    std::ofstream ofs(cfg.visualization.outputDir + "/interpolated_data.csv");
                    ofs << "freq,real,imag\n";
                    for (size_t i = 0; i < interpData.freq.size(); ++i) {
                        ofs << std::setprecision(12) << interpData.freq[i] << ','
                            << interpData.values[i].real() << ','
                            << interpData.values[i].imag() << '\n';
                    }
                }

                // b) 最近邻映射做“插值 vs 原始”对比
                {
                    auto nearest = [&](double f) {
                        size_t jbest = 0;
                        double dmin = std::numeric_limits<double>::infinity();
                        for (size_t j = 0; j < interpData.freq.size(); ++j) {
                            double d = std::abs(interpData.freq[j] - f);
                            if (d < dmin) { dmin = d; jbest = j; }
                        }
                        return jbest;
                    };
                    std::ofstream ofs(cfg.visualization.outputDir + "/interpolation_vs_original.csv");
                    ofs << "freq_raw,raw_real,raw_imag,freq_interp,interp_real,interp_imag,abs_error\n";
                    for (size_t i = 0; i < rawData.freq.size(); ++i) {
                        size_t j = nearest(rawData.freq[i]);
                        std::complex<double> e = interpData.values[j] - rawData.values[i];
                        ofs << std::setprecision(12)
                            << rawData.freq[i] << ','
                            << rawData.values[i].real() << ','
                            << rawData.values[i].imag() << ','
                            << interpData.freq[j] << ','
                            << interpData.values[j].real() << ','
                            << interpData.values[j].imag() << ','
                            << std::abs(e) << '\n';
                    }
                }

                data = std::move(interpData); // 后续流程走插值后的数据
            }

            // 4. 构建角频率 ω 与观测向量 b
            std::vector<double> omega(data.freq.size());
            Eigen::VectorXcd   b(data.values.size());
            for (size_t i = 0; i < data.freq.size(); ++i) {
                omega[i] = 2.0 * M_PI * data.freq[i];
                b[i] = data.values[i];
            }

            // 5. 构造 tau 网格（对数等距）
            std::vector<double> taus;
            std::vector<double> l1Weights;
            {
                // 5.1 基础对数等距网格
                int    numTau = cfg.kernel.num_tau;
                double tauMin = cfg.kernel.tau_min;
                double tauMax = cfg.kernel.tau_max;
                taus.reserve(numTau);
                double logMin = std::log10(tauMin);
                double logMax = std::log10(tauMax);
                for (int i = 0; i < numTau; ++i) {
                    double t = logMin + (logMax - logMin) * i / (numTau - 1.0);
                    taus.push_back(std::pow(10.0, t));
                }

                // 5.2 如启用峰检测 → 插入种子
                std::vector<int>    peakIdx;            // 记录在 τ‑grid 中的索引

                auto pCfg = cfg.preprocess.find_peaks;  // Config 子结构已解析
                if (pCfg.enable) {
                    PeakSeedDetector::Options psOpt;
                    psOpt.interp_type = pCfg.interp_type;
                    psOpt.smooth_window = pCfg.smooth_window;
                    psOpt.peak_prominence = pCfg.peak_prominence;
                    psOpt.peak_dist_dec = pCfg.peak_dist_dec;
                    psOpt.interp_factor = pCfg.interp_factor;
                    PeakSeedDetector detector(psOpt);

                    std::vector<double> tauSeed = detector(data.freq, data.values);

                    // —— 输出检测到的峰列表及数量 ——
                    std::ofstream peakOut(cfg.visualization.outputDir + "/detected_peaks.txt");
                    peakOut << "# Detected " << tauSeed.size() << " peaks (tau in s) \n";
                        for (double t : tauSeed) peakOut << std::setprecision(12) << t << '\n';
                            std::cout << "[PeakDetect] found " << tauSeed.size() << " peak(s)." << std::endl;

                    // —— 合并 τ_seed 到网格并去重 ——
                    taus.insert(taus.end(), tauSeed.begin(), tauSeed.end());
                    std::sort(taus.begin(), taus.end());
                    taus.erase(std::unique(taus.begin(), taus.end(),
                        [](double a, double b) { return std::fabs(std::log10(a / b)) < 1e-6; }), taus.end());

                    // —— 生成 λ₁ 权重：峰点减小惩罚系数以鼓励被选中 ——
                    l1Weights.assign(taus.size(), 1.0);                    // default 1.0
                    const double wSeed = std::clamp(pCfg.weight_factor, 0.05, 1.0); // 峰位的目标权重(<1)
                    const double sigma = 0.07;                                      // 以 decade 为单位
                    for (size_t i = 0; i < taus.size(); ++i) {
                        double w = 1.0; // 默认无偏置
                        for (double tSeed : tauSeed) {
                            double d = std::log10(tSeed / taus[i]);
                            double g = std::exp(-0.5 * (d * d) / (sigma * sigma));  // 0..1, 峰位=1
                            // 峰附近 → wSeed；远离峰 → 1.0
                            w = std::min(w, 1.0 - (1.0 - wSeed) * g);
                        }
                        l1Weights[i] = w;  // 0<w<=1
                    }
                }
                else {
                    l1Weights.assign(taus.size(), 1.0);                    // no weighting
                }
            }

            // 6. 构建 beta 网格
            std::vector<double> betas;
            betas.reserve(cfg.kernel.num_gamma);
            if (cfg.kernel.gamma_scale == "log") {
                double logmin = std::log10(cfg.kernel.gamma_min);
                double logmax = std::log10(cfg.kernel.gamma_max);
                for (int j = 0; j < cfg.kernel.num_gamma; ++j) {
                    double f = double(j) / (cfg.kernel.num_gamma - 1);
                    betas.push_back(std::pow(10.0, logmin + f * (logmax - logmin)));
                }
            }
            else {
                double gmin = cfg.kernel.gamma_min;
                double gmax = cfg.kernel.gamma_max;
                for (int j = 0; j < cfg.kernel.num_gamma; ++j) {
                    double f = double(j) / (cfg.kernel.num_gamma - 1);
                    betas.push_back(gmin + f * (gmax - gmin));
                }
            }

            // === 把 1D 的 tau 权重扩展为 2D，并叠加 β 先验 ===
            const int Nt = (int)taus.size();
            const int Nb = (int)betas.size();
            std::vector<double> l1w2d;
            l1w2d.reserve(Nt * Nb);

            // β 先验：离 1 越远 → 罚越重（阈值越高）
            auto beta_weight = [&](double beta) {
                double z = (beta - cfg.priors.beta_center) / std::max(1e-12, cfg.priors.beta_sigma);
                double g = std::exp(-0.5 * z * z);                 // 1 at beta=1, decays away
                return 1.0 + cfg.priors.beta_strength * (1.0 - g); // ≥1
            };

            // 列顺序：β 外、τ 内
            for (int j = 0; j < Nb; ++j) {
                double wb = beta_weight(betas[j]);
                for (int i = 0; i < Nt; ++i) {
                    double wt = (i < (int)l1Weights.size() ? l1Weights[i] : 1.0);
                    l1w2d.push_back(wt * wb);
                }
            }

            /*// 归一到均值=1，避免整体抬高/降低 λ1
            double meanw = 0.0; for (double w : l1w2d) meanw += w;
            meanw = (l1w2d.empty() ? 1.0 : meanw / l1w2d.size());
            if (meanw > 0) for (double& w : l1w2d) w /= meanw;*/



            // 7. 生成字典矩阵 A (M × (numTau·numBeta))
            DictionaryConfig dcfg;
            dcfg.tau_list = taus;
            dcfg.gamma_list = betas;
            dcfg.enable_cache = false;
            dcfg.include_constant_basis = false;
            dcfg.cache_path = "results/cache.csv";
            Eigen::MatrixXcd A = DictionaryGenerator(dcfg).generate(omega);

            // 8. 构造 2D TV 差分算子 D2D
            Eigen::SparseMatrix<double> D2D = trspv::build2DTV(Nt, Nb);
            double dlogt = (Nt > 1 ? std::log(taus[1] / taus[0]) : 1.0);
            double dbeta = (Nb > 1 ? (betas[1] - betas[0]) : 1.0);
            //trspv::scaleTVBySteps(D2D, Nt, Nb, dlogt, dbeta);


            // 设定 group 放大因子（在做 ParamSelector 前设置 or 在 acfg 确认后设置均可）
            double gs = std::max(1, cfg.admm.group_size_tau) * std::max(1, cfg.admm.group_size_beta);
            if (gs > 1) cfg.admm.lambda1 *= std::sqrt((double)gs);

            // β 外、τ 内的列顺序 → stride = τ 个数
            cfg.admm.gamma_stride = (int)taus.size();
            cfg.admm.Nt = (int)taus.size();
            cfg.admm.Nb = (int)betas.size();

            // 9. ADMM 配置
            ADMMConfig strictCfg = cfg.admm;
            ADMMConfig scanCfg = strictCfg;
            scanCfg.max_iters = cfg.param_selection.scan_max_iters;
            scanCfg.tol_primal = cfg.param_selection.scan_tol;
            scanCfg.tol_dual = cfg.param_selection.scan_tol;
            scanCfg.l1_weights = l1w2d;

            // === 计算 block-Lasso 的 λ_max，按相对刻度设置 λ 扫描 ===
            // —— 带“组均权”的 λ1,max —— 与 z1 的 k = (λ1*wavg)/ρ 一致
            auto lambda1_max_block = [&](const Eigen::MatrixXcd& A,
                const Eigen::VectorXcd& b,
                int Nt, int Nb,
                int gs_tau, int gs_beta,
                int col_offset,                    // 0 or 1 (常数列偏移)
                const std::vector<double>& w2d)    // (β外、τ内) 展平权重，可空
                -> double {
                const Eigen::VectorXcd c = A.adjoint() * b;  // Aᴴ b
                const int Gt = std::max(1, gs_tau);
                const int Gb = std::max(1, gs_beta);
                const int stride = Nt;                       // β外、τ内
                const int betaCount = Nb;
                const int Ncols = (int)A.cols();

                auto group_wavg = [&](int bStart, int tStart, int bSpan, int tSpan) {
                    if (w2d.empty()) return 1.0;
                    double s = 0.0; int cnt = 0;
                    for (int gb = 0; gb < bSpan; ++gb)
                        for (int gt = 0; gt < tSpan; ++gt) {
                            int col = col_offset + (bStart + gb) * stride + (tStart + gt);
                            if ((unsigned)col < (unsigned)w2d.size()) { s += w2d[col]; ++cnt; }
                        }
                    return cnt ? (s / cnt) : 1.0;
                };

                double Lmax = 0.0;
                for (int bStart = 0; bStart < betaCount; bStart += Gb) {
                    for (int tStart = 0; tStart < stride; tStart += Gt) {
                        const int bSpan = std::min(Gb, betaCount - bStart);
                        const int tSpan = std::min(Gt, stride - tStart);

                        double n2 = 0.0;
                        for (int gb = 0; gb < bSpan; ++gb)
                            for (int gt = 0; gt < tSpan; ++gt) {
                                int col = col_offset + (bStart + gb) * stride + (tStart + gt);
                                if (col >= 0 && col < Ncols) n2 += std::norm(c[col]);
                            }

                        const double num = std::sqrt(n2);
                        const double wavg = group_wavg(bStart, tStart, bSpan, tSpan);
                        const double den = std::max(1e-12, wavg);        // 不用 √G 就别在这除

                        Lmax = std::max(Lmax, num / den);
                    }
                }
                return Lmax;
            };


            const int gs_tau = std::max(1, cfg.admm.group_size_tau);
            const int gs_beta = std::max(1, cfg.admm.group_size_beta);
            const int col_offset = dcfg.include_constant_basis ? 1 : 0;
            double Lmax = lambda1_max_block(A, b, Nt, Nb, gs_tau, gs_beta,col_offset, l1w2d);


            // 用 α·λmax 做扫描范围（把绝对上下限覆盖掉）
            double alph_min = 1e-5, alph_max = 1e-2;
            cfg.param_selection.lambda1_min = alph_min * Lmax;
            cfg.param_selection.lambda1_max = alph_max * Lmax;

            // （可选）同步给 TV 的扫描范围：固定比例相对 λ1
            cfg.param_selection.lambdat_min = 0.02 * cfg.param_selection.lambda1_min * dlogt;
            cfg.param_selection.lambdat_max = 0.08 * cfg.param_selection.lambda1_max * dlogt;
            cfg.param_selection.lambdab_min = 0.05 * cfg.param_selection.lambda1_min * dbeta;
            cfg.param_selection.lambdab_max = 0.2 * cfg.param_selection.lambda1_max * dbeta;


            ADMMConfig acfg = scanCfg;


            if (cfg.param_selection.enable) {
                ParamSelector selector(A, b, D2D, acfg, cfg.param_selection);
                acfg = selector.select();
                std::cout << "Auto-selected ADMM params:\n"
                    << "  lambda1       = " << acfg.lambda1 << "\n"
                    << "  lambda_tv_tau = " << acfg.lambda_tv_tau << "\n"
                    << "  lambda_tv_beta= " << acfg.lambda_tv_beta << "\n";
            }

            if (cfg.param_selection.refine_after) {
                ADMMConfig best = acfg;        // 先备份自动选择结果
                acfg = strictCfg;              // 换回严格迭代参数（iters/tols 等）
                acfg.lambda1 = best.lambda1;
                acfg.lambda_tv_tau = best.lambda_tv_tau;
                acfg.lambda_tv_beta = best.lambda_tv_beta;
                acfg.l1_weights = best.l1_weights;   // 别丢
                std::cout << "Refine ADMM with strict params: max_iters="
                    << acfg.max_iters << ", tol=" << acfg.tol_primal << "\n";
            }


            // 10. 运行 2D 反演
            acfg.l1_weights = l1w2d;
            //acfg.lambda_tv_tau *= std::sqrt(cfg.admm.group_size_tau);
            Solver2D solver(omega, b, taus, betas, acfg);
            Eigen::VectorXcd x2d = solver.solve();

            // 11. 输出矩阵形式的 rho 分布：实部、虚部、幅值
            auto dumpMatrix = [&](const std::string& filename,
                auto selector)
            {
                std::ofstream ofs(cfg.visualization.outputDir + "/" + filename);
                // 表头：tau, beta1, beta2, ...
                ofs << "tau";
                for (double beta : betas) ofs << ',' << beta;
                ofs << '\n';

                int N_tau = (int)taus.size();
                for (int i = 0; i < N_tau; ++i) {
                    ofs << taus[i];
                    for (int j = 0; j < (int)betas.size(); ++j) {
                        ofs << ',' << selector(x2d[i + j * N_tau]);
                    }
                    ofs << '\n';
                }
            };

            dumpMatrix("rho_real_matrix.csv", [](const auto& z) { return z.real(); });
            dumpMatrix("rho_imag_matrix.csv", [](const auto& z) { return z.imag(); });
            dumpMatrix("rho_abs_matrix.csv", [](const auto& z) { return std::abs(z); });

            // 12. 输出边际 tau 分布
            {
                std::ofstream ofs(cfg.visualization.outputDir + "/marginal_tau.csv");
                ofs << "tau,marginal_abs_rho\n";
                int N_tau = (int)taus.size(), N_beta = (int)betas.size();
                for (int i = 0; i < N_tau; ++i) {
                    double sum = 0.0;
                    for (int j = 0; j < N_beta; ++j) {
                        sum += std::abs(x2d[i + j * N_tau]);
                    }
                    ofs << taus[i] << ',' << sum << '\n';
                }
            }

            // 13. 输出边际 beta 分布
            {
                std::ofstream ofs(cfg.visualization.outputDir + "/marginal_beta.csv");
                ofs << "beta,marginal_abs_rho\n";
                int N_tau = (int)taus.size(), N_beta = (int)betas.size();
                for (int j = 0; j < N_beta; ++j) {
                    double sum = 0.0;
                    for (int i = 0; i < N_tau; ++i) {
                        sum += std::abs(x2d[i + j * N_tau]);
                    }
                    ofs << betas[j] << ',' << sum << '\n';
                }
            }

            // 14. 输出拟合 vs 原始数据对比
            {
                std::ofstream ofs(cfg.visualization.outputDir + "/fitted_vs_original.csv");
                ofs << "freq,orig_real,orig_imag,pred_real,pred_imag,abs_error\n";
                Eigen::VectorXcd b_pred = A * x2d;
                for (size_t i = 0; i < data.freq.size(); ++i) {
                    auto orig = data.values[i];
                    auto pred = b_pred[i];
                    ofs
                        << data.freq[i] << ','
                        << orig.real() << ','
                        << orig.imag() << ','
                        << pred.real() << ','
                        << pred.imag() << ','
                        << std::abs(pred - orig)
                        << '\n';
                }
            }

            // === 关键组分提取并落盘 ===
            auto comps = extract_components_cc(x2d, taus, betas, /*topK=*/6);

            // 计算连续光照(阶跃加光)的稳态总强度：sum A_k
            double total_signed = 0.0, total_abs = 0.0;
            for (auto& c : comps) { total_signed += c.amp; total_abs += std::abs(c.amp); }
            {
                std::ofstream ofs(cfg.visualization.outputDir + "/components.csv");
                ofs << "id,log10_tau,beta,amp,prominence,it,jb,frac_signed_on_ss,frac_abs_on_ss\n";
                for (size_t k = 0; k < comps.size(); ++k) {
                    double f_signed = (std::abs(total_signed) > 0.0)
                        ? (comps[k].amp / total_signed)
                        : std::numeric_limits<double>::quiet_NaN(); // 避免 0 作分母
                    double f_abs = (total_abs > 0.0)
                        ? (std::abs(comps[k].amp) / total_abs)
                        : std::numeric_limits<double>::quiet_NaN();

                    ofs << (k + 1) << ','
                        << std::log10(comps[k].tau) << ','
                        << comps[k].beta << ','
                        << comps[k].amp << ','
                        << comps[k].prominence << ','
                        << comps[k].it << ','
                        << comps[k].jb << ','
                        << f_signed << ','
                        << f_abs << '\n';
                }
            }

            // === 时域重建 ===
            if (!comps.empty()) {
                double tmin = 1e-3 * *std::min_element(taus.begin(), taus.end());
                double tmax = 10.0 * *std::max_element(taus.begin(), taus.end());
                int Nt = 300;
                std::vector<double> ts(Nt);
                double logt0 = std::log10(tmin), logt1 = std::log10(tmax);
                for (int i = 0; i < Nt; ++i) {
                    ts[i] = std::pow(10.0, logt0 + (logt1 - logt0) * i / (Nt - 1.0));
                }

                // a) 总曲线
                std::ofstream onTot(cfg.visualization.outputDir + "/transient_on_total.csv");
                std::ofstream offTot(cfg.visualization.outputDir + "/transient_off_total.csv");
                onTot << "t,SPV\n"; offTot << "t,SPV\n";
                for (double t : ts) {
                    double y_on = 0, y_off = 0;
                    for (auto& c : comps) {
                        y_on += c.amp * h_on(t, c.tau, c.beta);
                        y_off += c.amp * h_off(t, c.tau, c.beta);
                    }
                    onTot << std::setprecision(12) << t << ',' << y_on << '\n';
                    offTot << std::setprecision(12) << t << ',' << y_off << '\n';
                }

                // b) 各组件
                for (size_t k = 0; k < comps.size(); ++k) {
                    std::ofstream onK(cfg.visualization.outputDir + "/transient_on_comp_" + std::to_string(k + 1) + ".csv");
                    std::ofstream offK(cfg.visualization.outputDir + "/transient_off_comp_" + std::to_string(k + 1) + ".csv");
                    onK << "t,SPV\n"; offK << "t,SPV\n";
                    for (double t : ts) {
                        double y_on = comps[k].amp * h_on(t, comps[k].tau, comps[k].beta);
                        double y_off = comps[k].amp * h_off(t, comps[k].tau, comps[k].beta);
                        onK << std::setprecision(12) << t << ',' << y_on << '\n';
                        offK << std::setprecision(12) << t << ',' << y_off << '\n';
                    }
                }
            }

            // === 评估指标 ===
            {
                Eigen::VectorXcd b_pred = A * x2d;  // 复用/或重新计算
                const size_t N = data.freq.size();
                auto w_of = [&](size_t i) {
                    return (i < data.weights.size() ? data.weights[i] : 1.0);
                };

                // 残差与加权
                double rss_w = 0.0, wsum = 0.0;
                double rss_r = 0.0, rss_i = 0.0, tss_r = 0.0, tss_i = 0.0;
                double mean_r = 0.0, mean_i = 0.0, wsum_mean = 0.0;

                // 先算加权均值
                for (size_t i = 0; i < N; ++i) {
                    double w = w_of(i);
                    mean_r += w * data.values[i].real();
                    mean_i += w * data.values[i].imag();
                    wsum_mean += w;
                }
                mean_r /= std::max(1e-16, wsum_mean);
                mean_i /= std::max(1e-16, wsum_mean);

                for (size_t i = 0; i < N; ++i) {
                    double w = w_of(i);
                    auto e = b_pred[i] - data.values[i];
                    rss_w += w * std::norm(e);
                    wsum += w;

                    rss_r += w * std::pow(b_pred[i].real() - data.values[i].real(), 2);
                    rss_i += w * std::pow(b_pred[i].imag() - data.values[i].imag(), 2);
                    tss_r += w * std::pow(data.values[i].real() - mean_r, 2);
                    tss_i += w * std::pow(data.values[i].imag() - mean_i, 2);
                }
                double wrmse = std::sqrt(rss_w / std::max(1e-16, wsum));
                double R2_real = 1.0 - rss_r / std::max(1e-16, tss_r);
                double R2_imag = 1.0 - rss_i / std::max(1e-16, tss_i);

                // 有效自由度≈非零系数数（阈值）
                int nz = 0;
                double thr_nz = 1e-6 * x2d.cwiseAbs().maxCoeff();
                for (int k = 0; k < x2d.size(); ++k) if (std::abs(x2d[k]) > thr_nz) ++nz;
                int nobs = int(2 * N); // 实部+虚部
                int dof = std::max(1, nobs - nz);

                double chi2_red = (rss_w / std::max(1e-16, wsum)) * (nobs / double(dof));
                double RSS = 0.0; for (size_t i = 0; i < N; ++i) RSS += std::norm(b_pred[i] - data.values[i]);
                double AIC = 2.0 * nz + nobs * std::log(std::max(1e-300, RSS / nobs));
                double BIC = nz * std::log(std::max(1, nobs)) + nobs * std::log(std::max(1e-300, RSS / nobs));

                // 写 metrics.json
                using json = nlohmann::json;
                json mj;
                mj["weighted_rmse"] = wrmse;
                mj["R2_real"] = R2_real;
                mj["R2_imag"] = R2_imag;
                mj["chi2_reduced"] = chi2_red;
                mj["nz_coeffs"] = nz;
                mj["AIC"] = AIC;
                mj["BIC"] = BIC;
                std::ofstream jm(cfg.visualization.outputDir + "/metrics.json");
                jm << mj.dump(2) << std::endl;

                // 写 summary.json（含关键组分）
                json sj;
                sj["num_points"] = N;
                sj["tau_range"] = { *std::min_element(taus.begin(),taus.end()),
                                    *std::max_element(taus.begin(),taus.end()) };
                sj["beta_range"] = { *std::min_element(betas.begin(),betas.end()),
                                     *std::max_element(betas.begin(),betas.end()) };
                sj["metrics"] = mj;
                auto& arr = sj["components"] = json::array();
                for (size_t k = 0; k < comps.size(); ++k) {
                    json c;
                    c["id"] = int(k + 1);
                    c["tau"] = comps[k].tau;
                    c["log10_tau"] = std::log10(comps[k].tau);
                    c["beta"] = comps[k].beta;
                    c["amp"] = comps[k].amp;
                    c["prominence"] = comps[k].prominence;
                    c["it"] = comps[k].it;
                    c["jb"] = comps[k].jb;
                    arr.push_back(c);
                }
                std::ofstream js(cfg.visualization.outputDir + "/summary.json");
                js << sj.dump(2) << std::endl;
            }



            std::cout << "All results written to " << cfg.visualization.outputDir << std::endl;
            return 0;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }


