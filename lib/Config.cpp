#include "Config.h"
#include <fstream>
#include <stdexcept>

static trspv::CompletionMethod parseMethod(const std::string& s) {
    if (s == "PCHIP") return trspv::CompletionMethod::PCHIP;
    if (s == "Akima") return trspv::CompletionMethod::Akima;
    return trspv::CompletionMethod::None;
}

using json = nlohmann::json;

namespace trspv {

    void from_json(const json& j, FindPeaksConfig& fp) {
        fp.enable = j.value("enable", true);
        fp.interp_type = j.value("interp_type", std::string("akima"));
        fp.smooth_window = j.value("smooth_window", 5);
        fp.peak_prominence = j.value("peak_prominence", 0.02);
        fp.peak_dist_dec = j.value("peak_dist_dec", 0.15);
        fp.interp_factor = j.value("interp_factor", 4);
        fp.weight_factor = j.value("weight_factor", 0.2);
    }

    void from_json(const json& j, PreprocessConfig& p) {
        if (j.contains("find_peaks"))
            j.at("find_peaks").get_to(p.find_peaks);
    }

    void from_json(const json& j, Config& c)
    {
        if (j.contains("preprocess"))
            j.at("preprocess").get_to(c.preprocess);   
    }

    Config ConfigLoader::from_file(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            throw std::runtime_error("无法打开配置文件: " + path);
        }
        json j;
        try {
            ifs >> j;
        }
        catch (const json::parse_error& e) {
            throw std::runtime_error(std::string("JSON 解析错误: ") + e.what());
        }

        Config cfg; // 默认值已在结构体中初始化

        // 顶层兼容
        if (j.contains("inputFile"))       cfg.inputFile = j["inputFile"].get<std::string>();
        if (j.contains("noiseWeighted"))   cfg.noiseWeighted = j["noiseWeighted"].get<bool>();
        if (j.contains("spectrum_input_type"))
            cfg.spectrum_input_type = j["spectrum_input_type"].get<std::string>();


        // data 段
        if (j.contains("data")) {
            auto& jd = j["data"];
            if (jd.contains("input_file")) cfg.inputFile = jd["input_file"].get<std::string>();
            if (jd.contains("noise_weighted")) cfg.noiseWeighted = jd["noise_weighted"].get<bool>();
            if (jd.contains("input_type")) cfg.spectrum_input_type = jd["input_type"].get<std::string>();
        }

        // kernel 段
        if (j.contains("kernel")) {
            auto& jk = j["kernel"];
            if (jk.contains("tau_min"))    cfg.kernel.tau_min = jk["tau_min"].get<double>();
            if (jk.contains("tau_max"))    cfg.kernel.tau_max = jk["tau_max"].get<double>();
            if (jk.contains("num_tau"))    cfg.kernel.num_tau = jk["num_tau"].get<int>();
            if (jk.contains("gamma_min"))   cfg.kernel.gamma_min = jk["gamma_min"].get<double>();
            if (jk.contains("gamma_max"))   cfg.kernel.gamma_max = jk["gamma_max"].get<double>();
            if (jk.contains("num_gamma"))   cfg.kernel.num_gamma = jk["num_gamma"].get<int>();
            if (jk.contains("gamma_scale")) cfg.kernel.gamma_scale = jk["gamma_scale"].get<std::string>();
        }

        // admm 段
        if (j.contains("admm")) {
            auto & ja = j["admm"];
            if (ja.contains("lambda1"))         cfg.admm.lambda1 = ja["lambda1"].get<double>();
            if (ja.contains("lambda_tv_tau"))   cfg.admm.lambda_tv_tau = ja["lambda_tv_tau"].get<double>();
            if (ja.contains("lambda_tv_beta"))  cfg.admm.lambda_tv_beta = ja["lambda_tv_beta"].get<double>();
            if (ja.contains("rho"))             cfg.admm.rho = ja["rho"].get<double>();
            if (ja.contains("tol_primal"))      cfg.admm.tol_primal = ja["tol_primal"].get<double>();
            if (ja.contains("tol_dual"))        cfg.admm.tol_dual = ja["tol_dual"].get<double>();
            if (ja.contains("max_iters"))       cfg.admm.max_iters = ja["max_iters"].get<int>();
            if (ja.contains("group_size_tau"))  cfg.admm.group_size_tau = ja["group_size_tau"].get<int>();
            if (ja.contains("group_size_beta")) cfg.admm.group_size_beta = ja["group_size_beta"].get<int>();
            if (ja.contains("gamma_stride"))    cfg.admm.gamma_stride = ja["gamma_stride"].get<int>();

        }

        // priors 段
        if (j.contains("priors")) {
            auto& jp = j["priors"];
            if (jp.contains("beta_center"))   cfg.priors.beta_center = jp["beta_center"].get<double>();
            if (jp.contains("beta_sigma"))    cfg.priors.beta_sigma = jp["beta_sigma"].get<double>();
            if (jp.contains("beta_strength")) cfg.priors.beta_strength = jp["beta_strength"].get<double>();
        }


        // normalization 段
        if (j.contains("normalization")) {
            auto& jn = j["normalization"];
            if (jn.contains("enabled")) cfg.normalization.enabled = jn["enabled"].get<bool>();
        }

        // spectrum_completion 段
        if (j.contains("spectrum_completion")) {
            auto& jc = j["spectrum_completion"];
            if (jc.contains("method")) {
                std::string m = jc["method"].get<std::string>();
                cfg.completion.method = parseMethod(m);
            }
            if (jc.contains("interpolate"))
                cfg.completion.interpolate = jc["interpolate"].get<bool>();
            if (jc.contains("log_space"))
                cfg.completion.log_space = jc["log_space"].get<bool>();
            if (jc.contains("weight"))
                cfg.completion.weight = jc["weight"].get<double>();
            if (jc.contains("num_points"))
                cfg.completion.num_points = jc["num_points"].get<int>();
        }

        // logging 段
        if (j.contains("logging")) {
            auto& jl = j["logging"];
            if (jl.contains("file"))             cfg.logging.file = jl["file"].get<std::string>();
            if (jl.contains("level"))            cfg.logging.level = jl["level"].get<std::string>();
            if (jl.contains("snapshotInterval")) cfg.logging.snapshotInterval = jl["snapshotInterval"].get<int>();
        }

        // visualization 段
        if (j.contains("visualization")) {
            auto& jv = j["visualization"];
            if (jv.contains("enabled")) cfg.visualization.enabled = jv["enabled"].get<bool>();
            if (jv.contains("outputDir")) cfg.visualization.outputDir = jv["outputDir"].get<std::string>();
        }

        // param_selection 段
        if (j.contains("param_selection")) {
            auto & jp = j["param_selection"];
            auto & psc = cfg.param_selection;
            if (jp.contains("enable"))       psc.enable = jp["enable"].get<bool>();
            if (jp.contains("num_lambda1"))  psc.num_lambda1 = jp["num_lambda1"].get<int>();
            if (jp.contains("lambda1_min"))  psc.lambda1_min = jp["lambda1_min"].get<double>();
            if (jp.contains("lambda1_max"))  psc.lambda1_max = jp["lambda1_max"].get<double>();
            if (jp.contains("num_lambdat"))  psc.num_lambdat = jp["num_lambdat"].get<int>();
            if (jp.contains("lambdat_min"))  psc.lambdat_min = jp["lambdat_min"].get<double>();
            if (jp.contains("lambdat_max"))  psc.lambdat_max = jp["lambdat_max"].get<double>();
            if (jp.contains("num_lambdab"))  psc.num_lambdab = jp["num_lambdab"].get<int>();
            if (jp.contains("lambdab_min"))  psc.lambdab_min = jp["lambdab_min"].get<double>();
            if (jp.contains("lambdab_max"))  psc.lambdab_max = jp["lambdab_max"].get<double>();
                    // 默认写到 visualization.outputDir，JSON 可覆盖
                psc.outputDir = cfg.visualization.outputDir;
            if (jp.contains("outputDir"))    psc.outputDir = jp["outputDir"].get<std::string>();
            if (jp.contains("scan_max_iters")) psc.scan_max_iters = jp["scan_max_iters"].get<int>();
            if (jp.contains("scan_tol"))       psc.scan_tol = jp["scan_tol"].get<double>();
            if (jp.contains("refine_after"))   psc.refine_after = jp["refine_after"].get<bool>();
            
        }
        return cfg;
    }

} // namespace trspv