#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "KernelFunction.h"
#include "Config.h"

namespace trspv {

    // Configuration for dictionary generation and caching
    struct DictionaryConfig {
        std::vector<double> tau_list;
        std::vector<double> gamma_list;
        bool enable_cache = false;
        bool include_constant_basis = false;
        std::string cache_path;
        int gamma_stride = 0;
    };

    class DictionaryGenerator {
    public:
        DictionaryGenerator(const DictionaryConfig& cfg);
        // Generate or load the dictionary matrix of size (omega.size() x (tau_list.size() * gamma_list.size()))
        Eigen::MatrixXcd generate(const std::vector<double>& omega);

    private:
        DictionaryConfig config_;
        // Serialize to binary cache
        void saveCache(const Eigen::MatrixXcd& A);
        // Load from binary cache (return true if successful)
        bool loadCache(Eigen::MatrixXcd& A);
    };

} // namespace trspv
