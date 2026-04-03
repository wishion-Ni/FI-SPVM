#include "DictionaryGenerator.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace trspv {

DictionaryGenerator::DictionaryGenerator(const DictionaryConfig& cfg)
    : config_(cfg) {}

Eigen::MatrixXcd DictionaryGenerator::generate(const std::vector<double>& omega) {
    const size_t M = omega.size();
    const size_t T = config_.include_constant_basis
        ? config_.tau_list.size() - 1
        : config_.tau_list.size();
    const size_t G = config_.gamma_list.size();
    const size_t N = T * G + (config_.include_constant_basis ? 1 : 0);

    Eigen::MatrixXcd A;
    /*if (config_.enable_cache && loadCache(A)) {
        return A;
    }*/

    A.resize(M, N);

    // const double sigma = config_.basis_sigma_dec;
    const double sigma = 0.0;
    size_t col = 0;

    // if (config_.include_constant_basis) { ... }

    // beta outer loop, tau inner loop
    // config_.gamma_stride = static_cast<int>(config_.tau_list.size());
    for (double gamma : config_.gamma_list) {
        for (double tau_center : config_.tau_list) {
            for (size_t i = 0; i < M; ++i) {
                A(i, col) = KernelFunction::evaluate(
                    omega[i],
                    tau_center,
                    tau_center,
                    gamma,
                    sigma);
            }

            // Normalize each column to reduce scale differences across beta.
            double nrm2 = 0.0;
            for (size_t i = 0; i < M; ++i) {
                const auto& c = A(i, col);
                nrm2 += std::norm(c);
            }
            const double nrm = std::sqrt(nrm2);
            if (nrm > 0.0) {
                for (size_t i = 0; i < M; ++i) {
                    A(i, col) /= nrm;
                }
            }

            ++col;
        }
    }

    if (config_.enable_cache) {
        saveCache(A);
    }
    return A;
}

void DictionaryGenerator::saveCache(const Eigen::MatrixXcd& A) {
    const auto& path_str = config_.cache_path;
    if (path_str.empty()) {
        std::cerr << "[DictGen] cache_path is empty, skipping saveCache\n";
        return;
    }

    std::filesystem::path p(path_str);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) {
            std::cerr << "[DictGen] Failed to create cache directory "
                      << p.parent_path() << ": " << ec.message() << "\n";
            return;
        }
    }

    // Store the matrix as text CSV for readability and portability.
    std::ofstream ofs(config_.cache_path, std::ios::out);
    if (!ofs.is_open()) {
        std::cerr << "[DictGen] Failed to open cache file for writing: "
                  << config_.cache_path << std::endl;
        return;
    }

    const int rows = static_cast<int>(A.rows());
    const int cols = static_cast<int>(A.cols());
    ofs << "# rows=" << rows << ", cols=" << cols << "\n";

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const auto& c = A(i, j);
            ofs << c.real() << "," << c.imag();
            if (j + 1 < cols) {
                ofs << ",";
            }
        }
        ofs << "\n";
    }
    ofs.close();
    std::cout << "[DictGen] Cached matrix A to CSV: "
              << config_.cache_path << std::endl;
}

bool DictionaryGenerator::loadCache(Eigen::MatrixXcd& A) {
    namespace fs = std::filesystem;
    if (!fs::exists(config_.cache_path)) {
        return false;
    }

    std::ifstream ifs(config_.cache_path);
    if (!ifs.is_open()) {
        return false;
    }

    std::string header;
    std::getline(ifs, header);
    const auto pos_r = header.find("rows=");
    const auto pos_c = header.find("cols=");
    if (pos_r == std::string::npos || pos_c == std::string::npos) {
        std::cerr << "[DictGen] Unexpected cache header: " << header << std::endl;
        return false;
    }

    const int rows = std::stoi(header.substr(pos_r + 5));
    const int cols = std::stoi(header.substr(pos_c + 5));
    std::cout << "[DictGen] Parsed cache header rows=" << rows
              << ", cols=" << cols << std::endl;

    A.resize(rows, cols);
    std::string line;
    for (int i = 0; i < rows; ++i) {
        if (!std::getline(ifs, line)) {
            std::cerr << "[DictGen] Unexpected EOF at row " << i << std::endl;
            break;
        }
        std::stringstream ss(line);
        for (int j = 0; j < cols; ++j) {
            double real = 0.0;
            double imag = 0.0;
            char comma = 0;
            ss >> real;
            ss >> comma;
            ss >> imag;
            if (j + 1 < cols) {
                ss >> comma;
            }
            A(i, j) = std::complex<double>(real, imag);
        }
    }

    ifs.close();
    std::cout << "[DictGen] Loaded matrix A from CSV cache: "
              << config_.cache_path << std::endl;
    return true;
}

} // namespace trspv
