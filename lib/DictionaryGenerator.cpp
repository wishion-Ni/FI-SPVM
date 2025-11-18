#include "DictionaryGenerator.h"
#include <fstream>
#include <filesystem>
#include <iostream>

namespace trspv {

    DictionaryGenerator::DictionaryGenerator(const DictionaryConfig& cfg)
        : config_(cfg) {}

    Eigen::MatrixXcd DictionaryGenerator::generate(const std::vector<double>& omega) {
        size_t M = omega.size();
        size_t T = config_.include_constant_basis
            ? config_.tau_list.size() - 1
            : config_.tau_list.size();
        size_t G = config_.gamma_list.size();
        size_t N = T * G + (config_.include_constant_basis ? 1 : 0);

        Eigen::MatrixXcd A;
        /*if (config_.enable_cache && loadCache(A)) {
            return A;
        }*/
        
        A.resize(M, N);

 //       const double sigma = config_.basis_sigma_dec;

        const double sigma = 0.0;                 // 先关闭展宽
        size_t col = 0;

 //       if (config_.include_constant_basis) { ... }   // 原常数列保持

        /* ---- β 外层，τ 内层 ---- */
        //config_.gamma_stride = static_cast<int>(config_.tau_list.size());   // 写回 stride
        for (double gamma : config_.gamma_list) {           // β 外
            for (double tau_center : config_.tau_list) {    // τ 内
                for (size_t i = 0; i < M; ++i) {
                    A(i, col) = KernelFunction::evaluate(
                        omega[i],
                        tau_center, tau_center,   // row = center
                        gamma, sigma);
                }
                // === 列 L2 归一化（消除不同 β 的列范数差异）===
                double nrm2 = 0.0;
                for (size_t i = 0; i < M; ++i) {
                    const auto& c = A(i, col);
                    nrm2 += std::norm(c);
                }
                double nrm = std::sqrt(nrm2);
                if (nrm > 0.0) {
                    for (size_t i = 0; i < M; ++i) A(i, col) /= nrm;
                }

                ++col;
            }
        }


        saveCache(A);

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
        // 创建父目录（如果有的话）
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
            if (ec) {
                std::cerr << "[DictGen] Failed to create cache directory "
                    << p.parent_path() << ": " << ec.message() << "\n";
                return;
            }
        }
        // 原来的二进制写入改为文本 CSV 写入
        std::ofstream ofs(config_.cache_path, std::ios::out);
        if (!ofs.is_open()) {
            std::cerr << "[DictGen] Failed to open cache file for writing: "
                << config_.cache_path << std::endl;
            return;
        }

        const int rows = static_cast<int>(A.rows());
        const int cols = static_cast<int>(A.cols());
        // 可选：在文件头写一行尺寸信息
        ofs << "# rows=" << rows << ", cols=" << cols << "\n";

        // 写入每一行，每列格式为 real,imag
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                const auto& c = A(i, j);
                ofs << c.real() << "," << c.imag();
                if (j + 1 < cols) ofs << ",";
            }
            ofs << "\n";
        }
        ofs.close();
        std::cout << "[DictGen] Cached matrix A to CSV: "
            << config_.cache_path << std::endl;
    }

    bool DictionaryGenerator::loadCache(Eigen::MatrixXcd& A) {
        namespace fs = std::filesystem;
        // 1. 检查文件是否存在
        if (!fs::exists(config_.cache_path)) return false;

        std::ifstream ifs(config_.cache_path);
        if (!ifs.is_open()) return false;

        // 2. 读取并解析 header 行，格式示例： "# rows=50 , cols=21"
        std::string header;
        std::getline(ifs, header);
        auto pos_r = header.find("rows=");
        auto pos_c = header.find("cols=");
        if (pos_r == std::string::npos || pos_c == std::string::npos) {
            std::cerr << "[DictGen] Unexpected cache header: " << header << std::endl;
            return false;
        }
        // 提取数字部分
        int rows = std::stoi(header.substr(pos_r + 5));
        int cols = std::stoi(header.substr(pos_c + 5));
        std::cout << "[DictGen] Parsed cache header rows=" << rows
            << ", cols=" << cols << std::endl;

        // 3. 按行读取 CSV 数据，填充矩阵
        A.resize(rows, cols);
        std::string line;
        for (int i = 0; i < rows; ++i) {
            if (!std::getline(ifs, line)) {
                std::cerr << "[DictGen] Unexpected EOF at row " << i << std::endl;
                break;
            }
            std::stringstream ss(line);
            for (int j = 0; j < cols; ++j) {
                double real = 0.0, imag = 0.0;
                char comma;
                ss >> real;           // 读实部
                ss >> comma;          // 读第一个逗号
                ss >> imag;           // 读虚部
                if (j + 1 < cols) ss >> comma; // 如果不是本行最后一列，再读一个列间逗号
                A(i, j) = std::complex<double>(real, imag);
            }
        }
        ifs.close();
        std::cout << "[DictGen] Loaded matrix A from CSV cache: "
            << config_.cache_path << std::endl;
        return true;
    }

} // namespace trspv
