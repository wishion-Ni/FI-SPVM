#include "Logger.h"
#include <iostream>

namespace trspv {

    void Logger::init(const std::string& log_file, LogLevel level) {
        try {
            // 如果已有名为 file_logger 的 logger，先移除它
            if (auto existing = spdlog::get("file_logger"); existing) {
                spdlog::drop("file_logger");
            }
            auto file_logger = spdlog::basic_logger_mt("file_logger", log_file);
            spdlog::set_default_logger(file_logger);
            switch (level) {
            case LogLevel::Debug: spdlog::set_level(spdlog::level::debug); break;
            case LogLevel::Info:  spdlog::set_level(spdlog::level::info);  break;
            case LogLevel::Warn:  spdlog::set_level(spdlog::level::warn);  break;
            case LogLevel::Error: spdlog::set_level(spdlog::level::err);   break;
            }
            // 当级别为 info 或更高时，自动 flush
            spdlog::flush_on(spdlog::level::info);
        }
        catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        }
    }

    void Logger::debug(const std::string& msg) {
        spdlog::debug(msg);
    }

    void Logger::info(const std::string& msg) {
        spdlog::info(msg);
    }

    void Logger::warn(const std::string& msg) {
        spdlog::warn(msg);
    }

    void Logger::error(const std::string& msg) {
        spdlog::error(msg);
    }

} // namespace trspv
