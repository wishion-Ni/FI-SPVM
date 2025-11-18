#pragma once

#include <string>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

namespace trspv {

    enum class LogLevel {
        Debug,
        Info,
        Warn,
        Error
    };

    class Logger {
    public:
        static void init(const std::string& log_file, LogLevel level = LogLevel::Info);
        template<typename... Args>
        static void info(fmt::format_string<Args...> fmt, Args&&... args) {
            spdlog::info(std::move(fmt), std::forward<Args>(args)...);
        }

        static void debug(const std::string& msg);
        static void info(const std::string& msg);
        static void warn(const std::string& msg);
        static void error(const std::string& msg);
    };

} // namespace trspv
