cmake_minimum_required(VERSION 3.24)

project(FI_SPVM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FI_SPVM_ENABLE_OPENMP "Enable OpenMP acceleration when available." ON)
option(FI_SPVM_BUILD_TESTS "Build test targets." ON)

include(CTest)
if(NOT FI_SPVM_BUILD_TESTS)
    set(BUILD_TESTING OFF CACHE BOOL "Enable testing" FORCE)
endif()

find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

set(FI_SPVM_CORE_SOURCES
    lib/ADMMOptimizer.cpp
    lib/ComponentAnalysis.cpp
    lib/Config.cpp
    lib/DictionaryGenerator.cpp
    lib/KernelFunction.cpp
    lib/Logger.cpp
    lib/ParamSelector.cpp
    lib/PeakSeedDetector.cpp
    lib/ResultWriter.cpp
    lib/Solver2D.cpp
    lib/SpectrumCompletion.cpp
    lib/SpectrumData.cpp
    lib/Utils.cpp
)

add_library(fi_spvm_core STATIC ${FI_SPVM_CORE_SOURCES})
target_link_libraries(fi_spvm_core
    PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)
target_include_directories(fi_spvm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(FI_SPVM_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fi_spvm_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(fi_spvm_core PUBLIC FI_SPVM_HAS_OPENMP=1)
        message(STATUS "OpenMP enabled for fi_spvm_core")
    else()
        message(STATUS "OpenMP not found; building with serial fallback")
    endif()
endif()

add_library(fi_spvm_app STATIC src/SolverApp.cpp)
target_link_libraries(fi_spvm_app PUBLIC fi_spvm_core)
target_include_directories(fi_spvm_app
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(trspv_cli apps/trspv_cli/main.cpp)
target_link_libraries(trspv_cli PRIVATE fi_spvm_app)

if(BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)
    include(GoogleTest)

    add_executable(config_tests tests/config_tests.cpp)
    target_link_libraries(config_tests PRIVATE fi_spvm_core GTest::gtest_main)
    gtest_discover_tests(config_tests DISCOVERY_MODE PRE_TEST)

    add_executable(loader_tests tests/loader_tests.cpp)
    target_link_libraries(loader_tests PRIVATE fi_spvm_core GTest::gtest_main)
    gtest_discover_tests(loader_tests DISCOVERY_MODE PRE_TEST)

    add_executable(completion_tests tests/completion_tests.cpp)
    target_link_libraries(completion_tests PRIVATE fi_spvm_core GTest::gtest_main)
    gtest_discover_tests(completion_tests DISCOVERY_MODE PRE_TEST)

    add_executable(solver_tests tests/solver_tests.cpp)
    target_link_libraries(solver_tests PRIVATE fi_spvm_core GTest::gtest_main)
    gtest_discover_tests(solver_tests DISCOVERY_MODE PRE_TEST)

    add_executable(cli_smoke_test tests/cli_smoke_test.cpp)
    target_link_libraries(cli_smoke_test PRIVATE fi_spvm_app nlohmann_json::nlohmann_json)
    file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" FI_SPVM_SOURCE_DIR_CMAKE)
    target_compile_definitions(cli_smoke_test
        PRIVATE
            FI_SPVM_SOURCE_DIR="${FI_SPVM_SOURCE_DIR_CMAKE}"
    )

    add_test(NAME cli_smoke_test COMMAND cli_smoke_test)
endif()
