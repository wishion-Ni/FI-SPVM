# FI-SPVM Architecture

## Overview

The repository is organized around a thin CLI entry, a solver orchestration layer,
and reusable library modules:

- `apps/trspv_cli/main.cpp`: top-level executable entry.
- `src/SolverApp.*`: argument handling, config loading, solver orchestration.
- `lib/`: domain logic, optimization, config parsing, IO writers, and utilities.
- `tests/`: unit tests and runtime smoke tests.

## Build Targets

The CMake build is split into three main targets:

- `fi_spvm_core`: all reusable library implementations from `lib/`.
- `fi_spvm_app`: orchestration layer (`src/SolverApp.cpp`) linked to `fi_spvm_core`.
- `trspv_cli`: executable entry (`apps/trspv_cli/main.cpp`) linked to `fi_spvm_app`.

Compatibility file `src/trSPVSolver.cpp` is intentionally not part of build targets.

## Runtime Flow

1. `main` forwards CLI arguments to `SolverApp::run`.
2. `ConfigLoader::from_file` loads and validates the JSON configuration.
3. `ConfigLoader` resolves config-relative paths for input, logging, and outputs.
4. `SolverApp` loads input spectra and optionally runs interpolation/completion.
5. `SolverApp` builds the dictionary matrix, TV operator, and ADMM scan settings.
6. `Solver2D` performs optional parameter scanning and the final optimization.
7. `ComponentAnalysis` extracts domain components from the solved 2D distribution.
8. `ResultWriter` emits backward-compatible output files.

## Module Responsibilities

### Entry and orchestration

- `apps/trspv_cli/main.cpp`: top-level exception boundary.
- `src/SolverApp.*`: CLI compatibility, pipeline sequencing, high-level workflow.

### Domain and math

- `lib/ComponentAnalysis.*`: component extraction and transient response helpers.
- `lib/DictionaryGenerator.*`: basis/dictionary construction.
- `lib/KernelFunction.*`: kernel definitions.
- `lib/Utils.*`: TV operator helpers.

### Optimization

- `lib/ADMMOptimizer.*`: low-level ADMM solver.
- `lib/ParamSelector.*`: coordinate-style parameter scan for `lambda1`, `lambda_tv_tau`, and `lambda_tv_beta`.
- `lib/Solver2D.*`: orchestration wrapper around parameter selection and final solve.

### Data and config

- `lib/SpectrumData.*`: spectrum loading.
- `lib/SpectrumCompletion.*`: interpolation/completion path.
- `lib/Config.*`: config parsing, compatibility mapping, and validation.

### Output and logging

- `lib/ResultWriter.*`: file output generation.
- `lib/Logger.*`: runtime logging utilities.

## Test Layering

- `config_tests`: verifies config parsing compatibility and validation behavior.
- `loader_tests`: verifies CSV parsing, input type handling, and weighting semantics.
- `completion_tests`: verifies interpolation behavior on tiny fixtures.
- `solver_tests`: verifies ADMM convergence and deterministic scan behavior.
- `cli_smoke_test`: executes the solver path with fixture input and checks key output files.

## Notes

- OpenMP is an optional acceleration path. When unavailable, build falls back to serial scans.
- `ParamSelector` writes a machine-readable `param_selection_report.json` into the configured scan output directory.
