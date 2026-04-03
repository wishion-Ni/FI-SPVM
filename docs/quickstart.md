# Quickstart

## Prerequisites

- A C++ toolchain with C++17 support.
- Dependencies from `vcpkg.json`, including `eigen3`, `nlohmann-json`, `fftw3`, `spdlog`, and `gtest`.
- A local build system or IDE project configured to compile the sources in this repository.

## Basic Run

The unified CLI entry is `apps/trspv_cli/main.cpp`. The expected invocation pattern is:

```bash
trspv --conf config.json
```

Compatible legacy form:

```bash
trspv config.json
```

Optional overrides:

```bash
trspv --conf config.json --input path/to/data.csv --out results/run1
```

## Expected Outputs

The solver writes results under `visualization.outputDir` from the configuration, including:

- `admm_summary.txt`
- `components.txt`
- `metrics.json`
- `summary.json`
- `transient_on_total.csv`
- `transient_off_total.csv`

If interpolation is enabled, additional interpolation comparison files are generated.

## Development Notes

- The CLI entry should remain thin; workflow logic belongs in `SolverApp`.
- Output file generation should remain in `ResultWriter`.
- Config compatibility and validation should remain centralized in `ConfigLoader` and `validate_config`.

## Tests

The repository now contains `tests/config_tests.cpp` for config parsing and validation coverage.
Hook it into your local build once the project build files are available in version control.
