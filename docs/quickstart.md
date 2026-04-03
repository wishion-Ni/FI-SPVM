# Quickstart

## Prerequisites

- C++17 compiler toolchain.
- CMake 3.24 or newer.
- `vcpkg` checkout with `VCPKG_ROOT` set in the environment.
- On Windows: Visual Studio 2022 or Build Tools 2022 with MSVC.

## Configure and Build

Windows:

```powershell
# run inside "Developer PowerShell for VS 2022"
$env:VCPKG_ROOT="C:\\path\\to\\vcpkg"
cmake --preset dev-windows
cmake --build --preset build-dev-windows --config Release
ctest --preset test-dev-windows --output-on-failure
```

Linux:

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset dev-linux
cmake --build --preset build-dev-linux
ctest --preset test-dev-linux --output-on-failure
```

## Basic Run

Windows:

```powershell
out/build/dev-windows/Release/trspv_cli.exe --conf config.json
```

Linux:

```bash
out/build/dev-linux/trspv_cli --conf config.json
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

- Keep CLI entry thin (`apps/trspv_cli/main.cpp`).
- Keep orchestration in `SolverApp`.
- Keep file output logic in `ResultWriter`.
- Keep config parsing and validation centralized in `ConfigLoader` / `validate_config`.
