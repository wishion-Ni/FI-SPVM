# Quickstart

## Prerequisites

- C++17 compiler toolchain.
- CMake 3.24 or newer.
- Git (for automatic `vcpkg` bootstrap).
- On Windows: Visual Studio Build Tools (2019/2022) or Ninja toolchain.

## Configure and Build

Quick path (recommended):

Windows:

```powershell
.\scripts\build-test.ps1
```

Linux:

```bash
bash ./scripts/build-test.sh
```

Manual preset path:

Windows:

```powershell
.\scripts\bootstrap-vcpkg.ps1
cmake --workflow --preset workflow-dev-windows
```

Windows generator fallback presets (if needed):

- `dev-windows-vs2022`
- `dev-windows-vs2019`
- `dev-windows-ninja`
- `dev-windows-no-openmp`

Linux:

```bash
bash ./scripts/bootstrap-vcpkg.sh
cmake --workflow --preset workflow-dev-linux
```

Optional serial fallback validation:

Windows:

```powershell
cmake --workflow --preset workflow-dev-windows-no-openmp
```

Linux:

```bash
cmake --workflow --preset workflow-dev-linux-no-openmp
```

## Basic Run

Windows:

```powershell
out/build/dev-windows/Release/trspv_cli.exe --conf config.json
out/build/dev-windows/Release/trspv_cli.exe --conf examples/basic_run/config.json
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
- Config-relative paths are resolved against the config file directory.
