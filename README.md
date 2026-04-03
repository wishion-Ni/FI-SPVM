# FI-SPVM

The repository is organized into:

- `lib/`: core algorithms, config, optimization, and output writers.
- `src/`: application orchestration (`SolverApp`).
- `apps/`: thin executable entry points (`apps/trspv_cli/main.cpp`).
- `tests/`: unit and smoke tests.

## Build and test from scratch

This project uses `CMake + CMakePresets + vcpkg manifest`.

Prerequisites:

- C++17 compiler toolchain
- CMake 3.24+
- Git (for automatic `vcpkg` bootstrap)
- On Windows: Visual Studio Build Tools (2019/2022) or Ninja toolchain

Quick path (recommended for new users):

Windows (PowerShell):

```powershell
.\scripts\build-test.ps1
```

Linux (bash):

```bash
bash ./scripts/build-test.sh
```

Manual path (preset commands):

Windows (PowerShell):

```powershell
# first-time bootstrap
.\scripts\bootstrap-vcpkg.ps1

# single workflow entry
cmake --workflow --preset workflow-dev-windows
```

If your machine needs explicit generator selection, use one of:

- `dev-windows-vs2022`
- `dev-windows-vs2019`
- `dev-windows-ninja`
- `dev-windows-no-openmp`

Linux (bash):

```bash
bash ./scripts/bootstrap-vcpkg.sh
cmake --workflow --preset workflow-dev-linux
```

Optional serial fallback validation:

```powershell
cmake --workflow --preset workflow-dev-windows-no-openmp
```

```bash
cmake --workflow --preset workflow-dev-linux-no-openmp
```

## Run CLI

```powershell
out/build/dev-windows/Release/trspv_cli.exe --conf config.json
```

Compatible legacy form:

```powershell
out/build/dev-windows/Release/trspv_cli.exe --conf examples/basic_run/config.json
```

## Notes

- The solver pipeline is unified through `SolverApp::run(argc, argv)`.
- `src/trSPVSolver.cpp` is a compatibility placeholder and is not part of build targets.
- Config paths are resolved relative to the config file location by default.
- CLI overrides (`--input`, `--out`) are resolved relative to the invocation directory.

## Refactor task hub

Task checklists and log templates are tracked under:

- `task-hub/README.md`
- `task-hub/2026-04-02_v1.0/`
