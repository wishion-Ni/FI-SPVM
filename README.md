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

# auto generator selection
cmake --preset dev-windows
cmake --build --preset build-dev-windows --config Release
ctest --preset test-dev-windows --output-on-failure
```

If your machine needs explicit generator selection, use one of:

- `dev-windows-vs2022`
- `dev-windows-vs2019`
- `dev-windows-ninja`

Linux (bash):

```bash
bash ./scripts/bootstrap-vcpkg.sh
cmake --preset dev-linux
cmake --build --preset build-dev-linux
ctest --preset test-dev-linux --output-on-failure
```

## Run CLI

```bash
trspv_cli --conf config.json
```

Compatible legacy form:

```bash
trspv_cli config.json
```

## Notes

- The solver pipeline is unified through `SolverApp::run(argc, argv)`.
- `src/trSPVSolver.cpp` is a compatibility placeholder and is not part of build targets.

## Refactor task hub

Task checklists and log templates are tracked under:

- `task-hub/README.md`
- `task-hub/2026-04-02_v1.0/`
