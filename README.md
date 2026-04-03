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
- `vcpkg` checkout with `VCPKG_ROOT` set
- On Windows: Visual Studio 2022 (or Build Tools 2022 with MSVC)

Windows (PowerShell):

```powershell
# run inside "Developer PowerShell for VS 2022"
$env:VCPKG_ROOT="C:\\path\\to\\vcpkg"
cmake --preset dev-windows
cmake --build --preset build-dev-windows --config Release
ctest --preset test-dev-windows --output-on-failure
```

Linux (bash):

```bash
export VCPKG_ROOT=/path/to/vcpkg
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
