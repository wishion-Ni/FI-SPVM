# FI-SPVM

This repository is organized into two directories:

- `lib/`: Core library headers and implementations for the spectral variational model.
- `src/`: Application orchestration code, including `SolverApp`.
- `apps/`: Thin executable entry points such as `apps/trspv_cli/main.cpp`.

Configuration samples (`config.json`) remain at the repository root.

## Unified entry

The CLI entry has been converged to a thin `main` in `apps/trspv_cli/main.cpp`.
Argument parsing, config loading, solver orchestration, and output generation now
flow through `SolverApp::run(argc, argv)` only.

`src/trSPVSolver.cpp` is kept as a temporary compatibility placeholder so older
build references do not immediately break, but it no longer owns solver logic.

## Refactor task hub

To make the refactor roadmap directly discoverable in this code repository, task checklists and log templates are tracked under:

- `task-hub/README.md`
- `task-hub/2026-04-02_v1.0/`

Start from `task-hub/2026-04-02_v1.0/00_INDEX_2026-04-02_v1.0.md` and create execution logs using `LOG_TEMPLATE_2026-04-02_v1.0.md` in the same batch folder.
