# PR1 任务清单：入口收敛（Single Entry）

- 时间（UTC）：2026-04-02T00:00:00Z
- 版本：v1.0
- 任务ID：PR1-ENTRY-UNIFICATION
- 建议分支：`refactor/pr1-entry-unification`

## 目标
将执行入口收敛为“薄 main + 单一业务流程入口”，避免 `src/trSPVSolver.cpp` 与 `src/SolverApp.cpp` 双流程并存。

## 任务拆解
### T1. 新建统一 CLI 入口
- [ ] 新增 `apps/trspv_cli/main.cpp`
- [ ] 仅保留参数透传与异常处理
- [ ] 调用 `SolverApp::run(argc, argv)`

### T2. 切换构建入口
- [ ] 在构建系统中将可执行入口改为 `apps/trspv_cli/main.cpp`
- [ ] 移除旧入口编译项，避免 duplicate main

### T3. 处理旧入口文件
- [ ] `src/trSPVSolver.cpp` 改为薄适配器或归档移除
- [ ] 不保留算法流程代码（避免分叉）

### T4. 兼容性校验
- [ ] `--help` 输出参数保持兼容
- [ ] `--conf/--input/--out` 参数行为保持一致

### T5. 文档更新
- [ ] README 增加“统一入口说明”

## 验收标准（DoD）
- 只有一条主流程实现（`SolverApp::run`）。
- CLI 功能兼容；关键输出仍可生成。
- PR 描述含回滚方案。

## 风险与回滚
- 风险：构建脚本残留旧入口导致链接错误。
- 回滚：恢复旧入口编译项并保留新入口代码（两步回滚）。
