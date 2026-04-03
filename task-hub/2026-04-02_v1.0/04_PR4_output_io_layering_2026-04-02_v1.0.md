# PR4 任务清单：输出与IO分层（Output/IO Layering）

- 时间（UTC）：2026-04-02T00:00:00Z
- 版本：v1.0
- 任务ID：PR4-OUTPUT-IO-LAYERING
- 建议分支：`refactor/pr4-output-io-layering`

## 目标
把输出文件生成逻辑从流程编排层抽离，形成独立 IO 层，便于测试与扩展。

## 任务拆解
### T1. ResultWriter 抽象
- [ ] 新增 `lib/.../ResultWriter.*`
- [ ] 迁移 `write_interpolation_outputs`
- [ ] 迁移 `write_transient_outputs`
- [ ] 迁移 `write_metrics`

### T2. 参数扫描结构化结果
- [ ] 新增 `param_scan.csv` / `param_scan.json`
- [ ] 固化字段：参数值、指标、是否最优

### T3. 向后兼容
- [ ] 保持已有输出文件命名不变
- [ ] 对新增文件在文档中说明可选依赖

## 验收标准（DoD）
- `SolverApp` 仅保留流程编排，文件输出细节下沉。
- 旧脚本可继续消费原输出文件。

## 风险与回滚
- 风险：CSV 列顺序改变影响下游。
- 回滚：保留旧 Writer 适配层并提供开关。
