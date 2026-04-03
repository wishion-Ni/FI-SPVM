# PR5 任务清单：测试、文档与发布准备（Tests/Docs/Release）

- 时间（UTC）：2026-04-02T00:00:00Z
- 版本：v1.0
- 任务ID：PR5-TESTS-DOCS-RELEASE
- 建议分支：`chore/pr5-tests-docs-release`

## 目标
补齐测试闭环与文档资产，形成可复现发布流程。

## 任务拆解
### T1. 单元测试
- [ ] Config 解析与校验测试
- [ ] PeakSeedDetector 稳定性测试
- [ ] Component 提取边界测试

### T2. 回归测试
- [ ] 建立黄金数据样例（固定输入与配置）
- [ ] 校验关键指标范围（WRMSE/R2/组件数量）

### T3. 文档完善
- [ ] `docs/architecture.md`
- [ ] `docs/quickstart.md`
- [ ] 输出文件说明与故障排查

### T4. 发布检查
- [ ] 生成版本说明（变更、风险、兼容性）
- [ ] 预演回滚步骤

## 验收标准（DoD）
- CI 可跑核心测试集。
- 新成员可按 quickstart 在独立环境跑通。

## 风险与回滚
- 风险：回归样例覆盖不足导致漏检。
- 回滚：增加 smoke case 并设置合并门禁。
