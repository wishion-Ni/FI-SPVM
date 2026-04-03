# PR2 任务清单：组分模型去重（Component Dedup）

- 时间（UTC）：2026-04-02T00:00:00Z
- 版本：v1.0
- 任务ID：PR2-COMPONENT-DEDUP
- 建议分支：`refactor/pr2-component-dedup`

## 目标
统一 `Component` 领域模型定义，删除重复结构，消除语义漂移风险。

## 任务拆解
### T1. 模型收口
- [ ] 保留 `lib/ComponentAnalysis.*` 作为唯一 `Component` 定义来源
- [ ] 删除 `src` 层重复 `Component` 定义

### T2. 引用替换
- [ ] 全量替换重复定义引用为 `trspv::Component`
- [ ] 保证命名空间一致性

### T3. 功能回归
- [ ] 检查 `components.txt` 与 `summary.json` 字段稳定
- [ ] 检查瞬态输出与组分提取流程不变

## 验收标准（DoD）
- `rg "struct Component"` 仅保留领域模型定义（测试夹具除外）。
- 构建通过，运行后组分输出字段完整。

## 风险与回滚
- 风险：命名空间冲突导致编译错误。
- 回滚：短期恢复局部别名（非重复结构），再逐步迁移。
