# PR3 任务清单：配置解析标准化（Config Standardization）

- 时间（UTC）：2026-04-02T00:00:00Z
- 版本：v1.0
- 任务ID：PR3-CONFIG-STANDARDIZATION
- 建议分支：`refactor/pr3-config-standardization`

## 目标
统一配置解析风格并增加配置校验，提升稳定性与可维护性。

## 任务拆解
### T1. 统一解析范式
- [ ] 选择单一范式（推荐全 `from_json`）
- [ ] 为关键子配置补齐解析函数

### T2. 校验器
- [ ] 新增 `validate_config(cfg)`
- [ ] 添加范围/关系校验（如 `tau_min < tau_max`）

### T3. 错误信息改进
- [ ] 解析/校验失败信息包含字段名与非法值
- [ ] 保留历史配置兼容映射并输出 warning

### T4. 配置文档
- [ ] 补充 `docs/config-reference.md`
- [ ] 提供最小可运行示例配置

## 验收标准（DoD）
- 合法配置可运行；非法配置快速失败且错误清晰。
- 现有 `config.json` 兼容。

## 风险与回滚
- 风险：历史配置字段兼容不完整。
- 回滚：保留 `legacy_parse` 兼容路径一个版本周期。
