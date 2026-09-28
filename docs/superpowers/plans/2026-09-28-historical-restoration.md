# 历史分析结构恢复 Implementation Plan

**Goal:** 恢复六月完整阅读体验，并使五步证据摘要实际贯穿三模型流程。

**Architecture:** 保留现有采集、panel、JSON 发布及静态页面。仅调整 panel 提示/字段传递和页面布局，版本更新隔离旧缓存。按用户要求由主线程实现和验收，子代理限只读；本次子代理路由故障已记录。

**Tech Stack:** Python、pytest、原生 JavaScript、Node 测试、HTML/CSS。

**Spec:** ../specs/2026-09-28-historical-restoration.md

## 约束

模型、接口、Secrets、历史数据及用户清空的 README 不变。无 1X2/泊松预测，无本地改比分，无固定 13% 定价，无额外模型请求。

## 1. 证据摘要闭环

- [x] tests/test_score_policy.py 增加完整 run_predictions 测试：初审 reading_summary 必须进入 Gemini packet，终审不同比分及摘要必须原样发布。缺失/错误类型字段不补造摘要。
- [x] 运行 `python -m pytest tests/test_score_policy.py tests/test_panel.py -q --allow-hosts=127.0.0.1,::1 --allow-unix-socket`，确认新增断言先失败。
- [x] scripts/score_policy.py 定义五项摘要协议并从 raw_item 提取；scripts/panel.py 在 analyst_outputs 传递，更新提示缓存版本。scripts/predict.py 更新版本入口以保持新版走无改分 adapter。
- [x] 重跑目标测试，确认请求顺序/预算、失败状态、来源和比分保真均通过。

## 2. 六月卡片布局

- [x] tests/frontend_fixtures.cjs 加入 observe/no_bet 及 C/D 排除、五步转义、旧版缺失和完整正文可见检查。
- [x] `node tests/frontend_fixtures.cjs` 确认新增检查失败。
- [x] assets/dashboard.js 恢复精选、五步摘要、居中比分、完整判决、模型和风险；index.html 与 assets/terminal.css 恢复六月版层级及响应式布局。技术信息仍折叠。
- [x] 重跑前端测试。使用实际数据和清楚标记的演示数据在桌面/手机检查排版、筛选及空状态。

## 3. 发布验收

- [x] 完整 Python 与 Node 回归；检查 diff 无模型/API/历史数据改动、无旧后处理重新生效。
- [ ] 保存浏览器截图与验证结果，提交分支并通过已登录 GitHub 发布。
- [ ] 检查远端 CI、真实预测及 Pages；分开报告工作流成功和有效预测覆盖，避免将运行正常表述成命中率提高。
