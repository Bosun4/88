# Prompt 逆向审计与反诱盘裁判修复报告 — 2026-06-11

## 目标

按“AI 自己作为裁判演练”的方式排查 `scripts/predict.py` prompt，不只修字面，而是用真实抓包观察模型为何被盘口/文案带偏，并引入外部反向盘口与多智能体结构化输出原则，重构最小可验证约束。

## 联网学习结论

- Reverse Line Movement/RLM 不能被当作魔法词；外部资料强调它必须同时验证公众热度、变盘幅度、时间窗口、sharp book/市场共识/资金来源。轻微公众偏向、单一盘口移动或早盘移动都可能是噪声。
- 多智能体系统应按角色拆分能力与 schema；分析员负责子任务，裁判负责整合。若分析员 prompt 又说“不需要给预测”，同时 schema 又强制 `predicted_score`，会诱发模型补假字段或违背角色职责。

## 本地抓包演练

使用 `data/predictions.json` 最新世界杯 slate 演练 top4：

1. 沙特 vs 乌拉圭：模型给 `0-2 away A main 75`，理由高度依赖“实力碾压、最低比分赔率、2球模态”。虽然风险候选已列出 `0-1/1-2/0-0/1-1`，推荐层仍被最低赔率与强队名气拉回 A 级主推。
2. 伊朗 vs 新西兰：模型给 `1-0 home A main 75`，理由依赖 1-0 最低赔率、世界杯首战保守、锋线缺阵。风险候选完整，但 `sharp_alignment=0` 与多条 prematch 降级 warning 没能阻止 A/main。
3. 葡萄牙 vs 尼日利亚：模型能识别友谊赛/大热/客胜 steam 风险，已降到 `B small`，说明“反证闸门”有效但还不够显式。
4. 卡塔尔 vs 瑞士：无 web sources、无 1X2，只靠比分簇与让球盘推 `0-3 B small`；说明缺外部市场确认时应明确限制推荐资格。

核心错因：prompt 的裁判流程是“先选漂亮比分，再解释风险”；应改为“先判下注资格，再判比分”。

## 已修复

- `scripts/predict.py`
  - 新增 `REVERSE_AUDIT_GATE_PROTOCOL`：强制先输出反证清单，先判可不可以买，再判比分。
  - 新增 `LEAGUE_STYLE_PROMPT`：Phase1 与 Final 共用同一联赛/战意规则，避免 Final prompt 落后于 Phase1。
  - 新增 `DATA_BACKED_PROMPT_TUNING`：把本地数据库已证实/已证伪结论写入 prompt。保留分位簇塌缩、国际赛零封税/进球上修、判平二段裁决、L3 便宜盘防火墙；明确禁止机械读盘骨架、静态阈值杀平/博平复活。
  - RLM/聪明钱规则改为证据分级：极端热度、变盘幅度、时间窗口、sharp book/国际盘或本地 change/steam 同时具备才可升级；否则写 `unclear/noise`。
  - 修正 Phase1 角色冲突：GPT/Grok 仍按统一 schema 输出 prediction，但明确 `predicted_score` 是审计假设，不代表最终推荐。
  - 修正 abstain 契约：schema 允许 `final_direction=abstain`，并要求 `predicted_score=弃权`、`bet_action=observe/no_bet`。
  - 移除/弱化 “彻底无视/强行将大球带提升为 A/S” 这类会打穿风控的措辞，改成候选进入高优先级，但必须过下注资格闸门。
  - 修复 `import scripts.predict` 场景下增强 evidence 注入失效问题：`build_evidence_packet()` 现在兼容包导入与脚本导入，避免 `league_intel` 绝对导入失败后静默丢失 `score_cluster_diagnostics_v203/dual_market_divergence_calibration/local_quantitative_intelligence`。
- `tests/test_dual_market_divergence.py`
  - 新增 prompt 契约测试：abstain schema、Phase1 角色不冲突、Phase1/Final 共用逆向闸门与联赛风格、RLM 必须证据分级、Gemini 不再强推高比分 main。
- `tests/test_predict_package_import_evidence.py`
  - 新增包导入回归测试：验证 `import scripts.predict` 时增强 prompt facts 不会缺失。

## 验证

- `python -m py_compile scripts/predict.py` 通过。
- `python -m pytest tests/test_dual_market_divergence.py tests/test_predict_package_import_evidence.py -q` → `19 passed in 0.37s`。
- `python -m pytest -q` → `111 passed in 9.40s`。
- 静态 prompt guard：GPT/Grok/Final prompt 均包含 `先判可不可以买，再判比分`、`强队低赔若缺乏真实资金/盘口动态确认`、`a5/a4<=1.70`、`a4>5.3 是排除线`；不再包含 `直接断定`、`彻底无视`、`强行将` 等危险触发词。

## 尚未做

- 未调用真实 AI 网关重跑全 slate；本轮是 prompt 静态约束与历史抓包演练修复。
- 未 commit、未 push；当前改动停留在新分支 `fix/prompt-reverse-audit-referee-20260611` 本地工作树。
