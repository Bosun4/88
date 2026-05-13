# PR #4: Risk Severity Grading and Matrix Conflict Display

## 1. 目标
本次 PR 旨在优化风险预警机制的前端展现形式。之前的所有场次直接输出 `HIGH_RISK` 和大量尾部告警，造成用户视觉疲劳与风险阈值过宽的错觉。本次不修改任何核心概率生成逻辑或预测主干，而是引入 **Severity Grading（风险分级）** 模型。

## 2. 不修改的部分
- 不修改 `scripts/predict.py` 预测层。
- 不改 `final_direction`。
- 不改 `predicted_score`。
- 不改 `confidence`。
- 不重跑预测或调用 AI。
- 不调用任何破坏性 API。

## 3. 风险分级规则

*最高优先级判定原则，一场比赛命中更高级别后不再向下覆盖*

**SEV-1（致命风险/结构错乱）**
- `matrix_shadow_error` 非空。
- `predicted_score` 与 `final_direction` 的逻辑直接冲突。
- `final_direction` 与最大主/平/客概率产生严重偏离。

**SEV-2（强矩阵对抗）**
- `MATRIX_CONFLICT` 与 `TAIL_RISK` 同时存在。
- 矩阵推荐方向（`matrix_recommended_direction`）与 `final_direction` 截然反向。
- 矩阵首选比分方向与 `final_direction` 冲突。

**SEV-3（单边风险预警）**
- 只有 `MATRIX_CONFLICT`。
- 只有 `TAIL_RISK`。
- 风险比分候选区包含反方向比分。

**SEV-4（置信度下调/弱尾部）**
- 只有 `HIGH_RISK` 或置信度下降原因（`confidence_downgrade_reason`）。
- 矩阵与最终方向一致但提示部分尾部抛高。

**NORMAL（可信常规）**
- 无任何以上风险。

## 4. 输出产物
由于环境中无标准现代前端，本 PR 将附带生成静态可视化报告：`reports/live_predictions/risk_dashboard_latest.html`。
通过该 HTML，可以直接从已落盘的 Live Predictions JSON 渲染出结构化视觉。展示包含了分离的“主预测区”与“矩阵诊断区”，有效缓解用户的错误解读（把 Matrix 当作 Final 预测）。
