# P1 Score Tail Protection Implementation

生成时间：2026-05-13
分支：p1-score-tail-protection

## 实现范围

本次升级基于 `reports/design_p1_score_tail_protection_2026_05_13.md`，针对“弱主胜 + BTTS=yes + 客胜概率不低 + 高比分风险”场景补充协议层尾部风险保护，避免 AI 输出在弱主胜条件下过度锁死单一 `2-1`。

## 代码变更

### 1. `scripts/predict.py`

- 在 v20.3 输出 schema 中新增可选字段：
  - `risk_score_candidates`
  - `tail_risk_flags`
  - `confidence_downgrade_reason`
- 在 Phase1 / Gemini final prompt 中新增弱主胜尾部约束：
  - 当主胜概率 <= 52%、客胜概率 >= 23%、且 BTTS=yes 时，不得将 `1-2`、`2-2`、`2-3` 视为无关尾部。
  - 若最终仍选主胜 `2-1`，必须说明为什么排除客队反打与 `4+` 尾部。
- 新增协议层保护函数：
  - `apply_weak_home_tail_risk_protection()`
  - 只做风险展示和推荐置信度降级，不改 `final_direction`，不改 `predicted_score`。
- 触发条件：
  - `final_direction == home`
  - `home_win_pct <= 52`
  - `away_win_pct >= 23`
  - `BTTS=yes` 或已有高比分/尾部风险信号
- 触发后：
  - 将 `1-2`、`2-2`、`2-3` 加入 `risk_score_candidates`。
  - 添加 `weak_home_favorite_btts_tail` 等 `tail_risk_flags`。
  - 若 `recommendation.bet_confidence >= 70`，降级至最高 60。
  - 写入 `confidence_downgrade_reason = "Weak home favorite with BTTS tail risk"`。

## 测试变更

### 2. `tests/test_review_20260512_regressions.py`

新增单测：

- `test_tail_risk_protection_for_weak_home_favorite()`

覆盖内容：

- 弱主胜概率场景下保留原始方向和比分，不改足球判断。
- 自动补充 `1-2`、`2-2`、`2-3` 风险候选。
- 自动添加 `weak_home_favorite_btts_tail` 风险标记。
- 高置信度从 78 降至 60 以下或等于 60。
- 前端适配输出同步保留风险字段与降级后的 confidence。

## 约束遵守

- 未硬编码任何球队名。
- 未将 `2-3` 写死为最终赛果或最终预测。
- 未针对 2026-05-12 具体 8 场做过拟合。
- 逻辑仅基于概率阈值、BTTS/尾部风险信号触发。
- 未修改 OpenClaw 配置。
- 未修改 Hermes 配置。
- 未重启 Gateway。
- 未直接 push main/master。
- 未使用 `git push --force`。

## 本地验证

已执行：

```bash
python3 -m py_compile scripts/predict.py
/tmp/oc88-pytest-venv/bin/python -m pytest -q tests
```

结果：

```text
1 passed
```
