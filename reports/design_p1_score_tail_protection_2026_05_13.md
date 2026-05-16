# P1 设计报告：预测尾部分布与高比分保护机制

生成时间：2026-05-13
项目路径：/root/.openclaw/workspace/repos/88
报告名称：reports/design_p1_score_tail_protection_2026_05_13.md

## 1. 本轮反泄漏盲测结论

在通过完全隔离的文件系统与赛前数据进行严格盲测（Blind Sandbox）后，得出的预测结果与原始预测高度吻合。
反泄漏测试的结果表明，当前项目主代码的模型策略尚未能摆脱“弱主队偏好”以及“低频比分过滤”的困境。

## 2. 上轮泄漏沙盒为什么无效

上一轮沙盒重测时，由于在预测启动前未能严格隔离文件系统中的真实赛果（实际赛果 `actual_score` 和比分均以字符串形式存在于工作区报告和测试中），导致模型通过检索历史痕迹发生了**严重的数据泄露与事后迎合**。最终生成的 100% 方向命中率和 75% 比分命中率并不能反映模型的预测改进，已作为无效实验废弃。

## 3. PR #1 的真实价值

PR #1 的真实价值在于**工程加固，而不是命中率提升**。
它的修改包含：
- `async_call_ai_json` 中增强了空 JSON 或解析异常情况下的回退逻辑（parse_failed 拦截）。
- 下调 `AI_MAX_PROMPT_CHARS_PER_CHUNK` 以防止 token 溢出和注意力稀释。
这些措施让流水线更加健壮，避免脏数据入库，并未触及深层概率到比分的推导链路。

## 4. 当前真实基准指标 (Blind Sandbox Baseline)

根据最新一轮严格隔离盲测的输出评分：
- 方向命中率：37.5% (3/8)
- 比分命中率：0.0% (0/8)
- 总进球区间命中率：37.5% (3/8)
- BTTS 命中率：50.0% (4/8)

## 5. 塞尔塔 2-3 莱万特失败链路分析

模型输出了 2-1（主胜）。
实际赛果：2-3（客胜）。
在此链路中存在以下致命推导缺陷：
- **置信度未受限：** 主胜概率约为 48%，但比分池完全锁死了 2-1 这一单一选项。
- **客队反打被吞噬：** 尽管模型探测到了 BTTS=yes 和防线风险，莱万特的胜率（24%）和高比分客胜尾部（1-2，2-3）未能被拉入任何 backup score candidates。
- **高比分尾部压制：** 预测仅提供 goal_band=3 的 2-1，没有因为防线风险向 goal_band=4+ （如 2-2，3-2，2-3）释放候选。

## 6. 主要模型偏差

1. **弱主胜过度锁定：** 当 `home_win_pct` 在 40%-52% 区间徘徊时，直接推导单一的主胜强方向比分（如 2-1, 1-0）。
2. **常见比分权重偏高：** `2-1` 与 `1-1` 在 AI 策略中充当了保守的默认 fallback，吸收了太多不明确的概率分布。
3. **客队反打比分权重不足：** 对客队概率 >= 20% 的长尾抵抗情况缺乏足够的比分选项支撑，如 `1-2`、`2-3`、`0-1` 被系统性抑制。
4. **BTTS=yes 未转化为高比分尾部保护：** 当预测大概率双方进球且防守皆弱时，没有自动向 `goal_band=4+` 抛出保护分支。
5. **低置信主胜没有触发风险降级：** 胜率不足一半的情况仍在最终 JSON 中以满配“高确定性”输出，影响下注评级。

## 7. P1 升级方案

必须采用软性候选扩展和风险降级的原则，坚决不能“针对塞尔塔 2-3 硬编码”。
升级逻辑如下：
1. **风险降级与多元候选 (Risk Downgrade & Candidate Diversity)：**
   如果 `home_win_pct < 52%` 且预测 `BTTS = yes`：
   - 不允许只锁定 `2-1`。
   - 必须抛弃单一比分倾向，强制把 `1-2`、`2-2` 或 `2-3` 推入新增字段 `risk_score_candidates`。
   - 在 `confidence` 上做惩罚性降级（Downgrade），并填充 `confidence_downgrade_reason`。
2. **高比分尾部保护 (Tail Risk Guard)：**
   当 `BTTS = yes` 且存在进球战意/防线薄弱标记时，不仅要支持 `goal_band=2-3`，还需要将 `goal_band=4+` 的相关比分补充在尾部风险项中。
3. **客队反打容忍度 (Away Fight-back Tolerance)：**
   当 `away_win_pct >= 23%` 且主胜未超过 50%：
   - 客胜不应再被视为“极低概率”，`1-2` 或 `0-1` 应当进入 `risk_score_candidates`。
4. **最终比分选择流水线 (Score Selection Pipeline)：**
   决策顺序重构：Direction First -> Goal Shape (BTTS) Second -> Risk-adjusted Score Candidates Third。
   如置信度低，不能输出单比分，必须搭配风险候选阵列。

## 8. 最小代码修改建议

主要涉及文件：`scripts/predict.py`
需要进行如下结构的非侵入性增强：
- 在 `normalize_ai_predictions` 函数或 AI JSON schema 定义中，新增 `risk_score_candidates`, `tail_risk_flags`, `confidence_downgrade_reason` 字段。
- 在 `build_evidence_packet` 和模型 prompt (如 `generate_prompt`) 内加入**风险降级**（Risk Downgrade）的约束性提示说明，教导模型如果置信度不足以支撑单一比分，请在 `risk_score_candidates` 内填入包含 `1-2`, `2-2`, `2-3` 等长尾反打/高比分候选。

## 9. 新增测试建议

应在 `tests/test_review_20260512_regressions.py` 中补充以下单测：
- `test_tail_risk_protection_for_weak_home_favorite()`：构造 `home_win_pct=48, away_win_pct=24, BTTS=yes` 的场景，校验 AI 包装函数能否正确要求或抛出 `risk_score_candidates` 且不单独强锁 `2-1`。
- `test_confidence_downgrade_when_risk_flags_present()`：校验方向未超阈值时置信度是否能够有效折损。

## 10. 不过拟合约束

- **不引入球队白名单**（如：不允许写 `if away_team == '莱万特'`）。
- **不引入赛果锚定**（如：绝不把 2-3 硬编码给任何模型 Prompt）。
- **仅基于赔率倒推与进球模型概率阈值触发**（如：`home_prob < 52` 和 `away_prob >= 23` 是普适的概率风控指标）。
