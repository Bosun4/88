# vMAX Top-Tier Upgrade Plan

## 1. 执行摘要

vMAX 下一步不应继续堆叠单点预测 prompt，而应升级为 AI-Market-Intel Research OS：研究索引沉淀方法论，市场结构层处理赔率和 overround，比分分布层提供概率主干，AI residual 层处理非结构化信息，风险路由层把冲突和尾部风险显性化，forward ledger 层做盲测评价，UI 层负责清晰展示而不改变预测。

当前最新输出证明 PR1/2/3 已产生结构价值：字段闭环、风险候选、matrix shadow 都存在。但它也暴露新问题：风险标签过宽、tail candidates 几乎全场存在、matrix conflict 需要 severity、confidence 仍容易被误读。

## 2. 回测缺陷

传统回测会被三类污染：

- 信息泄漏：赛后阵容、赛果、closing line、新闻复盘进入赛前样本。
- Prompt overfit：根据历史失败不断改 prompt，直到回测漂亮。
- 样本幸存：只展示命中场或 top4，忽略失败和 C 级。

因此，回看报告只能用于提出假设，不能证明系统可盈利。

## 3. 盲测必要性

forward-only blind ledger 是下一阶段的核心。每条预测必须赛前锁定 input/prompt/output/normalized JSON 的 SHA256，赛后只追加 settlement。指标必须包含 Brier、Log Loss、calibration bins、exact score/top-N score、risk routing precision、CLV。没有盲测，就无法判断 PR2/PR3 的风险信号是真增益还是提示噪音。

## 4. 当前主干状态

- Gemini 是 final 主预测/终审。
- GPT/Codex 适合逻辑审计、代码和 JSON consistency。
- Grok 适合另类 critic 和尾部挑战。
- Matrix shadow 已成为旁路诊断，但不接管 final。
- `confidence` 已声明不是历史校准命中率。
- 当前 JSON 字段足够支撑 PR4 UI 升级和 PR5 ledger 设计。

## 5. PR1/2/3 价值

PR1：方向/比分/结果闭环，避免 predicted_score 与 final_direction 自相矛盾；同时降低 evidence packet 泄漏旧预测字段的风险。

PR2：风险候选与尾部保护，让弱主胜、BTTS、客队反打、高比分尾部不被 final 单点掩盖。

PR3：Matrix shadow 提供独立 score grid 诊断，能发现 final 与概率分布冲突，尤其适合 UI 风险提示。

## 6. 当前问题

- 风险标签过宽：若 15/15 都 HIGH_RISK，标签失去排序功能。
- Tail risk 过泛：候选几乎全场存在，需要 severity 和 supporting signal。
- Matrix conflict 数量口径不统一：direction conflict、任意 matrix flag、low-confidence warning 必须分开。
- 高比分/2-1 路径较多，需要 goal-band calibration。
- 缺少真实 forward ledger，无法评价阈值和候选是否有效。
- 市场去水方法和 overround sensitivity 还未系统化。

## 7. 检索结论

- Calibration vs Accuracy：命中率不等于盈利；需要校准概率和 +EV。
- Market efficiency：强式效率下模型概率不应无证据偏离 bookmaker probabilities。
- Overround：multiplicative/additive/power/Shin 会导致不同 fair probabilities，必须记录方法。
- Score distribution：Poisson/Dixon-Coles/xG lambda 应形成完整概率网格，不是只给单点比分。
- Lineups/news：阵容、伤停、临场新闻会影响 odds movement 和 closing line，必须有时间戳。
- Evaluation leakage：AI 预测最怕赛后 prompt 倒推和 sandbox/mock 冒充 blind result。

## 8. 架构路线图

1. Research Index Layer：sources + knowledge cards + protocols。
2. Market Structure Layer：赔率、overround、公平概率、盘口语义、时序。
3. Score Distribution Layer：Poisson/DC grid、方向概率、goal bands、BTTS。
4. AI Residual Layer：Gemini/GPT/Grok 解释和 critic，但不盖概率主干。
5. Risk Routing Layer：MATRIX_CONFLICT / TAIL_RISK / DATA_QUALITY / LOW_CONFIDENCE severity。
6. Forward Ledger Layer：赛前 SHA 锁定，赛后 settlement，长期指标。
7. UI Layer：Final 与 Matrix 分区展示，风险标签中文映射，候选折叠。

## 9. 下一步 PR 建议

### PR4: Severity grading / UI presentation

范围：前端展示和 copy；不改预测逻辑。

内容：Final vs Matrix shadow、风险标签分级、risk candidates Top3、confidence downgrade tooltip、matrix flags 中文映射、gate pass 数说明。

### PR5: Forward ledger

范围：新增 ledger schema、hash locking、settlement append-only、基础 evaluator。

内容：input/prompt/output/normalized SHA256，赛前锁定，赛后追加 actual_score，输出 Brier/Log Loss。

### PR6: RAG / Research Index integration

范围：把 sources/knowledge cards 接入 prompt 构建或研究摘要，但要防止把赛后评价泄漏到赛前预测。

内容：source IDs、method snippets、protocol guardrails、prompt hygiene。

### PR7: Calibration and fair probability layer

范围：概率校准、overround methods、Brier/Log Loss dashboards、risk threshold experiments。

内容：multiplicative/additive/power/Shin sensitivity，calibration bins，risk precision/recall，阈值调整必须基于 forward 样本。

## 10. 不建议做的事

- 不建议现在修改 `scripts/predict.py` 的预测阈值。
- 不建议重跑预测来迎合本次审计。
- 不建议把 matrix score 当 final score。
- 不建议把所有风险同级显示为 HIGH_RISK。
- 不建议在无 forward ledger 前宣称 PR2/PR3 提升命中率。
- 不建议用赛后信息补全赛前 prompt 或 sources。
- 不建议推送远端或开 PR，除非用户明确要求。
