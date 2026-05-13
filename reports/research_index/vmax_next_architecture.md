# vMAX Next Architecture

## 总体原则

vMAX 下一阶段应从“AI 给出单点比分”升级为“市场结构 + 概率分布 + AI 残差 + 风险路由 + 盲测账本”的 Research OS。AI 不盖概率主干；UI 不盖风险；回测不盖 forward ledger。

## A. Research Index Layer

职责：维护 sources、knowledge_cards、protocols、PR rationale。

输入：研究 URL、论文、方法文章、历史审计、PR 设计文档。
输出：`sources.jsonl`、知识卡片、协议文档、升级计划。
约束：每条知识结论必须引用 source ID；不大段复制原文；研究结论必须转化为 vMAX 规则。

## B. Market Structure Layer

职责：把赔率和市场信息变成可审计特征。

核心对象：

- 1X2 raw odds / implied / fair probs / overround。
- 正确比分 odds clusters。
- 总进球 odds and goal bands。
- 让球语义、半全场、odds movement。
- lineup/news/injury timestamps。

关键设计：fair probability 方法可配置，至少记录 multiplicative；后续比较 additive/power/Shin sensitivity。

## C. Score Distribution Layer

职责：生成完整比分概率网格。

组件：

- Poisson lambda_home/lambda_away。
- Dixon-Coles low-score correction。
- Renormalisation。
- Score top-N, direction probs, total-goal probs, BTTS。

输出不直接覆盖 final，而是形成 `matrix_*` / `score_distribution_*` 诊断字段。

## D. AI Residual Layer

职责：解释非结构化信息，但不盖概率主干。

AI 可以处理：伤停语义、战意、赛程、风格、来源冲突、模型 critic。
AI 不应处理：赛后信息、伪造 web sources、无依据 smart money、覆盖 locked probability trunk。

规则：当 AI final 与 matrix trunk 冲突时，记录冲突并降级，而不是让 AI 的叙事自动胜出。

## E. Risk Routing Layer

职责：把风险转为可展示、可评估的标签。

优先级：

1. `MATRIX_CONFLICT`: final direction/score 与 matrix 明显冲突。
2. `TAIL_RISK`: draw/away/high-goal/BTTS 尾部概率或风险候选密集。
3. `DATA_QUALITY`: no web validation, stale lineup, missing sharp data。
4. `LOW_CONFIDENCE`: confidence 或 score concentration 过低。
5. `HIGH_RISK`: 聚合标签，但不能替代具体原因。

设计要求：HIGH_RISK 不应泛化到所有比赛；必须有 severity grading，避免 UI 全红。

## F. Forward Ledger Layer

职责：赛前锁定、赛后评价。

字段：prediction, input hashes, prompt hashes, output hashes, model versions, market snapshots, risk flags, settlement。
指标：Brier, Log Loss, exact score, top-N score, calibration bins, CLV, risk routing precision/recall。

规则：赛前记录 append-only；赛后只追加 settlement；禁止倒推 prompt 和 mock/sandbox 造假。

## G. UI Layer

职责：清晰展示 final、matrix、risk 和评价边界。

展示原则：

- Final/AI-native 是主预测。
- Matrix shadow 是诊断，不是最终预测。
- Risk candidates 是替代风险，不是多重推荐。
- Confidence 不是历史命中率。
- 本轮 gate pass 数量应如实展示，例如“仅 1 场通过推荐 gate”。

PR4 应优先做 severity grading 与解释映射，而不是先把所有风险无差别铺到卡片上。
