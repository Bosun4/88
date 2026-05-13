# OpenClaw / Hermes Division Protocol

## 1. 目标

建立 AI-Market-Intel Research OS 的多代理分工：OpenClaw 做入口与轻量编排，Hermes 做深度研究、代码审计和 PR 设计；预测模型各自承担明确角色，避免一个模型既生成预测又无约束地评价自己。

## 2. OpenClaw 职责

- 入口编排：接收用户任务、确认硬性约束、维护任务队列。
- 摘要压缩：把长报告压成执行摘要、风险清单和下一步建议。
- 防重复：检查是否已有同名报告、已有 PR 计划、已有 blind ledger entry。
- 轻联网：做快速来源搜集、URL 去重、来源 freshness 标记。
- UI/产品桥接：把 Hermes 的审计结论转成前端展示需求和 copy。
- 禁止：不得擅自重跑预测、不得推送远端、不得绕过 Hermes 代码审计直接改预测主干。

## 3. Hermes 职责

- 深度代码：阅读 `scripts/predict.py`、测试、JSON schema、报告生成器；提出可执行 PR 范围。
- 研究主管：维护 research_index、knowledge_cards、market_protocols、forward ledger 规范。
- JSON 审计：读取 live prediction JSON，核对字段完整性、风险标签、matrix shadow、方向/比分闭环。
- PR 设计：把发现拆成 PR4/PR5/PR6/PR7 等可独立验证的改动。
- 验证纪律：每次文件生成后检查路径、JSONL 合法性、git diff；不 push，除非用户明确要求。

## 4. 模型分工

- Gemini：主预测/终审裁判。负责最终 `predicted_score`, `final_direction`, `recommendation`，但必须受到 schema、ledger 和 matrix shadow 约束。
- GPT/Codex：逻辑审计、代码修改、测试、schema consistency、prompt leakage 检查。适合发现字段闭环和 PR 风险。
- Grok：另类视角/反共识 critic。用于挑战主路径、提出冷门/尾部风险，但不单独接管 final。
- Matrix/Poisson/DC：概率主干和旁路诊断，不是 LLM；输出方向概率、score grid、tail risk。
- Human researcher：决定是否把模型建议升级为 PR，不以单次命中率做产品结论。

## 5. 决策层级

1. Evidence packet: 事实、赔率、新闻、阵容、时间戳。
2. Market/Score trunk: fair probabilities + score distribution。
3. AI residual: 对 trunk 无法表达的信息做解释或轻量调整。
4. Risk router: MATRIX_CONFLICT / TAIL_RISK / DATA_QUALITY 优先于推荐美化。
5. Forward ledger: 赛前锁定，赛后评价。
6. UI: 展示 final 与 risk，不改变预测。

## 6. 协作规则

- OpenClaw 可以发起 PR 任务，但 Hermes 必须给出边界：改哪些文件、不改哪些文件、如何验证。
- Hermes 的研究报告不得直接声称模型有效；必须指向 forward-only 指标。
- 若模型之间冲突，优先记录冲突并降级，不强行合并成单点高置信。
- 每次 PR 都要声明是否影响 prediction logic；UI PR 默认不得影响预测 JSON 生成。
