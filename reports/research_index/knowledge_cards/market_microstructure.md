# 知识卡片：Market Microstructure

## 核心摘要

足球盘口不是单一数字，而是由 1X2、让球、总进球、正确比分、半全场、交易时序和信息事件共同构成的市场结构。S6 指出阵容、伤停、临场消息会改变价格，尤其接近开赛时会反映到赔率移动和 closing line。S2/S3 则提示：市场概率需要先处理 bookmaker margin，且不同去水方法会影响公平概率解释。

对 vMAX 来说，microstructure 的重点不是让 AI 根据盘口讲故事，而是把盘口拆成可审计事实：开盘/即时/临场价格、overround、去水方法、方向变化、比分簇、总进球簇、是否有 lineup/news 时间戳。

## 对 vMAX 的启发

- Market Structure Layer 应保存 raw odds、implied probabilities、fair probabilities、overround、method、timestamp。
- 任何“聪明钱/反向波动/RLM”判断必须有时间序列证据；没有时只能写 `unclear`，不能补叙事。
- closing line 应作为 forward ledger 的重要后验基准，但不能在赛前预测生成时泄漏给模型。
- 阵容/伤停/news 必须有采集时间；赛后补录的 team news 不能用于赛前样本评价。
- UI 的风险标签应区分市场信号、信息缺失、matrix 分歧和 AI 自身低置信，避免全部合并成一个泛化 HIGH_RISK。

## 相关来源

- S2: bookmaker probabilities and efficiency.
- S3: overround allocation.
- S6: lineup/news/injury impact on odds movement and closing line.
