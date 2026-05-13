# 知识卡片：Score Distribution Models

## 核心摘要

S4 介绍 Poisson 与 Dixon-Coles 思路：用球队攻击/防守强度估计主客队进球 lambda，生成完整比分概率网格；Dixon-Coles 对 0-0、1-0、0-1、1-1 等低比分相关性做修正。S5 进一步把 extended xG、Poisson grids、Dixon-Coles rho 和 renormalisation 组合起来，强调概率网格必须归一化且可解释。

这类模型的价值在于提供“概率主干”：方向概率、总进球概率、BTTS、比分 top-N、尾部质量，而不是单点比分。AI 可以解释为什么某场 lambda 应上调/下调，或指出阵容/节奏等残差信息，但不能覆盖概率主干。

## 对 vMAX 的启发

- Score Distribution Layer 应输出完整 grid：score, prob, direction, total_goals, btts。
- Matrix shadow 不应接管 final，但应成为冲突检测器：final 单点与 grid top1/topN/方向概率不一致时打标。
- 对低比分联赛/小样本球队，应考虑 Dixon-Coles 风格低比分修正；修正后必须 renormalise。
- 不要只展示 `predicted_score`；需要展示 Top 3/Top 10 分布与尾部概率，避免用户误读单点预测。
- 若 AI 选择不在 matrix top-N 的比分，必须记录解释和风险等级。

## 相关来源

- S4: Dixon-Coles & Poisson Models in Football Prediction.
- S5: Extended xG lambdas, Poisson grids, DC correction, renormalisation.
