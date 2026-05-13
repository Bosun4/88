# 知识卡片：Calibration vs Accuracy

## 核心摘要

足球预测里，“猜中胜平负/比分”的命中率不是最终目标。来源 S1 强调，能选出较多赢家并不等于长期盈利；只有当模型概率经过校准，并且与市场价格比较后仍存在正期望值（+EV），才有投注意义。S2 从市场效率角度补充：若市场强式有效，模型归一化概率应接近 bookmaker-implied probabilities；模型与市场的偏离必须由真实信息优势解释，而不是由叙事信心解释。

因此，vMAX 里 `confidence` 只能是模型当前判断强度或推荐等级，不应被展示成历史命中率、利润率或可下注置信。Brier Score、Log Loss、Calibration Curve、ECE/MCE 这类概率指标要比“top1 命中”更关键。

## 对 vMAX 的启发

- UI 文案必须持续说明：`confidence` 非历史校准命中率。
- 推荐 gate 不能只看 final direction 是否强，还要看 fair probability vs market price 是否存在可解释 edge。
- 不应把 15 场赛后命中率当成系统升级成功证据；样本太小且容易被 prompt/阈值回看污染。
- PR7 应建立校准层：记录 1X2 概率、比分概率、Brier、Log Loss、reliability bins，并按联赛/赔率区间分桶。
- 若没有市场价格或 closing line，模型可以做“预测展示”，但不能宣称 +EV。

## 相关来源

- S1: Why Your Soccer Betting Model Isn’t Profitable.
- S2: Comparing two methods for testing efficiency.
