# 知识卡片：Overround & Fair Probabilities

## 核心摘要

赔率隐含概率之和通常大于 100%，多出的部分是 overround/bookmaker margin。S3 指出 margin 的分配不是唯一的：multiplicative、additive、power、Shin 等方法会给出不同 fair probabilities。S2 的效率检验也依赖 normalized probabilities 与 bookmaker probabilities 的比较，因此去水方法必须被记录，而不能只写“市场概率”。

vMAX 当前若只用原始赔率或简单归一化，会在强弱悬殊、长尾比分和冷门赔率上产生偏差。正确做法是把方法作为元数据，并在关键场次做 sensitivity check：如果不同方法下结论不稳，应降级而不是强推。

## 对 vMAX 的启发

- 每个市场记录：raw odds、raw implied、overround、fair method、fair probs、method_version。
- 1X2、总进球、正确比分的 overround 不同，不能混用一个去水结果。
- Power/Shin 方法适合纳入 PR7 calibration 实验；PR4 UI 不应改算法。
- fair probability 与 AI probability 的 edge 必须显示 method，否则 edge 不可复现。
- 若无法计算 overround 或市场字段缺失，应触发 data_quality 风险，而不是生成伪精确 EV。

## 相关来源

- S2: normalized probabilities vs bookmaker probabilities.
- S3: multiplicative/additive/power/Shin overround allocation.
