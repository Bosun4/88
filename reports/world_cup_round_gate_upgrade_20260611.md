# 世界杯分轮闸门与 06-07~06-10 国际赛验证升级报告

> 日期: 2026-06-11
> 仓库: `/root/.openclaw/workspace/repos/88`
> 分支: `fix/prompt-reverse-audit-referee-20260611`
> 原则: 只写真实验证结果；不把小样本吹成定律；不迎合“R1正常/R3必小”的直觉。

## 1. 真实仓库状态

- 工作路径: `/root/.openclaw/workspace/repos/88`
- 本地分支: `fix/prompt-reverse-audit-referee-20260611`
- 本地 HEAD: `a19f72206fe30562ca50f64fdfb49cc3d745b925`
- 远端 `origin/main`: `a19f72206fe30562ca50f64fdfb49cc3d745b925`
- 当前改动未 commit/push。

## 2. 06-07~06-10 已完赛国际热身赛闭环

### 纳入样本

只纳入“本地有赛前预测快照 + web_search 有稳定赛果摘要”的比赛；葡萄牙 vs 尼日利亚只搜到赛程/实时页，未纳入闭环。

| 比赛 | 赛前预测 | 实际赛果 | 方向 | 比分 | 关键偏差 |
|---|---:|---:|---|---|---|
| 哥伦比亚 vs 约旦 | 2-0 | 2-0 | 中 | 中 | 无 |
| 法国 vs 北爱尔兰 | 3-1 | 3-1 | 中 | 中 | 无 |
| 秘鲁 vs 西班牙 | 0-3 | 1-3 | 中 | 错 | 零封被击穿，BTTS 漏判 |
| 阿根廷 vs 冰岛 | 3-1 | 3-0 | 中 | 错 | BTTS 高估 |
| 匈牙利 vs 哈萨克 | 2-0 | 3-1 | 中 | 错 | 零封被击穿，总进球低估2球 |

### 指标

- 方向: 5/5
- 精确比分: 2/5
- 预测 X-0 零封: 3场；实际被 BTTS 击穿 2场
- 平均总进球误差: 实际比预测高 0.4 球

### 独立结论

1. 不能因为用户要求“全方位升级”就改坏方向链路。06-07~06-10 小样本继续支持：国际赛强弱/方向仍可读。
2. 真正要升级的是比分形状层：X-0 零封、BTTS、进球带上修。
3. 这个结论与 06-01~06-09 大样本报告一致：方向约 68%，但比分错多集中在零封幻觉和进球低估。
4. 因样本只有5场，不把它当成新铁律，只作为已有结论的最新复核。

## 3. 世界杯分轮独立修正

用户直觉: 第一轮/第二轮抢分，第三轮控分。

数据修正: `reports/wc_research/WC_READING_INTEL_2026.md` 的 5届320场世界杯分轮实证显示：

- R1 场均 2.34，最低；不是正常开放，而是最闷。
- R2 场均 2.69，最高；才是更开放的一轮。
- R3 场均 2.42，over2.5 反而 51%；“控分”不是总进球下降，而是净胜球收窄、已出线强队诱盘、小胜/小负/一球差路径增多。

因此本次升级不照抄“R3必小球”，而是强制先读轮次与出线形势。

## 4. 已落地改动

### `scripts/league_intel.py`

新增 `world_cup_round_gate(match)`：

- 输出结构化 evidence，不替 AI 判方向/比分。
- 包含 2026 48队赛制、R1/R2/R3/QF 分轮事实、must_not_assume、ai_required_audit、confidence_controls。
- R1: 强队赢不穿/一球小胜/防平；未知轮次降级。
- R2: 读首轮积分，0分/1分方是 BTTS/3+ 核心来源。
- R3: 三分类——双方已定、已出线强队 vs 有动机方、双方都需赢。
- 淘汰赛: 区分晋级判断与90分钟竞彩判断，点球/加时权重前置。

### `scripts/predict.py`

- `build_evidence_packet()` 注入 `local_quantitative_intelligence.world_cup_round_gate`。
- `DATA_BACKED_PROMPT_TUNING` 增加 06-07~06-10 实测结论：方向别改坏，比分形状防 BTTS 与上修。
- 新增 `WORLD_CUP_ROUND_GATE_PROMPT`，Phase1 和 Gemini Final 均强制读取。
- `LEAGUE_STYLE_PROMPT` 修正世界杯逻辑：R1最闷、R2最开放、R3控净胜球不是控总进球。

### Tests

- `tests/test_dual_market_divergence.py`: 增加 prompt 必须包含 06-07~06-10 实测结论、world_cup_round_gate、unknown round 降级、R3控分≠小球。
- `tests/test_predict_package_import_evidence.py`: 增加包导入时 `world_cup_round_gate` 注入测试，以及 R3 已出线强队诱盘闸门测试。

## 5. 后续验证边界

已验证的是 prompt/evidence 契约与历史赛果小闭环；未验证的是世界杯正赛实战收益。

开赛后应滚动做：

1. 每轮完整 slate 记录 `world_cup_round_gate.round/group_scenario`。
2. R1 单独统计: 强队低赔是否赢不穿、0-0/1-0/1-1 占比、X-0 是否继续被 BTTS 击穿。
3. R2 单独统计: 0分/1分球队是否显著提高 BTTS/3+。
4. R3 必须补积分/净胜球/同组同步赛，不补不得给 A/S。
