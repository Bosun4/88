# P3.5 最新预测合理性审计

## 审计对象

- JSON: `reports/live_predictions/live_predictions_9999_github_actions.json`
- 最新审计: `reports/live_predictions/live_prediction_audit_latest.md`
- JSON SHA256: `d5e1489b0c282d95571b6fe93f63a841256b296fe7af85487d87ea26585d875b`
- 硬性边界：本次只读现有预测与审计；未修改 `scripts/predict.py`，未重新运行预测，未推送远端。

## 1. 实测统计

- 总场数：15
- `risk_level` 分布：{'medium': 8, 'high': 6, 'low': 1}
- `recommendation_tier` 分布：{'C': 14, 'B': 1}
- final direction 分布：{'home': 7, 'away': 6, 'draw': 2}
- final score 分布：{'2-1': 3, '0-1': 1, '1-1': 2, '3-0': 3, '0-2': 1, '1-2': 4, '1-0': 1}
- 预测总进球 >= 3：10/15
- final 平局：2/15
- `tail_risk_flags` 非空：15/15
- `risk_score_candidates` 非空：15/15，平均每场 3.8 个
- `matrix_vs_final_direction_conflict`：3/15
- 任意 matrix active flag：4/15
- risk candidate type 统计：high_btts_tail:9, weak_home_favorite_btts_tail:5, away_fightback:4, draw_resistance:3, draw_live/low_score_btts:2, away_fightback/high_btts_tail:2, draw_live:2, lower_odds_near_base:2, home_resistance:1, stale_draw:1, away_btts_extension:1, home_response:1, under_finish:1, btts_tail:1, home_cover_tail:1, under_tail:1, btts_flip:1, weak_favorite_home_flip:1, draw_low_cluster:1, away_clean_sheet_variant:1, away_cover_tail:1, draw_anchor:1, low_tempo_stalemate:1, home_cover_variant:1, counter_punch_away:1, away_cover:1, home_clean_sheet:1, home_narrow_win:1, home_btts_win:1, away_low_score:1, home_fightback:1, away_cover_high_tail:1, high_goal_tail_btts_flip:1, four_plus_tail:1, lower_odds_rival:1, btts_branch:1

用户给定的上层摘要为“15 场 HIGH_RISK、TAIL 12、CONFLICT 4”。按当前 JSON 原始字段直接读取，`risk_level=high` 不是 15 场，而是 6 场；`tail_risk_flags` 非空是 15 场；matrix direction conflict 是 3 场。若 UI 或中间层显示 15/15 HIGH_RISK、12/15 TAIL、4/15 CONFLICT，说明它使用了更宽的聚合规则或额外 severity 映射，需要在 PR4 明确写入 rule_version。

## 2. 15/15 HIGH_RISK 是否过宽？

结论：如果把 15/15 都标为同级 `HIGH_RISK`，过宽。

理由：JSON 里的 `risk_level` 实际有 high/medium/low 分层，且只有 1 场 gate pass。多数 C 级和缺少外部验证确实需要风险提示，但“所有比赛同红色 HIGH_RISK”会损失排序能力。曼城 vs 水晶宫这种 final 3-0 但 matrix 2-2 且方向概率接近均分的强冲突，和普通 `missing_sharp_money_data` 不应同级。

建议：PR4 先做 severity grading：

- SEV1: MATRIX_CONFLICT + 高尾部/低集中度。
- SEV2: risk_level=high 或 weak_home_favorite_btts_tail。
- SEV3: tail candidates 存在但 matrix 不冲突。
- INFO: no_web_validation/missing_sharp_money_data 这类数据质量提示。

## 3. 12/15 TAIL_RISK 是否过泛？

结论：若 TAIL_RISK 定义为“存在任何 tail_risk_flags 或 risk_score_candidates”，它会非常泛；当前 JSON 中 tail flags 非空甚至达到 15/15，risk candidates 非空 15/15。

这说明 PR #2 的候选保留机制有效，但 UI 不能把“候选存在”直接等同于强尾部风险。风险候选是信息，不一定都是 actionable risk。应按以下维度再分级：

- 候选是否跨方向，例如 final 主胜但候选含 1-2/2-3。
- 候选是否与 matrix top scores 重合。
- matrix direction prob 是否支持该尾部。
- 是否来自明确规则，如 `weak_home_favorite_btts_tail`，还是只是 adjacent score。

## 4. 4/15 CONFLICT 是否有真实价值？

结论：matrix conflict 有真实价值，但当前 JSON 中 direction conflict 是 3 场；任意 matrix flag 是 4 场。若外层显示 4 场 conflict，可能把 `matrix_away_tail_warning`/`low_confidence` 也算入 broad conflict。

真实价值最高的不是数量，而是强弱排序：

- 西班牙人 vs 毕尔巴鄂：final 客胜 0-1，matrix 平局 1-1，典型方向冲突。
- 曼城 vs 水晶宫：final 主胜 3-0，matrix 平局 2-2，且 matrix 方向概率接近三分，属于最强 UI 警示。
- 赫塔费 vs 马洛卡：final 主胜 1-0，matrix 平局 1-1，并伴随 away/draw tail。

这些冲突能防止 UI 把 final 单点包装成确定性结论，因此应保留并突出。

## 5. 是否过度偏平局或高比分？

平局：不过度。final 方向中 draw 只有 2/15。相反，matrix 多次提示 draw risk，说明 final 可能并未充分体现平局尾部；这正是 matrix shadow 的价值。

高比分：存在偏向 3 球路径的倾向。预测总进球 >=3 的场次为 10/15，其中 2-1、3-0、1-2 很多。这不一定错误，因为赔率总进球/BTTS 可能支持，但需要 forward ledger 验证 exact score 和 goal band calibration。UI 上不应把 2-1/3-0 作为唯一结论，必须展示 1-1、2-2、1-2 等风险候选。

## 6. risk_candidates 是否噪音？

结论：不是纯噪音，但目前颗粒度偏宽。

它的价值：保留了弱主胜、BTTS、高比分、客队反打等替代路径，能补 final 单点的盲区。

噪音来源：

- 几乎每场都有候选，缺少 severity。
- adjacent score、lower_odds_rival、data-quality tail 混在一起。
- 没有和 matrix prob / market fair prob 绑定排序。

建议：每个 candidate 增加 `severity`, `source`, `supporting_signal`, `candidate_prob`（如可得），UI 默认只展示 severity>=2 或 Top 3。

## 7. 是否需要先调阈值再做 UI（PR #4）？

结论：不建议先调预测阈值；建议 PR #4 先做 UI severity grading 和解释映射，但不要改 `scripts/predict.py` 的预测逻辑。

理由：

- 当前问题主要是展示层把风险变成“全红/全尾部”的可读性问题。
- 阈值调整会改变预测主干，需要 forward-only ledger 支撑；现在尚未建立 PR5 盲测账本。
- PR4 可以在不改预测输出的前提下改善用户理解：Final vs Matrix、risk candidates、tail flags、downgrade reason、gate pass 数。
- 阈值和校准应放到 PR7，在足够 forward 样本后基于 Brier/Log Loss/risk precision 调整。

## 8. 逐场摘要

1. 周三003 西甲 比利亚雷 vs 塞维利亚 | final 主胜 2-1 conf 60 tier C risk medium | tail=3 candidates=4 | matrix home 2-1 flags=无
2. 周三004 西甲 西班牙人 vs 毕尔巴鄂 | final 客胜 0-1 conf 55 tier C risk high | tail=2 candidates=4 | matrix draw 1-1 flags=matrix_vs_final_direction_conflict,matrix_draw_risk_warning
3. 周三005 法甲 布雷斯特 vs 斯特拉斯 | final 平局 1-1 conf 45 tier C risk high | tail=2 candidates=3 | matrix draw 1-1 flags=matrix_away_tail_warning,matrix_low_confidence_warning
4. 周三006 英超 曼城 vs 水晶宫 | final 主胜 3-0 conf 75 tier C risk low | tail=2 candidates=3 | matrix draw 2-2 flags=matrix_vs_final_direction_conflict,matrix_away_tail_warning,matrix_draw_risk_warning
5. 周三007 意大利杯 拉齐奥 vs 国际米兰 | final 客胜 0-2 conf 65 tier C risk medium | tail=2 candidates=3 | matrix away 1-2 flags=无
6. 周三008 法甲 朗斯 vs 巴黎圣曼 | final 客胜 1-2 conf 60 tier C risk high | tail=3 candidates=4 | matrix away 1-2 flags=无
7. 周三009 西甲 阿拉维斯 vs 巴萨 | final 客胜 1-2 conf 62 tier C risk high | tail=3 candidates=4 | matrix away 1-2 flags=无
8. 周三010 西甲 赫塔费 vs 马洛卡 | final 主胜 1-0 conf 66 tier C risk medium | tail=6 candidates=7 | matrix draw 1-1 flags=matrix_vs_final_direction_conflict,matrix_away_tail_warning,matrix_draw_risk_warning
9. 周三011 美职 辛辛那提 vs 迈国际 | final 客胜 1-2 conf 60 tier C risk high | tail=2 candidates=3 | matrix away 1-2 flags=无
10. 周三012 美职 西雅图 vs 圣何塞 | final 主胜 2-1 conf 65 tier C risk medium | tail=2 candidates=3 | matrix home 2-1 flags=无
11. 周四001 西甲 巴伦西亚 vs 巴列卡诺 | final 平局 1-1 conf 60 tier C risk medium | tail=2 candidates=4 | matrix draw 1-1 flags=无
12. 周四002 沙职 达曼协定 vs 吉达联合 | final 客胜 1-2 conf 65 tier C risk medium | tail=2 candidates=5 | matrix away 1-2 flags=无
13. 周四003 沙职 胡巴卡德 vs 拉斯决心 | final 主胜 3-0 conf 75 tier C risk medium | tail=2 candidates=3 | matrix home 2-0 flags=无
14. 周四004 西甲 赫罗纳 vs 皇家社会 | final 主胜 2-1 conf 55 tier C risk high | tail=5 candidates=4 | matrix home 2-1 flags=无
15. 周四005 西甲 皇马 vs 奥维耶多 | final 主胜 3-0 conf 75 tier B risk medium | tail=2 candidates=3 | matrix home 2-0 flags=无

## 9. 最终审计判断

当前预测 JSON 字段完整，PR #1/#2/#3 的结构价值明确。但风险路由正在进入“过宽提示”阶段：HIGH_RISK/TAIL_RISK 若不分级，会让 UI 失去优先级。PR #4 应做 severity grading 与 presentation，不应改预测；PR #5 应建立 forward ledger；PR #7 再基于盲测数据处理 calibration 和阈值。
