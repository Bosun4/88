# 临场资金时间序列升降级 Prompt 升级审计报告

## 基本信息

- 分支：`feature/market-timeline-rerank-20260612`
- 基线/远端 SHA：`fab0990ecfcfd98183d3a16f1214e40839e5aef3`
- 备份 tag：`backup/pre-market-timeline-rerank-20260612134911`
- 改动范围：Prompt 层与 prompt 回归测试；不改核心比分/方向后处理逻辑；未 push。

## 升级动机

墨西哥 vs 南非与韩国 vs 捷克复盘显示：

- 墨西哥场：88 已能用 `2球模态 + 零封/BTTS=no + 主胜簇` 命中 `2-0`，说明原读盘闭环有效。
- 韩国场：88 已将 `2-1` 写入失败路径，但未能因临场 `韩国平手低水持续压低 + 大球水下压/小球水抬升` 将其从尾部升权。

本次升级目标是加入“临场资金时间序列升降级协议”，要求 AI 在联网能力可用时抓取多时间点、多公司、多市场的盘口变化，并将其用于比分簇动态重排。

## 改动内容

### `scripts/predict.py`

新增 `MARKET_TIMELINE_RERANK_ADDENDUM`，核心要求：

1. 抓取时间点：初盘、T-24h、T-6h、T-90m、T-60m、T-30m、临场最终；缺口必须写 `missing_timeline_points`。
2. 公司角色：百家平均、竞彩官方、威廉、澳门、Bet365、Pinnacle、立博。
3. 四市场闭环：欧赔定方向/分流，亚盘定赢球深度，大小定进球边界，BTTS/零封定双方进球结构，正确比分只做映射。
4. 聪明钱联动：最后 T-60m/T-30m 跨市场同步时必须判断是确认、诱导、分流还是回补。
5. 比分升降级：低分主线遇到主队低水保护 + 大球水下压时，提升 `2-1/1-2` 等三球边界；让球升深 + 大小压低 + 零封一致时提升 `2-0/3-0` 等零封穿盘比分。
6. 反证条件：必须写 `minimum_evidence_needed` / `why_this_can_fail`。
7. 没有真实时间序列数据时，只能写 `timeline_unavailable` 并降级信心。

注入位置：

- Grok 联网指令 `_web_research_instruction("grok")`
- `build_phase1_prompt`
- `build_gemini_final_prompt`
- `build_fallback_referee_prompt`
- `build_family_debate_referee_prompt`

### `tests/test_dual_market_divergence.py`

新增测试锁住协议生效：

- Grok 联网指令必须携带 `临场资金时间序列`、`T-60m`、`Bet365`、`Pinnacle`、`比分升降级`。
- Gemini final、fallback、family debate prompt 必须携带 `临场资金时间序列`、`四市场闭环`、`比分升降级`、`T-30m`。

## 验证结果

- 先跑红灯：新增测试在实现前失败，确认测试能捕捉缺失协议。
- 相关回归：`25 passed in 0.63s`
  - `tests/test_dual_market_divergence.py`
  - `tests/test_family_debate_referee.py`
  - `tests/test_final_referee_retry.py`
  - `tests/test_abstain_preserves_analysis.py`
  - `tests/test_two_one_hard_gate.py`
- 全量回归：`100 passed in 9.70s`

## 预期行为变化

- 墨西哥型盘口：欧赔主胜表面退热，但亚盘升深 + 大小压低 + 零封一致时，应维持/提升 `2-0/3-0`，不应误杀主胜穿盘结构。
- 韩国型盘口：早盘 `1-1` 低赔不再永久锁死；若最后 T-60m/T-30m 出现主队平手低水保护 + 大球水下压/小球水抬升，应把 `2-1` 从尾部提升为强防线或主线候选。

## 回退方式

如升级后预测表现变差，可执行：

```bash
git switch feature/market-timeline-rerank-20260612
git reset --hard backup/pre-market-timeline-rerank-20260612134911
```

或直接切回 `restore/pre-fix-fab0990` / `origin/main` 对比。

## 注意事项

- 本次未 commit、未 push。
- 当前工作区仍存在改动前已有的未跟踪报告文件，本次未触碰。
- 本次只改 Prompt 协议，不引入新爬虫、不新增本地采集链路、不改变本地后处理比分逻辑。
