# 世界杯读盘档接入 league_intel 喂prompt链路 — 前后对比报告

> 分支：`feat/wc-reading-intel-20260608`（基于 origin/main a0bc7be）
> 备份tag：`backup/pre-wc-intel-20260608214342`
> 日期：2026-06-08 | 状态：**改码+本地全回归完成，停在push前待批准**

## 一、改动目标
把已核验的世界杯读盘情报档（5届320场分轮实证 + 双窗口状态 + 2026新制）接入 AI 终审的 prompt 证据链，作 evidence 喂给AI，**不当第四裁判、不动方向/比分判定**。

## 二、改动文件与内容（103插入/3删除，纯增量）

### scripts/league_intel.py (+97/-2)
1. `detect_league_key`：新增识别 `世界杯/world cup` → `world_cup`，`国际赛/友谊赛/friendly` → `intl_friendly`。
2. `LEAGUE_PROFILES`：新增 `world_cup`(2.48均/47%over)、`intl_friendly`(2.60均) 两档，数值来自5届320场实证。
3. 新增 `WC_TEAM_FORM`（16队双窗口状态标签）、`WC_CLEAN_SHEET_FRAGILE`（零封脆弱名单）、`_wc_detect_round`（轮次推断）、`analyze_world_cup_context`（情报注入主函数）。
4. `build_league_intelligence`：世界杯/国际赛**跳过** analyze_motivation 联赛排名逻辑（避免 home_rank=0 触发的争四/降级/中游噪声污染），改注入世界杯读盘档。

### scripts/predict.py (+9)
- 在 `build_evidence_packet` 真实prompt链路中（`analyze_motivation` 旁），世界杯/国际赛时调 `analyze_world_cup_context`，结果注入 `evidence["local_quantitative_intelligence"]["world_cup_reading_intel"]`。

## 三、改前 vs 改后对比

| 项 | 改前 | 改后 |
|---|---|---|
| 世界杯识别 | detect_league_key 返回 `default` | 返回 `world_cup` |
| 世界杯prompt | 仅default联赛画像 + **错误的排名战意噪声**（MID-TABLE/RELEGATION，因home_rank=0） | 分轮先验+赛制+状态+零封脆弱，**排名噪声已剔除** |
| 喂AI链路 | build_league_intelligence是死码(无人调用) | 接在真实消费点 build_evidence_packet |
| 回归 | 98 passed | **98 passed（无破坏）** |
| 字节编译 | OK | league_intel.py + predict.py 均OK |

## 四、端到端验证（真实#555盘面）
用 origin/main 的真实世界杯盘面跑 `build_evidence_packet`，确认 `world_cup_reading_intel` 进入evidence包，例（德国vs库拉索）：
```
[WC2026] 48队/12组/前2+8个最佳第三名晋级...
[WC-R1] 首轮是最闷一轮(场均2.34最低)...盘面大让球/大球期待要打折
[WC-FORM:主 德国|split] WWW但场场丢球(0零封)...防线漏
[WC-CS-FRAGILE] 德国 零封能力差...防BTTS/补时击穿
```

## 五、回归对比
- 改前基线：`98 passed in 13.86s`
- 改后终检：`98 passed in 13.46s`
- 结论：纯增量，零破坏，新增逻辑全部走 `if lk in (world_cup, intl_friendly)` 分支，不影响任何现有联赛路径。

## 六、未做（守军规边界）
- ❌ 未 commit、未 push（等单独批准）
- ❌ 未碰 origin/main（分支独立）
- ❌ 未改方向/比分判定逻辑（只注入evidence）
- ⚠️ 数据时效：WC_TEAM_FORM 是6/8双窗口快照，开赛后状态会变，需滚动更新（非本次范围）

## 七、待批准事项
本地全回归通过。**是否批准 commit + push 到新分支 `feat/wc-reading-intel-20260608`（仅推新分支，不碰主干，不开PR除非你要）？**
按军规需你单独批准 push。
