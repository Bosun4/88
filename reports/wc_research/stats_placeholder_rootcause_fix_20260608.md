# 占位stats脏数据 + 1X2缺失 根因排查与最小修复报告

> 分支：`fix/placeholder-stats-flag-20260608`（基于 origin/main e28b709，已含世界杯情报PR#30）
> 日期：2026-06-08 | 状态：**改码+全回归完成，停在push前待批准**

## 一、现象（#555批次27场世界杯盘面）
- **6场竞彩1X2(sp_home/draw/away)全为0**：西班牙vs佛得角、法国vs北爱、秘鲁vs西班牙、伊拉克vs挪威、卡塔尔vs瑞士、德国vs库拉索。
- **4场home_stats==away_stats完全相同占位值**(25场11胜 form=WDLWD)：上述6场里的4场。

## 二、根因（已用代码复算确认，非猜测）

**根因链（单一源头）：**
1. 1X2来自数据源 `item.get("win/same/lose")`（fetch_data.py:154）。这6场**数据源未挂胜平负盘**——竞彩尚未对这些世界杯场次开1X2投注（开赛6/11临近才会挂全），但比分簇(CRS a系列)已挂。**属数据源时序问题，非代码bug。**
2. 脏stats是**下游连带症状**：当API无真实战绩 → 走 `generate_stats_from_context` 容灾。世界杯场次 `home_rank=away_rank=0`(兜底成10) **且** `sp_home=sp_away=0`(无赔率信号) → 主客两队 strength 算出完全相同(0.526) → 两队得到**完全相同的占位值**(25场11胜WDLWD)。

**复算验证：** rank兜底10 → strength=1-9/19=0.526 → win_rate→wins=11、form=WDLWD，与实际脏数据完全吻合。

## 三、最小修复（不碰数据源时序，只修"喂假数据当真"）

### scripts/fetch_data.py (+14/-3)
`generate_stats_from_context` 新增信号检测：当 `rank<=0 且 无1X2赔率` = 无任何强度信号时，给占位stats打标记：
- `estimated: True`（始终标记为估算）
- `data_available: False`（无真实信号时）
- `data_note`: "⚠️无真实战绩且无1X2赔率信号，此为排名兜底占位值，不可作读盘依据"

→ 不再让AI把虚构的"25场11胜WDLWD"当真实战绩读盘。

### scripts/predict.py (+4)
`build_evidence_packet` 的 `data_quality` 新增 `team_stats_reliable` 字段：任一方stats `data_available=False` 时为 `False`，让AI在证据层就知道战绩数据不可信。

## 四、改前 vs 改后

| 项 | 改前 | 改后 |
|---|---|---|
| WC无信号场stats | 虚构"25场11胜WDLWD"，AI无从分辨 | 同样占位但带 `data_available:False`+警示note |
| evidence数据质量 | 无stats可信度标识 | `team_stats_reliable` 明确True/False |
| 有真实1X2/排名场 | 正常 | 正常（data_available=True，不受影响）|
| 回归 | 98 passed | **98 passed** |

## 五、验证
- 三场景单测：无信号场→False、有1X2场→True、有排名场→True ✅
- 端到端：德国(脏)场 team_stats_reliable=False、比利时(正常)场=True ✅
- 字节编译：fetch_data.py + predict.py 均OK ✅
- 全回归：改前98 passed → 改后98 passed（零破坏）✅

## 六、守边界
- ❌ 未commit/push（等批准）❌ 未碰origin/main ❌ 未改方向/比分判定
- 纯增量 +18/-3，新逻辑只在"无真实数据"分支生效，不影响正常场次
- ⚠️ 1X2缺失本身是数据源时序问题，开赛临近竞彩挂全后自然消失；本修只确保"缺失期间不喂假数据当真"

## 七、待批准
本地全回归通过。**是否批准 commit + push 到新分支 `fix/placeholder-stats-flag-20260608`？**（仅推新分支，不碰主干）
