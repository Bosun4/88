# P0-A+P0-B 修复前后对比报告（世界杯国际欧赔接通）

- 时间：2026-06-15
- 分支：`fix/wc-intl-odds-sportkey-mapping-20260615`（off `1d9d03f`）
- 备份 tag：`backup/pre-wc-sportkey-20260615`
- 改动文件：`scripts/global_odds.py`、`scripts/fetch_data.py`（共 +24 行，0 删除既有逻辑）
- 状态：**仅本地分支，未 commit / 未 push / 未碰 main**

## 改了什么（最小改动）
1. `global_odds.py` `LEAGUE_SPORT_KEY` 增一行 `"世界杯": "soccer_fifa_world_cup"`。
2. `fetch_data.py` 新增 `NATIONAL_TEAM_MAPPING`（48+ 国家队中→英 SSOT），并在 `translate_team_name` 中**先查国家队映射再走 GoogleTranslator**——绕开 `.replace("联"," United")` 腐蚀路径与联网不稳。

## 修复前 → 修复后（真实 `data/predictions.json`，今日 16 场全世界杯）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| `世界杯` → sport_key | `None`（死） | `soccer_fifa_world_cup` ✅ |
| 路由到国际欧赔抓取的场次 | **0 / 16** | **16 / 16** ✅ |
| 48 队队名命中 SSOT（不走有损翻译） | 0 | **48 / 48** ✅ |
| 队名腐蚀风险（`联`→United 等） | 存在 | 国家队路径已绕开 ✅ |

队名样例（修复后）：阿根廷→Argentina、佛得角→Cape Verde、刚果金→DR Congo、库拉索→Curacao、阿尔及利→Algeria、波黑→Bosnia and Herzegovina。

## 验证
- 语法：两文件 `ast.parse` OK。
- 路由实证：`LEAGUE_SPORT_KEY.get('世界杯')` → `soccer_fifa_world_cup`；按真实 slate 分组 → `{soccer_fifa_world_cup: 16}`，16/16 场进入抓取队列。
- 队名实证：48 场 live 队名 100% 命中 SSOT，无一落到 GoogleTranslator。
- 回归：本地 `.venv` `pytest -q` → **122 passed**（与基线一致，零回归）。

## 说明（诚实边界）
- 沙盒 `ODDS_API_KEY` 为空 → 未发真实 API 请求；以上验证为**路由层 + 队名层**实证（沙盒可达的最强证据）。「实际欧赔数值是否拉回 + 背离是否点亮」需线上带 key 运行核验——**沙盒有效 ≠ 线上落地**，远端运行/SHA 为唯一事实源。
- 仅此一笔即解 P0-A+P0-B 串联死点；P1（WC DNA、轮次硬指令）、P2（L3/强强簇硬闸）仍在后续分支。

## 下一步（待批）
P0/commit/push 需单独批准。批准后：仅 push 本分支 `fix/wc-intl-odds-sportkey-mapping-20260615`（不碰 main），push 后 `git rev-parse origin/...` 核验远端 SHA。
