# 方案A 修复前后对比报告 — fix/wc-intl-odds-mapping

**日期**: 2026-06-10
**分支基点**: origin/main = `fab0990`（detached）
**备份 tag**: `backup/pre-wc-intl-mapping-20260610234200`
**目标分支**: `fix/wc-intl-odds-mapping`（待推送，未碰 main）

## P0 病灶（fab0990 基线实证）
- `LEAGUE_SPORT_KEY`（global_odds.py:35）原 26 条，**无世界杯/国际赛条目** → 世界杯场次 `.get()` 返回 None 被跳过，根本不拉盘口。
- `TEAM_NAME_MAPPING`（fetch_data.py:9）原约 49 条，**全是俱乐部，国家队 0 条** → 国家队走 GoogleTranslator 兜底，命名碰运气、匹配不上 the-odds-api event。
- `git log --all -S soccer_fifa_world_cup` 全仓库 0 命中 → 该修复从未进任何 commit，线上世界杯欧赔至今 0%。

## 改动内容（纯增量，0 改现有行）
| 文件 | 改动 | 行数 |
|---|---|---|
| scripts/global_odds.py | LEAGUE_SPORT_KEY 补 世界杯/国际赛/国际友谊赛 → soccer_fifa_world_cup | +3 |
| scripts/fetch_data.py | TEAM_NAME_MAPPING 补 48 支世界杯参赛队中英映射（FIFA标准名+沙特别名容错） | +15 |
| **合计** | | **+18 / -0** |

## 修复前后对比（真实导入验证，非回显）
| 指标 | 修复前 | 修复后 |
|---|---|---|
| LEAGUE_SPORT_KEY 条目 | 26 | 29 |
| TEAM_NAME_MAPPING 条目 | 49 | 102 |
| 世界杯 → sport_key | None（跳过） | soccer_fifa_world_cup |
| 国家队翻译来源 | GoogleTranslator 兜底 | 字典精确命中 |
| 巴西/阿根廷/美国/韩国 命中字典 | ✗（走谷歌） | ✓ Brazil/Argentina/USA/South Korea |

## 验证结果
- 语法检查（ast.parse 双文件）: **OK**
- 映射生效（真实 import 验证）: **全部命中字典，不再走谷歌翻译**
- 全回归 pytest: **98 passed in 9.51s，零失败、零 flaky**，改前改后一致

## 已知局限（诚实标注）
- 沙盒所有 API key 全空、API-Football 账号已封 → **无法实网拉世界杯 event 校对英文命名**。
- 48 队英文名按 the-odds-api/FIFA 标准命名 + 易混队别名容错（如沙特双写）。
- **实网命名待开赛后用真实 event 二次校验**；若有个别队命名偏差，开赛后按真实抓包微调即可（不影响 sport_key 与整体链路）。

## 待办
- [ ] P0 最终确认后推送 `fix/wc-intl-odds-mapping`，停在等用户网页合 PR，不碰 main。
