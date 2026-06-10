# 全方位检查与本地修复报告 — 2026-06-11

## 真实仓库与远端核验

- 工作路径：`/root/.openclaw/workspace/repos/88/`
- 本地 HEAD：`fab0990ecfcfd98183d3a16f1214e40839e5aef3`
- 远端 `origin/main`：`fab0990ecfcfd98183d3a16f1214e40839e5aef3`
- 远端 `origin/HEAD`：`fab0990ecfcfd98183d3a16f1214e40839e5aef3`
- 当前状态：detached HEAD，本轮只做本地工作树修复；未 commit、未 push、未碰 main。

## 已确认根因

1. 世界杯国际欧赔链路只在工作树中部分修复，远端 main 仍未落地；且 `国际赛/国际友谊赛` 误绑 `soccer_fifa_world_cup` 存在空抓/误抓风险。
2. 前端字段契约不完整：`over_under_2_5` 未展示，理由字段未兼容 `ai_native_reason/final_ai_analysis`，多处置信度绕过风控降级后的 `display_confidence`。
3. `steam_tracker._match_event()` 只按主队匹配，客队变量被计算但未参与命中，可能把同主队不同客队赛事错配。
4. `LEAGUE_DNA_PROFILES` 缺世界杯、荷甲、沙特、MLS、英超、西甲、意甲等重点联赛，赛前风控会落到 `generic` 或杯赛通用档。
5. 国家队英文名存在 API 命名差异风险，如 `美国/USA` vs `United States`、`捷克/Czech Republic` vs `Czechia`。

## 本地修复内容

- `scripts/global_odds.py`
  - 新增 `sport_key_for_league()`，支持 noisy league label 的包含式解析。
  - 仅明确世界杯标签映射到 `soccer_fifa_world_cup`，移除泛 `国际赛/国际友谊赛` 伪装世界杯的风险。
  - 新增国家队英文别名相似度：`USA/United States`、`South Korea/Korea Republic`、`Czech Republic/Czechia`、`Ivory Coast/Cote d'Ivoire` 等。
- `scripts/steam_tracker.py`
  - 统一复用 `sport_key_for_league()`。
  - `_match_event()` 改为主客队 pair score，主队相似但客队不符不再误命中。
- `scripts/predict.py`
  - 输出稳定前端契约别名：`ai_confidence`、`ai_btts`、`ai_over25`、`ai_score_reason`。
  - 补齐世界杯、荷甲、沙特、MLS、英超、西甲、意甲等 `LEAGUE_DNA_PROFILES`。
  - 世界杯优先识别为专属 profile，不再落 `cup_or_cross_context`。
- `index.html`
  - 新增 `getDisplayConfidence()`，全站优先展示 `recommendation.display_bet_confidence/display_confidence/risk_adjusted_confidence`。
  - 新增 `getOver25Label()` 并在卡片和底层字段展示 over2.5。
  - `finalReason()` 兼容 `ai_score_reason/ai_native_reason/final_ai_analysis`。
- 测试
  - 新增世界杯 mock 注入测试、泛国际友谊不误绑世界杯测试、国家队别名匹配测试。
  - 新增 steam 主客队组合匹配测试。
  - 新增 DNA 覆盖测试与前端契约别名测试。

## 验证结果

- 编译：`python -m py_compile scripts/fetch_data.py scripts/global_odds.py scripts/steam_tracker.py scripts/predict.py scripts/main.py` 通过。
- 聚焦回归：`python -m pytest tests/test_dual_market_divergence.py tests/test_steam_tracker.py tests/test_prematch_factor_gate.py tests/test_friendly_lessons_gate.py -q` → `38 passed in 0.60s`。
- 全量回归：`python -m pytest -q` → `104 passed in 9.31s`。

## 未处理但已记录

- P2：`scripts/predict.py` 仍有旧版 `_short_prediction_for_prompt` alias 赋值疑似无运行入口，可后续单独清理。
- P2：Phase1 与 Gemini final prompt 有重复联赛风格文案，可后续抽共享常量降低漂移风险。
- 当前工作区原本存在未跟踪报告/研究脚本与 `.backup_p0p1_20260607141824/`，本轮未清理，避免误删历史现场。
