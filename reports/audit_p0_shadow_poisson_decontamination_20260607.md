# P0 审计报告：切断泊松影子矩阵对 AI 读盘的污染注入

**日期**: 2026-06-07
**基线远端 HEAD**: 124199f
**备份**: tag `backup/pre-p0p1-shadow-cleanup-20260607141824` + 物理备份 `.backup_p0p1_20260607141824/`
**改动文件**: `scripts/predict.py`（+12 / -27，净 -15 行）

---

## 1. 问题（根因级铁证）

6/3 死码清理只切了"泊松生产死码"，但泊松实体仍活在 `build_evidence_packet`（真正生效版，行3465）的注入链里：

- 行3496 调用 `build_unified_score_matrix_shadow()`（内部自实现 `_poisson_pmf_shadow` 跑泊松矩阵 + 全盘口 Devig）
- 行3515 把泊松输出注入 `evidence["matrix_shadow_facts"]`：`top_mathematical_scores`、`shadow_recommended_score`、`shadow_recommended_direction`，外加一段 note 明令"**AI 必须利用此静态数理测算作为强事实基准**"
- 该 evidence 经 `_safe_json_line(e)` **整体序列化**进 phase1 + 终审 prompt
- GEMINI 终审协议第3条硬性要求"**必须读取 matrix_shadow_facts（静态数理泊松基准）**"，第4条整条背离探测建立在"静态数理概率"之上

**后果**：泊松通过影子暗线污染 AI 视野，把比分锚向保守簇 → 印证 13 场复盘的"比分仅15%命中、5/13进球低估、均衡盘方向50%"。

## 2. 改动（最小且精确）

| # | 位置 | 修复前 | 修复后 |
|---|------|--------|--------|
| 1 | 行3482 | `import advanced_models`（死 import，全仓零调用，scipy 唯一来源） | 删除 |
| 2 | 行3495-3499 | `matrix_pack = build_unified_score_matrix_shadow(...)` + top_scores/recommended_* 泊松计算 | 删除 |
| 3 | 行3514-3531 | `evidence["matrix_shadow_facts"]`（泊松比分+方向+屈从指令） | 替换为 `evidence["jingcai_market_facts"]`（仅保留 overround 抽水率纯市场事实，note 明确"系统不提供任何静态数理基准，比分与方向完全由 AI 读盘") |
| 4 | GEMINI 终审协议第3条 | "必须读取 matrix_shadow_facts（静态数理泊松基准）" | "必须读取 local_quantitative_intelligence 与 jingcai_market_facts；系统不提供静态数理/泊松基准" |
| 5 | 终审协议第4条 | 背离探测锚点=静态数理概率 | 背离探测锚点=真实市场事实（大众投票热度/变盘方向/聪明钱/国际盘偏斜） |
| 6 | protocol_notes | "matrix_shadow_facts 已注入" | "jingcai_market_facts 已注入；系统不提供数理基准" |

**保留**（未误伤）：
- `build_unified_score_matrix_shadow` 函数本体 → path2（行4425）的 `matrix_disagreement_flags` 后端诊断仍在用，有 `test_two_one_hard_gate.py` 测试覆盖
- `overround` 竞彩抽水率 → 纯市场事实，符合"物理极简只算事实"
- `numpy` → `quant_edge.SteamMoveDetector` 资金流检测在用（注：scipy 可随 advanced_models 在 P1 移除，numpy 不可动）
- `evidence_compiler_version` 版本串 → `test_review_20260512_regressions.py:46` 硬断言，保持不变

## 3. 验证（本地 .venv 全程实跑）

- **语法**：`py_compile scripts/predict.py` ✓ 通过
- **全回归**：`pytest tests/ -q` → **86 passed / 0 failed**（20.54s）
- **运行时证据包对比**（真实 match 跑 `build_evidence_packet`）：
  - `matrix_shadow_facts`：已移除 ✓
  - `jingcai_market_facts`：存在，overround=8.15%，note 无泊松字样 ✓
  - `local_quantitative_intelligence`（战意/经验/steam）：完整保留 ✓
- **实际 AI prompt 文本扫描**：
  - phase1(grok) prompt：泊松硬指令 = 无 ✓
  - 终审 addendum：泊松硬指令 = 无 ✓

## 4. 结论

泊松影子矩阵对 AI 读盘的**注入污染已彻底切断**，背离探测的锚点从"静态数理概率"重构为"真实市场事实"，与用户"废泊松、AI 像人读盘"范式一致。后端诊断标记（path2）保留以维持测试与背离审计能力。

## 5. 遗留（P1 待批，下一轮）
- 删 `advanced_models.py`（含 scipy 泊松，纯死模块）+ 从 requirements 移 scipy + 改 `test_shadow_integration.py:53` 冒烟测
- 删 `build_evidence_packet_v203`(行3295) 死码 + 三组重复同名函数旧版（921/1009/1115）
- 清根目录 37 个零引用一次性脚本 + 补 .gitignore
