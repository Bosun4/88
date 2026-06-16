# 88 全方位审计与升级研究报告（只读阶段）
日期：2026-06-16 | 远端 main：0c7038e | 性质：READ-ONLY 研究，未碰代码/未碰 git

## 0. 执行摘要
- 子代理（Hermes×3）因运行时缺文件/shell/web 工具全部空跑，已由主代理本地实跑接管。
- **赛果复盘联网路径不可用**：web_search 无 kimi key、无任何搜索 provider key，仓库无回填真实赛果。按「真实可验证最硬军规」，不编造赛果驱动调整 → 本轮赛果复盘暂缺事实基础。
- **代码审计发现 P0 泊松污染复活 + 重复函数死码**，pytest 122 全过。
- 读盘逻辑/prompt 升级：需在 P0 去污后再做（避免污染未清就调 prompt）。

## 1. 代码审计（已本地核验，证据带行号）

### 1.1 🔴 P0：泊松影子矩阵复活并污染读盘决策
6/8 PR#27-30 切除的泊松影子矩阵仍在 predict.py 活着，且比上次更深地嵌入下游决策：
- `predict.py:4701 _poisson_pmf_shadow()` — 泊松 PMF
- `predict.py:4758 build_unified_score_matrix_shadow()` — 用泊松算比分矩阵（4780 行 `_poisson_pmf_shadow(lam_h,h)*_poisson_pmf_shadow(lam_a,a)`）
- `predict.py:4846 attach_matrix_shadow_fields()` — 把 matrix_* 字段挂进 prediction
- `predict.py:5539` — 主流程实际调用

**污染链（matrix_* 被下游实际消费，非仅诊断）：**
- `predict.py:5357-5359`：`draw/away/home_prob = max(probs, matrix_direction_probs)` ← 泊松方向概率直接参与方向判定
- `predict.py:5362`：`high_tail` 用 `matrix_goal_probs` ← 泊松进球分布参与尾部判定
- `predict.py:4402-4403`：`matrix_probs` + `matrix_disagreement_flags` 被读取
- `predict.py:4967`：`matrix_top_scores`（泊松比分）进入比分候选合并

缓解事实：attach 函数结尾把 predicted_score/final_direction/confidence/result 等还原回原值（"protected"），即不直接改 AI 终审字段；但方向概率 max() 融合与候选合并仍构成读盘污染。**违背「机械数理锚反向污染盘感」最高军规。**

前端：index.html:601 读 `unified_matrix_top_scores`（非 matrix_top_scores，前端有兜底）。

### 1.2 🟡 重复函数定义（死码 / 隐性 bug）
predict.py 内 14 个同名函数重复定义（后者覆盖前者）：
`adapt_ai_to_frontend, build_evidence_packet, normalize_ai_predictions, _parse_score, _score_direction, _score_total, _score_goal_band, _score_btts, _normalize_score_text, _short_prediction_for_prompt, _exists, _f, _i, _json_compact`
- 已确证 `build_evidence_packet`：933 定义被 3720 定义覆盖 → **933 整段死码（永不执行）**。证据包是喂 AI 读盘的核心，重复定义风险高。
- 其余需逐一确认两份是否一致（若不一致=隐性 bug，加载顺序决定行为）。

### 1.3 ✅ pytest 全回归
`.venv/bin/python -m pytest -q` → **122 passed in 28.12s**，无回归。

## 2. 赛果复盘（受阻，事实缺口）
- 当前 slate（predictions.json，update 2026-06-15 23:50:47）：12 场世界杯小组赛，全部未推荐，信心 28–46。
- 同批 12 场在多日 history 重复出现，信心剧烈摆动（西班牙vs佛得角：6/13 信心88比分4-0 → 6/15 信心42比分3-0）→ 读盘稳定性存疑信号，但**无真实赛果无法判定对错**。
- `p.result='主胜'` 是预测方向中文映射，非真实赛果。
- 联网赛果：web_search 无 key（kimi missing_kimi_api_key），无 brave/tavily/perplexity key，子代理无 web 工具 → 当前环境无法可验证获取真实赛果。
- 结论：赛果驱动的判据调整需先解决赛果数据源；不在无赛果下编造调整。

## 3. 读盘/prompt 升级（待 P0 去污后）
- 现有读盘逻辑编码于 read_the_answer*.py + league_style_analyze.py + predict.py 证据包/prompt 构造。
- 升级前置条件：先切泊松污染（否则 prompt 调优会被污染矩阵反噬，重蹈 6/8 覆辙）。
- 世界杯轮次感知（首轮竞技/末轮出线控分默契战意）需核查 prompt 是否已含——待 P0 后专项审计。

## 4. 建议执行顺序（待用户审批）
1. **P0 去污**（最高优先）：切 matrix_shadow/泊松全链（4701/4758/4846/5539 + 5357-5362 max 融合 + 4402-4403 + 4967 消费点），保留 overround 纯抽水事实。参照 6/8 PR#27-30 手法。
2. **死码清理**：移除 14 处重复函数的死定义（先逐一 diff 确认一致性）。
3. **赛果数据源**：恢复 web_search key 或提供 raw 赛果文件，再做复盘 → 反推判据调整。
4. **读盘/prompt 升级 + 世界杯轮次感知**：在 1-2 完成、污染归零后进行。

每步：新分支 + 备份 tag + 本地 .venv 全回归 + 前后对比报告 + 远端 SHA 核验；P0/commit/push 单独批准、仅推新分支不碰主干。
