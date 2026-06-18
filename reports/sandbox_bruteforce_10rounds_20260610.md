# 88 vMAX 沙盒暴力测试报告（10 轮）

- **日期**: 2026-06-10
- **真实路径**: `/root/.openclaw/workspace/repos/88/`（exec 实跑，非回显）
- **基线**: 本地 HEAD `b155bc0`（注：origin/main 已到 `a8f50f1`，本地落后2个自动更新commit，未碰git）
- **测试性质**: 纯沙盒只读/测试，**未改任何生产代码、未碰 git、未碰主干**
- **环境**: .venv Python 3.12.3

---

## 总览

| 轮 | 维度 | 结果 |
|---|---|---|
| R0 | 基线全回归 | ✅ 98 passed / 9.7s |
| R1 | 静态体检（语法/import/泊松/幽灵依赖） | ✅ 编译全过，幽灵依赖=0，1-1锚定已弃用 |
| R2 | predict.py AST 死码审计 | ⚠️ **14函数 ~439行真死码** |
| R3 | 比分簇 gate 边界轰炸 | ✅ 5/5 gate正确 + 7/7异常不崩 |
| R4 | 证据包纯净度 | ✅ 零泊松/影子污染词 |
| R5 | 随机模糊 2000 条 | ✅ 100% 存活 |
| R6 | 联赛风格闸一致性 | ⚠️ prompt重复文本 |
| R7 | 联赛降级路径 | ⚠️ **荷甲/沙特联/MLS/英西意落generic** |
| R8 | 重复全回归 ×5（flaky检测） | ✅ 98×5，零flaky |
| R9 | 线上index.html一致性 | ⚠️ **4个BTTS/终审字段前后端错配** |
| R10 | 汇总 | ✅ 本报告 |

**结论：核心读盘逻辑健壮、确定性强、证据包纯净（核心军规全过）。发现3类非致命瑕疵需后续单独批准修复。**

---

## ✅ 通过项（核心军规全绿）

### R0/R8 测试稳定性
- 单轮 98 passed；连续 5 轮全 98 passed，耗时稳定 9.2–9.5s，**零 flaky，确定性强**。

### R1 静态健康
- `compileall` 全过（scripts/market_sentinel/forward_ledger/tests）
- **幽灵依赖 bs4/lxml/sklearn = 0**（前期清理成果保持）
- 1-1 锚定：`draw_anchor`/`1_1_anchor` 字面 = 0；第3753行存在**反锚定保护语句**（"不得仅因1-1/0-0最低赔就推翻共识走平局"）✅

### R3 读盘三模块 gate（边界轰炸）
全部正确触发（阈值 A6_MAX=11.0 / A7_MAX=14.0）：
- 排除线 a4>5.3 → `four_goals_exclusion_line` ✅
- 曲线塌缩 a5/a4≤1.70 → `big_goal_curve_collapse` ✅
- 尾部共振 a6≤11 / a7≤14 → `big_goal_tail_resonance` ✅
- 单点诱盘 a4低a5高 → `four_goals_single_point_low_caution` ✅
- 异常输入（空/全0/负值/字符串/None/1e9/缺字段）7/7 不崩溃 ✅

### R4 证据包纯净度（核心军规）
- `build_evidence_packet` 产出 11288 字符，**污染词命中 = NONE**（poisson/泊松基准/matrix_shadow/lambda_home/期望进球矩阵全无）
- 含读盘事实：读盘/锚点/抽水/聪明钱 ✅
- **泊松影子矩阵 `attach_matrix_shadow_fields`（5134行）经实测确认：仅作事后诊断对照，用 protected 字典锁死核心字段，不反向改读盘结果、不进prompt** ✅

### R5 随机模糊
- `_total_goal_anchor_facts` + `_cross_anchor_questions` 各灌 1000 条随机match（负值/None/字符串/1e12/空队名）→ **2000/2000 存活**

---

## ⚠️ 发现项（需单独批准后才改，本轮未动代码）

### 【发现1·R2】predict.py 14个函数 ~439行真死码
经 `__code__.co_firstlineno` 字节码级实证（inspect.getsourcelines 会误报，已校正）——以下早版定义被后版同名覆盖，**永不执行**：

| 函数 | 死码行(早版) | live版 |
|---|---|---|
| adapt_ai_to_frontend | 2250–2446（**197行**） | 5090 |
| build_evidence_packet | 932–1017（86行） | 3425 |
| normalize_ai_predictions | 1706–1763（58行） | 4176 |
| _short_prediction_for_prompt | 1042–1051 | 3710 |
| _f / _i / _exists / _json_compact / _normalize_score_text / _parse_score / _score_direction / _score_total / _score_goal_band / _score_btts（早版286–445） | ~93行 | 2726+ |

> 物理极简军规违规：5167行里约439行幽灵代码。**建议另起死码清理分支处理（仿前次 PR#28/#29 流程），需单独批准。**

### 【发现2·R7】联赛DNA表覆盖缺口
`LEAGUE_DNA_PROFILES` 仅11条。用户偏好投注联赛 **荷甲/沙特联/MLS** 及五大联赛中 **英超/西甲/意甲** 全部落入 `generic`（medium波动），丢失大球/小球盘感标签。
- 注：读盘核心靠 prompt 文字清单（已含这些联赛归类），DNA表仅辅助标签，故**非逻辑错误**，但表与prompt清单不一致。

### 【发现3·R9】线上4个字段前后端命名错配
前端 index.html 读取、但后端从不产出（永远 undefined，靠 `||` 兜底）：
- `p.ai_btts`（后端是 `btts_ai`，拼写顺序错配）
- `p.btts_decision`、`p.btts_label`、`p.final_audit_reason`

> 不崩溃，但**线上 BTTS 展示与终审理由可能空白**——触及"线上展示须与模型逻辑一致"军规。

### 【发现4·R6/R9 命名瑕疵】
- prompt 第3695行与3756行存在**完全重复的"防守绞肉联赛"文本**（冗余）
- 前端字段 `unified_matrix_top_scores` 命名像泊松矩阵，实际后端(2334行)赋的是 AI 读盘候选 `top_candidates`，命名误导

---

## 建议优先级
1. **P1**：发现3（线上BTTS/终审字段错配）—— 直接影响线上展示正确性
2. **P2**：发现1（439行死码清理）—— 物理极简，仿 PR#28/#29 推新分支
3. **P3**：发现2（联赛DNA补荷甲/沙特联/MLS/英西意）+ 发现4（去prompt重复文本）

> 以上均需 **commit/push 前单独批准、仅推新分支不碰 main**。本轮严格只读，origin/main(a8f50f1) 零改动。

---

# 追加：真实数据链路验证（用户追问"调API了吗"后补测）

## 凭证现实
- 沙盒**所有外部 key 全空**：API_FOOTBALL_KEY / THE_ODDS_API_KEY / ODDS_API_KEY / GROK_API_KEY / GPT_API_KEY / GEMINI_API_KEY 均未设置，无 .env/secrets。线上靠 GitHub Actions secrets，不在沙盒。
- 实网探测（端点存活）：football-data.org 200（185联赛，但具体比分403需付费）、the-odds-api 401(MISSING_KEY)、api-football 403。**网络通，瓶颈是凭证。**

## 替代方案：用真实生产抓包数据（data/history_*.json）跑真链路
共 **35 个真实历史批次、668 场真实场次**（真实 a0-a7 总进球赔率 + home/away_stats + h2h + change 资金流）。

### ✅ 读盘 gate 真实数据端到端（6/8 批 27 场世界杯）
- 27/27 **零崩溃**；gate 分布：排除线18 / 曲线塌缩8 / 单点诱盘1 / 尾部共振0
- 逐场判读全部可解释：
  - 德国vs库拉索 a4=4.2/a5=4.8(slope1.14) → 曲线塌缩=碾压真大球 ✅
  - 秘鲁vs西班牙 a4=4.1/a5=7.0(slope1.71) → 单点诱盘拦截 ✅
  - 科特迪瓦vs厄瓜多尔 a4=8.5 → 排除线 ✅
- **结论：读盘核心逻辑在真实赔率上判读正确、稳健。**

### 🔴 真链路发现（仅真数据能暴露）

**发现5：`change`（资金流/聪明钱方向）真实有效率仅32%**
- 全量668场硬统计：**67%空 / 32%有效**（印证记忆"change约7成为空"）
- 分月：5月69%空、6月51%空；6/8世界杯批 100%空
- 有效结构：`{'win':-1,'same':1,'lose':1}`（升降方向码）
- 影响：背离探测/Sharp资金审判在2/3场次实际空转，读盘退化为单靠赔率结构。

**发现6：24场世界杯全落 `cup_or_cross_context` 通用杯赛档**
- 因 LEAGUE_DNA_PROFILES 无世界杯条目 + global_odds.py 缺 soccer_fifa_world_cup（两修复均未落地），世界杯被当普通杯赛，丢专属读盘情报。

**发现7（远端SHA核验）：世界杯 sport_key 修复从未进任何commit**
- `git log --all -S soccer_fifa_world_cup` 全仓库0命中，origin/main 0命中。
- 记忆中"6/10沙盒复现+修复后27场0/27→24/27"的改动**只在沙盒验证过，代码从未提交**。世界杯国际欧赔命中至今仍0%。

## 仍需用户提供才能做的
真调外部AI/赔率端到端：需一个可用 GROK_API_KEY+GROK_API_URL（或任一odds key）。给则注入沙盒只用不打印不入库，跑真AI读盘/真拉赔率全链路。
