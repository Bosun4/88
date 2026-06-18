# 88 全方位世界杯审计报告（Superpowers / 只读）

- 审计时间：2026-06-15
- 真实路径：`/root/.openclaw/workspace/repos/88/`
- 审计分支：`fix-gemini-grok-full-spectrum-research-20260614`
- 本地 HEAD（远端核验一致）：`1d9d03f` == `origin/...20260614`
- `origin/main` 远端 SHA：`5d60693`（PR#46 已 squash 入 main = `46b1e43`）
- 基线回归：本地 `.venv` `pytest -q` → **122 passed**（绿）
- 方法：只读排查 + 真实 `data/predictions.json` 抓包验证 + 只读子代理深读 predict.py 提示链
- 纪律：未改任何代码、未做任何 git 操作；远端 SHA 为唯一事实源

---

## 0. 核心事实：线上 100% 是世界杯
`data/predictions.json`（MAX-v1.1, 2026-06-14 22:30）`matches.today` = **16 场，全部 `league=世界杯`**。
→ 任何"世界杯专属"缺口此刻不是边角，而是**作用于全部线上比赛**。

---

## P0-A｜世界杯国际欧赔 sport_key 仍然死（最高优先级）

- 证据：`scripts/global_odds.py:35` `LEAGUE_SPORT_KEY` **无 `soccer_fifa_world_cup`**。
- `git grep "soccer_fifa_world_cup" origin/main` → **NOT FOUND**；当前分支 HEAD → **NOT FOUND**。
- 调用点 `global_odds.py:153` `LEAGUE_SPORT_KEY.get("世界杯")` → `None` → 国际欧赔 event 拉不到。
- 连带：`steam_tracker.py:275/323` 同样 `None` → steam 线移动信号死。
- **线上实证**：16 场全部 `sharp_money_direction=unclear / sharp_detected=False / sharp_confidence=0`。旗舰「双盘背离/聪明钱」探测对全部世界杯比赛**完全失效**。
- 影响：系统在自己最高价值的信号（国际盘 vs 竞彩背离）上**全程瞎跑**。6/10 已诊断、批"全做"，但远端核验显示**从未进任何 commit，线上至今 0%**。

## P0-B｜国家队名缺失 → 即便补了 sport_key 也匹配不上

- `scripts/fetch_data.py:9` `TEAM_NAME_MAPPING` = **纯俱乐部映射，0 个国家队**（阿根廷/巴西/法国/西班牙/佛得角… 全无）。
- 兜底 `fetch_data.py:35` `name.replace("联"," United")` → 对含"联"的中文名会污染（腐蚀风险）；国家队走 `GoogleTranslator`（联网、有损、不稳）。
- `global_odds.py:173-174` 用 `translate_team_name` 做英文模糊匹配 → 国家队匹配大概率失败。
- 影响：P0-A 与 P0-B 是**串联依赖**——补 sport_key 不补国家队映射，euro-odds 仍对不上队名，等于白补。两者必须同批。

---

## P1-A｜世界杯没有 DNA / 风格闸（生态先验丢失）

- `predict.py:117-156` `LEAGUE_STYLE_RESEARCH_HINTS` 仅 德甲/意甲/英超/英冠/法甲/西甲/荷甲 → 世界杯**无键**。
- `predict.py:716-733` `_league_style_anchor_facts("世界杯")` 子串无命中 → `matched_style_hints=[]`。
- `predict.py:4002-4004` 散文前置闸列了 德甲/荷甲/挪超→大球、西甲/意甲→小球，**无世界杯小组赛桶**。
- `_league_dna_profile("世界杯")`（`predict.py:4955`）因"世界杯"含"杯"→落入 `cup_or_cross_context` 通用高方差档，**非世界杯专属**。
- 关键死码：`league_intel.py:32` `LEAGUE_PROFILES["world_cup"]`（场均 2.48 球、R1 最闷/R3 净胜球收窄的实测生态先验）只被 `build_league_intelligence` 消费，而该函数**predict.py 从不调用**（predict.py 只调 `detect_league_key/analyze_motivation/analyze_world_cup_context`）→ **2.48 生态先验是孤儿死码**。
- 影响：世界杯比赛拿到的是"任意未知联赛"的通用闸，无"中立场低分锦标赛足球"先验，盘感锚缺失。

## P1-B｜轮次闸：逻辑正确但"软"且依赖抓取串

- 好消息（subagent 行级核验）：`league_intel.py:316-326` `_wc_detect_round` 解析轮次；`:339-350` 把 R1/R3 映射成**正好是用户标准规则**：
  - R1：`[WC-R1] 首轮最闷…强队首战常打不开，大让球/大球期待打折`
  - R3：`[WC-R3] 控分=净胜球收窄非总进球降…2-1/1-0占R3 42.5%…已出线强队vs有动机方=诱盘高危`
  - 已经过 `predict.py:3746→3782→4028` 注入 `<evidence_batch>`，GPT/Grok/Gemini 都能看到。
- 缺陷：
  1. 它是**JSON 证据深处**（`local_quantitative_intelligence.world_cup_reading_intel`），**不是散文硬指令**。`predict.py:4002-4004` 的显式风格闸**只字不提世界杯轮次**，模型需自己在证据里翻到才会用 → 可靠性看模型读深度。
  2. `_wc_detect_round` 若 `baseface/match_num` 不含"第三轮/末轮"→ 返 `None` → `[WC-RND?] 默认按小组赛读`。**R3 控分/默契闸会静默不触发**。轮次闸只和抓取串一样可靠。

## P1-C｜中立场 / 主场优势对世界杯失真（待深核）

- 世界杯全程中立场，但引擎多处用 `home_rank/away_rank`、`home_stats` 权重、"主场"措辞。需确认是否在世界杯下错误赋主场优势（subagent 已标为待核点，本轮未逐行定论）。

---

## P2｜结构化盘感未落地为硬闸（只在报告里）

- `L3 防火墙（伪强队 86% 打不穿让球盘）`：`grep` 全 `scripts/*.py` → **NOT FOUND**（只在 `reports/`）。
- `强强高分簇 4球≤5.3 且 5球≤7.8 → 2-2/3-2`：`predict.py:806-819` 有 5.3 排除/单点诱盘锚，但 subagent 核验"强强簇"成对逻辑仅作为 **family-debate 回退角色**（`predict.py:4139`），主链非硬闸。
- 影响：6/9–6/10 结构化的盘感（L3 防火墙、强强簇）多数停留在文档，未进 AI 硬证据/闸。

---

## 测试与线上一致性
- 122 项回归全绿，说明**已有测试覆盖不到这些世界杯缺口**（缺 WC sport_key、缺国家队映射、缺 WC DNA 都无对应断言）→ 测试存在"绿色盲区"。

---

## 结论（按优先级）
1. **P0-A + P0-B（必须同批）**：补 `soccer_fifa_world_cup` sport_key + 48 队国家队中英映射。这是 6/10 批了但从未落地的同一笔。无此，世界杯背离/聪明钱永久 0%。
2. **P1-A**：建世界杯专属 DNA（中立场/2.48 生态/无主场优势）并把 `league_intel.LEAGUE_PROFILES["world_cup"]` 从孤儿死码接回主链。
3. **P1-B**：把 R1/R3 轮次规则从"JSON 深处"升级为散文硬指令；并对 `_wc_detect_round=None` 增加从出线形势/对阵推断轮次的兜底。
4. **P1-C**：核中立场主场优势失真。
5. **P2**：把 L3 防火墙 + 强强簇落为主链硬证据/闸 + 补对应回归测试，消除绿色盲区。
