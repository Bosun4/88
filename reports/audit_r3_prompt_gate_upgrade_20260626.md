# 项目88 全方位审计 + 世界杯第三轮 Prompt/闸门升级报告

> 触发：用户"重点审计 predict.py / prompt / 规则 / 闸门，根据世界杯第三轮（已出线大轮换、控分爆冷）升级 prompt"
> 时间：2026-06-26 03:2x GMT+8
> 仓库：`/root/.openclaw/workspace/repos/88`（已 `git remote -v` 核实 = `git@github.com:Bosun4/88.git`）
> 备份：`/root/backups/88_predict.py.20260626_032120.bak`
> 回归：`.venv` 内 `pytest tests/ -q` → **138 passed**（升级前后均通过，零回归）

---

## 一、审计范围与产物

| 模块 | 文件 | 行数 | 结论 |
|---|---|---|---|
| 主引擎 | `scripts/predict.py` | 5558 | 架构清晰：证据包→三模型分析(GPT/Grok)→互审→Gemini终审→本地协议闸门 |
| 经验规则 | `scripts/experience_rules.py` | 421 | 含 upset/draw/goals 规则库；R3 维度此前无专属规则 |
| 闸门 | predict.py 内 9 个 `_apply_*_gate` | — | 见第三节 |
| 真实赛果对账 | `reports/audit_real_results_worldcup_202606.md` | — | n=23，方向 52.2%，比分 21.7%，**ECE 22.6（校准差）** |
| 30 场汇总 | `reports/audit_summary_20260624.md` | — | 方向 56.7%，比分 20.0%，4 大系统问题 |

---

## 二、审计核心发现（基于项目自有真实赛果，非臆测）

### 🔴 发现1：世界杯专属"平局识别失败"（最致命，与本次需求高度吻合）
高信心强队胜全部踢平，集中在第二/三轮：
- 西班牙 预测3-0(信心76) → 实际 **0-0**
- 葡萄牙 预测3-1(信心78) → 实际 **1-1**
- 卡塔尔 预测0-2(信心77) → 实际 **1-1**
- 沙特 预测0-2(信心67) → 实际 **1-1**
- 厄瓜多尔 预测3-1 → 实际 **0-0**

**根因链**：强队已出线 → 大面积轮换 → 阵容生疏/慢热/控分 → 不打净胜球 → 闷平或被反杀。
这正是用户指出的"第三轮控分爆冷"机制，模型此前未对其加权。

### 🔴 发现2：信心系统性虚高（70-79 档宣称75%、实测25%，缺口+50）
ECE 22.6 远超理想<5。R3 已出线热门方的胜信心必须硬性下调。

### 🔴 发现3：负方安慰球过量（2-1→2-0、4-1→3-0、1-2→0-1）
已有 `_apply_lopsided_consolation_goal_gate`，但因第一轮回测净收益=0 被摘除调用（保留代码）。本次不动它。

### 🟡 发现4：强队火力上限被压（荷兰2-1→实5-1、德国4-0→实7-1）
属大球带释放问题，已有曲线塌缩/尾部共振判据覆盖，本次不改。

---

## 三、闸门盘点（9 个，调用顺序见 predict.py 5498-5538）

1. `apply_two_one_home_hard_no_bet_gate` — 2-1 尾部硬不推
2. `apply_deep_favorite_loose_euro_guard` — 深盘强队欧赔不实压
3. `apply_pre_match_factor_v2_gate` — **综合赛前因子（含 R3 三分法 + 本次新增 R3 保平先验）**
4. `_apply_external_fact_source_gate` — 外部来源质量
5. `_apply_direction_candidate_consistency_gate` — 方向/候选一致
6. `_apply_contrarian_market_claim_gate` — 反向市场主张
7. `_apply_lopsided_consolation_goal_gate` — **已摘除调用**（回测无收益）
8. `_apply_low_confidence_draw_guard` — 低信心平局降 observe
9. `_sync_gate_with_bet_action` — bet_action 同步

特点：本地闸门**只降推荐/加风险标签，从不改 AI 的方向与比分**（防止本地成为"第四个裁判"）。

---

## 四、本次升级（5 处，+41 行，零删除逻辑，零回归）

所有改动均挂在世界杯/R3 识别下，**不污染俱乐部联赛读盘**。

### Prompt 层（4 处）
1. **`_cross_anchor_questions` R3 硬审计**：四问扩为五问（新增"同组并行开赛/默契球/算计净胜球"维度）；新增"R3 爆冷高发实证先验"——列出 5 场真实翻车案例，要求把【被逼平/小负/0进球】列首要风险、热门方向信心上限 ≤60、显式审计收窄比分簇。
2. **Gemini 终审 R3 三分法**：在 A/B/C 三类后新增"轮换·控分·爆冷实证先验"段（同 5 案例）+ "联网硬要求"扩为四件事（出线形势/锁定名次/教练发布会轮换意图/默契球争议），无来源时 recommendation 最高 B。
3. **phase1 prompt 第4点（R3）**：补入实证案例与原因链，热门方胜信心上限硬限 60（除非联网证明派主力争名次），重点释放弱队抢分/定位球爆冷路径。
4. **`LEAGUE_STYLE_RESEARCH_HINTS["世界杯"].risk_note`**：嵌入实证（出线+轮换场默认下调热门信心 ≤60，优先防 0-0/1-1/1-0/0-1/被逼平）。

### 闸门层（1 处，确定性后处理）
5. **`apply_pre_match_factor_v2_gate` 新增 R3 热门保平先验**：
   - 触发：`worldcup_r3` + `name_favorite` + 方向为胜 + 预测净胜≥2 + 非(必须取胜且已联网确认)。
   - 动作：胜方向最高降 **B**；若无联网 / 已出线 / 有轮换风险 → 进一步降 **C** 并标记防平。
   - 价值：即使 AI 未在文本写"已出线/轮换"，只要结构上是 R3 名气热门推穿盘大胜，也会被自动降级。专治发现1的高信心翻车。
   - 安全：只改推荐等级与风险标签，**绝不改方向/比分**（与全项目军规一致）。

---

## 五、联网能力说明
- 项目 prompt 已要求各模型（GPT/Grok/Gemini）在具备联网能力时执行 Web-Augmented Research，所有影响方向/比分/推荐的 claim 必须进 `external_fact_table` / `web_research.sources`（含 url + published_at）。
- 本次升级把 R3 的出线形势/轮换/锁定名次/默契球**纳入联网硬要求**：无来源时保留盘口方向但 recommendation 封顶 B，禁止把战意推演包装成 main。
- 注：本机 `web_search` 工具当前被禁用，故 R3 实时赛果未联网回填；升级所用先验全部来自项目自有真实赛果对账（n=23/30），可复核。

---

## 六、未做 / 待确认
- **未 commit / 未 push**：按项目军规，改生产码后 commit/push 需用户单独明确批准，且只精确 `git add scripts/predict.py`，绝不 `git add .`。
- 未改 `_apply_lopsided_consolation_goal_gate`（已知回测无收益，保持摘除）。
- 建议：06-18 之后 R3 真实赛果联网回填后，对新增 R3 保平先验做命中率/净收益回测再决定是否升权为硬降级。
