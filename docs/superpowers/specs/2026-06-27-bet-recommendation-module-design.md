# 下注推荐模块设计文档（bet_recommendation）

日期：2026-06-27
分支：`feat/bet-recommendation-module-20260627`（基于 origin/main）
作者：赛后复盘 + 功能开发会话

---

## 1. 目标

在终审结论之后，为每场比赛新增一个**本地确定性下注推荐模块**：基于真实盘口赔率计算各玩法期望值，输出**激进**与**稳健**两套下注组合（前端可切换），每场总预算 200 元、单注最低 20 元。前端 `index.html` 新增「下注推荐」模块展示。

### 用户确认的配置
- **1C** 两套都要：激进 + 稳健，前端可切换，**默认显示激进**，每套组合及每条腿附中文推荐理由
- **2 abcd** 覆盖玩法：正确比分(区分0-0/1-1/2-2) + 胜平负让球 + 总进球/大小球 + 半全场（不含 BTTS）
- **3 C/200** 每场总预算 200 元，按期望值/信心自动分配，单注≥20
- **4A** 数据来源：predict.py 输出 `bet_recommendation` 字段，index.html 渲染真实数据
- **5A** 基于 main 开新分支，PR 到 main，脏改动已隔离

### 免责声明（落实到前端 UI）
本模块为算法推荐展示，不构成盈利保证。下注金额仅为基于期望值的分配建议，是否下注由用户自行决定。前端必须显著展示此提示。

---

## 2. 核心原则（不可违背）

1. **本地闸门原则**：本模块只**新增 `bet_recommendation` 字段**，**绝不修改** AI 终审的 `final_direction`、`predicted_score`、`recommendation.tier`、`bet_action`。
2. **确定性纯函数**：不调用 AI、不联网、无随机；同样输入必得同样输出（可单测、可回归）。
3. **真实赔率驱动**：所有期望值基于 `match_obj` 原生赔率字段（CRS/总进球/让球/半全场），无赔率的玩法直接跳过，不臆造。
4. **尊重风控**：当终审 `bet_action ∈ {no_bet, observe}` 或低区分度闸门触发时，下注推荐降为「不建议下注」，金额置 0，仅展示理由。

---

## 3. 数据来源（赔率字段映射）

| 玩法 | 字段来源 | 说明 |
|------|----------|------|
| 正确比分 CRS | `CRS_FULL_MAP`（s00/s11/s22/w10/l01…）→ `match_obj` 对应键 | 含 0-0/1-1/2-2 三态区分 |
| 胜平负/让球 | 1X2 欧赔 + HHAD 让球（hhad_win/same/lose, give_ball） | |
| 总进球/大小球 | `a0`-`a7`（总进球数赔率）→ 推导大小球 2.5 等 | `STANDARD_TOTAL_GOAL_ODDS` 作兜底参考 |
| 半全场 | `HFTF_MAP`（ss/sp/sf/ps/pp/pf/fs/fp/ff） | |

赔率缺失时该玩法 candidate 不生成（标记 `available:false`）。

---

## 4. 算法设计

### 4.1 候选生成（每场）
从终审已产出的结构化字段提取「模型看好」的标的，与真实赔率交叉：

- **正确比分候选**：取 `top3` + `predicted_score` + `score_elimination_audit` 中 keep 的比分；其赔率从 CRS 字段取。
- **胜平负候选**：取 `final_direction` 及 `direction_probs`；赔率取对应 1X2。
- **让球候选**：依 `give_ball` 与 hhad 赔率，方向同 final_direction。
- **总进球/大小球候选**：取 `goal_band` 推导（如 goal_band=2 → 总进球2球；≥3 → 大2.5），赔率取 a{n}。
- **半全场候选**：依 final_direction + 是否慢热（contextual_logic.tempo / worldcup risk）保守取 平/主 或 主/主，赔率取 HFTF。

### 4.2 模型概率估计（去赔率隐含 + 模型信心融合）
对每个候选估计「模型主观命中概率」 `p_model`：

- 方向类（胜平负/让球/半全场）：用 `direction_probs` 为基；
- 比分类：用 `top3[i].prob`（已是百分比）为基，若无则用簇审计估一个保守值；
- 总进球类：用 goal_band 对应一个保守 band 概率（如主选 band 给 0.45，相邻 band 0.25）。

**这是模型主观概率，不是市场隐含概率**——期望值 = 模型认为的边际，正是"模型 vs 市场"的价值判断。

### 4.3 期望值（EV）
对每个候选：

```
ev = p_model * (odds - 1) - (1 - p_model) * 1
ev_ratio = ev   # 每 1 元投注的期望净收益
```

- `ev_ratio > 0`：正期望（有价值）
- 激进组合优先选 `ev_ratio` 高的（哪怕命中率低、赔率高）
- 稳健组合优先选 `p_model` 高且 `ev_ratio ≥ 0` 的（命中率优先，避免高赔长尾）

### 4.4 两套组合的区分

**稳健组合（steady）**
- 入选条件：`p_model ≥ 0.35` 且 `ev_ratio ≥ -0.05`（允许微负，命中优先）
- 最多 2-3 个标的，偏向：主选比分、主方向、主选总进球
- 不含高赔长尾（odds > 8 的标的排除）
- 预算分配：按 `p_model` 加权

**激进组合（aggressive）**
- 入选条件：`ev_ratio > 0`（必须正期望）
- 最多 3-4 个标的，允许高赔长尾（含 1-4/1-5/0-4/0-5 等大比分，若 CRS 有赔率且 EV>0）
- 预算分配：按 `ev_ratio` 加权（追求期望最大化）

### 4.5 预算分配（每场 200，单注≥20）
对一套组合内的 N 个标的：

1. 计算权重 `w_i`（稳健=p_model，激进=ev_ratio 截断到≥0.01）
2. 归一化 → 初始金额 `amount_i = 200 * w_i / Σw`
3. **单注下限钳制**：`amount_i < 20` 的标的，若组合内标的数 > 可容纳数（200/20=10，实际远小于），按 EV 排序砍掉最弱标的后重算；保证每个保留标的 ≥ 20
4. **取整**：金额向下取整到 5 元的倍数（20/25/30…），余额补给 EV 最高的标的
5. 总和不超过 200

边界：
- 若正期望标的为 0（激进） → 激进组合返回空 + 理由「无正期望标的」
- 若 `bet_action ∈ {no_bet, observe}` → 两套都返回「不建议下注」，amount=0

### 4.6 输出结构（写入 pred["bet_recommendation"]）

```json
"bet_recommendation": {
  "available": true,
  "per_match_budget": 200,
  "min_stake": 20,
  "disclaimer": "算法推荐，非盈利保证，下注自负",
  "no_bet": false,
  "no_bet_reason": "",
  "default_view": "aggressive",
  "steady": {
    "total_stake": 200,
    "expected_return": 236.5,
    "reason": "中文组合理由：为何这套偏稳健、命中逻辑",
    "legs": [
      {"market":"correct_score","selection":"0-0","odds":8.5,"p_model":0.40,"ev_ratio":2.40,"stake":120,"potential_payout":1020,"reason":"中文单腿理由：为何选这个标的"}
    ]
  },
  "aggressive": {
    "total_stake": 200,
    "expected_return": 410.0,
    "reason": "中文组合理由：为何这套追期望、风险点",
    "legs": [ ... ]
  },
  "version": "bet_reco_v1"
}
```

字段说明：
- `potential_payout = stake * odds`（命中该腿的返还）
- `expected_return = Σ stake_i * odds_i * p_model_i`（组合期望返还，腿间独立近似）
- `market` 枚举：`correct_score / one_x_two / handicap / total_goals / over_under / half_full`

---

## 5. 集成点（predict.py）

### 5.1 常量与映射（文件顶部 CRS_FULL_MAP 附近）
- 复用 `CRS_FULL_MAP`、`HFTF_MAP`、`STANDARD_TOTAL_GOAL_ODDS`
- 新增市场枚举常量 `BET_MARKETS`

### 5.2 新增纯函数
- `_extract_market_odds(match_obj) -> dict`：把 match_obj 原始赔率规整成 {market: {selection: odds}}
- `_build_bet_candidates(pred, odds_map) -> list`：生成候选 + p_model + ev
- `_allocate_budget(legs, budget, min_stake, weight_key) -> list`：预算分配 + 单注下限 + 取整
- `_apply_bet_recommendation_gate(pred, match_obj) -> pred`：主入口，组装 steady/aggressive，写 `pred["bet_recommendation"]`

### 5.3 调用位置
在每场预测处理函数末尾、`_sync_gate_with_bet_action(pred)` **之后**、`pred["engine_version"]=...` **之前**插入：

```python
try:
    _apply_bet_recommendation_gate(pred, match_obj)
except Exception as e:
    pred.setdefault("validation_warnings", []).append(f"bet_recommendation_gate_error:{str(e)[:120]}")
```

理由：此时所有 gate 已跑完，能读到最终 tier/bet_action/risk_tags，下注推荐基于最终结论。

---

## 6. 前端设计（index.html）

### 6.1 数据流
`data/predictions.json` 的每个 match.prediction 现在多了 `bet_recommendation`。前端在 `renderDetails(m,p)` 中新增渲染块 `renderBetRecommendation(p)`。

### 6.2 模块 UI
- 标题：「💰 下注推荐」+ 免责小字
- Tab 切换：[激进] / [稳健]（**默认激进**）
- 组合理由：Tab 下方显示该套组合的中文推荐理由
- 每条腿一行：玩法标签 + 选项 + 赔率 + 金额（高亮）+ 命中可得 + 单腿推荐理由（小字/可展开）
- 底部汇总：总投注 200 / 组合期望返还 / 若不建议下注则显示理由卡
- 配色沿用现有 dark 主题（`.json-box`/`.mb-r` 风格），新增 `.bet-leg` `.bet-tab` class
- 无 `bet_recommendation` 或 available=false 时显示「本场无下注推荐」

### 6.3 交互
纯前端 Tab 切换（JS toggle class），不发请求。

---

## 7. 测试计划

### 7.1 单元测试（tests/ 下新增 test_bet_recommendation.py）
- `_allocate_budget`：单注下限钳制、取整到5、总和≤200、砍最弱标的
- `ev` 计算正确性
- no_bet 场景返回空组合
- 赔率缺失玩法被跳过
- 激进含正EV长尾、稳健排除高赔

### 7.2 回归
- 运行现有 `pytest`（138 passed 基线），确认零回归
- mock 模式跑一遍 run_predictions，确认 `bet_recommendation` 字段生成且不破坏既有字段

### 7.3 真实数据验证
- 用 data/predictions.json 现有真实场次离线注入测试函数，人工核对佛得角型(0-0)/大胜型的推荐是否合理

---

## 8. 风险与边界

- **不改 AI 判断**：仅加字段，回归保证既有 138 测试不挂
- **赔率噪声**：体彩赔率可能缺字段 → available:false 兜底
- **p_model 非校准概率**：明确标注是"模型主观"，前端不宣称真实命中率
- **金额取整误差**：余额补给最高 EV 腿，保证总和=预算或≤预算
- **预算溢出**：硬上限 200，单注下限 20，二者冲突时（标的过多）按 EV 砍标的

---

## 9. 交付物
1. predict.py：新增 4 个纯函数 + 1 处调用
2. index.html：新增渲染块 + 样式 + Tab 交互
3. tests/test_bet_recommendation.py：单元测试
4. 设计文档（本文件）
5. PR 到 origin/main
