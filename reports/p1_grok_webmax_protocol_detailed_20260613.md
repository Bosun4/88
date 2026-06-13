# P1 细化版：Grok Web-Max 外部事实审判层执行稿

日期：2026-06-13
状态：方案稿，仅用于评审；未改生产代码、未推送。

---

## 0. 核心目标

当前 88 分析质量的瓶颈不是“不会读盘口”，而是：

> 盘面结构已经能读，但外部事实证据层太薄，导致部分 B/main 或 B/small 仍是赔率簇自洽，而非事实闭环。

P1 的目标不是让 Grok 裁比分，而是让 Grok 把联网能力打满，成为：

1. 外部事实猎手
2. 来源质量审判员
3. 反证补全器
4. Gemini 终审的事实弹药库

一句话：

> GPT 读盘面结构，Grok 查外部世界，Gemini 负责终审收敛。

---

## 1. P1 不做什么

### 1.1 不做临场时间序列

暂时不要求 Grok 输出：

- T-24h
- T-6h
- T-90m
- T-60m
- T-30m
- 临场最终

原因：当前 evidence 没有本地真实盘口时间轴。若 prompt 强行要求，模型很容易编出“临场资金流入/回补/诱导”的故事。

### 1.2 不让 Grok 当比分裁判

Grok 可以给：

- 外部事实
- 反证
- 风险
- 信息质量
- 方向影响

但不应该主导最终比分。

最终比分仍由 Gemini 结合：

- GPT 市场结构
- Grok 外部事实
- raw evidence
- recommendation gates

综合裁定。

### 1.3 不让无来源事实升权

如果 Grok 提到：

- 核心伤停
- 战意明确
- 轮换严重
- 官方首发
- 天气影响
- 旅行疲劳

但没有 URL / 来源 / 时间，则只能作为低可信噪音，不得用于升推荐等级。

---

## 2. Grok 搜索维度：从粗到细

### 2.1 人员层 Personnel

必须优先搜索：

- 最新伤停名单
- 停赛名单
- 疑似伤缺
- 复出球员
- 预计首发
- 官方首发（如果已公布）
- 门将变化
- 中卫/后腰/核心中轴缺失
- 头号射手/核心组织者缺失

推荐 query 方向：

- `{home} {away} injury news`
- `{home} predicted lineup {away}`
- `{team} squad news`
- `{team} official lineup`
- `{player} injury update`

### 2.2 战意层 Motivation

必须搜索：

- 是否生死战
- 是否已出线/已淘汰
- 是否争冠/保级/晋级
- 教练发布会态度
- 是否练兵/轮换/保护主力
- 是否存在德比、复仇、主场告别等特殊动机

重点不是“战意强/弱”的空话，而是找可验证事件：

- 积分形势
- 小组排名
- 出线条件
- 教练原话
- 队长/主力采访

### 2.3 赛程层 Schedule & Fatigue

必须检查：

- 一周双赛
- 3-4 天内连续比赛
- 连续客场
- 跨洲旅行
- 长途飞行
- 海拔/气候适应
- 是否刚踢完加时/点球/高强度比赛

影响标签：

- downgrade_favorite
- downgrade_pressing_intensity
- upgrade_late_goal_risk
- upgrade_rotation_risk

### 2.4 环境层 Weather / Venue / Pitch

必须搜索：

- 天气
- 大风/暴雨/高温/低温
- 草皮状态
- 人工草
- 中立场
- 主场实际优势是否存在
- 空场/安保/政治因素

这些只影响：

- 节奏
- 大小球
- 传控/高压是否受限
- 冷门/平局风险

不直接推比分。

### 2.5 媒体与官方层 Media / Official

来源优先级：

1. 官方俱乐部 / 国家队 / FIFA / UEFA / 联赛官网
2. 主流体育媒体
3. 跟队记者
4. 数据站 / 比分站
5. 普通预测站
6. 社媒/论坛/转载

Grok 必须区分：

- official
- mainstream_media
- beat_reporter
- data_site
- prediction_site
- social_rumor

### 2.6 市场快照层 Market Snapshot

允许搜索：

- 当前欧赔快照
- 当前亚盘快照
- 当前大小球快照
- 当前 BTTS / clean sheet / team total 相关信息

但禁止输出：

- 时间序列
- 资金持续流入
- 临场回补
- 诱盘闭环

除非本地 evidence 或来源明确给出多时间点变化。

市场快照只回答：

- 当前市场是否支持 GPT 的方向？
- 当前市场是否和比分簇冲突？
- 当前市场是否提示保守/大球/一边倒？

---

## 3. Grok 输出结构设计

### 3.1 `external_fact_table`

每条事实必须结构化：

```json
{
  "category": "injury | lineup | motivation | schedule | weather | venue | media | market_snapshot | referee",
  "claim": "事实描述",
  "source_type": "official | mainstream_media | beat_reporter | data_site | prediction_site | social_rumor",
  "source_title": "来源标题",
  "source_url": "https://...",
  "published_at": "YYYY-MM-DD or unknown",
  "freshness": "same_day | recent_3d | stale | unknown",
  "confidence": "high | medium | low",
  "impact_direction": "upgrade_home | downgrade_home | upgrade_draw | upgrade_away | upgrade_over | downgrade_over | risk_only | no_clear_impact",
  "why_it_matters": "为什么这条事实影响判断"
}
```

### 3.2 `source_conflict_audit`

用于记录冲突：

```json
{
  "has_conflict": true,
  "conflicts": [
    {
      "topic": "核心前锋是否出战",
      "source_a": "官方称待定",
      "source_b": "媒体称缺阵",
      "resolution": "unresolved",
      "recommendation_impact": "downgrade_confidence"
    }
  ]
}
```

### 3.3 `evidence_quality_score`

0-100 分，建议分层：

- 85-100：官方/主流/多源交叉，同日或近 24h，信息直接影响首发/伤停/战意
- 70-84：可靠媒体 + 数据站交叉，但无官方确认
- 50-69：有来源但偏旧、或来源一般、或只间接影响判断
- 30-49：来源少、信息碎、预测站较多
- 0-29：无来源、社媒传闻、URL 缺失、过旧新闻

### 3.4 `minimum_evidence_needed`

Grok 必须说还缺什么：

- missing_confirmed_lineup
- missing_key_injury_confirmation
- missing_motivation_context
- missing_weather_or_pitch_context
- missing_market_snapshot
- missing_reliable_sources

### 3.5 `external_facts_decision_impact`

Grok 不能直接说“主推 2-1”。它只能输出影响：

```json
{
  "direction_impact": "supports_home | supports_draw | supports_away | mixed | unclear",
  "goal_impact": "supports_over | supports_under | mixed | unclear",
  "recommendation_impact": "can_upgrade | hold | downgrade | no_bet",
  "main_reason": "外部事实如何影响终审"
}
```

---

## 4. 评分闸门：怎么把 Grok 结果用于推荐层

### 4.1 推荐升权规则

只有满足以下条件，Grok 外部事实才允许帮助升权：

- `evidence_quality_score >= 70`
- 至少 2 个可靠来源
- 至少 1 个来源为 official / mainstream_media / beat_reporter
- 事实和 GPT 盘面结构同向
- 无重大 source_conflict

否则只能维持或降级。

### 4.2 必须降级规则

以下任一触发，不能给 main：

- `evidence_quality_score < 50`
- `sources=[]`
- source_url 为空或为 `#`
- 使用了伤停/首发/战意/轮换论据但没有来源
- source_conflict unresolved 且影响核心球员/战意/盘口解释

### 4.3 无来源推荐上限

建议：

- `sources=[]` 且涉及世界杯/杯赛/友谊赛/国际赛：最高 `B/small`
- `sources=[]` 且有强队低赔/跨洲/轮换风险：最高 `C/observe`
- `sources=[]` 但纯盘口结构极强：允许 `B/small`，但必须标记 `pure_market_structure_only`
- `sources=[]` 不允许 `B/main` 或 `A/main`

### 4.4 来源质量对最终推荐的映射

| evidence_quality_score | 推荐影响 |
|---|---|
| 85-100 | 可支持升到 main，但仍需盘面同向 |
| 70-84 | 可支持 small/main 之间选择 |
| 50-69 | 只能辅助，不升权 |
| 30-49 | 降一级 |
| 0-29 | 强制 observe/no_bet |

---

## 5. 用当前 6/10 数据反推 P1 价值

当前线上 `2026-06-10 21:04:22` 数据显示：

- 26 场
- 16 场 gate_pass
- 8 场有 web sources
- 17 场三 AI 比分分歧
- 多个 B/small 或 B/main 无外部来源

典型问题样本：

### 5.1 葡萄牙-刚果金

当前：

- `3-0`
- `B/main`
- `gate=True`
- `websrc=0`

问题：

- 盘口结构自洽，但没有外部事实确认
- 强弱悬殊 + 大球塌缩 + 让球深度推 3-0，本身可以理解
- 但无来源情况下给 main 偏激

P1 后应变为：

- 若 Grok 找到可靠首发/战意/阵容完整：可维持 B/main
- 若无来源：降为 B/small 或 C/observe

### 5.2 卡塔尔-瑞士

当前：

- `0-3`
- `B/small`
- `websrc=0`
- Top4 入选

问题：

- 触发了多个风险规则，包括客胜缺外部确认
- 但仍进 Top4

P1 后应要求：

- 客胜大胜必须至少有来源证明阵容/战意/实力兑现路径
- 无来源时应保留比分判断，但不应进入 Top4 或只能 small/observe

### 5.3 加拿大-波黑

当前回滚版：

- `1-0`
- `B/small`
- `websrc=0`

问题：

- 盘口能解释 1-0，也能解释 1-1
- 缺少外部事实时，终审对“东道主优势/保守首轮”的收敛不够硬

P1 后：

- Grok 应查 Davies/核心伤停、双方阵容、媒体预期、战意
- 若关键进攻事实支持不足，应更倾向降低主胜推荐强度，而非强收 1-0

### 5.4 荷兰-日本

当前：

- `1-1`
- `C/observe`
- `websrc=0`
- 盘口背离逻辑较强

这个处理较合理：

- 有市场背离
- 没有外部确认
- 降到 observe

这类应作为 P1 的正样本。

---

## 6. Prompt 改造要点

### 6.1 Grok instruction 核心句

建议：

> 你是联网外部事实总参谋，不是比分裁判。你必须最大化搜索可验证赛前事实，并用 sources 证明每条会影响判断的 claim。无来源事实不得用于升权；来源冲突未解决时必须降级。禁止编造盘口时间序列，当前赔率只能作为快照，不能推导临场资金流。

### 6.2 Gemini final prompt 核心句

建议：

> 你必须读取 Grok 的 external_fact_table、source_conflict_audit 与 evidence_quality_score。若 Grok 外部证据质量低，任何依赖伤停/首发/战意/赛程的高等级推荐必须降级。若 GPT 盘面结构强但 Grok 证据不足，可保留比分方向，但 recommendation 不得升为 main。

### 6.3 输出约束

必须明确：

- 没有来源，不等于没有风险
- 来源质量低，只能降权
- 盘口结构强但事实层薄，最多小注
- 外部事实与盘面冲突，必须进入 `source_conflict_audit`

---

## 7. 测试设计

### 7.1 Grok prompt 测试

断言包含：

- `external_fact_table`
- `source_conflict_audit`
- `evidence_quality_score`
- `minimum_evidence_needed`
- `禁止编造盘口时间序列`
- `当前赔率只能作为快照`

断言不包含：

- `T-60m`
- `T-30m`
- `临场回补`
- `资金持续流入`

### 7.2 schema 透传测试

`normalize_ai_predictions()` 和 `adapt_ai_to_frontend()` 必须保留：

- `external_fact_table`
- `source_conflict_audit`
- `evidence_quality_score`
- `minimum_evidence_needed`
- `external_facts_decision_impact`

### 7.3 无来源降级测试

构造：

- tier=A 或 B/main
- reason 包含伤停/首发/战意
- web_research.sources=[]

预期：

- `recommend_gate_pass=False` 或 `final_action != main`
- `validation_warnings` 包含 `external_fact_without_source`

### 7.4 高质量来源保留测试

构造：

- `evidence_quality_score >= 80`
- official source + mainstream source
- 与盘口结构同向

预期：

- 不触发来源降级
- 可保留 B/small 或 A/main，具体仍看推荐层

---

## 8. 执行优先级

### P1a：替换错误方向

删除当前 feature 的 `MARKET_TIMELINE_RERANK_ADDENDUM`，替换为 `GROK_WEBMAX_EXTERNAL_INTELLIGENCE_ADDENDUM`。

### P1b：结构化字段透传

让 `normalize_ai_predictions()` / `adapt_ai_to_frontend()` 透传 Grok 外部事实字段。

### P1c：无来源降级闸

新增推荐层规则：

- 无来源 + 上下文论据 = 降级
- URL 空/`#` = 低质量来源
- 来源冲突未解决 = 降级

### P1d：测试闭环

先补测试，再改 prompt 和透传逻辑。

---

## 9. 最终判断

Grok Web-Max 的优势不是“让文案更长”，而是让系统知道：

- 哪些事实是真的
- 哪些事实是传闻
- 哪些事实缺失
- 哪些判断只能靠盘面
- 哪些推荐必须降级

当前 88 已经能读盘，但还需要 Grok 把外部事实层补硬。

因此 P1 的正确目标是：

> 把 Grok 打造成外部事实审判层，让 Gemini 终审拥有可验证的事实弹药，而不是让 AI 继续凭漂亮文案强行收敛。
