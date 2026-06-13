# P1 方案草案：Grok Web-Max External Intelligence Protocol

日期：2026-06-13
范围：仅方案设计，不改代码、不推送。
目标：废弃“临场资金时间序列”方向，把 Grok 定位为**联网外部事实总参谋**，最大化利用原生联网能力，但严格限制在“可验证事实层”，不允许它伪造盘口时间故事或替代终审裁判。

---

## 1. 为什么要改方向

当前本地 feature 的问题不在于“联网不够多”，而在于它把 Grok 的联网任务朝“临场时间序列盘口解释”推得太远：

- 本地 evidence 中没有真实的多时间点盘口时间轴（T-24h/T-60m/T-30m 等）
- prompt 却要求 Grok 解释“临场资金确认/回补/诱导/分流”
- 这会诱导模型用语言补出一条并不存在的数据链

这与 88 的核心原则冲突：

- 没有数据，就明确说没有
- 可以联网补事实，但不能编时间序列
- AI 读盘可以大胆，证据链必须真实

因此 P1 不该继续强化“时间点盘口曲线”，而应强化 **Grok 的外部事实侦察与冲突审判能力**。

---

## 2. 新定位：Grok = 联网事实总参谋，不是比分裁判

### 2.1 角色定义

Grok 的职责从：

- 查伤停、查首发、查热度、顺手讲盘口

升级为：

- 穷尽搜索所有可能改变赛前判断的外部事实
- 给每条事实附来源、发布时间、可信度、冲击方向
- 主动指出哪些“看起来像强信号”的信息其实证据不足
- 不负责裁定最终比分，只负责把“外部世界”带进终审

### 2.2 一句话原则

> Grok 负责找事实、拆传闻、做冲突审计；Gemini 才负责把这些事实和盘面一起收敛成最终判断。

---

## 3. Grok Web-Max 应搜什么

不追盘口时间序列，改为全覆盖搜下面这些事实层：

### A. 人员层
- 最新伤停
- 停赛
- 复出
- 预计首发
- 官方首发（如果已公布）
- 门将/中卫/后腰/核心前锋等关键位置是否缺失

### B. 战意层
- 是否争冠/保级/争四/晋级/淘汰赛生死战
- 是否提前出线/无欲无求
- 是否轮换窗口
- 教练发布会是否有明确态度

### C. 赛程层
- 一周双赛
- 连续客场
- 长途旅行/跨洲/时差
- 是否刚踢完高强度杯赛

### D. 环境层
- 天气
- 场地
- 海拔
- 人工草/糟糕草皮
- 中立场/空场/异常环境

### E. 舆情与新闻层
- 官方公告
- 主流媒体伤停新闻
- 赛前发布会
- 当地记者/跟队记者更新
- 可靠赔率资讯站的赛前综述

### F. 市场快照层（注意：只要快照，不要时间轴）
- 当前可见的欧赔/亚盘/大小球快照
- 不要求多时间点
- 不要求推导资金流曲线
- 仅用于验证“市场是否大致支持/冲突”

---

## 4. Grok 必须输出的结构

建议 Grok 的联网输出不再是松散 prose，而是半结构化：

### 4.1 `external_fact_table`
每条事实至少包含：

- `category`: injury / lineup / motivation / schedule / weather / market_snapshot / media / referee / venue
- `claim`: 事实内容
- `source_title`: 来源标题
- `source_url`: 来源 URL
- `published_at`: 发布时间
- `confidence`: high / medium / low
- `impact_direction`: upgrade_home / downgrade_home / upgrade_draw / upgrade_away / upgrade_over / downgrade_over / risk_only
- `why_it_matters`: 这条事实为什么会影响比赛判断

### 4.2 `source_conflict_audit`
- 哪些来源互相冲突
- 哪些是官方、哪些只是媒体、哪些是二手转载
- 冲突未消解时，不允许给高置信推荐

### 4.3 `evidence_quality_score`
建议 0-100：

- 官方首发/官方伤停/俱乐部公告 → 高
- 主流媒体 + 多源交叉 → 中高
- 只有一家媒体/转载链条不清晰 → 中低
- 论坛、无出处博文、社媒传闻 → 低

### 4.4 `minimum_evidence_needed`
Grok 必须说明：

- 还缺什么，才能把这场从观察升到可买
- 比如：
  - 缺首发确认
  - 缺关键伤停二次确认
  - 缺官方战意/轮换信息
  - 缺可靠市场快照

---

## 5. Grok 不允许做什么

这是本方案最重要的一部分。

### 禁止 1：伪造盘口时间序列
不能写：
- T-60m 主队持续压低
- T-30m 大球水下压
- 临场回补明显

除非 evidence 里真的提供了这些时间点数据。

### 禁止 2：无来源硬说伤停/战意
不能写：
- 核心前锋伤缺
- 主队战意一般
- 客队轮换严重

如果没有 sources，只能写：
- `unverified_claim`
- 并降低 evidence_quality_score

### 禁止 3：用联网事实直接裁比分
Grok 可以说：
- “这条事实提升平局/小球/客队不败风险”

但不应替代 Gemini 直接完成最终收敛。

### 禁止 4：把赔率快照讲成资金故事
当前可见赔率可以写“支持/冲突”，但不能进一步脑补成：
- 机构确认
- 聪明钱流入
- 回补完成
- 诱盘闭环成立

除非本地有硬数据支持。

---

## 6. 对 GPT / Gemini 的配合关系

### GPT
继续做：
- 市场结构师
- 读静态赔率形态 / 正确比分簇 / 总进球结构 / 竞彩风控偏斜

### Grok
只做：
- 外部事实猎手
- 冲突来源审计员
- 赛前信息完整度评估器

### Gemini
负责：
- 读 GPT 的结构
- 读 Grok 的联网证据质量
- 读 raw evidence 的盘面事实
- 决定是否：主推 / 小注 / 防平 / 观察 / 放弃

即：

- GPT 决定“盘像什么”
- Grok 决定“外部世界有没有支持或打脸”
- Gemini 决定“能不能下注，以及下注到什么程度”

---

## 7. 最小代码改动设计（不写实现，只列点）

### 7.1 删除/替换内容
当前 feature 里的：
- `MARKET_TIMELINE_RERANK_ADDENDUM`

应整体废弃，替换为：
- `GROK_WEBMAX_EXTERNAL_INTELLIGENCE_ADDENDUM`

### 7.2 `_web_research_instruction("grok")`
从：
- 伤停/首发/临场新闻/资金热度/时间序列

改成：
- 伤停
- 首发
- 赛程
- 战意
- 天气场地
- 媒体与官方来源
- 当前赔率快照
- 来源冲突审计
- 缺证据时必须写 `timeline_unavailable` 或 `missing_external_confirmation`

### 7.3 最终 prompt builder
当前追加时间序列协议的位置：
- `build_gemini_final_prompt`
- `build_fallback_referee_prompt`
- `build_family_debate_referee_prompt`
- 如果 phase1 prompt 也有追加，则一并替换

改为追加新的 `Grok Web-Max` 协议。

### 7.4 schema 约束
建议增加可选字段：
- `external_fact_table`
- `source_conflict_audit`
- `evidence_quality_score`
- `minimum_evidence_needed`
- `missing_external_confirmation`

注意：
- 这些字段属于 AI 输出 schema 或 raw_item 补充字段
- 没有真实来源时必须允许为空，但不能伪造内容

---

## 8. 测试建议

现有 feature 测试只验证“协议字符串是否进 prompt”，这还不够。

P1 应新增：

### 8.1 指令测试
- Grok web instruction 必须包含官方/媒体/首发/伤停/天气/赛程/赔率快照/冲突审计
- 不再出现 `T-60m/T-30m/临场回补/时间序列` 等字样

### 8.2 终审 prompt 测试
- final/fallback/family debate prompt 必须要求 Gemini 读取 Grok 的 external facts
- 若 `evidence_quality_score` 低，推荐等级不得抬高

### 8.3 无来源降级测试
- 当 web_research.used=false 或 sources=[] 时
- 含“伤停/战意/首发”结论的推荐必须触发验证警告或降级

### 8.4 结构化字段保留测试
- `normalize_ai_predictions()` / `adapt_ai_to_frontend()` 必须能携带并透传
  - `external_fact_table`
  - `source_conflict_audit`
  - `evidence_quality_score`
  - `minimum_evidence_needed`

---

## 9. 我对优先级的判断

### 先做
1. 清理同名覆盖风险（结构债）
2. 替换 Grok 的 prompt 协议方向
3. 增加外部事实表与来源质量字段测试

### 后做
1. 若未来真的接入真实盘口时间轴数据，再考虑“临场时间序列模块”
2. 影子矩阵去泊松化
3. 风险闸门拆文件

---

## 10. 最终建议

如果只允许做一个 P1，我建议优先做：

> **把 Grok 从“临场盘口讲故事的人”改造成“联网外部事实总参谋”。**

这是最符合 88 原则的：
- 可验证
- 不伪造
- 不迎合
- 真正让联网能力服务终审，而不是制造新的幻觉层
