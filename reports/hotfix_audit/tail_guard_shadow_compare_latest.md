# Tail Guard Shadow Compare

- Source: `reports/live_predictions/live_predictions_9999_github_actions.json`
- Old threshold: `home<=52 and away>=23`
- New threshold: `home<=55 and away>=22`
- Match count: 15
- Old trigger count: 3
- New trigger count: 4
- New-only trigger count: 1

## New-only triggers

- 西雅图 vs 圣何塞 | final=home score=2-1 home=53.0 away=22.0
## 最新审查结论记录 (2026-05-14)

当前 shadow comparison 有参考价值，但使用的是 `live_predictions_9999_github_actions.json`，不算新的 forward-only 验证。

当前结论记录为：
- 旧阈值 52/23：触发 3 场
- 新阈值 55/22：触发 4 场
- 新增触发 1 场：西雅图 vs 圣何塞
初步看未造成大范围误报，但样本不足。

下一步建议设计为：
- `52/23 = strong_tail_guard`
- `55/22 = weak_tail_watch`

不要把 55/22 直接升级为强风险标签。它应作为弱观察层，只用于前端提示或 Forward Ledger 观察。

**注意：** 该结果基于已生成预测文件的 shadow comparison，不等同于新赛程 forward-only 验证。是否合并 hotfix，需要等待下一批新预测文件和赛后结果验证。
## Tail Guard Forward-Only Shadow Compare (Today)

- Source: `reports/live_predictions/live_predictions_9999_github_actions.json`
- Old threshold: `home<=52 and away>=23`
- New threshold: `home<=55 and away>=22`
- Match count: 15
- Old trigger count: 3
- New trigger count: 4
- New-only trigger count: 1

### New-only triggers (weak_tail_watch)

- **西雅图 vs 圣何塞** | final=home score=2-1 home_pct=53.0% away_pct=22.0%