# PR: feat: sub-50 weak-home structural risk transparency

## 1. P4.5 审计结论
针对预测日志的扫描显示，当模型判断主胜且 `home_win_pct < 50` 且 `(draw_pct + away_win_pct) > home_win_pct` 时，如果 `predicted_score` 为一球小胜（如 2-1）且平局拉力极强（如 1-1 同赔或同概率），AI 非常容易受“保级战意”等主观叙事干扰而无视概率结构。赫罗纳等 3 场比赛的实战复盘验证了这一系统性陷阱。

## 2. 为什么不硬锁 1-1
直接在后处理阶段拦截并强制改写 `final_direction=draw` 或 `predicted_score=1-1` 会破坏系统的不可变裁判原则，剥夺 AI 作为终审裁判的完整判断权。这种“拍脑袋拦截”可能会在某些强战意局（如赫塔费 3-1）导致反向错误。

## 3. 为什么采用 No-Bet Gate 与 Score Cluster
采用透明化策略：
- **不干预预测本身**，保持原本算出的 `2-1` 与 `55` 信心。
- **添加外挂诊断**：若符合条件，追加 `sub50_tiebreaker_warning` 和 `no_bet_reason`，告知前端此单“不适合作为强推单比分”。
- **增加 Score Cluster**：向用户输出一个 `["2-1", "1-1"]` 簇，将其明确为二元博弈，既反映了主胜意图，也锁定了最可能打出的反跑平局。

## 4. 新增字段说明
此 PR 添加了以下非侵入性字段：
- `decision_quality_flags` (list): 挂载 `SUB50_WEAK_HOME_TIEBREAKER`
- `sub50_tiebreaker_warning` (bool): 激活标志
- `no_bet_reason` (string): 提示主客概率倒挂
- `hedge_recommendation` (string): “防 1-1 / 防 X2 / 不建议强推主胜单比分”
- `score_cluster` (list): 如 `["2-1", "1-1"]`
- `narrative_override_warning` (bool): 审计 AI reason，若包含战意等词汇则预警

## 5. 不修改 final 的证明
测试用例 `test_sub50_weak_home_tiebreaker_adds_no_bet_without_overriding_final` 验证通过。函数输入与输出对比表明，`final_direction`、`predicted_score`、`confidence` 等核心状态保持绝对静止。

## 6. 已知局限
当前拦截机制依赖前端（Dashboard）的配合。如果前端只读取主预测，新加字段的价值将大打折扣。因此需要在后续 UI 升级中明确展示此 Warning。

## 7. 后续要求
必须使用 Forward Ledger 进行验证，观察明天及之后的比赛中，打上此标签的场次其实际 X2 打出率是否显著高于同评级的常规主胜场次。
