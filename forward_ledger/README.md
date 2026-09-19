# 前向锁档与复盘接口（审计01/02/03/08/13）

发布后集成（主程序已自动接入，路径与CI保持一致）：

```python
from forward_ledger.ledger import create_ledger_from_prediction
count = create_ledger_from_prediction(
    snapshot_path, "data/forward_ledger.jsonl", on_conflict="keep"
)
```

- 支持 `matches.today[].prediction`，也支持历史 `predictions` 数组。
- 返回本次新增数。同一个已锁赛事重复运行不追加；`keep` 保留首次事件版本，默认 `error` 对内容变化报错。不同赛事继续追加。
- 开赛时间要求带时区 `kickoff_at` 或 `kickoff_at_utc`；时间判断使用真实锁档时刻，不把历史 `created_at` 当补锁时间。
- 缺 kickoff 仍写记录：`is_abstain=true`, `strict_forward=false`, `lock_status=unverifiable_missing_kickoff`。已开赛为 `after_kickoff`。有比赛但无法识别结构/空数组会显式报错。
- 每条保留原始完整 `prediction_snapshot`、源文件SHA256、行内容SHA256、前条哈希与当前条哈希。追加与结算前都核验链；不修改已存在行。旁路 `.lock` 文件防并发写。哈希能检测意外修改，并非防管理员整体重写的外部公证；崩溃残留锁需人工检查后清除。

共享 `scripts.fixture_identity`：

- `prediction_rows(payload) -> list[dict]` 规范当前/旧产物结构。
- `fixture_key(row) -> str | None` 使用来源命名空间内 `match_id/fixture_id + UTC开赛日期`；无稳定ID时使用联赛、赛季、双方实体与事件日期。有 kickoff 时统一 UTC 日期，无 kickoff 才用显式 `event_date/fixture_date`；永远不使用业务 `date` 或列表顺序 `id/match_num`。
- `dedupe_predictions(rows)` 保留首次记录；不可识别的记录保留供弃权追溯，不能统计成独立比赛。
- `match_actual(row, actuals, market="1x2") -> dict | None` 检查日期、双方方向、双方提供的赛季/联赛、完整 kickoff；缺失或冲突不结算。支持双方之一缺赛季、但日期联赛与双方身份完整一致；不猜测球队别名。`api_football_fixture_id` 明确对接 API-Football 命名空间，`match_id=wencai:123` 不会与其他 provider 的裸ID强行比较。已确证同源事件ID时忽略中文/英文显示标签差异，仍检查双方实体ID与实际开赛时刻。
- `provider_ids(row)` 返回命名空间到事件ID的映射，API规范结果有 `source=api_football`。`auto_actuals` 保留输入 `source`，CSV出处独立放入 `result_source`；CSV匹配日期按联赛当地时区生成 `result_event_date`，不会改写事件身份日期。
- `api_actual(fixture)` 将 API-Football 常规90分钟 `score.fulltime` 规范化；AET/PEN 不能回退包含加时的 goals。

结算CSV必须包含 `fixture_id` 或 `match_id` 加 `event_date/kickoff_at`，或者同日期联赛双方完整身份，并提供 `actual_score`。仅 `match_id,actual_score` 的旧CSV不再自动结算。

`self_learn.self_learn(pred_file=None, diary_file=None, actuals_fetcher=None)` 返回并保存日记；`verify` 复用同一逻辑。日记页面可使用：

```text
ledger.samples                     已结算分析数
ledger.direction_samples           有方向预测的分母
ledger.direction_accuracy_pct      方向命中率，无样本为null
ledger.main_score                  {samples,hits,accuracy_pct}
ledger.side_risk_score             {samples,hits,accuracy_pct}
ledger.risk_d                     {samples,hits,accuracy_pct,main_score,side_risk_score}
ledger.bettable                   {staked,candidates,unpriced,wins,win_rate,pnl,roi_pct}
ledger.strict_forward             同ROI账本结构，仅有效锁档与赛前报价
unresolved                        未结算记录与原因
```

主比分读取 `final_ai_score/predicted_score`；副风险比分只读取 `risk_score_candidates` 中字符串或 `{score: ...}`，同场候选去重，以每个有候选的比赛计一次覆盖率分母，不把正文任意比分、Top3、主比分命中合并。D级保留方向与比分分析，不作为有效推荐；缺预测不入对应分母。

收益是单位本金的**存储报价模拟**，不是宣称真实成交。选择匹配的报价依次来自 recommendation.odds/bet_odds、外层 sp_home/sp_draw/sp_away、嵌套 odds 方向；无赔率不能补2.0，也不把缺报价的输单单独计-1。严格前向还要求 `quoted_at/odds_captured_at/captured_at <= locked_at_utc < kickoff_at`。回顾入口不自行声称 strict_forward。

`baseline_259.collect(return_audit=True)` 返回 rows、unresolved、input_versions、deduplicated_records、strict_forward_samples。历史只有队名、无事件日期的 actuals 不可自动修复；基准默认标记 retrospective_unlocked。需要补齐独立核验的身份和事件日期后才可重新结算。
