# Forward-Only Evaluation Protocol for vMAX

## 1. 为什么传统回测易过拟合

传统回测容易把赛后已知事实、后来修好的 prompt、阈值微调和幸存样本混在一起。尤其是 AI 预测系统，研究者可能无意中把赛果、closing line、阵容最终名单或历史失败样本泄漏进 prompt。这样得到的“回测命中率”不是真实赛前能力，而是 retrospective fitting。

更危险的是“不验证”：如果系统只看漂亮的单场解释，不记录失败和概率校准，就会把叙事强度误当 edge。来源 S1/S2/S7 的共同结论是：准确率、叙事合理、模型一致都不足以证明有可下注价值；必须做 forward-only blind evaluation。

## 2. Forward-only blind ledger 定义

Forward-only blind ledger 是赛前冻结、赛后追加的不可回写记录。每场比赛在预测生成时写入一条赛前记录；赛前记录一经提交，不允许修改、覆盖、删除或补录赛后信息。赛后只允许追加 `settlement` 区块。

最小粒度：一场比赛一条 ledger entry。

## 3. SHA256 锁定要求

每条赛前记录必须保存以下 SHA256：

- `input_packet_sha256`: 原始赛前证据包。
- `prompt_sha256`: 发送给模型的完整 prompt。
- `model_output_sha256`: 原始模型返回 JSON/text。
- `normalized_prediction_sha256`: 解析后的标准预测对象。
- `code_version_sha256` 或 `git_commit`: 生成预测的代码版本。
- `ledger_entry_sha256`: canonical JSON 序列化后的整条赛前记录。

锁定时间：必须早于 kickoff，且记录 `created_at_utc`、`lock_deadline_utc`、`kickoff_utc`。

## 4. 赛果后置规则

赛前字段禁止包含：

- 实际赛果、红黄牌、赛后 xG、赛后 lineup confirmation。
- closing line 或开赛后价格，除非单独标记为 `pre_kickoff_snapshot` 且有时间戳。
- 赛后媒体/伤停复盘。

赛后只能追加：

- `actual_score`, `actual_direction`, `settled_at_utc`。
- `closing_line_snapshot`（如有，必须独立标记为赛后评价基准）。
- `post_match_notes`，不得覆盖赛前 prediction。

## 5. 每场记录字段要求

必填基础字段：

- `ledger_id`, `match_id`, `match_num`, `league`, `home_team`, `away_team`, `kickoff_utc`。
- `created_at_utc`, `locked_at_utc`, `source_snapshot_time_utc`。
- `model_route`: Gemini/GPT/Grok/其他模型及版本。
- `input_sources`: odds/news/lineups/injuries/team_stats/source IDs。

预测字段：

- `direction`: home/draw/away。
- `score`: predicted_score。
- `direction_probs`: home/draw/away probability, if available。
- `score_probs`: top-N score probabilities, if available。
- `confidence`: model confidence with note that it is not calibrated hit rate。
- `recommendation_tier`, `recommend_gate_pass`, `recommend_gate_reasons`。

风险字段：

- `risk_level`: low/medium/high, with explicit rule version。
- `risk_candidates`: array of alternative scores with risk_type and reason。
- `tail_risk_flags`: e.g. weak_home_favorite_btts_tail, high_goal_tail_compressed。
- `matrix_flags`: booleans for matrix_vs_final_direction_conflict, away_tail, draw_risk, high_goal_tail, low_confidence。
- `matrix_direction_probs`, `matrix_top_scores`, `matrix_goal_probs`。
- `data_quality_flags`: missing_web_validation, missing_sharp_money_data, stale_lineups, no_closing_line。

Market fields：

- `raw_odds_1x2`, `raw_implied_1x2`, `overround_1x2`。
- `fair_probs_1x2`, `fair_method`: multiplicative/additive/power/Shin。
- `correct_score_odds`, `total_goals_odds`, `handicap_line`, `odds_timestamp`。

Settlement fields（赛后追加）：

- `actual_score`, `actual_direction`, `direction_hit`, `score_exact_hit`。
- `brier_1x2`, `log_loss_1x2`, `ranked_score_loss`, `topN_score_hit`。
- `closing_line_value` if odds snapshots exist。

## 6. 指标

主指标：

- Brier Score: 评估 1X2 概率校准。
- Log Loss: 惩罚对错误结果过度自信。
- Calibration Curve / Reliability bins: 按 40-50、50-60、60-70、70+ 等置信分桶。
- ECE/MCE: 汇总校准误差。

辅助指标：

- Direction accuracy: 只作辅助，不代表盈利。
- Exact score hit rate and Top-N score hit rate。
- Risk routing precision/recall: HIGH_RISK/TAIL_RISK/MATRIX_CONFLICT 是否真能识别异常结果或低质量推荐。
- CLV: closing line value，若有赛前 odds snapshots。
- EV backtest: 仅在 fair probability、可下注价格、stake rule 全部锁定时启用。

## 7. 明确禁止

- 禁止倒推 prompt：不得根据赛果修改赛前 prompt 再覆盖 ledger。
- 禁止用 sandbox/mock 输出冒充 blind prediction。
- 禁止删除失败样本或只保留命中的 top4。
- 禁止赛后补填 lineup/news 到赛前 input packet。
- 禁止未锁 SHA256 的预测进入正式评估。
- 禁止把 `confidence` 展示为历史命中率或盈利概率。
- 禁止在 UI PR 中调 prediction 阈值并声称是展示优化。

## 8. 当前文件锁定示例

本次审计读取的 `reports/live_predictions/live_predictions_9999_github_actions.json` SHA256：

`d5e1489b0c282d95571b6fe93f63a841256b296fe7af85487d87ea26585d875b`

该 hash 仅用于本研究文档引用；没有重新运行预测，也没有修改 `scripts/predict.py`。
