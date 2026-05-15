# Forward Ledger 赛前防泄漏与账本评分系统

## 1. 为什么 Forward Ledger 比传统回测更重要？
在竞彩足球和赔率预测中，传统的回溯测试（Backtesting）极易发生“幸存者偏差”或“过度拟合”。人们常常在看到冷门赛果后，回滚修改规则（例如把某场比赛的 3-0 强行降级为 1-1）来提高历史胜率。
Forward Ledger（前向账本）强制规定：只有“赛前生成的预测特征与赛果”才有资格接受检验。这迫使 AI 在面对未知比赛时进行真实的盲测，积累出毫无水分的期望收益率曲线。

## 2. 避免数据泄漏与赛前锁定
- **sha256 锁定**：`create_ledger_from_prediction` 会读取刚跑完的预测 JSON 文件，计算全文 `sha256` 并落盘至 `ledger.jsonl` 中，永远封存它出生那一刻的切片。
- **只追加，不修改**：赛后输入真实比分时，`score_ledger_with_actuals` 读取已锁定的 `jsonl` 并在内存打分，随后输出包含计分统计的全新 `csv` 账本，绝不允许反写修改赛前 JSON 预测。

## 3. 核心风控指标的长期追踪
此套 Ledger 系统能够无情地衡量几个核心防御逻辑在长周期的表现：
- **Matrix 是否优于 Final**：通过比对 `matrix_top_scores_covered` 和 `exact_score_hit`。
- **Risk Candidates 是否兜底**：通过追踪 `risk_candidate_covered`。
- **Deep Favorite Score Moderation 是在帮忙还是在帮倒忙**：
  - 如果降档（如 3-0 降为 2-0），且实际赛果确为 2-0（或距离更近），则记为 `score_moderation_helped = True`。
  - 如果降档后，赛果确为 3-0 或大胜，则记为 `score_moderation_hurt = True`，从而避免过度防冷。

## 4. 与主预测引擎零耦合
本模块仅是一个只读（Read-only）评分与锁档外挂。完全遵守了不修改 `scripts/predict.py` 和不篡改预测内核的硬性纪律。
