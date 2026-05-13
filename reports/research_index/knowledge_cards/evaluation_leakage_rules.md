# 知识卡片：Evaluation Leakage Rules

## 核心摘要

S7 指向 AI 预测流程的典型风险：prompt 泄漏、用赛后信息改赛前样本、反复调 prompt 直到历史命中、用 sandbox 或 mock 输出冒充 blind result。S1/S2 也间接提醒：若评价只看命中率而不看校准和市场价格，系统会被错误优化。

forward-only evaluation 的目标是把预测、证据、prompt、模型版本和输出在赛前锁定，赛后只追加结果，不允许修改赛前记录。任何赛后解释都必须进入 `post_match_notes`，不能覆盖原始 prediction。

## 对 vMAX 的启发

- 预测生成后立即写入 blind ledger，并对 input packet、prompt、output JSON、code version 计算 SHA256。
- 赛果后置：赛前 ledger 不含 final score、closing post-result notes 或赛后新闻。
- 禁止倒推 prompt、禁止用沙盒造假、禁止删除失败样本、禁止只保留 top winners。
- 评价要分层：direction、score、probability calibration、risk routing、market edge。
- 所有 PR 的效果都必须通过 forward ledger 验证；回看报告只能用于假设生成。

## 相关来源

- S7: AI prediction leakage and forward-only evaluation hygiene.
- S1: accuracy is not profitability.
- S2: market-efficiency probability comparison.
