# Deep Favorite Score Moderation

## 概述
这是一个通用的风控防护补丁。核心目的是拦截和降级在极端深盘或超低胜赔下的“盲目3-0过冲”。

由于大众资金倾向在豪门碾压局中博取大胜比分（特别是弱队已降级或无战意时），预测模型容易捕捉到市场表层的“3-0穿盘共识”。但由于轮换、战意衰减或豪门的“打卡下班”心态，这些比赛经常以 2-0 这种不穿盘的“赢球输盘”告终。

## 适用范围与限制
1. **这是比分降档保护，不是方向修正**。绝不改变主队胜或客队胜的大方向（`final_direction`）。
2. **不适用于所有 3-0**。如果矩阵数据、进球模态强烈支持 4+ 进球的大开大合局面，将保持原样。
3. **严格触发条件**：
   - 强方胜率 >= 58% 且 (平+弱) <= 42%。
   - `total_goals` 的 4+ 进球不具备压倒性优势（即小于 2 或 3 的分布）。
   - 2-0 与 3-0 在比分候选池中的概率极近（`<= 5%` 差距）。
4. **不硬编码球队或赛果**。所有判断均基于概率和盘口拓扑，彻底清除了类似“皇马”、“奥维耶多”或特定日期的硬编码。

## 输出
触发降档时，将在输出的预测 JSON 中添加：
```json
{
  "score_moderation_applied": true,
  "original_predicted_score": "3-0",
  "predicted_score": "2-0",
  "score_cluster": ["2-0", "3-0", "3-1"],
  "decision_quality_flags": ["DEEP_FAVORITE_SCORE_MODERATION"],
  "score_moderation_reason": "Downgraded from blowout due to candidate proximity and goal band limits."
}
```

## 后续建议
请在接下来的联赛尾声轮次中开启 Forward Ledger 验证，以观察 `DEEP_FAVORITE_SCORE_MODERATION` 是否准确避开了赢球输盘陷阱，或是错误地阻挡了真实大胜。
