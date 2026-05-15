# Market Sentinel 代码集成方案

未来 Market Sentinel 如何无缝、安全地接入 vMAX：

## 1. 原则限制
- **仅做旁路情报提供**，不改写最终赛果推断。
- 最终预测文件的核心主字段 (`final_direction`, `predicted_score`, `confidence`) 保持独立生成和验证逻辑，不受 Market Sentinel 强行接管。

## 2. 预测主流程读取输出
它将通过生成独立 JSON 的形式投喂给 `predict.py`，其中包含：
- `market_divergence_flags`
- `late_steam_flags`
- `exchange_volume_flags`
- `market_microstructure_summary`
- `no_bet_reason`（如果产生极限异动）

## 3. 拦截生效点
- 如发生诸如 `steam_critical` 与 `divergence_critical` 双重共振的情况，`predict.py` 的后处理层将读取 `no_bet_reason`，向前端挂载 `NO_BET` 及 risk flags。
- AI 的 Prompts 也可以挂载这段 summary，促使其审视是否存在庄家深度诱盘。
