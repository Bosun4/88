# 临场异动指标 (Late Steam Detector) 设计

## 1. 监控窗口
主要围绕临场关键信息期（如首发公布、大户资金涌入）：
- **监控区间**：T-60 到 T-0（开球前 60 分钟内）。
- **采集频率**：默认每 2 分钟记录一次 snapshot。
- **重点时间窗**：T-30 / T-15 / T-5。

## 2. Velocity 速率计算
衡量盘口变化剧烈程度：
```text
odds_velocity = log(odds_t / odds_t_minus_delta) / minutes
prob_velocity = fair_prob_t - fair_prob_t_minus_delta
volume_velocity = traded_volume_t - traded_volume_t_minus_delta
```

## 3. Alert 规则与标签
基于 5 分钟滑动窗口下的百分点（pp）变幅：
- `5 分钟 fair probability 变化 >= 4pp`：**steam_watch**
- `5 分钟 fair probability 变化 >= 7pp`：**steam_warning**
- `5 分钟 fair probability 变化 >= 10pp`：**steam_critical**
- 若伴随 `traded_volume` 同步猛增，追加标签：**sharp_confirmed**
- 只有价格骤变而无明显成交量配合，可能被标记为：伪异动/假动作（清理散户筹码）。

## 4. 输出字段集合
- `late_steam_flags`
- `steam_direction`
- `steam_strength`
- `steam_window`
- `steam_evidence`
