# 欧亚背离指标 (Euro-Asian Divergence Index) 设计

## 1. 1X2 Fair Probability (欧赔公平概率)
根据剥水算法推导主/平/客真实概率：
- 支持 `multiplicative`, `power`, `additive`, 或带有资金倾向修正的 `shin_approx` (不称真 Shin)。

## 2. Asian Handicap Expected Probability Proxy
利用亚洲让球盘的配置反推比赛强弱预期：
- 使用让球线段 `line` 与 `handicap odds`，计算主队覆盖盘口的预估概率 `expected_home_strength_from_handicap`。
- 若无亚洲盘数据支持，系统将使用 degraded mode 仅依赖欧赔动态。

## 3. Divergence Index 公式
```text
DI_home = fair_home_prob_from_1x2 - expected_home_strength_from_handicap
DI_abs = abs(DI_home)
```

## 4. Alert 规则阈值
- `DI_abs >= 0.05`：**watch**
- `DI_abs >= 0.08`：**warning**
- `DI_abs >= 0.12`：**critical**

## 5. 市场方向解释
- **欧赔强、亚盘不升**：欧赔造热 / 盘口阻力。庄家利用 1X2 引诱资金去所谓的热门方，而在亚盘封锁更深让幅。
- **亚盘升、欧赔不动**：盘口提前反应 / sharp handicap move。内幕资金或职业玩家更早在让球盘布局。
- **欧赔降、交易所 lay 增大**：可能是假热。表面上看大流公司赔率下降，但交易所挂出的不看好大单激增。
