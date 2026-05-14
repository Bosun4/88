# 市场哨兵 (Market Sentinel) 数据源审计报告

## 1. Betfair Exchange API
- **用途**：作为监控市场的核心引擎，提供真实的市场微结构数据。
- **主要字段**：`market id`, `runner id`, `back/lay prices`, `traded volume`, `available sizes`, `market status`, `stream updates`。
- **价值**：能够透视大户“虚假挂单”、“诱盘撤单”及“买卖档位压制”，捕捉纯正的 Sharp Money 和真实资金意图。

## 2. Betfair Historical Data
- **用途**：在离线环境下使用真实市场重放数据（Tick级或分钟级快照）。
- **包含内容**：历史盘口微结构、traded volume replay、BSP (Betfair Starting Price) / settlement。
- **价值**：用于回测及验证“欧亚背离”和“Late Steam”等风控算法的有效性，评估其止损避险能力，无需实盘消耗资金。

## 3. The Odds API
- **用途**：聚合几十家传统与主流 Bookmaker 赔率数据。
- **包含内容**：bookmaker 1X2, spreads / Asian-like handicap proxy (亚指折算概率), totals (大小球)。多公司横向比较与差价分析。
- **价值**：便于快速构建欧赔公平概率与亚盘净胜期望概率，进而计算“欧亚背离度”。

## 4. Pinnacle API 限制
- **状态**：公众访问受限，且门槛极高，暂不将其作为开源预测系统（vMAX）的默认依赖。
- **规划**：如果在未来获得可用授权或商业通道，将采用独立的 Adapter 模式接入作为高级验证模块。

## 5. 当前项目缺口
- 缺乏实时的 **exchange order book**。
- 没有提取及比对 **closing line**。
- 没有在赛前 30 分钟内高频抓取 **odds snapshot**。
- 尚未实现跨平台的 **event id**（sportsbook vs exchange）对齐工具。
