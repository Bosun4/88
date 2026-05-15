# Market Sentinel 研究与 PR 路线计划

本文档总结了构建 vMAX Market Sentinel (市场微观结构嗅探器) 的分阶段 PR 提交路线：

## PR A: schema + offline detector (当前完成阶段)
只做 schema + offline detector，定义 `market_snapshot_schema`，实装核心计算公式（欧亚背离计算器、临场变盘速率探测器），跑通本地的 mock 环境，为后续接数据夯实底座。此阶段坚守红线，不调用任何外部网络，不修改主预测 `predict.py`，不篡改最终裁定方向与信心。

## PR B: The Odds API adapter
再接 The Odds API，编写 SDK/请求模块，抓取各 bookmaker 的 odds、spreads、totals，并将实时数据清洗并转换为 MarketSnapshot。这能够让我们验证跨书商之间对立盘口的方差以及计算实时欧洲主流庄家倾向。

## PR C: Betfair adapter
再接 Betfair Exchange，深入获取 Order Book 和成交资金，计算 `traded_volume` 和真实的挂单 `lay_size` / `back_size`，解决 Late Steam 判定模型中至关重要的 `sharp_confirmed` 数据缺口。

## Pinnacle 暂不作为默认依赖
公众 API 访问门槛较高，暂不将其作为 vMAX 的原生抓取依赖。

## PR D: late steam scheduler
在运行机器上部署监控层引擎，支持 T-30, T-15, T-5 的自动抓取并产生变动 Velocity，输出报警至文件。

## PR E: dashboard display
前台增加模块解读报警标志，将预警（Critical/Warning）透传到卡片页，给操作者阻断下单的指示。

## PR F: forward ledger 验证
禁止过度拟合（回溯找冷门造规则）。要求在未来几周部署在线但隐形（影子模式）运行，复盘验证是否真的抓住了资金的陷阱，避免“假资金诱盘”。
