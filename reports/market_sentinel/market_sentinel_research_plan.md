# Market Sentinel 研究与 PR 路线计划

本文档总结了构建 vMAX Market Sentinel (市场微观结构嗅探器) 的分阶段 PR 提交路线：

## PR A: schema + offline detector (当前阶段)
定义了 `market_snapshot_schema`，实装核心计算公式（欧亚背离计算器、临场变盘速率探测器），跑通本地的 mock 环境，为后续接数据夯实底座。此阶段不调用外部网络。

## PR B: The Odds API adapter
接入 The Odds API 的 SDK/请求模块，解决将欧赔、亚指折算后统一塞进 `market_snapshot_schema` 的数据归一化问题。

## PR C: Betfair adapter
攻克最核心的微结构获取，解析 Order Book、Volume 和 BSP，产出 `late_steam_flags` 和 `sharp_confirmed`。

## 暂不接入：Pinnacle
公众访问受限，暂不作为开源默认依赖。

## PR D: late steam scheduler
在运行机器或 Github Actions 层面，编写在开球前一小时激活、并能高频（T-30, T-15, T-5）执行抓拍的比对引擎调度脚本。

## PR E: dashboard display
在前端 `risk_dashboard` 增加专门的 `Market Sentinel` 异动监视板块。

## PR F: forward ledger 验证
禁止用单场过去的冷门倒推规则，必须将算法上线后，用未赛的数据做 Forward Validation。
