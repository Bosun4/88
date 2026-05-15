# The Odds API Adapter

## 1. Adapter 做了什么
将 `The Odds API` 的赛事与盘口数据接入 `Market Sentinel`，实时计算公平概率与欧亚背离度，清洗格式并输出标准化的 `MarketSnapshot`，为离线的异动嗅探器提供数据弹药。

## 2. 支持的 Markets
- `h2h`: 提供欧赔 1X2，计算主/平/客的 `implied_prob_fair`。
- `spreads`: 提供亚指（Asian Handicap）折算，抓取 `line` 和 `odds`。
- `totals`: 提供大小球盘口。

## 3. 设置 API Key
程序不会在代码或日志中打印 Key。运行时会自动从环境变量中读取：
```bash
export ODDS_API_KEY="your_api_key_here"
# Also supports ODDS_API_BASE for custom endpoints, and THE_ODDS_API_KEY / OODS_API_KEY as fallbacks.
```

## 4. 运行 Mock 模式
在无 Key 状态下，支持完全断网脱机运行 `--mock` 模式：
```bash
python -m market_sentinel.adapters.the_odds_api --mock --out reports/market_sentinel/snapshots/the_odds_api_mock.jsonl
```

## 5. 为什么不直接接入 predict.py？
遵守架构底线，保持系统的低耦合与绝对安全。Adapter 的职责是纯粹的“倾听市场”并“抛出报警标志 (`alerts`)”，主推演核心 `predict.py` 目前仍将以沙盒环境运行，防范外部网络 I/O 的阻断风险。未来可以通过读取落地生成的 `alerts.json` 间接获得风险情报。

## 6. 下一步：接入真实赛事调度
在测试通过后，我们将通过 cron 或 GitHub Actions 构建赛前 T-60 和 T-30 的自动化执行 Pipeline。针对目标比赛（如 `soccer_epl`, `soccer_spain_la_liga` 等）发起请求，实时保存异动警报。

## 7. 当前已知限制
- 不同 Bookmaker 之间的赔率开售情况不一致（部分可能没有亚洲盘 `spreads`）。
- API 获取的队伍名称与竞彩/vMAX 中文队伍名称映射暂未完全打通（Team Name Mapping / ID mismatches），后续需在主系统中引入名称模糊对齐库。
