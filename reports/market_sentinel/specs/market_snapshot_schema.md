# Market Snapshot Schema 契约设计

该 Schema 统一定义并承载多端（Betfair、Odds API 或手工录入）盘口及资金快照：

```json
{
  "match_id": "string",
  "source": "betfair|odds_api|manual|other",
  "timestamp_utc": "string",
  "minutes_to_kickoff": 30,
  "market_type": "1x2|asian_handicap|totals|correct_score",
  "selection": "home|draw|away|over|under|score",
  "odds": 2.1,
  "implied_prob_raw": 0.476,
  "implied_prob_fair": 0.455,
  "line": -0.25,
  "back_price": null,
  "lay_price": null,
  "back_size": null,
  "lay_size": null,
  "traded_volume": null,
  "bookmaker": "string",
  "exchange_market_id": "string",
  "runner_id": "string"
}
```
