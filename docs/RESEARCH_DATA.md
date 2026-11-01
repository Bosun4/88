# 赛前证据与历史导入

索引位于忽略提交的 `data/research/history.sqlite`。原预测、快照、前向账本不作为可修改的数据表。预测前读取全部 history/snapshots，按文件内容哈希断点续传，并保存本轮独立报价。Actions 缓存保留索引；丢失缓存可由归档重建，未归档观察记录可能丢失。

## 导入（不会调用 GPT）

```sh
python -m scripts.history_import --archive data --report work/coverage.json
python -m scripts.history_import --archive data --start 2026-09-01 --end 2026-10-31 --report work/range.json
python -m scripts.history_import --input work/sourced-matches.json --report work/import.json
python -m scripts.history_import --public --start 2026-10-01 --end 2026-10-13 --report work/public.json
```

`--db` 可指定独立研究库。日期范围筛选归档文件业务日/公开查询日；`--input` 导入整个文件。文件日期不代替实际开赛时间。公开源按天检查点续传，失败退出码2并报告缺口，不标记完成、不反复请求被拒绝的来源。公开适配器仅通过离线字段夹具验证；本次真实访问仍被拒绝，不能宣称取得完整历史或已验证最新接口格式。

输入接受普通列表或 `{"matches":[...]}`，单条格式示例（不是历史观测）：

```json
{
  "match_id": "sporttery:来源赛事ID",
  "source": "sporttery",
  "source_url": "https://www.sporttery.cn/实际来源路径",
  "league": "来源联赛名", "season": 2026,
  "home_team": "主队名", "away_team": "客队名",
  "kickoff_at": "2026-10-01T20:00:00+08:00",
  "captured_at": "2026-10-01T12:00:00+08:00",
  "quoted_at": "2026-10-01T11:50:00+08:00",
  "quote_kind": "observation", "give_ball": -1,
  "market_quotes": {
    "result": {"home": 2.0, "draw": 3.4, "away": 3.2},
    "handicap": {"home_cover": 3.0, "draw": 3.5, "away_cover": 2.0},
    "correct_score": {"2-1": 8.0, "胜其他": 40.0},
    "total_goals": {"2": 3.5, "7+": 30.0},
    "half_full_time": {"ss": 4.0, "sf": 15.0}
  }
}
```

缺项省略，不能猜测来源/时间。半全场两字符依次代表半场/全场，s=胜、p=平、f=负。28个精确比分加“胜其他/平其他/负其他”才是完整比分市场。让球必须为整数，未知不转换为0。

另行导入同一赛事的赛后结果可加：

```json
"actual": {
  "score": "2-1", "period": "90min", "source": "sporttery",
  "source_url": "https://www.sporttery.cn/实际赛果路径",
  "published_at": "2026-10-01T22:15:00+08:00"
}
```

必须是真实来源的90分钟赛果；若仅能确认采集时已可见，可保守使用赛果采集时间，不倒填为赛前。API-Football 的 AET/PEN 只读 fulltime，不用含加时的 goals。事件未取得不填“无红牌”，红牌仅取常规时间事件。

## 身份、时间及检索

报价必备稳定赛事身份、带时区开赛/采集时间、来源网址；报价时间晚于采集时间则拒绝。内容哈希去重；赛事以供应商ID和实际开赛日绑定。跨来源要求时间、联赛、赛季、球队身份严格相容；同名异场不合并，冲突赛果不参与比较。

只有 source=sporttery、sporttery.cn 来源、明确 `quote_kind=official_opening` 及来源报价时间同时具备才标初盘。导入者必须具备明确初盘标记的来源依据，URL本身不能证明内容真实性。首次观察不是初盘，不同来源/让球值分别比较；至少两个不同的已知报价时间才标为可核验序列。

`collection_mode=prematch/retrospective` 按采集是否早于开赛区分。补录可用于现在的历史研究，但不能混入过去截止时点的真实赛前评估。查询排除采集晚于分析截止的证据；当前推理还排除已开赛赛事。不从预测文字推断赛果。

历史比较按同联赛、相同让球及1X2报价对数比例平均距离≤0.25筛选，优先同赛季及相近总进球最低报价档；这是检索距离，未按预测收益优化。展示统一样本数、时间范围、低赔方向兑现/未兑现数量及两类案例，不称作模型命中率。初盘比较仅用明确初盘。前向账本成绩与赛后补录研究分开统计。

## 本次覆盖

2026-11-01 整理118份历史文件、96份快照，共214份，文件日期2026-05-06至2026-10-13。有效报价覆盖150场、827条，实际赛事跨度2026-09-19至2026-10-13。独立可核验赛果0场，明确初盘0条，当前可用历史比较案例0。缺来源/身份/时间的早期记录保留原文，但不参与研究。

逐文件报告见 [history-coverage-v25.json](history-coverage-v25.json)。`complete_jingcai_history=false`；并非体彩竞彩完整历史库。
