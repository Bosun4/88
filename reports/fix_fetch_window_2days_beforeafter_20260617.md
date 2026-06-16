# 抓取窗口收敛修复前后对比报告：只抓今天+明天两天

- **日期**：2026-06-17
- **分支**：`fix/fetch-window-2days-20260617`
- **基线**：`origin/main` @ `d4591c2`
- **备份 tag**：`backup/pre-fetch-window-20260617`（指向 d4591c2）
- **范围**：仅抓取窗口过滤（`scripts/fetch_data.py`），零碰 AI/预测逻辑、零 prompt 改动

---

## 一、根因（.venv 实跑问财接口亲验）

`scrape_wencai_jczq_async` 用 `?date=X` 请求问财接口，但**接口一次性返回横跨多日的全部赛程**，`date` 参数基本不约束范围。

实测 2026-06-17 凌晨查询（业务日 2026-06-16）：

- 接口返回 **12 场**，业务日分布：`{6/16: 3, 6/17: 5, 6/18: 4}`（跨周二/三/四）。
- 这 12 场**全部**进入 `enrich_match_data`（每场打 API-Football 多个端点）+ 后续 AI 终审。
- 后果：① 把昨天/更远的比赛也拉进来重复抓；② 每天重复抓未来几天同一批比赛；③ enrich + AI 双重 token 浪费。

每场带可靠日期字段：`stime`（开赛 Unix 秒）、`week`/`week_num`（周几）。

---

## 二、修复方案

在 `scrape_wencai_jczq_async` 解析出 `football_list` 后、进入 enrich 前，新增窗口过滤：

- `_business_day_from_stime(stime)`：由开赛 Unix 秒按**竞彩业务日口径**（复用 main.py 同款 `VMAX_DATE_SHIFT_HOURS=11` 偏移）算业务日。
- `filter_matches_by_window(...)`：只保留业务日 ∈ `[today, today+days_ahead]` 的场次。
- env 旋钮 `VMAX_FETCH_DAYS_AHEAD`，**默认 1**（今天+明天两天）。设 0 = 只当天；设 2/3 = 多留几天。
- **Fail-safe**：`stime` 缺失/为 0/解析失败 → **保留**该场（宁可多跑也不误杀真实比赛）。

为什么用业务日而非自然日：凌晨 00:00–10:59 开赛的深夜场，自然日已是"明天"但竞彩业务上仍属"今天"。业务日口径与 main.py 的 `get_target_date` 完全一致，避免误杀今晚深夜场。

---

## 三、修复前后对比（真实接口数据）

| 项 | 修复前（基线 d4591c2） | 修复后（本分支） |
|---|---|---|
| 接口返回 | 12 场（6/16:3, 6/17:5, 6/18:4） | 12 场（不变，源头无法控制） |
| 进 enrich + AI | **12 场全进** | **8 场**（6/16:3 + 6/17:5） |
| 砍掉 | 0 | 4 场（6/18 窗口外） |
| API-Football 调用 | 12 场 × N 端点 | 8 场 × N 端点（↓33%） |
| AI 终审场次 | 12 | 8 |

> 注：测试时为凌晨，业务日=6/16，窗口=6/16+6/17。白天运行时业务日=6/17，窗口=6/17+6/18，效果一致。

---

## 四、改动内容（单文件 `scripts/fetch_data.py`）

- 新增 `_env_int` / `_business_day_from_stime` / `filter_matches_by_window`（纯函数，可测）。
- `scrape_wencai_jczq_async` 在 `football_list` 非空后插入一行窗口过滤 + 空结果短路。
- 零改动 enrich/AI/预测/prompt。

---

## 五、验证

- `py_compile` → OK
- **TDD**：6 个测试（`tests/test_fetch_window_2days.py`）覆盖：今明两天保留/砍昨天与更远、days_ahead=0 只当天、days_ahead=2 三天、stime 缺失/非法 fail-safe 保留、深夜场业务日不按自然日误杀、env 默认 1。
- **真实接口端到端**：12 场 → 8 场，业务日分布精确。
- **.venv 全回归 126 passed**（基线 120 + 新 6，零破坏）。

---

## 六、军规合规

- ✅ 真实路径 exec 实跑（接口亲验 + 全回归）
- ✅ 独立干净新分支、从最新 origin/main 起、零碰 main
- ✅ 改前建远端备份 tag 并核验 SHA
- ✅ TDD + 全回归 + 真实数据前后对比
- ✅ 纯过滤、零碰预测逻辑、零 prompt 改动、可 env 调节、fail-safe 不误杀
- ⏸ **push 前停下，待用户决定**
