# P0 修复前后对比报告：伪 Steam 信号去污

- **日期**：2026-06-16
- **分支**：`fix/p0-steam-fake-signal-purge-20260616`
- **基线**：`origin/main` @ `ac3e566`
- **备份 tag**：`backup/pre-p0-steam-fix-20260616`（远端核验指向 ac3e566）
- **方案**：B（移除 SteamMoveDetector 主路注入，方向码事实收口至诚实单一事实源）

---

## 一、根因（已 .venv 亲手实证）

`scripts/quant_edge.py:23,33` 的 `SteamMoveDetector.detect` 把竞彩抓包 `change` 字段当作**浮点 odds 增量**处理：
```python
wc = float(change.get("win", 0))
if wc < -0.08:  # 把方向码 -1 当成 -1.00 暴跌
    ... strength = min(10, int(abs(wc) * 80))  # → 满格 10
```

**取证**：扫描全库 967 个真实 `change` 字典，distinct 值**只有 `-1 / 0 / 1` 三个方向码**，浮点增量 **0 个**。
`_movement_label`（predict.py:3170）正确地把 `-1` 解读为方向码"赔率下降"，但 SteamMoveDetector 走的是另一套单位，导致语义错配。

**后果**：`build_evidence_packet`（predict.py:3755→3787）把伪信号作为 `steam_movement_signals` 注入 AI 终审；Gemini 终审 prompt 第3条要求"必须读取聪明钱蒸汽作为客观事实"。几乎每一场只要 `change.win=-1`（赔率方向下行），就被喷一个 **strength=10 满格伪 Steam/sharp 信号**，污染读盘，违反"不造假"军规。

---

## 二、修复前后对比（真实样本：吉达国民 vs 穆拜征服，change.win=-1，散户主热74%）

| 字段 | 修复前（基线 ac3e566） | 修复后（本分支） |
|---|---|---|
| `steam_movement_signals` | `{"steam":true, "signal":"🚨 主胜反向Steam! 降水-1.00+热度74%=庄家造热", "direction":"upset_away", "strength":10}` | `None` |
| `compiler` | `v20.7.0_poisson_purged_pure_board_reading` | `v20.7.1_poisson_purged_steam_fakesig_purged_pure_board_reading` |
| 诚实 sharp 链 `reverse_line_movement` | （未受影响） | `{"detected":null, "status":"requires_open_current_odds", "note":"…没有开盘/即时赔率序列时不能硬判真实 RLM。"}` — 仍在 |

**结论**：伪满格 steam 信号被彻底消除；方向码事实由 `compile_sharp_money_facts(_movement_label)` 这条已存在的诚实链单一承担，零信息丢失、零假数据。

---

## 三、改动内容（diff +7 / -4，单文件 scripts/predict.py）

1. 移除 `steam_res = quant_edge.SteamMoveDetector().detect(...)` 调用（替换为去污注释，留痕）。
2. `steam_movement_signals` 注入改为 `None`（伪旁路移除）。
3. `compiler` 版本号升级标记去污。

`SteamMoveDetector` 类本身保留在 quant_edge.py（不删，避免牵连其他可能引用），仅切断主路注入。

---

## 四、验证

- `py_compile scripts/predict.py` → OK
- 无 `steam_res` 残留引用 → 确认
- **`.venv` 全回归：114 passed**（与基线一致，零破坏）
- 真实样本 build_evidence_packet 对比 → 伪信号消除、诚实链保留

---

## 五、军规合规

- ✅ 真实路径 `/root/.openclaw/workspace/repos/88/` exec 实跑
- ✅ 独立新分支，零碰 main
- ✅ 改前建远端备份 tag 并核验 SHA
- ✅ .venv 全回归 + 修复前后真实数据对比
- ⏸ **push 前停下，待用户单独批准**（本报告即批准依据）

> 报备：对比基线时使用了临时 `git stash -u` + 只读 worktree，已核验本分支文件完好（v20.7.1 标记在）、工作区 tracked 改动仅 predict.py、stash pop 正确恢复无产物污染。此操作仅为只读对比，未触及任何产物提交。
