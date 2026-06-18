# B 修复前后对比报告（世界杯专属 DNA + 接回 2.48 孤儿生态先验）

- 时间：2026-06-15
- 分支：`feat/wc-dna-reconnect-20260615`（off main `b9ab31d`，已含 A 的国际盘修复）
- 备份 tag：`backup/pre-wc-dna-20260615`
- 改动文件：`scripts/league_intel.py`、`scripts/predict.py`（+11 行，0 删既有逻辑）
- 状态：**仅本地分支，未 commit / 未 push / 未碰 main**
- 性质：纯 evidence/prior 注入（非裁判），符合「物理极简、读盘入 AI」军规

## 审计缺口（B 针对的两点）
1. **2.48 生态先验是孤儿死码**：`LEAGUE_PROFILES["world_cup"]=(2.48,47,28,...)` 只被 `build_league_intelligence` 消费，而 predict.py 从不调用它 → 世界杯场均偏低/中立场无主场优势的核心先验从未进 AI 提示。
2. **predict.py 无世界杯 DNA**：`_league_dna_profile("世界杯")` 因含"杯"落入通用 `cup_or_cross_context`（高方差杯赛档），无中立场/无主场优势/低分锦标赛先验。

## 改了什么（最小改动）
1. `league_intel.py:analyze_world_cup_context` 增 `[WC-ECO]` 行：从 `LEAGUE_PROFILES["world_cup"]` 实时取 2.48/O2.5/U2.5 注入提示，并显式声明「中立场无主场优势、不照搬联赛大球」。**孤儿先验接回主链。**
2. `predict.py:LEAGUE_DNA_PROFILES` 增 `"世界杯"` 专属 DNA（`draw_risk=high, btts=low, neutral_venue=True`）。因 `_league_dna_profile` 先遍历字典键、再走"杯"兜底，新增键**优先命中**，不再落通用档。

## 修复前 → 修复后

| 项 | 修复前 | 修复后 |
|---|---|---|
| `[WC-ECO]` 2.48 生态先验进提示 | ❌ 孤儿死码 | ✅ 注入（含中立场声明） |
| `_league_dna_profile("世界杯")` | `cup_or_cross_context`（通用） | `世界杯`（专属，`neutral_venue=True`）✅ |
| 中立场/无主场优势先验 | 缺失 | 双处显式声明 ✅ |

`analyze_world_cup_context` 实测输出（西班牙vs佛得角·首轮）：
```
[WC2026] 48队/12组/前2+8个最佳第三名晋级…
[WC-ECO] 世界杯生态先验(5届320场): 场均2.48球(偏低)、O2.5约47%、U2.5约28%、主场优势=0(中立场)。世界杯是中立场锦标赛足球,默认偏低分/防守反击,不得照搬高节奏联赛大球思维;无主场优势,主客仅代表名义顺位不代表地利。
[WC-R1] 首轮是最闷一轮…
[WC-FORM:主 西班牙|trust] …
```

## 验证
- 语法：两文件 `ast.parse` OK。
- 接回实证：`analyze_world_cup_context` 现输出 `[WC-ECO]` 含 2.48。
- DNA 实证：`_league_dna_profile("世界杯").key=="世界杯"` 且 `neutral_venue=True`（PASS，不再 generic）。
- 回归：本地 `.venv` `pytest -q` → **122 passed**（零回归）。

## 说明（诚实边界）
- 这是 evidence/prior 注入，不改 AI 终审方向/比分，仅给读盘补回中立场低分先验，消除"照搬联赛大球"的盘感污染风险。
- 实际对 AI 比分质量的提升须线上多场赛果回测核验——沙盒只能证"先验已进提示链"，不能证"读得更准"。远端 SHA / 线上赛果为唯一事实源。

## 下一步（待批）
P0/commit/push 需单独批准。批准后：仅 push 本分支 `feat/wc-dna-reconnect-20260615`（不碰 main），push 后核验远端 SHA。
