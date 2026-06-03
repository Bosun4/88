# 死码清理 — 修复前后对比报告 (2026-06-04)

## 范围
延续 PR#20 未尽的"泊松时代死码"清理。**生产代码逻辑零改动**，仅删除可达性证实的死码与依赖噪声。

## 删除前基线
- scripts/*.py: 29 个文件
- predict.py: 5069 行
- requirements.txt: 10 个依赖
- 测试: 83 passed / 0 failed

## 改动清单

### 1. 删除生产真死码模块（967 行，主链路零 import + 无测试牵连 + 无 workflow 引用）
| 文件 | 行数 | 判定依据 |
|---|---|---|
| scripts/models.py | 705 | AST 可达性图谱不可达；全仓零 import；无 `__main__` |
| scripts/odds_engine.py | 119 | 同上 |
| scripts/odds_history.py | 143 | 同上 |

> 保留 `self_learn.py`（有 `__main__`，可独立运行，按保守原则不删）。
> 保留 `predict.py` 内 `_poisson_pmf_shadow`/`build_unified_score_matrix_shadow`——读盘范式下 AI 的静态对照基准（背离探测用），是活路径。

### 2. 删除幽灵依赖（生产+测试均零引用）
- requirements.txt: 移除 `beautifulsoup4` / `lxml` / `scikit-learn`
- main.py REQUIRED_PACKAGES: 同步移除同三项（双真值源对齐）

### 3. 删除 ENGINE_VERSION/ENGINE_ARCHITECTURE 僵尸赋值
- predict.py line58~63 的旧赋值（`20.2.1-FULL-ANCHOR`）被 line3461 覆盖，运行时从未生效。
- AST 核验：line58~3461 间 5 处使用全在函数体内（调用时 line3461 已赋值），删除无 NameError 风险。
- 以指针注释替代，保留版本常量注释头。

## 删除后验证
- scripts/*.py: 26 个文件（-3）
- 全量语法编译: 56/56 .py 通过 ✅
- predict.py import 冒烟: OK，`ENGINE_VERSION = vMAX 20.6.0-READING-PARADIGM`（生效值不变）✅
- 全量测试: **83 passed / 0 failed**（与基线一致，零回归）✅

## diff 汇总
`6 files changed, 2 insertions(+), 982 deletions(-)` — 净删 980 行。

## 回滚
- 备份 tag: `backup/pre-deadcode-cleanup-<TS>`（指向 06cbf76）
- 物理备份: `/tmp/88_deadcode_backup_<TS>/`
