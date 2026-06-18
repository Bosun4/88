# P0 发布链路修复前后对比报告（JSON 原子写入 + 去 autostash + Browser JSON 硬闸）

- 时间：2026-06-16
- 分支：`fix/p0-publish-pipeline-json-guard-20260616`（off latest main `8c9e5d5`）
- 备份 tag：`backup/pre-p0-publish-pipeline-20260616`
- 改动文件：`.github/workflows/predict.yml`、`scripts/main.py`
- 状态：**仅本地分支，未 commit / 未 push / 未碰 main**

## 背景
上一轮线上事故根因是发布链路在产物生成后执行：

```bash
git pull --rebase --autostash origin main
```

远端 main 同时更新 `data/predictions.json` 时，`autostash pop` 产生冲突标记，损坏 JSON 被盲 `git add/commit/push` 进 main，导致前端解析失败。

## 改动 1：主程序写 JSON 改为校验后原子替换
`scripts/main.py`

- 新增 `_json_safe()`：把 `NaN/Infinity` 转为 `null`，避免 Python 非标准 JSON 污染前端。
- 新增 `write_json_atomic()`：写入 `path.tmp` → `json.load` 自校验 → `os.replace()` 原子替换。
- 删除启动时先覆盖写空 skeleton 的逻辑，避免中途崩溃留下“合法但空”的线上数据。
- 无比赛数据默认不覆盖上一份好数据；只有显式 `VMAX_ALLOW_EMPTY_PUBLISH=true` 才允许发布空 slate。

## 改动 2：CI 发布步骤去掉 autostash，改为 fail-safe
`.github/workflows/predict.yml`

- 删除 `git pull --rebase --autostash origin main`。
- 改为 `git fetch origin main` 后比较 `HEAD == origin/main`：如果 main 在预测运行期间前进，直接失败并要求重跑，**不再尝试合并生成产物**。
- commit 前用 Node 执行浏览器级 `JSON.parse` 校验 `data/predictions.json` 和最新 history 文件。
- commit 前对文件内容检查 `<<<<<<< / ======= / >>>>>>>` 冲突标记。

## 修复前 → 修复后

| 风险 | 修复前 | 修复后 |
|---|---|---|
| autostash pop 冲突污染 JSON | 可能被盲 commit | main 前进则失败重跑，不合并产物 ✅ |
| 冲突标记进入产物 | 只靠 staged diff 检查 | 文件内容硬闸 ✅ |
| `NaN/Infinity` 前端解析失败 | Python 默认允许写出 | 写入前转 `null` + `allow_nan=False` ✅ |
| 中途崩溃写空/半截 JSON | 直接覆盖目标文件 | tmp 校验后 `os.replace` ✅ |
| 抓不到数据覆盖好数据 | 会成功写空 slate | 默认 fail，需显式 opt-in ✅ |

## 验证
- `scripts/main.py` AST：OK
- `.github/workflows/predict.yml` YAML：OK
- 当前 `data/predictions.json`：Node `JSON.parse` OK
- 原子写入 smoke：`NaN/Infinity` → `null`，tmp 文件清理 OK
- 本地 `.venv` 回归：**122 passed**

## 诚实边界
- 这次修的是发布链路 P0，不改变预测算法。
- 如果 GitHub Actions 运行期间 main 发生变化，新的行为是 fail-safe：本次不更新产物，需重新触发；这是有意取舍，符合“宁可不更新也不推坏数据”。

## 下一步（待批）
批准后：commit 本分支并 push `fix/p0-publish-pipeline-json-guard-20260616`，push 后核验远端 SHA；不直接写 main。
