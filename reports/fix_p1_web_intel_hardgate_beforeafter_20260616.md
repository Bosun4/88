# P1 修复前后对比报告：联网情报链硬闸

- **日期**：2026-06-16
- **分支**：`fix/p1-web-intel-hardgate-20260616`
- **基线**：`origin/main` @ `ac3e566`
- **备份 tag**：`backup/pre-p1-web-intel-20260616`（远端核验指向 ac3e566）

---

## 一、根因（行号亲验 @ ac3e566）

`_apply_external_fact_source_gate`（predict.py:1796）当前只有**一条真硬闸**：`valid_sources==0 且依赖外部事实 → capped C + observe`。其余三项是 prompt 君子协定，gate 零代码强制：

1. **新鲜度无强制**：`_valid_external_source_count`（1732）只检查 `url + (title|claim)`，完全不看 `published_at`/`freshness`。三年前旧闻与当天新闻同等有效。
2. **来源冲突无处理**：`source_conflict_audit` 是 AI 自填字段（schema @ 3930），gate 里零校验。
3. **quality<50 仅 warning**：基线 `elif 0 < evidence_quality < 50: warnings.append(...)`（1837）只贴标签，tier/gate 不动。

---

## 二、修复前后对比（三场景，端到端经 adapt_ai_to_frontend）

| 场景 | 修复前（基线 ac3e566） | 修复后（本分支） |
|---|---|---|
| external_fact 全 stale（freshness=stale，3年前旧闻） | tier=**A** / gate_pass=**True** / **main** | tier=C / gate_pass=False / observe |
| source_conflict_audit.has_conflict=true（未解决冲突） | tier=**A** / gate_pass=**True** / **main** | tier=C / gate_pass=False / observe |
| evidence_quality_score=35 | tier=**A** / gate_pass=**True** / **main** | tier=C / gate_pass=False / observe |

三条漏洞全部从"君子协定"升级为"代码硬闸"。

---

## 三、改动内容（diff +79 / -6，单文件 scripts/predict.py）

在 `_apply_external_fact_source_gate` 内新增统一降级动作 `_cap_to_c_observe`（tier 封顶 C + 不推荐 + observe + gate 不过），并接入三条硬闸：
1. **硬闸1 新鲜度**：`_all_external_facts_stale` — external_fact_table 非空且**所有**条目 freshness∈{stale/expired/old}（谨慎便：任一 fresh 或 unknown 即不判 stale，避免误杀）。
2. **硬闸2 冲突**：`_has_unresolved_source_conflict` — has_conflict=true 且 conflicts 非空。
3. **硬闸3 质量**：`0 < evidence_quality < 50` 且依赖外部事实 → 硬降级（基线只 warning）。

三条均仅在 `needs_external_source`（命中外部事实上下文）时触发，纯比分判断不受影响。

---

## 四、验证

- `py_compile` → OK
- **TDD**：先写 6 个失败测试锁定期望（`tests/test_web_intel_hardgate_p1.py`），改码后转 green。
- **新测试 6 passed**（3 硬闸生效 + 3 不误杀）
- **.venv 全回归 120 passed**（基线 114 + 新 6，零破坏）
- 真实场景前后对比 → 三漏洞全部修复

---

## 五、军规合规

- ✅ 真实路径 exec 实跑、独立新分支、零碰 main
- ✅ 改前建远端备份 tag 并核验 SHA
- ✅ TDD（红→绿）+ .venv 全回归 + 修复前后对比
- ✅ 基线对比用只读 worktree（零 stash 撞产物风险）
- ⏸ **push 前停下，待用户单独批准**
