# P1a 审计报告：移除泊松死模块 + scipy 依赖 + v203 死码

**日期**: 2026-06-07
**基线**: 接 P0 分支 `fix/p0-shadow-poisson-decontam-20260607` (HEAD a988494)
**备份**: 复用 `.backup_p0p1_20260607141824/`（追加 advanced_models/main/test_shadow_integration 的 .p1bak）
**改动**: 5 文件，净 -299 行

---

## 1. 改动清单

| # | 操作 | 文件 | 说明 |
|---|------|------|------|
| 1 | 删除模块 | `scripts/advanced_models.py`（-226行） | 含 `ZeroInflatedBivariatePoisson`（scipy 泊松）等，predict.py 零调用（P0 已删死 import），纯死模块 |
| 2 | 删死码 | `scripts/predict.py`（-60行） | `build_evidence_packet_v203`（3295-3354）全仓零调用。保留 `normalize_match_v203`（别处仍用） |
| 3 | 清依赖 | `requirements.txt` | 移除 `scipy>=1.12.0`（已无任何活引用） |
| 4 | 清依赖 | `scripts/main.py` | REQUIRED_PACKAGES 自动安装列表移除 scipy。保留 numpy(quant_edge)/pandas(grabber)/deep-translator(fetch_data) |
| 5 | 修测试 | `tests/test_shadow_integration.py`（-12行） | 移除第3段 advanced_models 泊松冒烟测，保留 league_intel/experience_rules/quant_edge 三活模块测试 |

## 2. 验证（本地 .venv 实跑）

- **全回归**：`pytest tests/ -q` → **86 passed / 0 failed**（14.63s）
- **残留扫描**：scripts/ + tests/ 内 `advanced_models`/`scipy` 活引用 = **0**
- **运行时解耦铁证**：import 拦截器验证核心 predict 链路（build_evidence_packet + build_phase1_prompt）对 scipy/advanced_models **零触发**
- **编译**：predict.py / main.py / test_shadow_integration.py 全部 py_compile 通过
- **证据包**：jingcai_market_facts 正常生成

## 3. 纠错记录（真实可验证军规）

P0 报告中两处判断在 P1 实查中被修正：
1. ~~"scipy 唯一来源是 advanced_models"~~ → **错**：`main.py` REQUIRED_PACKAGES 也硬编码了 scipy（已一并清理）
2. ~~"根目录30+脚本仍在版本控制内 = 仓库污染"~~ → **错**：37 个孤儿脚本**全部已被 .gitignore 覆盖、未被 git 跟踪**，从版本控制看仓库本就干净，无需处理

## 4. 未处理（标注，不在本轮范围）

- **legacy/v18_1/predict_v18_1.py** 仍 `from advanced_models import ...`，但在 try/except 内（import 失败仅 warning + 提供 fallback 函数），且不在生产/测试链路。删 advanced_models 后无害（86 passed 已证）。legacy 是冻结历史版本，删除属范围扩大，本轮不动。
- **三组交错重复同名函数**（build_evidence_packet/build_phase1_prompt/_canonical_output_schema_text/build_critic_prompt/build_gemini_final_prompt/adapt_ai_to_frontend 各有旧版+新版）：旧版 `build_evidence_packet`(921) 被 `_BASE_BUILD_EVIDENCE_PACKET_V2021` 保存着是**活码**，其余旧版疑似死码但需逐个用 _BASE 引用核验 + 单元覆盖。属高风险结构清理，建议独立分支（P1b）专项处理，不在本轮赶工。

## 5. 结论

泊松死模块（advanced_models.py）+ scipy 重依赖 + v203 死码已干净移除，核心链路与泊松/scipy 彻底解耦，全回归无回退。重复函数死码（P1b）留待独立专项。
