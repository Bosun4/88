# P1b 可行性研究报告（只读核验，未改任何代码）

**日期**: 2026-06-07
**性质**: 纯只读研究，零代码改动。为后续安全执行 P1b 提供精确切除清单。
**基线**: P1a 分支 HEAD a186a56

---

## 1. 背景

predict.py 存在大量"同名函数重复定义"——Python 后定义覆盖前定义。需判定每个旧版是死码（可删）还是被 `_BASE_*` 别名保存的活码（碰不得）。

## 2. 判定规则（铁律）

- 文件加载到 **行2667-2680** 时执行 14 个 `_BASE_X = 函数名` 保存语句，此刻全局名绑定的是**先定义的旧版**（行号小者）→ **这些旧版是活码，永远不能删**。
- 重复定义但**无任何 _BASE 保存**的旧版 → 纯被覆盖死码，可删。

## 3. 核验结果

### 3.1 被 _BASE 保存的旧版（活码，禁删）—— 14个
build_evidence_packet(921)、adapt_ai_to_frontend(2330)、normalize_ai_predictions(1804)、_short_prediction_for_prompt(1136)、_f/_i/_exists/_json_compact/_normalize_score_text/_parse_score/_score_direction/_score_total/_score_btts/_score_goal_band（均为先定义版）。

### 3.2 无 _BASE 保存的死码旧版（可删）—— 7个函数
`_canonical_output_schema_text`(1009)、`_phase1_system`(1100)、`build_phase1_prompt`(1115)、`build_critic_prompt`(1148)、`build_gemini_final_prompt`(1172)、`build_fallback_referee_prompt`(1217)、`build_consistency_judge_prompt`(1234)。新版全部在 3538+ 区段，已实测生效。

## 4. ⚠️ 关键风险：死码非连续块，活码钉在中间

死码区段(1009-1253)被两个活码"钉穿"，**严禁整段删除**：
- **行1082-1099 `_web_research_instruction`**：唯一定义活码。被旧版(1121,死)和新版(3654,活)调用；删旧调用方后仍被3654活调用 → 必须保留
- **行1136-1147 `_short_prediction_for_prompt`**：被 _BASE 保存的活码 → 必须保留

## 5. 精刀切除清单（三段，跳过两处活码）

| 段 | 行范围 | 内容 | 行数 |
|----|--------|------|------|
| A | 1009-1081 | `_canonical_output_schema_text` 旧版 | 73 |
| **跳过** | 1082-1099 | `_web_research_instruction` **活码** | — |
| B | 1100-1135 | `_phase1_system` 旧版 + `build_phase1_prompt` 旧版 | 36 |
| **跳过** | 1136-1147 | `_short_prediction_for_prompt` **活码** | — |
| C | 1148-1253 | `build_critic/gemini_final/fallback_referee/consistency_judge` 四旧版 | 106 |
| **跳过** | 1254+ | `_clean_env_key` 等 **活码** | — |

**预计净删 ~215 行**（必须从后往前删 C→B→A，避免行号偏移）。

边界已核验：1081/1082、1135/1136、1253/1254 三处交界均为"死码尾 + 空行 + 活码 def 头"，吻合。

## 6. 执行前置条件 & 建议

1. **建议等 P0/P1a 合并入 main 后**，在干净基线上单独切 P1b 分支执行，避免未合并分支链堆叠高风险结构改动。
2. 执行步骤（军规）：备份 tag → 从后往前删三段 → py_compile → 本地 .venv 全回归(必须仍 86 passed) → import 烟测 build_phase1_prompt/build_critic_prompt/build_gemini_final_prompt 实际可调且输出正常 → 出修复前后对比 → 推新分支 → 等批准。
3. **额外烟测**：因删的是 prompt 构建函数，回归后必须实跑一次 `build_phase1_prompt([ev],"grok")` 等6个新版函数，确认无 NameError/逻辑回退。

## 7. 结论

P1b 技术上可行、收益明确（净删~215行死码），但**有"活码钉穿死码区"的真实陷阱**，必须按三段精刀清单执行，绝不可整段删。建议待主干合并后专项执行。本轮只研究、不动手。
