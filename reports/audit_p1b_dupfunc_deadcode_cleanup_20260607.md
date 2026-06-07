# P1b 审计报告：清理重复同名函数死码

**日期**: 2026-06-07
**基线**: 接 P1a (HEAD a186a56)
**备份**: tag `backup/pre-p1b-dupfunc-cleanup-20260607152730` + `.backup_p0p1_20260607141824/predict.py.pre_p1b`
**改动**: scripts/predict.py 净删 215 行 (5080 → 4865)

## 1. 删除内容（7个重复函数的死码旧版，无 _BASE 保存）
从后往前删，避免行号偏移：
- 段C (1148-1253, 106行): build_critic_prompt / build_gemini_final_prompt / build_fallback_referee_prompt / build_consistency_judge_prompt 旧版
- 段B (1100-1135, 36行): _phase1_system / build_phase1_prompt 旧版
- 段A (1009-1081, 73行): _canonical_output_schema_text 旧版

新版全部在 3500+ 区段，Python 后定义覆盖前定义，旧版本就永不可达。

## 2. 关键风险规避：活码钉在死码区中间，跳过保留
- _web_research_instruction (1082-1099): 唯一定义活码，新版 prompt 仍调用 → 保留
- _short_prediction_for_prompt (1136-1147): 被 _BASE_SHORT_PREDICTION_FOR_PROMPT_V2021 保存的活码 → 保留

## 3. 验证（本地 .venv 实跑）
- 编译: py_compile 通过
- 死码归一: 7个函数现各只剩1个定义(新版)
- 活码完好: _web_research_instruction=1, _short_prediction_for_prompt=2(含_BASE旧版,正确)
- 6个新版 prompt 函数实跑烟测全过(防NameError): _canonical(3690字)/_phase1_system(364)/build_phase1_prompt(13572)/build_critic_prompt(8623)/build_gemini_final_prompt(14663)/build_consistency_judge_prompt(8509)/build_fallback_referee_prompt(12271)
- 全回归: 86 passed / 0 failed (18.16s)

## 4. 结论
重复同名函数死码已按三段精刀清单安全清除，活码零误伤，全回归无回退。predict.py 从 5155(P0前) → 4865，累计瘦身约290行结构死码。
