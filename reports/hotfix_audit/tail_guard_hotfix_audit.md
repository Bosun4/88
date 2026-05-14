# Tail Guard Hotfix Audit

## 背景
赛后复盘发现多场 2-1 主胜预测被 2-3 客胜反打击穿，例如比利亚雷 vs 塞维利亚、塞尔塔 vs 莱万特。有人提出放宽弱主胜尾部保护阈值。

## 审计原则
这属于赛后发现问题后的规则修改，必须作为待验证 hotfix，不得直接宣传为命中率提升。

## 必须回答
1. 当前 `scripts/predict.py` 是否已经被修改？
是，本地已被修改。
2. 修改在哪个分支？
在 `main` 分支。
3. 是否已经 commit？
否，代码修改还未提交。
4. 是否已经 push？
否，代码未推送到远程仓库。
5. 是否污染 main？
是，直接在 main 分支修改了代码。
6. 修改是否只影响 risk candidates / flags？
是，根据 `scripts/predict.py` 内 `apply_weak_home_tail_risk_protection` 函数的逻辑，修改只影响 `risk_score_candidates`、`tail_risk_flags` 等附加风险标记。
7. 修改是否会改变 final_direction？
否，它在 final 裁决之后对结果加风控标签，不回写主干预测方向。
8. 修改是否会改变 predicted_score？
否，不改变预测的具体比分数值。
9. 修改是否会改变 confidence？
不改变原始 `bet_confidence`，但可能会触发 `confidence_downgrade_reason` 或影响前端的推荐评级显示。
10. 是否有测试覆盖？
运行过一段 Python 脚本 `tests/test_predict_patch.py` 做简单验证，但并不在标准测试用例中。
11. 是否存在过度触发风险？
是。将 `home_pct` 上限从 52 提高到 55，`away_pct` 下限从 23 降到 22，意味着会有更多的“主胜”比赛被强制加入 `1-2`, `2-2`, `2-3` 这些尾部比分。这可能会让大量本身正常的赢球比赛（甚至胜面较大的比赛）被过度标记风险。
12. 是否建议作为独立 PR？
强烈建议。
13. 是否应先做 forward-only 验证？
是的，不应该依赖过去一天的结果去调整今天的数据。需要正向滚动验证修改后的规则。

## 初步判断
允许将该修复作为“风险候选增强”，但不允许让它直接改 final prediction。
如果它只增加 risk_score_candidates / tail_risk_flags，不改 final，则可考虑独立 PR。
如果它会影响 final_direction / predicted_score / confidence，则不建议合并。
