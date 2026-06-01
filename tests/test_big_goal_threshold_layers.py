# -*- coding: utf-8 -*-
"""
大球判据三层一致性回归 (读盘范式 v2.1 / 阈值 1.45->1.70 + 尾部共振)

背景: c023b59 把观察层阈值从 1.45 改成 1.70 并提取常量, 但执行层(_handicap_anchor_facts)
和提示层(_cross_anchor_questions)漏改, 仍硬编码 1.45 — 而旧的 57 项回归无一覆盖大球注入
逻辑, 导致"假绿"通过. 本文件专门堵住这个盲区, 锁死三层口径一致:

  观察层 _total_goal_anchor_facts  -> anchor 标签
  执行层 _handicap_anchor_facts    -> big_goal_injection.triggered + must_audit_scores
  提示层 _cross_anchor_questions   -> 大球曲线塌缩/尾部共振 提问文案

核心覆盖: slope ∈ (1.45, 1.70] 这个本次校准的"全部价值区间"必须三层都触发大球.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict
from scripts.predict import (
    BIG_GOAL_SLOPE_THRESHOLD,
    BIG_GOAL_TAIL_A6_MAX,
    BIG_GOAL_TAIL_A7_MAX,
    BIG_GOAL_PRIMARY,
)


def _match(a4, a5, a6=18.0, a7=29.0, give_ball="-1.0"):
    """构造最小 match_obj, 只填大球判据相关字段."""
    return {
        "a0": 10.0, "a1": 5.0, "a2": 3.35, "a3": 3.45,
        "a4": a4, "a5": a5, "a6": a6, "a7": a7,
        "give_ball": give_ball,
        "home_team": "A", "away_team": "B", "league": "瑞超",
    }


# ---------- 常量自检 ----------

def test_thresholds_are_the_calibrated_values():
    assert BIG_GOAL_SLOPE_THRESHOLD == 1.70
    assert BIG_GOAL_TAIL_A6_MAX == 11.0
    assert BIG_GOAL_TAIL_A7_MAX == 14.0


# ---------- 观察层 ----------

def _anchors(m):
    facts = predict._total_goal_anchor_facts(m)
    return {o["anchor"] for o in facts.get("observations_for_ai", [])}


def test_observation_collapse_at_slope_1_60():
    # slope = 6.0/4.0 = 1.50, 在 (1.45, 1.70] 区间 — 旧阈值会漏, 新阈值必中
    assert "big_goal_curve_collapse" in _anchors(_match(4.0, 6.0))


def test_observation_single_point_when_slope_above_threshold():
    # slope = 9.0/4.0 = 2.25 > 1.70 且无尾部共振 -> 单点诱盘
    a = _anchors(_match(4.0, 9.0, a6=20.0, a7=35.0))
    assert "four_goals_single_point_low_caution" in a
    assert "big_goal_curve_collapse" not in a


def test_observation_tail_resonance_exempts():
    # slope 高(2.25)但 a7=12 <=14 触发尾部共振豁免
    a = _anchors(_match(4.0, 9.0, a6=20.0, a7=12.0))
    assert "big_goal_tail_resonance" in a


# ---------- 执行层 (核心: 这层之前漏改) ----------

def _injection(m):
    facts = predict._handicap_anchor_facts(m)
    tpl = facts.get("score_shape_template", {})
    return tpl.get("big_goal_injection"), tpl.get("must_audit_scores", [])


def test_exec_injection_triggers_in_calibration_band():
    """slope=1.50 ∈ (1.45,1.70]: 旧执行层(1.45)会漏注入, 新执行层必须触发.
    这是堵"假绿"的关键断言 — 175 场校准的全部收益都在这个区间."""
    inj, must = _injection(_match(4.0, 6.0))
    assert inj is not None and inj.get("triggered") is True
    for sc in BIG_GOAL_PRIMARY:
        assert sc in must


def test_exec_injection_not_triggered_above_threshold():
    inj, _ = _injection(_match(4.0, 9.0, a6=20.0, a7=35.0))
    assert inj is None


def test_exec_injection_via_tail_resonance():
    """执行层之前完全没有尾部共振逻辑 — 巴西6-2/大阪6-1 类碾压大球会被漏掉.
    slope 高但 a7<=14 必须靠尾部共振触发注入."""
    inj, must = _injection(_match(4.0, 9.0, a6=20.0, a7=12.0))
    assert inj is not None and inj.get("triggered") is True
    for sc in BIG_GOAL_PRIMARY:
        assert sc in must


def test_exec_exclusion_line_no_injection():
    # a4 > 5.3 排除线, 不注入
    inj, _ = _injection(_match(6.0, 8.0))
    assert inj is None


# ---------- 提示层 ----------

def _questions(m):
    return " | ".join(predict._cross_anchor_questions(m))


def test_prompt_collapse_question_in_band():
    q = _questions(_match(4.0, 6.0))
    assert "大球曲线塌缩确认" in q
    # 文案口径必须是 1.70, 不能再出现写死的 <=1.45 判据口径
    assert "1.7" in q


def test_prompt_tail_resonance_question():
    q = _questions(_match(4.0, 9.0, a6=20.0, a7=12.0))
    assert "尾部共振" in q


def test_prompt_high_tier_abandons_exact_score():
    # HIGH 档弃精确: 提问必须明确不押精确比分
    q = _questions(_match(4.0, 6.0))
    assert "弃精确" in q or "不要勉强押准" in q or "主推" in q


# ---------- 三层一致性: 同一 case 三层同步触发 ----------

def test_all_three_layers_consistent_in_band():
    """slope=1.50 这个本次校准核心区间, 观察/执行/提示三层必须同步判为大球.
    任何一层回退到 1.45 都会让这个测试失败 — 永久防止再次出现执行层撕裂."""
    m = _match(4.0, 6.0)
    assert "big_goal_curve_collapse" in _anchors(m)          # 观察层
    inj, _ = _injection(m)
    assert inj is not None and inj.get("triggered") is True   # 执行层
    assert "大球曲线塌缩确认" in _questions(m)                 # 提示层
