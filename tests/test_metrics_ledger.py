# -*- coding: utf-8 -*-
"""Tests for the dual-ledger metrics (ROI vs win-rate separation)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import metrics_ledger as ml


def _pred(direction, action="main", tier="A", odds=2.0, risk_tags=None, reason=""):
    return {
        "result": direction,
        "recommendation": {
            "bet_action": action,
            "tier": tier,
            "odds": odds,
            "risk_tags": risk_tags or [],
            "why_recommended": reason,
        },
    }


def test_observe_not_counted_in_roi():
    s = ml.settle_one(_pred("home", action="observe", tier="C"), 2, 0)
    assert s["bettable"] is False
    assert s["profit"] == 0.0
    agg = ml.aggregate([s])
    assert agg["bettable"]["staked"] == 0
    # direction still tracked
    assert agg["direction_accuracy_pct"] == 100.0


def test_value_bet_win_loss_roi():
    win = ml.settle_one(_pred("home", odds=2.0), 1, 0)   # hit -> +1.0
    loss = ml.settle_one(_pred("home", odds=2.0), 0, 1)  # miss -> -1.0
    agg = ml.aggregate([win, loss])
    assert agg["value_bets"]["staked"] == 2
    assert agg["value_bets"]["pnl"] == 0.0
    assert agg["value_bets"]["roi_pct"] == 0.0
    assert agg["value_bets"]["win_rate"] == 50.0


def test_upset_low_winrate_positive_roi():
    # 3 upset bets at odds 8.0, only 1 hits -> winrate 33% but ROI positive
    rows = [
        ml.settle_one(_pred("away", odds=8.0, risk_tags=["反打"]), 0, 2),  # hit +7
        ml.settle_one(_pred("away", odds=8.0, risk_tags=["反打"]), 2, 0),  # miss -1
        ml.settle_one(_pred("away", odds=8.0, risk_tags=["反打"]), 1, 0),  # miss -1
    ]
    agg = ml.aggregate(rows)
    u = agg["upset_bets"]
    assert u["staked"] == 3
    assert u["wins"] == 1
    assert round(u["win_rate"], 0) == 33.0
    # PnL = 7 - 1 - 1 = 5 ; ROI = 5/3 = 166.7%
    assert u["pnl"] == 5.0
    assert u["roi_pct"] > 100.0
    # upset bets must NOT pollute value-bet book
    assert agg["value_bets"]["staked"] == 0


def test_upset_negative_roi_triggers_concern():
    # all upset bets miss -> negative ROI (this is when system SHOULD pull back)
    rows = [ml.settle_one(_pred("away", odds=5.0, reason="背离反打"), 2, 0) for _ in range(3)]
    agg = ml.aggregate(rows)
    assert agg["upset_bets"]["roi_pct"] == -100.0


def test_mixed_books_isolated():
    rows = [
        ml.settle_one(_pred("home", action="main", tier="A", odds=1.8), 2, 0),       # value win
        ml.settle_one(_pred("away", action="small", tier="B", odds=6.0,
                             risk_tags=["upset"]), 0, 1),                              # upset win
        ml.settle_one(_pred("draw", action="observe", tier="D"), 1, 1),               # not bet
    ]
    agg = ml.aggregate(rows)
    assert agg["bettable"]["staked"] == 2
    assert agg["value_bets"]["staked"] == 1
    assert agg["upset_bets"]["staked"] == 1
    assert agg["samples"] == 3


def test_missing_odds_is_unpriced():
    s = ml.settle_one(_pred("home", odds=0), 1, 0)
    assert s["odds"] is None
    assert s["profit"] is None
    assert ml.aggregate([s])["bettable"]["unpriced"] == 1


def test_outer_direction_prices_and_risk_denominators():
    rows = []
    for direction, goals, price in [('home', (2, 0), 1.22), ('draw', (1, 1), 3.5), ('away', (0, 1), 4.0)]:
        pred = _pred(direction, odds=0)
        pred['predicted_score'] = '2-0'
        pred['risk_score_candidates'] = [{'score': '1-1'}, {'score': '1-1'}]
        rows.append(ml.settle_one({'prediction': pred, 'sp_' + direction: price}, *goals))
    assert abs(rows[0]['profit'] - .22) < 1e-9
    d = _pred('home', tier='D')
    d['predicted_score'] = '2-0'
    d['recommend_gate_pass'] = True
    rows.append(ml.settle_one(d, 2, 0))
    assert rows[-1]['bettable'] is False
    agg = ml.aggregate(rows)
    assert agg['risk_d']['samples'] == 1
    assert agg['main_score']['samples'] == 4
    assert agg['side_risk_score']['samples'] == 3
    assert agg['side_risk_score']['hits'] == 1


def test_english_direction_normalized():
    s = ml.settle_one({"final_direction": "home",
                       "recommendation": {"bet_action": "main", "tier": "A", "odds": 2.0}}, 3, 1)
    assert s["pred_dir"] == "主胜"
    assert s["hit"] is True


def test_saved_quote_profit_missing_and_invalid_prices():
    priced_win = ml.settle_one({'prediction': _pred('home', odds=0), 'sp_home': 1.22}, 1, 0)
    priced_loss = ml.settle_one({'prediction': _pred('home', odds=0), 'sp_home': 1.22}, 0, 1)
    missing_loss = ml.settle_one(_pred('home', odds=None), 0, 1)
    assert missing_loss['profit'] is None
    book = ml.aggregate([priced_win, priced_loss, missing_loss])['bettable']
    assert book['pnl'] == -.78
    assert book['roi_pct'] == -39.0
    assert book['unpriced'] == 1
    for odds in (float('nan'), float('inf'), -1, 1):
        assert ml.settle_one(_pred('home', odds=odds), 1, 0)['odds'] is None


def test_strict_roi_requires_quote_at_or_before_lock():
    row = {'fixture_id': 'x', 'kickoff_at': '2099-01-02T00:00:00Z',
           'locked_at_utc': '2026-01-01T00:00:00Z', 'strict_forward': True,
           'odds_captured_at': '2099-01-01T00:00:00Z',
           'prediction': _pred('home', odds=1.22)}
    result = ml.settle_one(row, 1, 0)
    assert result['strict_forward'] is False
