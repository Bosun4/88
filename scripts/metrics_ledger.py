"""Separate analysis, stored-price ROI and strictly forward performance.

A unit stake is a simulation at a saved quote, never a claimed execution price.
No odds or unverified time => no strictly forward ROI. D remains analysis-only.
"""
from __future__ import annotations

import math
import re

try:
    from .fixture_identity import fixture_key, kickoff_time, parse_time
except ImportError:
    from fixture_identity import fixture_key, kickoff_time, parse_time

BETTABLE_ACTIONS = {'main', 'small', 'hedge'}
BETTABLE_TIERS = {'S', 'A', 'B'}
UPSET_HINT_TERMS = ('反打', '博冷', '冷门', 'upset', 'contrarian', 'reverse',
                    '背离', 'sharp', 'rlm', 'fade')
DIRS = {'home': '主胜', 'draw': '平局', 'away': '客胜'}


def is_bettable(action, tier):
    return (str(action or '').lower() in BETTABLE_ACTIONS
            and str(tier or '').upper() in BETTABLE_TIERS)


def is_upset_bet(action, risk_tags, reason=''):
    text = ' '.join([str(action or ''), str(reason or ''), str(risk_tags or '')]).lower()
    return any(term in text for term in UPSET_HINT_TERMS)


def normalize_score(value):
    text = str(value or '').strip()
    m = re.fullmatch(r'(\d+)\s*[-:]\s*(\d+)', text)
    return f'{int(m[1])}-{int(m[2])}' if m else None


def score_channels(pred):
    """Main exact score vs structured side-risk coverage; no prose mining."""
    main = normalize_score(pred.get('final_ai_score') or pred.get('predicted_score'))
    raw = pred.get('risk_score_candidates') or []
    side = []
    if isinstance(raw, list):
        for value in raw:
            score = normalize_score(value.get('score') if isinstance(value, dict) else value)
            if score and score not in side:
                side.append(score)
    return main, side


def _price(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 1 else None
    except (TypeError, ValueError):
        return None


def settle_one(pred, gh, ga):
    """Accept a full match row (preferred) or a legacy prediction dict."""
    row = pred
    pred = row.get('prediction', row)
    rec = pred.get('recommendation') or {}
    action = str(rec.get('bet_action') or pred.get('final_action') or pred.get('selection_layer') or '').lower()
    tier = str(pred.get('recommendation_tier') or rec.get('tier') or 'D').upper()
    direction = str(pred.get('final_direction') or pred.get('result') or '')
    direction = {v: k for k, v in DIRS.items()}.get(direction, direction.lower())
    actual_dir = 'home' if gh > ga else ('draw' if gh == ga else 'away')
    main, side = score_channels(pred)
    score = f'{gh}-{ga}'
    hit = direction == actual_dir if direction in DIRS else None
    bettable = (is_bettable(action, tier) and not pred.get('is_abstain')
                and pred.get('recommend_gate_pass') is not False and hit is not None)
    market = rec.get('market') or pred.get('market') or '1x2'
    quote = pred.get('odds') if isinstance(pred.get('odds'), dict) else {}
    selection = rec.get('selection') or direction
    selection = {v: k for k, v in DIRS.items()}.get(selection, selection)
    price = None
    source = None
    if market in ('1x2', 'regulation', '90min') and selection == direction:
        for source_name, value in [
            ('recommendation.odds', rec.get('odds')),
            ('recommendation.bet_odds', rec.get('bet_odds')),
            ('match.sp_' + direction, row.get('sp_' + direction)),
            ('prediction.odds.' + direction, quote.get(direction)),
            ('prediction.odds.sp_' + direction, quote.get('sp_' + direction)),
        ]:
            price = _price(value)
            if price is not None:
                source = source_name
                break
    quote_at = rec.get('quoted_at') or row.get('odds_captured_at') or row.get('captured_at')
    kickoff = kickoff_time(row)
    quoted = parse_time(quote_at)
    # An explicitly late quote must not be used even for stored-price simulation.
    if kickoff and quoted and quoted >= kickoff:
        price = None
        source = None
    profit = (price - 1 if hit else -1.) if bettable and price else (None if bettable else 0.)
    locked = parse_time(row.get('locked_at_utc'))
    strict = bool(row.get('strict_forward') is True and kickoff and quoted and locked
                  and quoted <= locked < kickoff)
    return {
        'fixture_key': fixture_key(row), 'pred_dir': DIRS.get(direction, ''),
        'actual_dir': DIRS[actual_dir], 'actual_score': score, 'hit': hit,
        'bettable': bettable, 'upset': is_upset_bet(action, rec.get('risk_tags') or pred.get('tail_risk_flags'), pred.get('reason') or rec.get('why_recommended')),
        'tier': tier, 'action': action, 'market': market, 'selection': selection,
        'odds': price, 'odds_source': source, 'quoted_at': quote_at, 'profit': profit,
        'roi_eligible': bool(bettable and price), 'strict_forward': strict,
        'roi_basis': 'saved_quote_simulation',
        'main_score': main, 'main_score_hit': main == score if main else None,
        'side_risk_scores': side, 'side_risk_hit': score in side if side else None,
    }


def aggregate(settled):
    # Identity-less legacy rows cannot safely be merged; callers expose that scope.
    seen = set()
    rows = []
    for item in settled:
        key = item.get('fixture_key')
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        rows.append(item)

    def accuracy(group, field):
        valid = [s for s in group if s.get(field) is not None]
        hits = sum(bool(s[field]) for s in valid)
        return {'samples': len(valid), 'hits': hits,
                'accuracy_pct': round(hits / len(valid) * 100, 1) if valid else None}

    def book(group):
        priced = [s for s in group if s.get('roi_eligible')]
        count = len(priced)
        wins = sum(bool(s['hit']) for s in priced)
        pnl = sum(s['profit'] for s in priced)
        return {'staked': count, 'candidates': len(group), 'unpriced': len(group) - count,
                'wins': wins, 'win_rate': round(wins / count * 100, 1) if count else None,
                'pnl': round(pnl, 3) if count else None,
                'roi_pct': round(pnl / count * 100, 1) if count else None}

    bets = [s for s in rows if s['bettable']]
    direction = accuracy(rows, 'hit')
    risk_d = [s for s in rows if s['tier'] == 'D']
    return {
        'samples': len(rows), 'direction_samples': direction['samples'],
        'direction_accuracy_pct': direction['accuracy_pct'],
        'bettable': book(bets),
        'value_bets': book([s for s in bets if not s['upset']]),
        'upset_bets': book([s for s in bets if s['upset']]),
        'strict_forward': book([s for s in bets if s.get('strict_forward')]),
        'main_score': accuracy(rows, 'main_score_hit'),
        'side_risk_score': accuracy(rows, 'side_risk_hit'),
        'risk_d': {**accuracy(risk_d, 'hit'), 'main_score': accuracy(risk_d, 'main_score_hit'),
                   'side_risk_score': accuracy(risk_d, 'side_risk_hit')},
        'roi_basis': 'saved_quote_simulation_not_execution',
    }


def coaching_summary(agg):
    def pct(v):
        return f'{v}%' if v is not None else '不可计算'
    lines = [f"全样本方向命中率(仅分析): {pct(agg['direction_accuracy_pct'])} (n={agg['direction_samples']})"]
    for key, label in [('bettable', '存储报价模拟合账'), ('value_bets', '价值单'), ('upset_bets', '博冷单'), ('strict_forward', '严格前向')]:
        b = agg[key]
        lines.append(f"{label}: {b['staked']}单 ROI={pct(b['roi_pct'])} PnL={b['pnl']} 缺有效报价={b['unpriced']}")
    for key, label in [('main_score', '主比分'), ('side_risk_score', '副文风险比分'), ('risk_d', 'D级分析方向')]:
        b = agg[key]
        lines.append(f"{label}: {b['hits']}/{b['samples']} {pct(b['accuracy_pct'])}")
    return '\n'.join(lines)
