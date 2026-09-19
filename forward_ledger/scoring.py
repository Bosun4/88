"""Score verified locks without modifying the original prediction ledger."""
import csv
import json
from pathlib import Path

from forward_ledger.ledger import read_verified_entries
from scripts.fixture_identity import match_actual
from scripts.metrics_ledger import aggregate, normalize_score, settle_one
from scripts.post_review import score_prediction, summarize_reviews


def _score_distance(first, second):
    first, second = normalize_score(first), normalize_score(second)
    if not first or not second:
        return 999
    a, b = map(int, first.split('-'))
    c, d = map(int, second.split('-'))
    return abs(a - c) + abs(b - d)


def score_ledger_with_actuals(ledger_jsonl, actual_results_csv, output_csv, output_md):
    with open(actual_results_csv, encoding='utf-8-sig', newline='') as stream:
        actuals = list(csv.DictReader(stream))
    scored, metrics, reviews = [], [], []
    seen = set()
    for original in read_verified_entries(ledger_jsonl):
        entry = dict(original)
        key = entry.get('fixture_key') or entry.get('lock_key')
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        actual = match_actual(entry, actuals)
        score = normalize_score(actual.get('actual_score')) if actual else None
        entry.update({'settlement_status': 'unresolved', 'actual_score': None,
                      'actual_direction': None, 'direction_hit': None, 'exact_score_hit': None,
                      'risk_candidate_covered': None, 'goal_band_hit': None, 'btts_hit': None,
                      'evaluation_scope': 'strict_forward' if entry.get('strict_forward') else 'legacy_unverified'})
        if score:
            snapshot = dict(entry.get('prediction_snapshot') or entry)
            snapshot['strict_forward'] = bool(entry.get('strict_forward'))
            snapshot['locked_at_utc'] = entry.get('locked_at_utc')
            review = score_prediction(snapshot, score)
            gh, ga = map(int, score.split('-'))
            metric = settle_one(snapshot, gh, ga)
            entry.update({'settlement_status': 'settled', 'actual_score': score,
                          'actual_direction': review['actual_direction'],
                          'direction_hit': review['direction_hit'], 'exact_score_hit': review['score_hit'],
                          'goal_band_hit': review['goal_band_hit'], 'btts_hit': review['btts_hit'],
                          'risk_candidate_covered': review['side_risk_hit'],
                          'matrix_top_scores_covered': score in entry.get('matrix_top_scores', []),
                          'score_cluster_covered': score in entry.get('score_cluster', []),
                          'score_moderation_helped': False, 'score_moderation_hurt': False,
                          'profit': metric['profit'], 'odds': metric['odds'],
                          'roi_eligible': metric['roi_eligible']})
            if entry.get('score_moderation_applied'):
                old = _score_distance(entry.get('original_predicted_score'), score)
                new = _score_distance(entry.get('predicted_score'), score)
                entry['score_moderation_helped'] = new < old
                entry['score_moderation_hurt'] = old < new
            reviews.append((bool(entry.get('strict_forward')), review))
            metrics.append(metric)
        scored.append(entry)
    csv_path, md_path = Path(output_csv), Path(output_md)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for entry in scored for k in entry))
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(scored)
    report = {
        'records': len(scored), 'settled': sum(e['settlement_status'] == 'settled' for e in scored),
        'strict_forward_analysis': summarize_reviews([r for strict, r in reviews if strict]),
        'legacy_unverified_analysis': summarize_reviews([r for strict, r in reviews if not strict]),
        'ledger': aggregate(metrics),
    }
    md_path.write_text('# Forward Ledger Scoring Report\n\n'
                       '主比分和副文风险比分分别计算分母；D级仅分析。\n'
                       'ROI是记录报价模拟，缺有效报价不计算；无有效赛前锁档归旧口径。\n\n'
                       '```json\n' + json.dumps(report, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    return scored
