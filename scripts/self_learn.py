"""Saved-prediction review with exact event binding and separate analysis books."""
import json
from pathlib import Path

import requests

try:
    from .config import API_FOOTBALL_KEY, API_FOOTBALL_BASE, GPT_API_KEY, GPT_API_URL
    from . import metrics_ledger as ml
    from .fixture_identity import api_actual, dedupe_predictions, event_date, fixture_key, match_actual, prediction_rows
except ImportError:
    from config import API_FOOTBALL_KEY, API_FOOTBALL_BASE, GPT_API_KEY, GPT_API_URL
    import metrics_ledger as ml
    from fixture_identity import api_actual, dedupe_predictions, event_date, fixture_key, match_actual, prediction_rows

ROOT = Path(__file__).resolve().parents[1]
PRED_FILE = str(ROOT / 'data/predictions.json')
DIARY_FILE = str(ROOT / 'data/ai_diary.json')


def fetch_actual_results(date_str):
    try:
        response = requests.get(
            f'{API_FOOTBALL_BASE}/fixtures', headers={'x-apisports-key': API_FOOTBALL_KEY},
            params={'date': date_str}, timeout=15)
        response.raise_for_status()
        return response.json().get('response', [])
    except (requests.RequestException, ValueError) as exc:
        print(f'赛果获取失败: {exc}')
        return []


def evaluate_matches(matches, actuals):
    """Pure review; accepts normalized actuals or API-Football fixtures."""
    normalized = []
    for actual in actuals:
        result = api_actual(actual) if 'fixture' in actual else actual
        if result:
            normalized.append(result)
    settled, unresolved, review_log = [], [], []
    for match in dedupe_predictions(matches):
        found = match_actual(match, normalized)
        if not found:
            unresolved.append({'fixture_key': fixture_key(match),
                               'match': f"{match.get('home_team')} vs {match.get('away_team')}",
                               'settlement_status': 'unresolved',
                               'reason': 'missing_identity' if not fixture_key(match) else 'no_exact_result'})
            continue
        score = ml.normalize_score(found.get('actual_score'))
        if not score:
            unresolved.append({'fixture_key': fixture_key(match), 'reason': 'invalid_regulation_score'})
            continue
        gh, ga = map(int, score.split('-'))
        # Only the immutable forward-ledger scoring path may declare strictness.
        item = ml.settle_one({**match, 'strict_forward': False}, gh, ga)
        settled.append(item)
        review_log.append({**item, 'match': f"{match.get('home_team')} vs {match.get('away_team')}"})
    return {'ledger': ml.aggregate(settled), 'reviews': review_log, 'unresolved': unresolved,
            'evaluation_scope': 'retrospective_unlocked'}


def self_learn(pred_file=None, diary_file=None, actuals_fetcher=None):
    path = Path(pred_file or PRED_FILE)
    if not path.exists():
        print('暂无预测记录，跳过复盘。')
        return None
    matches = prediction_rows(json.loads(path.read_text(encoding='utf-8')))
    fetcher = actuals_fetcher or fetch_actual_results
    actuals = []
    for day in sorted({event_date(m) for m in matches if event_date(m)}):
        fetched = fetcher(day)
        actuals.extend(fetched.values() if isinstance(fetched, dict) else fetched)
    review = evaluate_matches(matches, actuals)
    summary = ml.coaching_summary(review['ledger'])
    print(summary)
    diary = {'reflection': '仅客观账本；未结算样本不参与学习。', 'risk_adjustment': '中性'}
    if review['ledger']['samples'] and GPT_API_KEY:
        prompt = (
            '请依据以下账本复盘。ROI是存储报价模拟，不是成交收益；不可计算不得当作0。'
            '方向、主比分、副文风险比分各自分母；D级仅分析，不能升级为有效推荐。'
            '旧记录不是严格前向验证。博冷以收益和样本充分性评估，不仅看胜率。\n'
            + summary + '\n' + json.dumps(review['reviews'], ensure_ascii=False)
            + '\n返回纯JSON: {"reflection":"反思与策略(120字内)","risk_adjustment":"稳健/进取/中性"}')
        try:
            response = requests.post(
                GPT_API_URL, headers={'Authorization': f'Bearer {GPT_API_KEY}', 'Content-Type': 'application/json'},
                json={'model': 'gpt-5.4', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': .5}, timeout=20)
            response.raise_for_status()
            text = response.json()['choices'][0]['message']['content']
            parsed = json.loads(text[text.find('{'):text.rfind('}') + 1])
            diary.update({k: parsed[k] for k in ('reflection', 'risk_adjustment') if k in parsed})
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            print(f'AI反思失败，仍保留客观账本: {exc}')
    agg = review['ledger']
    roi = agg['bettable']['roi_pct']
    accuracy = agg['direction_accuracy_pct']
    diary.update(review)
    diary.update({'yesterday_summary': summary,
                  'yesterday_direction_accuracy': f'{accuracy}%' if accuracy is not None else '不可计算',
                  'yesterday_bettable_roi': f'{roi}%' if roi is not None else '不可计算',
                  'yesterday_win_rate': f'方向 {accuracy}% | 存储报价ROI {roi}%（非成交/非严格前向）'})
    target = Path(diary_file or DIARY_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(diary, ensure_ascii=False, indent=2), encoding='utf-8')
    return diary


if __name__ == '__main__':
    self_learn()
