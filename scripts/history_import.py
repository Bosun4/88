"""Import archived predictions or sourced JSON, optionally fetch public Sporttery history.

python -m scripts.history_import --archive data --db data/research/history.sqlite --report report.json
python -m scripts.history_import --input sourced.json --db data/research/history.sqlite
python -m scripts.history_import --public --start 2026-10-01 --end 2026-10-02 --db ...
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    from .research_data import ResearchStore, MARKETS
except ImportError:
    from research_data import ResearchStore, MARKETS

BASE = 'https://webapi.sporttery.cn/gateway/uniform/football/'
BJ = timezone(timedelta(hours=8))


def official_time(day, clock):
    try:
        return datetime.fromisoformat(str(day) + 'T' + str(clock)).replace(tzinfo=BJ).isoformat()
    except ValueError:
        return None


def normalize_official(match, bonus, captured):
    """Conservative field adapter, fixture-tested; live schema requires verification."""
    kickoff = official_time(match.get('matchDate'), match.get('matchTime'))
    if not match.get('matchId') or not kickoff:
        return []
    identity = {'match_id': 'sporttery:' + str(match['matchId']), 'source': 'sporttery',
                'kickoff_at': kickoff, 'league': match.get('leagueName') or match.get('leagueNameAbbr'), 'season': match.get('season'),
                'home_team': match.get('homeTeamName'), 'away_team': match.get('awayTeamName'),
                'captured_at': captured, 'give_ball': match.get('goalLine'),
                'source_url': BASE + 'getFixedBonusV1.qry?clientCode=3001&matchId=' + str(match['matchId'])}
    rows = []
    lists = {'result': 'hadList', 'handicap': 'hhadList', 'correct_score': 'crsList',
             'total_goals': 'ttgList', 'half_full_time': 'hafuList'}
    for market, field in lists.items():
        for item in bonus.get(field, []) or []:
            if not isinstance(item, dict):
                continue
            quoted = official_time(item.get('updateDate'), item.get('updateTime'))
            if not quoted:
                continue
            if market in ('result', 'handicap'):
                labels = ('home','draw','away') if market == 'result' else ('home_cover','draw','away_cover')
                prices = {k: item.get(v) for k, v in zip(labels, ('h','d','a'))}
            else:
                if market == 'correct_score':
                    mapping = {k: {'w':'h','s':'d','l':'a'}[v[0]] + v[1:] for k, v in MARKETS[market].items()}
                elif market == 'total_goals':
                    mapping = {k: 's' + str(i) for i, k in enumerate(MARKETS[market])}
                else:
                    mapping = {k: ''.join({'s':'h','p':'d','f':'a'}[c] for c in k) for k in MARKETS[market]}
                prices = {k: item.get(v) for k, v in mapping.items()}
                if market == 'correct_score':
                    prices.update({k: item.get(v) for k, v in [('胜其他','hOther'),('平其他','dOther'),('负其他','aOther')]})
            # A history list may be incomplete. Only explicit opening metadata is accepted.
            rows.append({**identity, 'quoted_at': quoted, 'give_ball': item.get('goalLine', identity['give_ball']),
                         'quote_kind': 'official_opening' if item.get('isOpening') is True else 'observation',
                         'market_quotes': {market: prices}})
    # Result response is 90-minute full-field result; no extra-time score fallback.
    if match.get('fullFieldScore'):
        rows.append({**identity, 'actual': {'score': str(match['fullFieldScore']).replace(':','-'),
                     'period': '90min', 'published_at': captured, 'source': 'sporttery',
                     'source_url': BASE + 'getUniformMatchResultV1.qry?matchId=' + str(match['matchId'])}})
    return rows


def fetch_public(store, start, end, checkpoint_dir, session=None):
    import requests
    session = session or requests.Session()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    day = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if last < day:
        raise ValueError('end must be on or after start')

    def get(endpoint, params):
        response = session.get(BASE + endpoint, params=params,
                               headers={'Referer': 'https://www.sporttery.cn/'}, timeout=20)
        response.raise_for_status()
        body = response.json()
        if body.get('success') is not True or not isinstance(body.get('value'), dict):
            raise ValueError('public_source_rejected_or_schema_changed')
        return body['value']

    while day <= last:
        path = checkpoint_dir / (day.isoformat() + '.json')
        try:
            if path.exists():
                status = store.import_file(path)
                statuses.append({'date': str(day), 'status': 'cached', **status})
            else:
                data = get('getUniformMatchResultV1.qry', {'startDate': str(day), 'endDate': str(day)})
                if not isinstance(data.get('matchResult'), list):
                    raise ValueError('match_result_schema_changed')
                rows = []
                for match in data['matchResult']:
                    bonus = get('getFixedBonusV1.qry', {'clientCode': '3001', 'matchId': match['matchId']})
                    if not isinstance(bonus.get('oddsHistory'), dict):
                        raise ValueError('odds_history_schema_changed')
                    normalized = normalize_official(match, bonus['oddsHistory'], datetime.now(timezone.utc).isoformat())
                    if not normalized:
                        raise ValueError('unrecognized_match_or_quote_schema')
                    rows.extend(normalized)
                    time.sleep(.25)
                payload = {'matches': rows}
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
                temp.replace(path)
                statuses.append({'date': str(day), 'status': 'ok', **store.import_file(path)})
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            statuses.append({'date': str(day), 'status': 'source_unavailable', 'error_type': type(error).__name__,
                             'http_status': getattr(getattr(error, 'response', None), 'status_code', None),
                             'source_url': BASE, 'checked_at': datetime.now(timezone.utc).isoformat()})
            # Do not hammer an unavailable source or mark the missing day complete.
            break
        day += timedelta(days=1)
    return statuses


def import_archives(store, directory, start=None, end=None):
    root = Path(directory)
    files = sorted({*root.glob('history_*.json'), *root.glob('snapshots/*.json')})
    reports = []
    for path in files:
        text = path.name.removeprefix('history_')[:10]
        if (start and text < start) or (end and text > end):
            continue
        reports.append({'file': path.name, **store.import_file(path)})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/research/history.sqlite')
    parser.add_argument('--archive')
    parser.add_argument('--input', action='append', default=[])
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--report')
    args = parser.parse_args()
    if args.public and (not args.start or not args.end):
        parser.error('--public requires --start and --end')
    for value in (args.start, args.end):
        if value:
            date.fromisoformat(value)
    store = ResearchStore(args.db)
    try:
        files = import_archives(store, args.archive, args.start, args.end) if args.archive else []
        files.extend({'file': Path(path).name, **store.import_file(path)} for path in args.input)
        public = fetch_public(store, args.start, args.end, Path(args.db).parent / 'public') if args.public else []
        report = {'coverage': store.coverage(), 'files': files, 'public_source': public}
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'coverage': report['coverage'], 'public_source': public}, ensure_ascii=False))
    finally:
        store.close()
    if any(item['status'] == 'source_unavailable' for item in public):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
