"""Deterministic, sourced research evidence. Never generate a forecast probability."""
from __future__ import annotations

import hashlib
import copy
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

try:
    from .fixture_identity import fixture_key, kickoff_time, parse_time, prediction_rows, same_fixture
except ImportError:
    from fixture_identity import fixture_key, kickoff_time, parse_time, prediction_rows, same_fixture

SCORES = {f'{h}-{a}': ('s' if h == a else 'w' if h > a else 'l') + str(h) + str(a)
          for h, a in [(1,0),(2,0),(2,1),(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),
                       (5,0),(5,1),(5,2),(0,0),(1,1),(2,2),(3,3),(0,1),(0,2),
                       (1,2),(0,3),(1,3),(2,3),(0,4),(1,4),(2,4),(0,5),(1,5),(2,5)]}
MARKETS = {
    'result': {'home': 'sp_home', 'draw': 'sp_draw', 'away': 'sp_away'},
    'handicap': {'home_cover': 'hhad_win', 'draw': 'hhad_same', 'away_cover': 'hhad_lose'},
    'correct_score': SCORES,
    'total_goals': {str(i) if i < 7 else '7+': f'a{i}' for i in range(8)},
    'half_full_time': {k: k for k in ('ss','sp','sf','ps','pp','pf','fs','fp','ff')},
}
IDENTITY = ('match_id', 'fixture_id', 'api_football_fixture_id', 'source', 'league', 'league_id', 'season',
            'home_team', 'away_team', 'home_id', 'away_id', 'kickoff_at', 'event_date')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def valid_url(value):
    try:
        url = urlsplit(str(value or ''))
        return url.scheme in ('https', 'http') and bool(url.hostname) and not url.username and not url.password
    except ValueError:
        return False


def odds(value):
    try:
        number = float(value)
        return number if not isinstance(value, bool) and math.isfinite(number) and number > 1 else None
    except (TypeError, ValueError):
        return None


def handicap_value(value):
    try:
        number = float(value)
        return int(number) if not isinstance(value, bool) and math.isfinite(number) and number.is_integer() else None
    except (ValueError, TypeError):
        return None


def markets(row):
    merged = {**(row.get('v2_odds_dict') or {}), **row}
    output = {market: {label: odds(merged.get(key)) for label, key in mapping.items()
                       if odds(merged.get(key)) is not None} for market, mapping in MARKETS.items()}
    # Explicit normalized imports may contain the three aggregate score outcomes.
    for market, quotes in (row.get('market_quotes') or {}).items():
        if market in output and isinstance(quotes, dict):
            allowed = set(MARKETS[market]) | ({'胜其他', '平其他', '负其他'} if market == 'correct_score' else set())
            output[market].update({k: odds(v) for k, v in quotes.items() if k in allowed and odds(v) is not None})
    return output


def quote_records(row):
    key, kickoff = fixture_key(row), kickoff_time(row)
    captured, quoted = parse_time(row.get('captured_at')), parse_time(row.get('quoted_at'))
    if not key or not kickoff or not captured or not row.get('source') or not valid_url(row.get('source_url')):
        return []
    if quoted and quoted > captured:
        return []
    identity = {k: row.get(k) for k in IDENTITY}
    identity['kickoff_at'] = kickoff.isoformat()
    host = urlsplit(row['source_url']).hostname or ''
    kind = 'official_opening' if (row.get('quote_kind') == 'official_opening'
                                  and row.get('source') == 'sporttery' and quoted
                                  and (host == 'sporttery.cn' or host.endswith('.sporttery.cn'))) else 'observation'
    return [{'fixture_key': key, 'identity': identity, 'market': market, 'odds': values,
             'handicap': handicap_value(row.get('give_ball')) if market == 'handicap' else None,
             'source': row['source'], 'source_url': row['source_url'],
             'quoted_at': quoted.isoformat() if quoted else None, 'captured_at': captured.isoformat(),
             'quote_kind': kind, 'collection_mode': 'prematch' if captured < kickoff else 'retrospective'}
            for market, values in markets(row).items() if values
            and (market != 'handicap' or handicap_value(row.get('give_ball')) is not None)]


def result_record(row):
    """Only independently sourced regulation results; never read prediction text."""
    import re
    actual = row.get('actual')
    if not isinstance(actual, dict) or actual.get('period') != '90min':
        return None
    published = parse_time(actual.get('published_at'))
    kickoff = kickoff_time(row)
    if (not fixture_key(row) or not kickoff or not published or published < kickoff + timedelta(minutes=90)
            or not valid_url(actual.get('source_url'))
            or not re.fullmatch(r'\d{1,2}-\d{1,2}', str(actual.get('score', '')))):
        return None
    return {'fixture_key': fixture_key(row), 'identity': {k: row.get(k) for k in IDENTITY},
            'score': actual['score'], 'period': '90min', 'published_at': published.isoformat(),
            'source_url': actual['source_url'], 'source': actual.get('source', row.get('source'))}


class ResearchStore:
    """SQLite index outside public JSON; exact record hashes make imports repeatable."""
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS records (hash TEXT PRIMARY KEY, kind TEXT NOT NULL,
                fixture TEXT NOT NULL, body TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS fixture_records ON records(fixture, kind);
            CREATE TABLE IF NOT EXISTS imports (name TEXT PRIMARY KEY, hash TEXT, report TEXT);
        ''')

    def close(self):
        self.db.close()

    def add(self, kind, record):
        cursor = self.db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?)',
                                (digest(record), kind, record['fixture_key'],
                                 json.dumps(record, ensure_ascii=False, allow_nan=False)))
        return cursor.rowcount

    def ingest(self, payload):
        report = Counter(rows=0, quote_records=0, result_records=0, missing_quote_provenance=0,
                         missing_result_provenance=0)
        for row in prediction_rows(payload):
            report['rows'] += 1
            quotes = quote_records(row)
            if not quotes:
                report['missing_quote_provenance'] += 1
            for record in quotes:
                report['quote_records'] += self.add('quote', record)
            actual = result_record(row)
            if actual:
                report['result_records'] += self.add('result', actual)
            else:
                report['missing_result_provenance'] += 1
        self.db.commit()
        return dict(report)

    def import_file(self, path):
        path = Path(path)
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        prior = self.db.execute('SELECT hash,report FROM imports WHERE name=?', (str(path.resolve()),)).fetchone()
        if prior and prior[0] == sha:
            return {**json.loads(prior[1]), 'cached': True}
        report = self.ingest(json.loads(content))
        self.db.execute('INSERT OR REPLACE INTO imports VALUES(?,?,?)',
                        (str(path.resolve()), sha, json.dumps(report)))
        self.db.commit()
        return {**report, 'cached': False}

    def records(self, kind, key=None):
        query, params = 'SELECT body FROM records WHERE kind=?', [kind]
        if key:
            query += ' AND fixture=?'
            params.append(key)
        return [json.loads(row[0]) for row in self.db.execute(query, params)]

    def timeline(self, row, cutoff):
        key = fixture_key(row)
        records = self.records('quote', key) if key else []
        records = [q for q in records if _available(q, cutoff)]
        records.sort(key=lambda q: (q['quoted_at'] or q['captured_at'], q['market'], q['source']))
        groups = {}
        for quote in records:
            group = (quote['market'], quote['source'], quote.get('handicap'))
            groups.setdefault(group, []).append(quote)
        selected = []
        for values in groups.values():
            # Keep opening/first/latest and the newest ten observations, bounded per market.
            keep = {digest(q): q for q in [values[0], *[q for q in values if q['quote_kind'] == 'official_opening'][:1], *values[-10:]]}
            for q in keep.values():
                selected.append({**q, 'display_kind': 'official_opening' if q['quote_kind'] == 'official_opening'
                                 else 'first_observed' if q is values[0] else 'subsequent_observation'})
        return {'records': selected, 'available_count': len(records), 'shown_count': len(selected),
                'verified_series': any(len({q['quoted_at'] for q in group if q['quoted_at']}) >= 2
                                       for group in groups.values()),
                'official_opening_available': any(q['quote_kind'] == 'official_opening' for q in records),
                'note': '首次采集不等于初盘；不同来源或让球值分别比较，缺少报价时间不能推断变盘时刻。'}

    def similar(self, row, cutoff, opening_only=False):
        current = markets(row)
        key = fixture_key(row)
        results = {}
        for result in self.records('result'):
            if parse_time(result['published_at']) <= cutoff:
                results.setdefault(result['fixture_key'], []).append(result)
        quotes = {}
        for quote in self.records('quote'):
            ident = quote['identity']
            kickoff = kickoff_time(ident)
            if (quote['fixture_key'] == key or same_fixture(ident, row) or not kickoff or kickoff >= cutoff
                    or ident.get('league') != row.get('league') or not _available(quote, cutoff)
                    or (parse_time(quote['quoted_at']) or parse_time(quote['captured_at'])) >= kickoff
                    or (opening_only and quote['quote_kind'] != 'official_opening')):
                continue
            quotes.setdefault(quote['fixture_key'], {}).setdefault(quote['market'], []).append(quote)
        cases = []
        for fixture, grouped in quotes.items():
            actuals = results.get(fixture, [])
            identity = next(iter(grouped.values()))[0]['identity']
            actuals = [r for values in results.values() for r in values if same_fixture(identity, r['identity'])]
            if not actuals or len({r['score'] for r in actuals}) != 1:
                continue
            latest = {k: max(v, key=lambda q: q['quoted_at'] or q['captured_at']) for k, v in grouped.items()}
            result = actuals[0]
            if current['handicap'] and ('handicap' not in latest or
                    latest['handicap']['handicap'] != handicap_value(row.get('give_ball'))):
                continue
            reference = latest.get('result')
            if not reference or len(current['result']) != 3 or len(reference['odds']) != 3:
                continue
            # Ratio distance is a quote comparison, never a win probability.
            distance = sum(abs(math.log(current['result'][k] / reference['odds'][k])) for k in current['result']) / 3
            if distance > .25:
                continue
            same_season = str(reference['identity'].get('season') or '') == str(row.get('season') or '')
            totals = latest.get('total_goals', {}).get('odds', {})
            goal_match = bool(totals and current['total_goals'] and
                              min(totals, key=totals.get) == min(current['total_goals'], key=current['total_goals'].get))
            h, a = map(int, result['score'].split('-'))
            direction = 'home' if h > a else 'away' if a > h else 'draw'
            favorite = min(reference['odds'], key=reference['odds'].get)
            cases.append({'fixture_key': fixture, 'identity': reference['identity'], 'actual': result,
                          'reference_quotes': {k: v['odds'] for k, v in latest.items()},
                          'quote_evidence': latest,
                          'quote_distance': round(distance, 4), 'same_season': same_season,
                          'same_goal_structure': goal_match, 'favorite_realized': direction == favorite,
                          'evaluation_mode': 'historical_comparison',
                          'opening_verified': reference['quote_kind'] == 'official_opening'})
        cases.sort(key=lambda c: (not c['same_season'], not c['same_goal_structure'], c['quote_distance'], c['fixture_key']))
        unique = []
        for case in cases:
            if not any(same_fixture(case['identity'], previous['identity']) for previous in unique):
                unique.append(case)
        cases = unique
        # Include both favorable and contrary examples when either exists.
        selected = cases[:4]
        for realized in (True, False):
            opposite = next((c for c in cases if c['favorite_realized'] is realized and c not in selected), None)
            if opposite:
                selected.append(opposite)
        dates = sorted(c['identity']['kickoff_at'] for c in cases)
        return {'sample_count': len(cases), 'date_from': dates[0] if dates else None,
                'favorite_realized_count': sum(c['favorite_realized'] for c in cases),
                'contrary_count': sum(not c['favorite_realized'] for c in cases),
                'date_to': dates[-1] if dates else None, 'opening_only': opening_only, 'cases': selected,
                'note': '历史案例只作情景反证，不是预测胜率；缺少独立赛果、身份或时间证据的样本不参与。'}

    def coverage(self):
        quotes, results = self.records('quote'), self.records('result')
        fixtures = {q['fixture_key'] for q in quotes}
        dates = sorted(q['identity']['kickoff_at'] for q in quotes)
        return {'fixtures': len(fixtures), 'quote_records': len(quotes),
                'result_fixtures': len({r['fixture_key'] for r in results}),
                'official_opening_records': sum(q['quote_kind'] == 'official_opening' for q in quotes),
                'prematch_observations': sum(q['collection_mode'] == 'prematch' for q in quotes),
                'retrospective_observations': sum(q['collection_mode'] == 'retrospective' for q in quotes),
                'date_from': dates[0] if dates else None, 'date_to': dates[-1] if dates else None,
                'imported_files': self.db.execute('SELECT COUNT(*) FROM imports').fetchone()[0],
                'complete_jingcai_history': False}


def _available(record, cutoff):
    captured, quoted = parse_time(record.get('captured_at')), parse_time(record.get('quoted_at'))
    return bool(captured and captured <= cutoff and (not quoted or quoted <= captured))


def recent_form(fixtures, team_id, cutoff, provenance, target_kickoff=None):
    """Last 30d plus last 5/10 (at most 90d), excluding all future/result leakage."""
    records, conflicting = {}, set()
    if not provenance.get('source') or not valid_url(provenance.get('source_url')) or not _available(provenance, cutoff):
        return {'available': False, 'reason': 'observation_after_cutoff', 'matches': []}
    for raw in fixtures if isinstance(fixtures, list) else []:
        if not isinstance(raw, dict):
            continue
        f, teams = raw.get('fixture', {}), raw.get('teams', {})
        when = parse_time(f.get('date'))
        if (not when or not cutoff - timedelta(days=90) <= when <= parse_time(provenance['captured_at']) - timedelta(minutes=90)
                or f.get('status', {}).get('short') not in ('FT', 'AET', 'PEN')):
            continue
        side = next((s for s in ('home','away') if teams.get(s, {}).get('id') == team_id), None)
        score = raw.get('score', {}).get('fulltime') or (raw.get('goals') if f.get('status', {}).get('short') == 'FT' else {}) or {}
        if not side or not f.get('id') or any(type(score.get(s)) is not int or score[s] < 0 for s in ('home','away')):
            continue
        opponent = 'away' if side == 'home' else 'home'
        key = str(f['id'])
        record = {'fixture_id': f['id'], 'kickoff_at': when.isoformat(), 'venue_side': side,
                               'opponent': teams.get(opponent), 'score': f"{score['home']}-{score['away']}",
                               'goals_for': score[side], 'goals_against': score[opponent],
                               'red_cards': raw.get('red_cards'), **provenance}
        if key in records and records[key] != record:
            conflicting.add(key)
        records[key] = record
    for key in conflicting:
        records.pop(key, None)
    rows = sorted(records.values(), key=lambda r: r['kickoff_at'], reverse=True)
    def summary(items):
        return {'sample_count': len(items), 'wins': sum(r['goals_for'] > r['goals_against'] for r in items),
                'draws': sum(r['goals_for'] == r['goals_against'] for r in items),
                'losses': sum(r['goals_for'] < r['goals_against'] for r in items),
                'goals_for': sum(r['goals_for'] for r in items), 'goals_against': sum(r['goals_against'] for r in items)}
    recent = [r for r in rows if parse_time(r['kickoff_at']) >= cutoff - timedelta(days=30)]
    return {'available': bool(rows), 'cutoff': cutoff.isoformat(), 'window_days': 30, 'max_lookback_days': 90,
            'last_30_days': summary(recent), 'last_5': summary(rows[:5]), 'last_10': summary(rows[:10]),
            'home': summary([r for r in recent if r['venue_side'] == 'home']),
            'away': summary([r for r in recent if r['venue_side'] == 'away']),
            'rest_days': round(((target_kickoff or cutoff) - parse_time(rows[0]['kickoff_at'])).total_seconds()/86400, 2) if rows else None,
            'rest_days_basis': 'kickoff_to_kickoff' if target_kickoff else 'last_kickoff_to_cutoff',
            'matches': rows[:10], 'missing': ['xG', 'travel_distance']}


def sourced_evidence(match):
    """Final input boundary: exclude facts observed after the analysis cutoff.

    Text without source/timestamp remains unknown. No retroactively collected
    injuries or results may be used as a real prematch observation.
    """
    row = copy.deepcopy(match)
    kickoff = kickoff_time(row)
    cutoff = parse_time(row.get('analysis_cutoff')) or datetime.now(timezone.utc)
    if kickoff:
        cutoff = min(cutoff, kickoff)
    diagnostics = row.setdefault('collection_diagnostics', [])

    def verified(value):
        captured = parse_time(value.get('captured_at')) if isinstance(value, dict) else None
        return bool(captured and captured <= cutoff and (not kickoff or captured < kickoff)
                    and value.get('source') and valid_url(value.get('source_url'))
                    and (not value.get('published_at') or
                         (parse_time(value['published_at']) and parse_time(value['published_at']) <= captured)))

    for key in ('home_stats', 'away_stats', 'intelligence'):
        value = row.get(key)
        row[key] = value if verified(value) else {'available': False, 'reason': 'missing_source_or_time'}
    row['h2h'] = [v for v in (row.get('h2h') or []) if verified(v)
                  and parse_time(v.get('date')) and parse_time(v['date']) < cutoff]
    forms = {}
    for side, form in (row.get('recent_form') or {}).items():
        if (isinstance(form, dict) and parse_time(form.get('cutoff'))
                and parse_time(form['cutoff']) <= cutoff
                and all(verified(v) and parse_time(v.get('kickoff_at'))
                        and parse_time(v['kickoff_at']) + timedelta(minutes=90) <= parse_time(v['captured_at'])
                        for v in form.get('matches', []))):
            forms[side] = form
    row['recent_form'] = forms
    details = row.get('fixture_details') or {}
    row['fixture_details'] = {k: v for k, v in details.items()
                              if k in ('injuries', 'lineups') and verified(v)}
    lineup = row['fixture_details'].get('lineups', {}).get('records', [])
    expected = {row.get('home_id'), row.get('away_id')}
    row['fixture_details']['confirmed_lineup'] = bool(None not in expected and len(expected) == 2
        and expected <= {v.get('team', {}).get('id') for v in lineup if isinstance(v, dict)
                        and len(v.get('startXI') or []) == 11})
    captured = parse_time(row.get('captured_at'))
    if captured and (captured > cutoff or (kickoff and captured >= kickoff)):
        for mapping in MARKETS.values():
            for key in mapping.values():
                row.pop(key, None)
        row['v2_odds_dict'], row['market_quotes'] = {}, {}
        diagnostics.append({'source': row.get('source'), 'status': 'quotes_after_cutoff_excluded'})
    if not forms:
        diagnostics.append({'source': 'recent_form', 'status': 'missing_verified_recent_form'})
    row['analysis_cutoff'] = cutoff.isoformat()
    return row


def attach_research(matches, data_dir, cutoff=None):
    """Collect immutable observations, then compile bounded evidence for live analysis."""
    try:
        from .history_import import import_archives
    except ImportError:
        from history_import import import_archives
    cutoff = cutoff or datetime.now(timezone.utc)
    store = ResearchStore(Path(data_dir) / 'research' / 'history.sqlite')
    try:
        import_archives(store, data_dir)
        store.ingest(matches)
        coverage = store.coverage()
        for row in matches:
            row['analysis_cutoff'] = cutoff.isoformat()
            row['quote_timeline'] = store.timeline(row, cutoff)
            row['historical_cases'] = {'observations': store.similar(row, cutoff),
                                       'opening': store.similar(row, cutoff, opening_only=True)}
            row['history_coverage'] = coverage
        return coverage
    finally:
        store.close()
