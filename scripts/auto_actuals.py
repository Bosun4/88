#!/usr/bin/env python3
"""Fetch regulation-time results keyed by league, season, event day and teams.

Rows without a verifiable event day remain explicitly unresolved. `date` in the
prediction payload is a business/capture day and must never select a result.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import urllib.request
from zoneinfo import ZoneInfo

try:
    from .fixture_identity import event_date, kickoff_time, prediction_rows
except ImportError:
    from fixture_identity import event_date, kickoff_time, prediction_rows

CODES = {'英超': 'E0', '西甲': 'SP1', '意甲': 'I1', '法甲': 'F1', '德甲': 'D1'}
CSV_TIMEZONES = {'英超': 'Europe/London', '西甲': 'Europe/Madrid',
                 '意甲': 'Europe/Rome', '法甲': 'Europe/Paris', '德甲': 'Europe/Berlin'}
BASE = 'https://www.football-data.co.uk/mmz4281/'
CACHE = str(Path(tempfile.gettempdir()) / 'project88-fdcsv')


def result_event_date(row):
    """football-data CSV dates are competition-local, not capture or UTC days."""
    kickoff = kickoff_time(row)
    if kickoff and row.get('league') in CSV_TIMEZONES:
        return kickoff.astimezone(ZoneInfo(CSV_TIMEZONES[row['league']])).date().isoformat()
    return event_date(row)


def season_for_date(day):
    dt = datetime.strptime(day, '%Y-%m-%d')
    start = dt.year if dt.month >= 7 else dt.year - 1
    return f'{start % 100:02d}{(start + 1) % 100:02d}'


def fetch_csv(code, season):
    if code not in CODES.values() or len(season) != 4 or not season.isdigit():
        raise ValueError('Invalid league/season')
    path = Path(CACHE) / season / f'{code}.csv'
    if path.exists():
        raw = path.read_text(encoding='utf-8')
    else:
        req = urllib.request.Request(f'{BASE}{season}/{code}.csv', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read().decode('utf-8-sig', errors='replace')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw, encoding='utf-8')
    return list(csv.DictReader(io.StringIO(raw)))


def build_eng_results(season):
    results = {}
    for code in CODES.values():
        try:
            rows = fetch_csv(code, season)
        except Exception as exc:
            print(f'[warn] {code}/{season}: {exc}')
            continue
        for row in rows:
            day = None
            for fmt in ('%d/%m/%Y', '%d/%m/%y'):
                try:
                    day = datetime.strptime(row.get('Date', ''), fmt).date().isoformat()
                    break
                except ValueError:
                    continue
            try:
                gh, ga = int(row['FTHG']), int(row['FTAG'])
            except (KeyError, ValueError, TypeError):
                continue
            home, away = row.get('HomeTeam'), row.get('AwayTeam')
            if home and away and day and gh >= 0 and ga >= 0:
                results[(code, season, day, home.strip(), away.strip())] = f'{gh}-{ga}'
    return results


def load_aliases(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return {k: v for k, v in data.items() if not k.startswith('_')}


def cn_to_eng(cn, aliases, eng_teams):
    keyword = aliases.get(cn, cn)
    if keyword in eng_teams:
        return keyword
    candidates = [team for team in eng_teams if keyword and keyword.casefold() in team.casefold()]
    return candidates[0] if len(candidates) == 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--aliases', default='reports/audit_backfill_20260531/team_aliases.json')
    args = parser.parse_args()
    rows = prediction_rows(json.loads(Path(args.pred).read_text(encoding='utf-8')))
    aliases = load_aliases(args.aliases)
    extra = Path('reports/audit_backfill_20260531/team_aliases_big5.json')
    if extra.exists():
        aliases.update(load_aliases(extra))
    results = {}
    for season in sorted({season_for_date(result_event_date(r)) for r in rows if result_event_date(r) and r.get('league') in CODES}):
        results.update(build_eng_results(season))
    teams = {team for key in results for team in key[-2:]}
    out = []
    for row in rows:
        if row.get('league') not in CODES:
            continue
        day = result_event_date(row)
        season = season_for_date(day) if day else None
        home = cn_to_eng(row.get('home_team'), aliases, teams)
        away = cn_to_eng(row.get('away_team'), aliases, teams)
        score = results.get((CODES[row['league']], season, day, home, away))
        identity = {k: row[k] for k in ('fixture_id', 'match_id', 'api_football_fixture_id', 'source', 'fixture_source', 'match_num', 'league', 'league_id', 'home_id', 'away_id', 'home_team', 'away_team', 'kickoff_at', 'kickoff_at_utc') if k in row}
        out.append({**identity, 'event_date': event_date(row), 'result_event_date': day,
                    'season': row.get('season') or season,
                    'actual_score': score, 'market': '1x2',
                    'result_source': f'{BASE}{season}/{CODES[row["league"]]}.csv' if season else None,
                    'settlement_status': 'settled' if score else 'unresolved',
                    'unresolved_reason': None if score else ('missing_event_date' if not day else 'no_exact_result')})
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(out)} records; settled={sum(bool(r["actual_score"]) for r in out)}')


if __name__ == '__main__':
    main()
