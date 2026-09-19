import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_auto_actuals_season_and_event_day(monkeypatch):
    import auto_actuals as aa
    monkeypatch.setattr(aa, 'fetch_csv', lambda code, season: [
        {'HomeTeam': 'Napoli', 'AwayTeam': 'Bologna', 'Date': '11/05/2026', 'FTHG': '2', 'FTAG': '3'}])
    rows = aa.build_eng_results('2526')
    assert ('I1', '2526', '2026-05-11', 'Napoli', 'Bologna') in rows
    assert ('I1', '2526', '2026-09-12', 'Napoli', 'Bologna') not in rows
    assert aa.season_for_date('2026-09-12') == '2627'


def test_baseline_dedupe_and_no_cross_season(tmp_path, monkeypatch):
    import baseline_259 as b
    (tmp_path / 'data').mkdir()
    original = {'match_id': 'x', 'kickoff_at': '2026-05-11T18:00:00Z',
                'league': '意甲', 'home_team': 'A', 'away_team': 'B',
                'prediction': {'final_direction': 'home', 'predicted_score': '2-0'}}
    for day in (1, 2):
        (tmp_path / 'data' / f'history_{day}.json').write_text(json.dumps({'matches': {'today': [{**original, 'date': str(day)}]}}), encoding='utf-8')
    future = {**original, 'kickoff_at': '2026-09-12T18:00:00Z'}
    (tmp_path / 'data' / 'history_3.json').write_text(json.dumps({'matches': {'today': [future]}}), encoding='utf-8')
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    monkeypatch.setattr(b, 'load_actuals', lambda: ([{**original, 'actual_score': '2-0'}], {}))
    rows = b.collect()
    assert len(rows) == 1
    assert rows[0]['evaluation_scope'] == 'retrospective_unlocked'


def test_baseline_unresolved_records_are_explicit(tmp_path, monkeypatch):
    import baseline_259 as b
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/history_x.json').write_text(json.dumps({'matches': {'today': [{
        'league': '意甲', 'home_team': 'A', 'away_team': 'B', 'date': '2026-09-20',
        'prediction': {'final_direction': 'home'}}]}}), encoding='utf-8')
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    monkeypatch.setattr(b, 'load_actuals', lambda: ([{'home_team': 'A', 'away_team': 'B', 'actual_score': '2-0'}], {}))
    report = b.collect(return_audit=True)
    assert report['rows'] == []
    assert report['unresolved'][0]['reason'] == 'missing_event_identity'
    assert report['strict_forward_samples'] == 0


def test_real_enrichment_to_auto_actuals_readback(tmp_path, monkeypatch):
    import asyncio
    import auto_actuals as aa
    import fetch_data
    from fixture_identity import match_actual, api_actual
    fixture = {'fixture': {'id': 55, 'date': '2099-09-19T23:30:00Z', 'status': {'short': 'FT'}},
               'league': {'id': 39, 'season': 2099},
               'teams': {'home': {'id': 42, 'name': 'Arsenal'}, 'away': {'id': 49, 'name': 'Chelsea'}},
               'goals': {'home': 2, 'away': 0}}
    async def api(session, endpoint, params, sema):
        return [fixture] if endpoint == '/fixtures' else []
    monkeypatch.setattr(fetch_data, 'API_FOOTBALL_KEY', 'offline-test')
    monkeypatch.setattr(fetch_data, 'async_fetch_api', api)
    row = {'home_team': '阿森纳', 'away_team': '切尔西', 'league': '英超',
           'kickoff_at': '2099-09-20T07:30:00+08:00', 'event_date': '2099-09-20',
           'match_id': 'wencai:123', 'source': 'wencai'}
    enriched = asyncio.run(fetch_data.enrich_match_data(None, row, 0, '2099-09-19', None))
    assert enriched['api_football_fixture_id'] == 55
    assert match_actual(enriched, [api_actual(fixture)]) is not None
    pred, aliases, out = (tmp_path / n for n in ('pred.json', 'aliases.json', 'actual.json'))
    pred.write_text(json.dumps({'matches': {'today': [enriched]}}), encoding='utf-8')
    aliases.write_text(json.dumps({'阿森纳': 'Arsenal', '切尔西': 'Chelsea'}), encoding='utf-8')
    monkeypatch.setattr(aa, 'fetch_csv', lambda code, season: [
        {'HomeTeam': 'Arsenal', 'AwayTeam': 'Chelsea', 'Date': '20/09/2099', 'FTHG': '2', 'FTAG': '0'}])
    monkeypatch.setattr(sys, 'argv', ['auto_actuals', '--pred', str(pred), '--out', str(out), '--aliases', str(aliases)])
    aa.main()
    readback = json.loads(out.read_text(encoding='utf-8'))
    assert readback[0]['actual_score'] == '2-0'
    assert readback[0]['api_football_fixture_id'] == 55
    assert readback[0]['source'] == 'wencai'
    assert readback[0]['result_source'].endswith('/9900/E0.csv')
    assert match_actual(enriched, readback) is not None
