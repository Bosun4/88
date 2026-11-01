import copy
import json
from datetime import datetime, timezone

import pytest

from scripts.research_data import (ResearchStore, markets, quote_records, recent_form,
                                   result_record)
from scripts.history_import import normalize_official, fetch_public, import_archives


def match(day='2026-10-01', identity='1', score='2-1'):
    return {'match_id': 'sporttery:' + identity, 'source': 'sporttery',
            'source_url': 'https://webapi.sporttery.cn/quotes', 'league': '英超', 'season': 2026,
            'home_team': 'A' + identity, 'away_team': 'B' + identity,
            'kickoff_at': day + 'T20:00:00+00:00', 'captured_at': day + 'T12:00:00+00:00',
            'quoted_at': day + 'T11:00:00+00:00', 'give_ball': -1,
            'sp_home': 2.0, 'sp_draw': 3.4, 'sp_away': 3.2,
            'hhad_win': 3.0, 'hhad_same': 3.5, 'hhad_lose': 2.0,
            'a2': 3.5, 'a3': 4.0, 'w21': 8.0, 'ss': 4,
            'actual': {'score': score, 'period': '90min', 'published_at': day + 'T22:00:00+00:00',
                       'source_url': 'https://www.sporttery.cn/results', 'source': 'sporttery'}}


@pytest.fixture
def store(tmp_path):
    index = ResearchStore(tmp_path / 'research.sqlite')
    yield index
    index.close()


def test_all_five_markets_and_other_scores_are_retained():
    row = match()
    row['a7'] = 25
    row['market_quotes'] = {'correct_score': {'胜其他': 30, '平其他': 100, '负其他': 50}}
    values = markets(row)
    assert set(values) == {'result','handicap','total_goals','correct_score','half_full_time'}
    assert values['total_goals']['7+'] == 25
    assert values['correct_score']['胜其他'] == 30
    assert values['half_full_time']['ss'] == 4
    assert values['handicap']['home_cover'] == 3


def test_first_observation_is_not_opening_and_quotes_need_provenance(store):
    row = match()
    assert all(q['quote_kind'] == 'observation' for q in quote_records(row))
    row['quote_kind'] = 'official_opening'
    assert all(q['quote_kind'] == 'official_opening' for q in quote_records(row))
    row['source_url'] = 'https://invented.invalid/sporttery.cn'
    assert all(q['quote_kind'] == 'observation' for q in quote_records(row))
    row['captured_at'] = '2026-10-01 12:00:00'
    assert quote_records(row) == []
    row = match()
    row['quoted_at'] = '2026-10-02T12:00:00Z'
    assert quote_records(row) == []


def test_import_idempotent_and_timeline_orders_quotes_as_of_cutoff(store):
    first = match()
    second = {**first, 'captured_at': '2026-10-01T13:00:00Z', 'quoted_at': '2026-10-01T12:30:00Z', 'sp_home': 1.9}
    future = {**second, 'captured_at': '2026-10-03T13:00:00Z'}
    store.ingest([second, future, first])
    assert store.ingest([first, second])['quote_records'] == 0
    timeline = store.timeline(first, datetime(2026, 10, 1, 14, tzinfo=timezone.utc))
    records = [q for q in timeline['records'] if q['market'] == 'result']
    assert [q['odds']['home'] for q in records] == [2, 1.9]
    assert records[0]['display_kind'] == 'first_observed'
    assert not timeline['official_opening_available']
    assert timeline['available_count'] == 10


def test_result_requires_independent_90minute_result_not_prediction():
    row = match()
    row.pop('actual')
    row['prediction'] = {'predicted_score': '2-1', 'actual_score': '2-1'}
    assert result_record(row) is None
    row = match()
    row['actual']['period'] = 'extra_time'
    assert result_record(row) is None
    row = match()
    row['actual']['published_at'] = '2026-10-01T20:01:00Z'
    assert result_record(row) is None


def test_historical_cases_include_contrary_examples_and_block_lookahead(store):
    rows = [match(identity='1'), match('2026-10-02', '2', '0-1')]
    store.ingest(rows)
    target = match('2026-10-10', '3')
    cutoff = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    examples = store.similar(target, cutoff)
    assert examples['sample_count'] == 2
    assert {c['favorite_realized'] for c in examples['cases']} == {True, False}
    assert store.similar(target, cutoff, opening_only=True)['sample_count'] == 0
    assert store.similar(target, datetime(2026, 10, 1, 21, tzinfo=timezone.utc))['sample_count'] == 0
    conflict = match(identity='1', score='3-1')
    store.ingest([conflict])
    assert store.similar(target, cutoff)['sample_count'] == 1
    assert store.similar({**target, 'give_ball': 1}, cutoff)['sample_count'] == 0


def test_same_named_different_fixture_is_not_merged_or_target_leaked(store):
    a, b = match(identity='1'), match('2026-10-02', '2')
    b['home_team'], b['away_team'] = a['home_team'], a['away_team']
    store.ingest([a, b])
    assert store.coverage()['fixtures'] == 2
    assert store.similar(b, datetime(2026,10,3,tzinfo=timezone.utc))['sample_count'] == 1


def test_backfilled_quotes_are_not_prematch_observations(store):
    row = match()
    row['captured_at'] = '2026-11-01T12:00:00Z'
    store.ingest([row])
    coverage = store.coverage()
    assert coverage['retrospective_observations'] == 5
    assert coverage['prematch_observations'] == 0
    assert store.timeline(row, datetime(2026,10,2,tzinfo=timezone.utc))['available_count'] == 0


def test_file_import_resumes_and_tracks_missing_identity_and_sources(store, tmp_path):
    path = tmp_path / 'history_2026-10-01_today_morning.json'
    path.write_text(json.dumps({'matches': {'today': [match(), {'home_team':'A'}]}}), encoding='utf-8')
    result = import_archives(store, tmp_path)
    assert result[0]['missing_quote_provenance'] == 1
    assert result[0]['cached'] is False
    assert import_archives(store, tmp_path)[0]['cached'] is True
    assert import_archives(store, tmp_path, start='2026-10-02') == []
    assert store.coverage()['fixtures'] == 1


def fixture(fid, day, status='FT', fulltime=True):
    return {'fixture': {'id': fid, 'date': day+'T15:00:00Z', 'status': {'short': status}},
            'teams': {'home': {'id':1,'name':'A'}, 'away': {'id':2,'name':'B'}},
            'score': {'fulltime': {'home':2,'away':1} if fulltime else {}},
            'goals': {'home':4,'away':2}}


def test_recent_form_30days_5_10_and_90day_fallback_without_extra_time():
    rows = [fixture(1,'2026-10-10'),fixture(2,'2026-09-10'),fixture(3,'2026-07-01'),
            fixture(4,'2026-11-02'),fixture(5,'2026-10-12','NS'),
            fixture(6,'2026-10-13','AET',False)]
    rows.append(copy.deepcopy(rows[0]))
    cutoff = datetime(2026,11,1,tzinfo=timezone.utc)
    provenance = {'source':'api_football','source_url':'https://api.example/fixtures','captured_at':cutoff.isoformat()}
    form = recent_form(rows, 1, cutoff, provenance)
    assert form['last_30_days']['sample_count'] == 1
    assert form['last_5']['sample_count'] == form['last_10']['sample_count'] == 2
    assert form['last_5']['goals_for'] == 4  # only regulation score, not raw AET goals
    assert len(form['matches']) == 2
    assert recent_form(rows, 1, cutoff, {**provenance,'captured_at':'2026-11-02T00:00:00Z'})['available'] is False


def test_official_adapter_preserves_market_semantics_and_does_not_guess_opening():
    m = {'matchId':1,'matchDate':'2026-10-01','matchTime':'23:00:00','homeTeamName':'A',
         'awayTeamName':'B','leagueName':'英超','goalLine':-1,'fullFieldScore':'2:1'}
    timestamp = {'updateDate':'2026-10-01','updateTime':'10:00:00'}
    bonus = {'hadList':[{**timestamp,'h':2,'d':3,'a':4}],
             'ttgList':[{**timestamp,'s7':30}], 'hafuList':[{**timestamp,'ha':10}],
             'crsList':[{**timestamp,'h21':8,'a12':9,'hOther':40}]}
    rows = normalize_official(m,bonus,'2026-10-02T12:00:00Z')
    assert rows[0]['kickoff_at'] == '2026-10-01T23:00:00+08:00'
    quotes = [q for r in rows for q in quote_records(r)]
    assert next(q for q in quotes if q['market']=='total_goals')['odds'] == {'7+':30}
    assert next(q for q in quotes if q['market']=='half_full_time')['odds'] == {'sf':10}
    assert next(q for q in quotes if q['market']=='correct_score')['odds']['2-1'] == 8
    assert not any(q['quote_kind']=='official_opening' for q in quotes)
    assert result_record(rows[-1])['score'] == '2-1'


def test_public_source_failure_does_not_checkpoint_success(store, tmp_path):
    import requests
    class Session:
        calls = 0
        def get(self, *args, **kwargs):
            self.calls += 1
            raise requests.HTTPError('blocked')
    session = Session()
    statuses = fetch_public(store,'2026-10-01','2026-10-03',tmp_path/'public',session)
    assert session.calls == 1 and statuses[0]['status'] == 'source_unavailable'
    assert list((tmp_path/'public').glob('*.json')) == []


def test_cross_provider_result_joins_only_same_identity(store):
    row = match()
    actual = copy.deepcopy(row)
    row.pop('actual')
    actual.update(match_id='other:45', source='other')
    actual['market_quotes'] = {}
    store.ingest([row, actual])
    target = match('2026-10-10', 'target')
    result = store.similar(target, datetime(2026, 10, 10, tzinfo=timezone.utc))
    assert result['sample_count'] == 1
    assert result['cases'][0]['quote_evidence']['result']['source_url']


def test_invalid_handicap_does_not_become_zero_and_numeric_forms_agree(store):
    row = match()
    for handicap in (None, True, 'unknown', -.25):
        assert not any(q['market'] == 'handicap' for q in quote_records({**row, 'give_ball': handicap}))
    assert next(q for q in quote_records({**row, 'give_ball': '-1.0'}) if q['market'] == 'handicap')['handicap'] == -1


def test_analysis_cutoff_excludes_later_facts_and_quotes():
    from scripts.research_data import sourced_evidence
    row = match()
    row['analysis_cutoff'] = '2026-10-01T10:00:00Z'
    row['home_stats'] = {'played': 10, 'source': 'api_football', 'source_url': 'https://api.example/stats',
                         'captured_at': '2026-10-01T11:00:00Z'}
    row['intelligence'] = {'claim': '首发完整'}
    evidence = sourced_evidence(row)
    assert not markets(evidence)['result']
    assert evidence['home_stats']['available'] is False
    assert evidence['intelligence']['available'] is False
    assert row['sp_home'] == 2.0  # immutable input


def test_recent_conflicting_duplicate_results_are_excluded():
    a = fixture(1, '2026-10-10')
    b = copy.deepcopy(a)
    b['score']['fulltime']['home'] = 3
    cutoff = datetime(2026, 11, 1, tzinfo=timezone.utc)
    provenance = {'source': 'api_football', 'source_url': 'https://api.example/fixtures', 'captured_at': cutoff.isoformat()}
    assert recent_form([a,b], 1, cutoff, provenance)['last_10']['sample_count'] == 0


def test_normalized_import_markets_and_full_score_margin():
    from scripts.research_data import SCORES
    from scripts import predict
    row = match()
    row['market_quotes'] = {'correct_score': {k: 50 for k in (*SCORES, '胜其他', '平其他', '负其他')},
                            'total_goals': {str(i) if i < 7 else '7+': 8 for i in range(8)},
                            'half_full_time': {k: 9 for k in ('ss','sp','sf','ps','pp','pf','fs','fp','ff')}}
    evidence = predict.build_evidence_packet(row, 1)
    assert evidence['market_margin_audit']['correct_score']['available']
    assert evidence['total_goals_odds']['7+'] == 8
    assert evidence['half_full_time_odds']['主/负'] == 9
