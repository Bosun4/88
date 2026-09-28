"""Regressions for the overnight empty-window failure in production run 705."""
import asyncio
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fetch_data
from scripts import main
from test_wencai_post_api import FakeSession

BJ = timezone(timedelta(hours=8))
NOW = datetime(2099, 9, 29, 3, 7, tzinfo=BJ)


def fixture(day, hour=21, name='A'):
    kickoff = datetime(2099, 9, day, hour, tzinfo=BJ)
    return {'id': day * 100 + hour, 'home': name, 'guest': 'B',
            'cup': 'Fixture', 'stime': int(kickoff.timestamp())}


@pytest.fixture(autouse=True)
def schedule_env(monkeypatch):
    monkeypatch.setenv('VMAX_DATE_SHIFT_HOURS', '11')
    monkeypatch.setenv('VMAX_FETCH_DAYS_AHEAD', '0')
    monkeypatch.setenv('VMAX_DATE_MODE', 'next_available')
    monkeypatch.delenv('VMAX_TARGET_DATE', raising=False)


def test_overnight_selects_one_next_available_business_day():
    rows = [fixture(30), fixture(29), fixture(28)]
    kept, report = fetch_data.select_schedule_window(rows, '2099-09-28', now=NOW)
    assert [r['id'] for r in kept] == [2921]
    assert report['selected_date'] == '2099-09-29'
    assert report['requested_date'] == '2099-09-28'
    assert report['selection_reason'] == 'next_available'
    assert report['future_business_dates'] == ['2099-09-29', '2099-09-30']


def test_current_business_day_future_match_takes_precedence():
    kept, report = fetch_data.select_schedule_window(
        [fixture(29), fixture(29, hour=4)], '2099-09-28', now=NOW)
    assert [r['id'] for r in kept] == [2904]
    assert report['selected_date'] == '2099-09-28'


def test_explicit_date_never_silently_switches_days(monkeypatch):
    monkeypatch.setenv('VMAX_TARGET_DATE', '2099-09-28')
    kept, report = fetch_data.select_schedule_window([fixture(29)], '2099-09-28', now=NOW)
    assert kept == []
    assert report['selected_date'] == '2099-09-28'
    assert report['fixtures'][0]['reason'] == 'outside_window'
    assert report['fixtures'][0]['business_date'] == '2099-09-29'


def test_selection_does_not_resurrect_started_matches_or_far_future():
    far = {'stime': int(datetime(2099, 10, 20, 21, tzinfo=BJ).timestamp())}
    kept, report = fetch_data.select_schedule_window([fixture(28), far], '2099-09-28', now=NOW)
    assert kept == []
    assert report['fixtures'][0]['reason'] == 'already_started'


def test_manual_date_is_used_without_changing_business_day_shift(monkeypatch):
    monkeypatch.setenv('VMAX_TARGET_DATE', '2026-09-29')
    assert main.get_target_date() == '2026-09-29'
    assert main.get_target_date(1) == '2026-09-30'


@pytest.mark.parametrize('bad', ['2026-02-30', '2026-9-29', 'tomorrow', '../../data'])
def test_invalid_manual_date_is_rejected(monkeypatch, bad):
    monkeypatch.setenv('VMAX_TARGET_DATE', bad)
    with pytest.raises(ValueError):
        main.get_target_date()


def test_collector_reports_selected_date_and_fixture_diagnostics(monkeypatch):
    monkeypatch.setenv('WENCAI_AUTHORIZATION', 'test-auth')
    session = FakeSession({'code': 0, 'data': {'matches': {'1': [fixture(29)]}}})
    diagnostics = {}
    result = asyncio.run(fetch_data.scrape_wencai_jczq_async(
        session, '2099-09-28', diagnostics=diagnostics))
    assert len(result) == 1
    assert diagnostics['selected_date'] == '2099-09-29'
    assert diagnostics['source_status'] == 'ok'
    assert diagnostics['fixtures'][0]['reason'] == 'selected'


def test_source_failure_is_distinct_from_empty_schedule(monkeypatch):
    monkeypatch.setenv('WENCAI_AUTHORIZATION', 'test-auth')
    diagnostics = {}
    result = asyncio.run(fetch_data.scrape_wencai_jczq_async(
        FakeSession({'code': 301, 'msg': 'invalid'}), '2099-09-28', diagnostics=diagnostics))
    assert result == []
    assert diagnostics['source_status'] == 'source_error'


def test_auto_selection_does_not_assign_unknown_kickoff_to_selected_day():
    kept, report = fetch_data.select_schedule_window(
        [fixture(29), {'home': 'Unknown', 'stime': 'bad'}], '2099-09-28', now=NOW)
    assert [r['id'] for r in kept] == [2921]
    assert report['fixtures'][1]['reason'] == 'unknown_kickoff'


def test_selected_date_reaches_enriched_matches_and_collector_result(monkeypatch):
    monkeypatch.setenv('WENCAI_AUTHORIZATION', 'test-auth')
    monkeypatch.setattr(fetch_data, 'API_FOOTBALL_KEY', '')

    class Session(FakeSession):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(fetch_data.aiohttp, 'ClientSession', lambda: Session(
        {'code': 0, 'data': {'matches': {'1': [fixture(29)]}}}))
    import global_odds

    async def no_odds(matches):
        return matches

    monkeypatch.setattr(global_odds, 'enrich_with_global_odds_async', no_odds)
    result = asyncio.run(fetch_data.async_collect_all('2099-09-28'))
    assert result['date'] == '2099-09-29'
    assert result['matches'][0]['date'] == '2099-09-29'
    assert result['schedule']['selected_date'] == '2099-09-29'
