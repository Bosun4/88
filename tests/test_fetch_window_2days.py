# -*- coding: utf-8 -*-
"""抓取窗口测试：问财接口一次返回跨多日赛程，只保留今天+未来 N 天，砍掉昨天及更远。

背景缺陷：scrape_wencai_jczq_async 用 ?date=X 请求，但接口一次性返回跨周二/三/四
的全部场次（实测 6/17 查询返回 12 场跨三天），全部进 enrich + AI 终审 = 重复抓 + token 浪费。

修复原则：用每场 stime（开赛 Unix 秒）按竞彩业务日口径（VMAX_DATE_SHIFT_HOURS=11 偏移）
算业务日，只留 [today, today+days_ahead]。stime 缺失/解析失败的场次保留（fail-safe 不误杀）。
"""
import os
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import fetch_data

BJ = timezone(timedelta(hours=8))
SHIFT = 11  # 与 main.py VMAX_DATE_SHIFT_HOURS 默认一致


def _stime_for_bizday(day_str, hour=21):
    """构造一个开赛时间，使其竞彩业务日 == day_str（hour 为业务日当天北京时段）。"""
    # 业务日 day_str 21:00 北京 -> 该时刻减 shift 仍在同一业务日
    dt = datetime.strptime(day_str, "%Y-%m-%d").replace(hour=hour, tzinfo=BJ)
    return int(dt.timestamp())


def _match(day_str=None, stime=None, name="m"):
    m = {"home": name, "guest": name + "_a"}
    if stime is not None:
        m["stime"] = stime
    elif day_str is not None:
        m["stime"] = _stime_for_bizday(day_str)
    return m


def test_keeps_today_and_tomorrow_drops_yesterday_and_far():
    today = "2026-06-17"
    yesterday = "2026-06-16"
    tomorrow = "2026-06-18"
    day_after = "2026-06-19"
    matches = [
        _match(yesterday, name="yest"),
        _match(today, name="today"),
        _match(tomorrow, name="tom"),
        _match(day_after, name="far"),
    ]
    kept = fetch_data.filter_matches_by_window(matches, today=today, days_ahead=1, shift_hours=SHIFT)
    names = {m["home"] for m in kept}
    assert names == {"today", "tom"}


def test_days_ahead_zero_keeps_only_today():
    today = "2026-06-17"
    matches = [
        _match(today, name="today"),
        _match("2026-06-18", name="tom"),
    ]
    kept = fetch_data.filter_matches_by_window(matches, today=today, days_ahead=0, shift_hours=SHIFT)
    assert {m["home"] for m in kept} == {"today"}


def test_days_ahead_two_keeps_three_days():
    today = "2026-06-17"
    matches = [
        _match("2026-06-16", name="yest"),
        _match(today, name="today"),
        _match("2026-06-18", name="d1"),
        _match("2026-06-19", name="d2"),
        _match("2026-06-20", name="d3"),
    ]
    kept = fetch_data.filter_matches_by_window(matches, today=today, days_ahead=2, shift_hours=SHIFT)
    assert {m["home"] for m in kept} == {"today", "d1", "d2"}


def test_missing_or_bad_stime_is_kept_failsafe():
    today = "2026-06-17"
    matches = [
        _match("2026-06-16", name="yest_drop"),   # 昨天，砍
        {"home": "no_stime", "guest": "x"},          # 无 stime -> 保留
        _match(stime=0, name="zero_stime"),          # stime=0 -> 保留
        _match(stime="garbage", name="bad_stime"),   # 非法 -> 保留
    ]
    kept = fetch_data.filter_matches_by_window(matches, today=today, days_ahead=1, shift_hours=SHIFT)
    names = {m["home"] for m in kept}
    assert "yest_drop" not in names
    assert {"no_stime", "zero_stime", "bad_stime"} <= names


def test_late_night_match_business_day_not_naive_calendar_day():
    """6/17 凌晨开赛的场次（北京 6/17 03:00），业务日仍算 6/16，
    若 today=6/16 应保留（它是 6/16 业务日的深夜场，不该被当成"明天"砍掉）。"""
    today = "2026-06-16"
    # 北京 6/17 03:00 -> 减 11h = 6/16 16:00 -> 业务日 6/16
    stime = int(datetime(2026, 6, 17, 3, 0, tzinfo=BJ).timestamp())
    matches = [_match(stime=stime, name="late")]
    kept = fetch_data.filter_matches_by_window(matches, today=today, days_ahead=1, shift_hours=SHIFT)
    assert {m["home"] for m in kept} == {"late"}


def test_env_default_days_ahead_is_one(monkeypatch=None):
    """未显式传 days_ahead 时，默认窗口 = 今天 + 明天（VMAX_FETCH_DAYS_AHEAD 默认 1）。"""
    today = "2026-06-17"
    os.environ.pop("VMAX_FETCH_DAYS_AHEAD", None)
    matches = [
        _match("2026-06-16", name="yest"),
        _match(today, name="today"),
        _match("2026-06-18", name="tom"),
        _match("2026-06-19", name="far"),
    ]
    kept = fetch_data.filter_matches_by_window(matches, today=today, shift_hours=SHIFT)
    assert {m["home"] for m in kept} == {"today", "tom"}
