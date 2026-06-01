# -*- coding: utf-8 -*-
"""
steam_tracker 单元回归 [P1]

覆盖: 去抽水 / over2.5 概率抽取(Pinnacle优先+中位数回退) / event队名匹配 /
steam 四态(up/down/flat/unknown) / 联赛分层赋权 / graceful 降级。
全离线, 不打网络。
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import steam_tracker as st


# ---------- 去抽水 ----------

def test_devig_two_way_basic():
    # over 2.0 / under 2.0 -> 去 vig 后 over 概率 = 0.5
    assert abs(st._devig_two_way(2.0, 2.0) - 0.5) < 1e-9


def test_devig_two_way_skew():
    p = st._devig_two_way(1.5, 3.0)  # over 更热
    assert p is not None and 0.6 < p < 0.7


def test_devig_invalid():
    assert st._devig_two_way(0, 2.0) is None
    assert st._devig_two_way(2.0, 1.0) is None


# ---------- over2.5 概率抽取 ----------

def _ev(bookmakers):
    return {"home_team": "A FC", "away_team": "B FC", "bookmakers": bookmakers}


def _bm(key, over, under, point=2.5):
    return {
        "key": key,
        "markets": [{
            "key": "totals",
            "outcomes": [
                {"name": "Over", "price": over, "point": point},
                {"name": "Under", "price": under, "point": point},
            ],
        }],
    }


def test_extract_over_prefers_pinnacle():
    ev = _ev([
        _bm("pinnacle", 1.8, 2.0),
        _bm("bet365", 2.5, 1.5),   # 应被忽略(有 Pinnacle)
    ])
    p = st.extract_over_prob(ev)
    expect = st._devig_two_way(1.8, 2.0)
    assert abs(p - expect) < 1e-9


def test_extract_over_median_fallback():
    ev = _ev([
        _bm("bet365", 1.9, 1.9),
        _bm("williamhill", 2.1, 1.7),
    ])
    p = st.extract_over_prob(ev)
    assert p is not None and 0.0 < p < 1.0


def test_extract_over_missing_totals_returns_none():
    ev = {"home_team": "A", "away_team": "B", "bookmakers": [
        {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": []}]}
    ]}
    assert st.extract_over_prob(ev) is None


def test_extract_over_ignores_other_points():
    # 只有 3.5 线, 没有 2.5 -> None
    ev = _ev([_bm("pinnacle", 1.8, 2.0, point=3.5)])
    assert st.extract_over_prob(ev) is None


# ---------- event 匹配 ----------

def test_match_event_by_home():
    events = [
        {"home_team": "Bodo/Glimt", "away_team": "Molde", "bookmakers": []},
        {"home_team": "Rosenborg", "away_team": "Viking", "bookmakers": []},
    ]
    # translate_team_name 退化时原样返回, 用英文名直接测匹配
    ev = st._match_event("Rosenborg", "Viking", events)
    assert ev is not None and ev["home_team"] == "Rosenborg"


def test_match_event_no_hit():
    events = [{"home_team": "Xyz United", "away_team": "Qwe City", "bookmakers": []}]
    assert st._match_event("Totally Different", "Nope", events) is None


# ---------- steam 四态 + 赋权 ----------

def test_steam_up():
    s = st.compute_steam(0.50, 0.62, "挪超")
    assert s["steam_available"] is True
    assert s["steam_direction"] == "up"
    assert abs(s["steam_strength"] - 0.12) < 1e-6
    # 挪超权重 1.20
    assert abs(s["steam_weighted"] - 0.144) < 1e-6


def test_steam_down():
    s = st.compute_steam(0.55, 0.40, "意甲")
    assert s["steam_direction"] == "down"
    assert s["league_weight"] == 0.95


def test_steam_flat_below_eps():
    s = st.compute_steam(0.50, 0.505, "德甲")
    assert s["steam_direction"] == "flat"  # |Δ|=0.005 < EPS(0.015)


def test_steam_unknown_when_missing():
    s = st.compute_steam(None, 0.6, "芬超")
    assert s["steam_available"] is False
    assert s["steam_direction"] == "unknown"
    # 芬超未标定 -> 默认权重
    assert s["league_weight"] == st.DEFAULT_LEAGUE_WEIGHT


def test_unlisted_league_default_weight():
    s = st.compute_steam(0.40, 0.60, "沙特联")
    assert s["steam_direction"] == "up"
    assert s["league_weight"] == st.DEFAULT_LEAGUE_WEIGHT  # 不覆盖也不崩


# ---------- 快照 IO ----------

def test_snapshot_roundtrip(tmp_path):
    p = tmp_path / "snap.json"
    import json
    p.write_text(json.dumps({"ts": 1, "over25": {"A__B": 0.55}}), encoding="utf-8")
    d = st.load_snapshot(str(p))
    assert d.get("A__B") == 0.55


def test_load_snapshot_missing_file():
    assert st.load_snapshot("/nonexistent/path/xyz.json") == {}


# ---------- graceful 降级: 无 key 时 build 不崩 ----------

def test_build_signals_no_network_returns_neutral(monkeypatch):
    # 强制 fetch 返回空(模拟无 key / 网络失败), 不应抛异常
    monkeypatch.setattr(st, "fetch_totals_snapshot", lambda *a, **k: [])
    matches = [{"home_team": "A", "away_team": "B", "league": "挪超", "match_id": "m1"}]
    out = st.build_steam_signals(matches)
    assert "m1" in out
    assert out["m1"]["steam_available"] is False
    assert out["m1"]["steam_direction"] == "unknown"
