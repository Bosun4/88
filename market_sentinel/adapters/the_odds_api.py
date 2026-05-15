import os
import json
import urllib.request
import urllib.error
import argparse
from datetime import datetime, timezone
from dataclasses import asdict
from typing import List, Dict, Any, Optional
from market_sentinel.schemas import MarketSnapshot
from market_sentinel.fair_probs import fair_probs_from_1x2
from market_sentinel.divergence import euro_asian_divergence_index, classify_divergence
from market_sentinel.alerts import MarketAlert, generate_alert_json

def load_api_key() -> Optional[str]:
    return os.environ.get("THE_ODDS_API_KEY")

def fetch_odds(sport_key: str, regions: str, markets: str, api_key: Optional[str] = None) -> List[Dict]:
    if not api_key:
        api_key = load_api_key()
    if not api_key:
        raise ValueError("THE_ODDS_API_KEY not found in environment and not provided.")
        
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={regions}&markets={markets}&oddsFormat=decimal"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data
    except urllib.error.URLError as e:
        print(f"Error fetching odds: {e}")
        return []

def parse_the_odds_api_events_to_snapshots(events: List[Dict]) -> List[MarketSnapshot]:
    snapshots = []
    now_utc = datetime.now(timezone.utc).isoformat()
    
    for event in events:
        match_id = event.get("id", "unknown_match")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        
        for bookmaker in event.get("bookmakers", []):
            bm_name = bookmaker.get("key", "unknown_bookmaker")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                market_type = ""
                
                if market_key == "h2h":
                    market_type = "1x2"
                    # pre-calculate fair probs for this bookmaker's h2h
                    odds_dict = {}
                    for outcome in market.get("outcomes", []):
                        nm = outcome.get("name")
                        price = outcome.get("price", 0)
                        if nm == home_team: odds_dict["h"] = price
                        elif nm == "Draw": odds_dict["d"] = price
                        elif nm == away_team: odds_dict["a"] = price
                        
                    fair = fair_probs_from_1x2(odds_dict, method="power")
                    
                    for outcome in market.get("outcomes", []):
                        nm = outcome.get("name")
                        price = outcome.get("price", 0)
                        raw_prob = 1 / price if price > 0 else 0
                        
                        sel = ""
                        fair_p = 0
                        if nm == home_team: 
                            sel = "home"
                            fair_p = fair.get("h", 0)
                        elif nm == "Draw": 
                            sel = "draw"
                            fair_p = fair.get("d", 0)
                        elif nm == away_team: 
                            sel = "away"
                            fair_p = fair.get("a", 0)
                            
                        snapshots.append(MarketSnapshot(
                            match_id=match_id,
                            source="the_odds_api",
                            timestamp_utc=now_utc,
                            minutes_to_kickoff=0, # Would need commence_time parsing
                            market_type=market_type,
                            selection=sel,
                            odds=price,
                            implied_prob_raw=raw_prob,
                            implied_prob_fair=fair_p,
                            bookmaker=bm_name
                        ))
                        
                elif market_key == "spreads":
                    market_type = "asian_handicap"
                    for outcome in market.get("outcomes", []):
                        nm = outcome.get("name")
                        price = outcome.get("price", 0)
                        point = outcome.get("point")
                        raw_prob = 1 / price if price > 0 else 0
                        
                        sel = "home" if nm == home_team else ("away" if nm == away_team else nm)
                        snapshots.append(MarketSnapshot(
                            match_id=match_id,
                            source="the_odds_api",
                            timestamp_utc=now_utc,
                            minutes_to_kickoff=0,
                            market_type=market_type,
                            selection=sel,
                            odds=price,
                            implied_prob_raw=raw_prob,
                            implied_prob_fair=0.0, # difficult to calculate isolated
                            line=point,
                            bookmaker=bm_name
                        ))
                        
                elif market_key == "totals":
                    market_type = "totals"
                    for outcome in market.get("outcomes", []):
                        nm = outcome.get("name")
                        price = outcome.get("price", 0)
                        point = outcome.get("point")
                        raw_prob = 1 / price if price > 0 else 0
                        
                        snapshots.append(MarketSnapshot(
                            match_id=match_id,
                            source="the_odds_api",
                            timestamp_utc=now_utc,
                            minutes_to_kickoff=0,
                            market_type=market_type,
                            selection=nm.lower(),
                            odds=price,
                            implied_prob_raw=raw_prob,
                            implied_prob_fair=0.0,
                            line=point,
                            bookmaker=bm_name
                        ))
    return snapshots

def save_snapshots_jsonl(snapshots: List[MarketSnapshot], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in snapshots:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

def get_mock_events():
    return [
        {
            "id": "mock_match_123",
            "sport_key": "soccer_epl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 1.95},
                                {"name": "Draw", "price": 3.60},
                                {"name": "Chelsea", "price": 4.00}
                            ]
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Arsenal", "price": 2.05, "point": -0.5},
                                {"name": "Chelsea", "price": 1.85, "point": 0.5}
                            ]
                        }
                    ]
                }
            ]
        }
    ]

def generate_alerts_from_snapshots(snapshots: List[MarketSnapshot]) -> List[MarketAlert]:
    alerts = []
    matches = {}
    for s in snapshots:
        if s.match_id not in matches:
            matches[s.match_id] = {"1x2": {}, "ah": {}}
        if s.market_type == "1x2" and s.implied_prob_fair > 0:
            if s.selection == "home": matches[s.match_id]["1x2"]["h"] = s.implied_prob_fair
            elif s.selection == "away": matches[s.match_id]["1x2"]["a"] = s.implied_prob_fair
        elif s.market_type == "asian_handicap":
            if s.selection == "home": 
                matches[s.match_id]["ah"]["line"] = s.line
                matches[s.match_id]["ah"]["odds_h"] = s.odds
            elif s.selection == "away":
                matches[s.match_id]["ah"]["odds_a"] = s.odds
                
    now_utc = datetime.now(timezone.utc).isoformat()
    
    for match_id, data in matches.items():
        if "h" in data["1x2"] and "line" in data["ah"] and "odds_h" in data["ah"] and "odds_a" in data["ah"]:
            fair_h = data["1x2"]["h"]
            line = data["ah"]["line"]
            odds_h = data["ah"]["odds_h"]
            odds_a = data["ah"]["odds_a"]
            
            di_home, di_abs = euro_asian_divergence_index(fair_h, line, odds_h, odds_a)
            sev = classify_divergence(di_abs)
            
            if sev != "none":
                alerts.append(MarketAlert(
                    alert_type="EuroAsianDivergence",
                    severity=sev,
                    direction="home" if di_home > 0 else "away",
                    evidence=f"Match {match_id}: DI_abs={di_abs:.3f}. 1x2_home_fair={fair_h:.3f}, AH_line={line}",
                    source="the_odds_api_adapter",
                    timestamp_utc=now_utc
                ))
    return alerts

def run_the_odds_api_adapter(sport: str, regions: str, markets: str, out: str, mock: bool):
    if mock:
        events = get_mock_events()
        print("Using MOCK The Odds API events.")
    else:
        try:
            events = fetch_odds(sport, regions, markets)
            print(f"Fetched {len(events)} events from The Odds API.")
        except Exception as e:
            print(f"Failed to fetch from API: {e}")
            return
            
    snapshots = parse_the_odds_api_events_to_snapshots(events)
    save_snapshots_jsonl(snapshots, out)
    print(f"Saved {len(snapshots)} MarketSnapshots to {out}")
    
    alerts = generate_alerts_from_snapshots(snapshots)
    
    alert_out_json = out.replace("snapshots", "alerts").replace(".jsonl", "_alerts.json")
    alert_out_md = out.replace("snapshots", "alerts").replace(".jsonl", "_alerts.md")
    
    os.makedirs(os.path.dirname(alert_out_json), exist_ok=True)
    with open(alert_out_json, "w", encoding="utf-8") as f:
        f.write(generate_alert_json(alerts))
        
    with open(alert_out_md, "w", encoding="utf-8") as f:
        f.write("# The Odds API Market Alerts\n\n")
        if not alerts:
            f.write("No alerts generated.\n")
        for a in alerts:
            f.write(f"- **[{a.severity.upper()}]** {a.alert_type} ({a.direction}): {a.evidence}\n")
            
    print(f"Generated {len(alerts)} alerts -> {alert_out_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="soccer_epl")
    parser.add_argument("--regions", default="uk,eu")
    parser.add_argument("--markets", default="h2h,spreads,totals")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mock", action="store_true")
    
    args = parser.parse_args()
    run_the_odds_api_adapter(args.sport, args.regions, args.markets, args.out, args.mock)
