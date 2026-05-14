import json
import math
import os

class MarketSentinelOffline:
    def __init__(self):
        pass

    def fair_probs_from_1x2(self, odds_h, odds_d, odds_a):
        # Multiplicative margin removal
        margin = (1/odds_h) + (1/odds_d) + (1/odds_a)
        if margin <= 1.0: return 1/odds_h, 1/odds_d, 1/odds_a
        return (1/odds_h)/margin, (1/odds_d)/margin, (1/odds_a)/margin

    def euro_asian_divergence_index(self, fair_h_prob, ah_line, ah_odds_h, ah_odds_a):
        # A rough proxy for expected probability derived from Asian Handicap line and odds.
        # Simple proxy: base 50% + line adjustment
        base_expected = 0.50 - (ah_line * 0.15) 
        
        margin = (1/ah_odds_h) + (1/ah_odds_a)
        ah_fair_h = (1/ah_odds_h) / margin
        
        proxy_prob = base_expected + (ah_fair_h - 0.50)
        di_home = fair_h_prob - proxy_prob
        return di_home, abs(di_home)

    def late_steam_detector(self, snapshots):
        if len(snapshots) < 2:
            return {"flags": [], "steam_strength": 0, "evidence": "Not enough data"}
        
        latest = snapshots[-1]
        t_minus_5 = snapshots[0]
        
        f_h_now, _, _ = self.fair_probs_from_1x2(latest["odds_h"], latest["odds_d"], latest["odds_a"])
        f_h_old, _, _ = self.fair_probs_from_1x2(t_minus_5["odds_h"], t_minus_5["odds_d"], t_minus_5["odds_a"])
        
        prob_velocity = f_h_now - f_h_old
        vol_velocity = latest.get("volume", 0) - t_minus_5.get("volume", 0)
        
        flags = []
        delta_pp = abs(prob_velocity) * 100
        
        if delta_pp >= 10:
            flags.append("steam_critical")
        elif delta_pp >= 7:
            flags.append("steam_warning")
        elif delta_pp >= 4:
            flags.append("steam_watch")
            
        if flags and vol_velocity > 10000:
            flags.append("sharp_confirmed")
            
        return {
            "flags": flags,
            "steam_direction": "home" if prob_velocity > 0 else "away",
            "steam_strength": delta_pp,
            "steam_window": "5m",
            "evidence": f"Fair prob changed from {f_h_old:.3f} to {f_h_now:.3f}. Volume delta {vol_velocity}."
        }

    def process_match(self, match_data):
        alerts = []
        
        # 1. Divergence
        fh, fd, fa = self.fair_probs_from_1x2(match_data["1x2"]["h"], match_data["1x2"]["d"], match_data["1x2"]["a"])
        di, di_abs = self.euro_asian_divergence_index(fh, match_data["ah"]["line"], match_data["ah"]["odds_h"], match_data["ah"]["odds_a"])
        
        div_flag = "none"
        if di_abs >= 0.12: div_flag = "critical"
        elif di_abs >= 0.08: div_flag = "warning"
        elif di_abs >= 0.05: div_flag = "watch"
        
        if div_flag != "none":
            alerts.append({
                "type": "Euro-Asian Divergence",
                "level": div_flag,
                "detail": f"DI_abs={di_abs:.3f}. 1X2 Prob={fh:.3f}, but AH implies different expectation. (欧赔造热/亚盘阻力 可能)"
            })
            
        # 2. Late Steam
        steam_res = self.late_steam_detector(match_data["snapshots_5m"])
        if steam_res["flags"]:
            alerts.append({
                "type": "Late Steam",
                "level": steam_res["flags"][-1] if len(steam_res["flags"]) else "info",
                "detail": f"{steam_res['steam_direction']} side. {steam_res['evidence']}",
                "tags": steam_res["flags"]
            })
            
        return {
            "match_id": match_data["match_id"],
            "fair_home_prob": round(fh, 3),
            "divergence_index": round(di, 3),
            "alerts": alerts
        }

if __name__ == "__main__":
    sentinel = MarketSentinelOffline()
    
    mock_data = {
        "match_id": "test_girona_vs_sociedad",
        "1x2": {"h": 1.85, "d": 3.40, "a": 4.50},
        "ah": {"line": -0.5, "odds_h": 2.05, "odds_a": 1.85},
        "snapshots_5m": [
            {"time": "T-5", "odds_h": 2.10, "odds_d": 3.40, "odds_a": 3.80, "volume": 50000},
            {"time": "T-0", "odds_h": 1.85, "odds_d": 3.40, "odds_a": 4.50, "volume": 120000}
        ]
    }
    
    result = sentinel.process_match(mock_data)
    
    out_dir = "/root/.openclaw/workspace/repos/88/reports/market_sentinel/alerts"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "example_market_alerts.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    md_content = f"""# Market Alerts Report
## Match: {result['match_id']}
- **Fair Home Prob**: {result['fair_home_prob']}
- **Divergence Index**: {result['divergence_index']}

### Alerts:
"""
    for a in result['alerts']:
        md_content += f"- **[{a['level'].upper()}]** {a['type']}: {a['detail']}\n"
        
    with open(os.path.join(out_dir, "example_market_alerts.md"), "w") as f:
        f.write(md_content)
        
    print("Market Sentinel prototype generated alerts successfully.")
