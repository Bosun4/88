import json
import os
from datetime import datetime, timezone
from market_sentinel.fair_probs import fair_probs_from_1x2
from market_sentinel.divergence import euro_asian_divergence_index, classify_divergence
from market_sentinel.steam import detect_late_steam
from market_sentinel.alerts import MarketAlert, generate_alert_json

def run_demo():
    now_utc = datetime.now(timezone.utc).isoformat()
    
    mock_1x2 = {"h": 1.85, "d": 3.40, "a": 4.50}
    fair_probs = fair_probs_from_1x2(mock_1x2, method="power")
    fair_h = fair_probs["h"]
    
    di_home, di_abs = euro_asian_divergence_index(fair_h, handicap_line=-0.5, handicap_odds_h=2.05, handicap_odds_a=1.85)
    div_sev = classify_divergence(di_abs)
    
    alerts = []
    
    if div_sev != "none":
        alerts.append(MarketAlert(
            alert_type="EuroAsianDivergence",
            severity=div_sev,
            direction="home" if di_home > 0 else "away",
            evidence=f"DI_abs={di_abs:.3f}. Euro prob={fair_h:.3f}",
            source="offline_detector",
            timestamp_utc=now_utc
        ))
        
    old_fair_h = 0.430
    steam_res = detect_late_steam(fair_h, old_fair_h, volume_t=120000, volume_t_minus=50000, window_mins=5)
    
    if steam_res["late_steam_flags"]:
        alerts.append(MarketAlert(
            alert_type="LateSteam",
            severity=steam_res["late_steam_flags"][-1] if "steam" in steam_res["late_steam_flags"][-1] else steam_res["late_steam_flags"][0],
            direction=steam_res["steam_direction"],
            evidence=steam_res["steam_evidence"] + f" Flags: {steam_res['late_steam_flags']}",
            source="offline_detector",
            timestamp_utc=now_utc
        ))

    out_dir = "reports/market_sentinel/alerts"
    os.makedirs(out_dir, exist_ok=True)
    
    out_json = os.path.join(out_dir, "example_market_alerts.json")
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(generate_alert_json(alerts))
        
    out_md = os.path.join(out_dir, "example_market_alerts.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Market Alerts Demo\n\n")
        for a in alerts:
            f.write(f"- **[{a.severity.upper()}]** {a.alert_type} ({a.direction}): {a.evidence}\n")
            
    print("Demo execution completed. Check reports/market_sentinel/alerts/")

if __name__ == "__main__":
    run_demo()
