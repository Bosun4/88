#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-spectrum audit of project88 live predictions (data/predictions.json today slate)
vs verified real results (reports/actuals_worldcup_20260620.json).

Deterministic, no network, no secrets. Implements football-quant-research audit lens:
direction / exact-score / goal-band / BTTS / RPS / Brier / confidence calibration /
market-implied vs AI-claimed probability divergence / risk-flag retrospective.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data/predictions.json"
ACT = ROOT / "reports/actuals_worldcup_20260620.json"
DIR_IDX = {"home": 0, "draw": 1, "away": 2}


def parse_score(s):
    try:
        h, a = s.split("-"); return int(h), int(a)
    except Exception:
        return None


def adir(h, a):
    return "home" if h > a else ("away" if a > h else "draw")


def band(g):
    if g is None: return None
    if g <= 1: return "0-1"
    if g <= 3: return "2-3"
    return "4+"


def rps(probs, d):
    obs = [0.0, 0.0, 0.0]; obs[DIR_IDX[d]] = 1.0
    cp = co = s = 0.0
    for i in range(3):
        cp += probs[i]; co += obs[i]; s += (cp - co) ** 2
    return s / 2.0


def brier(probs, d):
    obs = [1.0 if i == DIR_IDX[d] else 0.0 for i in range(3)]
    return sum((probs[i] - obs[i]) ** 2 for i in range(3))


def implied_from_odds(r):
    """Devig 1X2 from global odds fields if present."""
    h = r.get("global_home"); d = r.get("global_draw"); a = r.get("global_away")
    try:
        h = float(h); d = float(d); a = float(a)
        if min(h, d, a) <= 1.0: return None
        ih, idr, ia = 1/h, 1/d, 1/a
        s = ih + idr + ia
        if s <= 0: return None
        return (ih/s, idr/s, ia/s)
    except Exception:
        return None


def load_actuals(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("results", obj) if isinstance(obj, dict) else {}


def parse_args():
    ap = argparse.ArgumentParser(description="Audit prediction snapshot vs verified actual results.")
    ap.add_argument("--pred", default=str(PRED), help="Prediction JSON snapshot path. Defaults to data/predictions.json for backward compatibility.")
    ap.add_argument("--actual", default=str(ACT), help="Actual-results JSON path. Supports either {'results': {...}} or a direct mapping.")
    ap.add_argument("--out", default=str(ROOT / "reports/audit_worldcup_live_20260620.json"), help="Output JSON path.")
    return ap.parse_args()


def main():
    args = parse_args()
    pred_path = Path(args.pred)
    actual_path = Path(args.actual)
    out_path = Path(args.out)

    act = load_actuals(actual_path)
    obj = json.loads(pred_path.read_text(encoding="utf-8"))
    rows = obj.get("matches", {}).get("today", [])

    scored = []
    for r in rows:
        key = f"{r.get('home_team')}||{r.get('away_team')}"
        if key not in act:
            continue
        sc = act[key]; ps = parse_score(sc)
        ah, aa = ps; ad = adir(ah, aa)
        p = r.get("prediction") or {}
        pscore = p.get("predicted_score"); pdir = p.get("final_direction")
        hp, dp, ap = p.get("home_win_pct"), p.get("draw_pct"), p.get("away_win_pct")
        probs = None; conf = None
        if None not in (hp, dp, ap):
            tot = hp + dp + ap
            if tot > 0:
                probs = (hp/tot, dp/tot, ap/tot)
                conf = max(hp, dp, ap)
        pp = parse_score(pscore) if pscore else None
        pg = (pp[0]+pp[1]) if pp else None
        ag = ah + aa
        imp = implied_from_odds(r)
        scored.append({
            "key": key, "match_num": r.get("match_num"), "actual": sc, "actual_dir": ad,
            "pred_score": pscore, "pred_dir": pdir,
            "dir_hit": pdir == ad, "score_hit": pp == (ah, aa) if pp else False,
            "band_hit": band(pg) == band(ag) if pg is not None else None,
            "btts_actual": ah > 0 and aa > 0, "btts_pred": (pp[0] > 0 and pp[1] > 0) if pp else None,
            "conf": conf, "ai_probs": probs, "implied": imp,
            "rps": rps(probs, ad) if probs else None, "brier": brier(probs, ad) if probs else None,
            "is_recommended": r.get("is_recommended"),
        })

    n = len(scored)
    if not n:
        out = {"n": 0, "pred_path": str(pred_path), "actual_path": str(actual_path), "rows": []}
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"No matched rows. Wrote {out_path}")
        return
    dh = sum(1 for s in scored if s["dir_hit"])
    sh = sum(1 for s in scored if s["score_hit"])
    bh = sum(1 for s in scored if s["band_hit"])
    bt = sum(1 for s in scored if s["btts_pred"] == s["btts_actual"])
    rv = [s["rps"] for s in scored if s["rps"] is not None]
    bv = [s["brier"] for s in scored if s["brier"] is not None]

    print(f"=== 项目88 预测快照审计 · 已匹配 {n} 场 ===")
    print(f"Prediction snapshot: {pred_path}")
    print(f"Actual results:       {actual_path}")
    print(f"方向命中:  {dh}/{n} = {dh/n*100:.1f}%")
    print(f"精确比分:  {sh}/{n} = {sh/n*100:.1f}%")
    print(f"进球带:    {bh}/{n} = {bh/n*100:.1f}%")
    print(f"BTTS:      {bt}/{n} = {bt/n*100:.1f}%")
    if rv: print(f"平均RPS:   {sum(rv)/len(rv):.4f}")
    if bv: print(f"平均Brier: {sum(bv)/len(bv):.4f}")
    print()
    print("逐场:")
    for s in scored:
        f = "✓" if s["dir_hit"] else "✗"
        sf = "S" if s["score_hit"] else " "
        ai = s["ai_probs"]; im = s["implied"]
        aistr = f"AI[{ai[0]*100:.0f}/{ai[1]*100:.0f}/{ai[2]*100:.0f}]" if ai else "AI[?]"
        imstr = f"盘口[{im[0]*100:.0f}/{im[1]*100:.0f}/{im[2]*100:.0f}]" if im else "盘口[?]"
        print(f"  {f}{sf} {s['match_num']} {s['key']:<22} 预测 {s['pred_score']:>4} {s['pred_dir']:<5} conf{(s['conf'] or 0):>4.0f}  实际 {s['actual']:>4} {s['actual_dir']:<5} | {aistr} {imstr}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n": n, "pred_path": str(pred_path), "actual_path": str(actual_path), "dir_hit": dh, "score_hit": sh, "rows": scored}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
