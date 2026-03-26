import json
import os
import re
import time
import asyncio
import aiohttp
import numpy as np
from datetime import datetime
from config import *
from models import EnsemblePredictor
from odds_engine import predict_match
from league_intel import build_league_intelligence
from experience_rules import ExperienceEngine, apply_experience_to_prediction
from advanced_models import upgrade_ensemble_predict

try:
    from odds_history import apply_odds_history
except Exception as e:
    print(f"  [WARN] odds_history加载失败,降级跳过: {e}")
    def apply_odds_history(m, mg): return mg

try:
    from quant_edge import apply_quant_edge
except Exception as e:
    print(f"  [WARN] quant_edge加载失败,降级跳过: {e}")
    def apply_quant_edge(m, mg): return mg

ensemble = EnsemblePredictor()
exp_engine = ExperienceEngine()

def calculate_value_bet(prob_pct, odds):
    if not odds or odds <= 1.05:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0:
        return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
    kelly = ((b * prob) - q) / b
    return {"ev": round(ev * 100, 2), "kelly": round(max(0.0, kelly * 0.25) * 100, 2), "is_value": ev > 0.05}

def parse_score(s):
    try:
        p = str(s).split("-")
        return int(p[0]), int(p[1])
    except:
        return None, None

def load_ai_diary():
    diary_file = "data/ai_diary.json"
    if os.path.exists(diary_file):
        try:
            with open(diary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {"yesterday_win_rate": "N/A", "reflection": "", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

def build_batch_prompt(match_analyses):
    diary = load_ai_diary()
    p = "[ROLE] Elite Football Quant AI. Analyze matches with ZERO mercy.\n"
    if diary.get("reflection"):
        p += f"[EVOLUTION] Prev: {diary.get('yesterday_win_rate','N/A')}. Log: {diary['reflection']}\n\n"
    p += "[RULES]\n1. Output ONLY raw JSON array. NO markdown.\n"
    p += "2. Fields: match(int), score(str), reason(60-110char analysis), ai_confidence(0-100), value_kill(bool), dark_verdict(one sentence)\n"
    p += "3. reason chain: public consensus trap → bookmaker xG/odds divergence → final kill verdict.\n\n"
    p += "[MATCHES]\n"
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma["engine"]
        stats = ma.get("stats", {})
        exp = ma.get("experience", {})
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", "UNK"))
        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)
        vote = m.get("vote", {})
        vh = int(vote.get("win", 33))
        vd = int(vote.get("same", 33))
        va = int(vote.get("lose", 34))
        info = m.get("intelligence", m.get("information", {}))
        bad_h = str(info.get("home_bad_news", ""))[:120]
        bad_a = str(info.get("guest_bad_news", ""))[:120]
        baseface = str(m.get("baseface", m.get("analyse", {}).get("baseface", "")))[:150]
        intro = str(m.get("expert_intro", m.get("intro", "")))[:120]
        p += f"[{i+1}] {h} vs {a} | {league} | HC:{m.get('give_ball',0)}\n"
        p += f"  Odds: H{sp_h} D{sp_d} A{sp_a} | Vote: H{vh}% D{vd}% A{va}%\n"
        p += f"  xG: H{eng.get('bookmaker_implied_home_xg','?')} A{eng.get('bookmaker_implied_away_xg','?')} | Gap:{eng.get('scissors_gap_signal','none')}\n"
        p += f"  Prob: H{eng.get('home_prob',33):.1f}% D{eng.get('draw_prob',33):.1f}% A{eng.get('away_prob',34):.1f}%\n"
        if bad_h: p += f"  H_bad: {bad_h}\n"
        if bad_a: p += f"  A_bad: {bad_a}\n"
        if baseface: p += f"  Context: {baseface}\n"
        smart = stats.get('smart_signals', [])
        if smart: p += f"  Signals: {', '.join(smart[:4])}\n"
        if exp.get("triggered_count", 0) > 0:
            p += f"  EXP: {','.join([t['name'] for t in exp.get('triggered',[])[:3]])}\n"
        p += f"  Scores: {', '.join(eng.get('top3_scores',['1-1','0-0','1-0']))}\n\n"
    p += '[OUTPUT] Exactly ' + str(len(match_analyses)) + ' JSON objects:\n'
    p += '[{"match":1,"score":"2-1","reason":"...","ai_confidence":75,"value_kill":true,"dark_verdict":"..."}]\n'
    return p

FALLBACK_URLS = [
    None,
    "https://api520.pro/v1","https://www.api520.pro/v1",
    "https://api521.pro/v1","https://www.api521.pro/v1",
    "https://api522.pro/v1","https://www.api522.pro/v1",
    "https://69.63.213.33:666/v1",
    "https://api523.pro/v1","https://api524.pro/v1"
]

def get_clean_env_url(name, default=""):
    v = os.environ.get(name, globals().get(name, default))
    v = str(v).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

async def async_call_one_ai_batch(session, prompt, url_env, key_env, models_list, num_matches, ai_name):
    key = get_clean_env_key(key_env)
    if not key: return ai_name, {}, "no_key"
    primary_url = get_clean_env_url(url_env)
    urls = [primary_url] + [u for u in FALLBACK_URLS if u and u != primary_url]
    for attempt in range(3):
        for mn in models_list:
            for base_url in urls:
                if not base_url: continue
                is_gem = "generateContent" in base_url
                url = base_url.rstrip("/")
                if not is_gem and "chat/completions" not in url:
                    url += "/chat/completions"
                headers = {"Content-Type": "application/json"}
                if is_gem:
                    headers["x-goog-api-key"] = key
                    payload = {"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.12}}
                else:
                    headers["Authorization"] = f"Bearer {key}"
                    payload = {"model":mn,"messages":[
                        {"role":"system","content":"Output ONLY valid JSON array. No markdown. reason field must be 60-110 chars analysis."},
                        {"role":"user","content":prompt}
                    ],"temperature":0.12}
                gw = url.split("/v1")[0][:40]
                print(f"  [AI] {ai_name.upper()}: {mn[:25]} @ {gw} | round {attempt+1}")
                try:
                    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=50)) as r:
                        if r.status == 200:
                            data = await r.json()
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip() if is_gem else data["choices"][0]["message"]["content"].strip()
                            clean = re.sub(r"```[\w]*","",raw_text).strip()
                            start=clean.find("[");end=clean.rfind("]")+1
                            results={}
                            if start!=-1 and end>start:
                                try:
                                    arr=json.loads(clean[start:end])
                                    if isinstance(arr,list):
                                        for item in arr:
                                            if item.get("match") and item.get("score"):
                                                results[item["match"]]={
                                                    "ai_score":item.get("score"),
                                                    "analysis":str(item.get("reason","")).strip()[:200],
                                                    "ai_confidence":int(item.get("ai_confidence",60)),
                                                    "value_kill":bool(item.get("value_kill",False)),
                                                    "dark_verdict":str(item.get("dark_verdict",""))
                                                }
                                except: pass
                            if len(results)>=max(1,num_matches*0.4):
                                print(f"    ✅ {ai_name.upper()}: {len(results)}/{num_matches} ({mn[:25]})")
                                return ai_name, results, mn
                        elif r.status==429:
                            await asyncio.sleep(2**attempt*5); continue
                        else:
                            print(f"    ⚠️ HTTP {r.status}")
                except asyncio.TimeoutError:
                    print(f"    ⏰ timeout")
                except Exception as e:
                    print(f"    ⚠️ {str(e)[:40]}")
                await asyncio.sleep(0.4)
        await asyncio.sleep(1.5)
    print(f"    ❌ {ai_name.upper()} ALL FAILED")
    return ai_name, {}, "failed"

async def run_ai_matrix(prompt, num_matches):
    ai_configs = [
        ("claude","CLAUDE_API_URL","CLAUDE_API_KEY",[
            "熊猫-按量-满血copilot-claude-opus-4.6-thinking",
            "熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking",
            "熊猫-按量-顶级特供-官max-claude-opus-4.6",
            "熊猫-特供-按量-Q-claude-opus-4.6",
            "熊猫-按量-特供顶级-官方正向满血-claude-sonnet-4.6-thinking",
            "熊猫-按量-满血copilot-claude-sonnet-4.6-thinking",
        ]),
        ("grok","GROK_API_URL","GROK_API_KEY",[
            "熊猫-A-6-grok-4.2-thinking",
            "熊猫-A-7-grok-4.2-多智能体讨论",
            
        ]),
        ("gpt","GPT_API_URL","GPT_API_KEY",[
            "熊猫-A-7-gpt-5.4",
            "熊猫-按量-gpt-5.3-codex-满血",
            "熊猫-A-10-gpt-5.3-codex",
            "熊猫-A-1-gpt-5.2",
        ]),
        ("gemini","GEMINI_API_URL","GEMINI_API_KEY",[
            "熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking",
            "熊猫-特供-X-12-gemini-3.1-pro-preview-thinking",
            "熊猫-顶级特供-X-17-gemini-3.1-pro-preview",
            "熊猫特供-按量-SSS-gemini-3.1-pro-preview",
        ]),
    ]
    all_results = {"gpt":{},"grok":{},"claude":{},"gemini":{}}
    async with aiohttp.ClientSession() as session:
        tasks = [async_call_one_ai_batch(session,prompt,ue,ke,ml,num_matches,an) for an,ue,ke,ml in ai_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
        if isinstance(res, tuple):
            ai_name, parsed, _ = res
            all_results[ai_name] = parsed
        else:
            print(f"  [CRITICAL] AI task crashed: {res}")
    return all_results

def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    sp_h=float(match_obj.get("sp_home",0) or 0)
    sp_d=float(match_obj.get("sp_draw",0) or 0)
    sp_a=float(match_obj.get("sp_away",0) or 0)
    engine_score=engine_result.get("primary_score","1-1")
    engine_conf=engine_result.get("confidence",50)
    ai_all={"gpt":gpt_r,"grok":grok_r,"gemini":gemini_r,"claude":claude_r}
    ai_scores=[];ai_conf_sum=0;ai_conf_count=0;value_kills=0
    weights={"claude":1.4,"grok":1.3,"gpt":1.1,"gemini":1.0}
    for name,r in ai_all.items():
        if not isinstance(r,dict):continue
        sc=r.get("ai_score","-")
        if sc and sc not in ["-","?",""]:
            ai_scores.append(sc)
            conf=r.get("ai_confidence",60)
            ai_conf_sum+=conf*weights.get(name,1.0)
            ai_conf_count+=weights.get(name,1.0)
            if r.get("value_kill"):value_kills+=1
    vote_count={}
    for sc in ai_scores:vote_count[sc]=vote_count.get(sc,0)+1
    final_score=engine_score
    if vote_count:
        best=max(vote_count,key=vote_count.get)
        if best in engine_result.get("top3_scores",[]) and vote_count[best]>=2:final_score=best
    avg_ai_conf=(ai_conf_sum/ai_conf_count) if ai_conf_count>0 else 60
    cf=engine_conf;cf=min(95,cf+int((avg_ai_conf-60)*0.4));cf=cf+value_kills*6
    has_warn=any("🚨" in str(s) for s in stats.get("smart_signals",[]))
    if has_warn:cf=max(35,cf-12)
    risk="低" if cf>=75 else ("中" if cf>=55 else "高")
    hp=engine_result.get("home_prob",33);dp=engine_result.get("draw_prob",33);ap=engine_result.get("away_prob",34)
    shp=stats.get("home_win_pct",33);sdp=stats.get("draw_pct",33);sap=stats.get("away_win_pct",34)
    fhp=hp*0.75+shp*0.25;fdp=dp*0.75+sdp*0.25;fap=ap*0.75+sap*0.25
    fhp=max(3,fhp);fdp=max(3,fdp);fap=max(3,fap);ft=fhp+fdp+fap
    if ft>0:fhp=round(fhp/ft*100,1);fdp=round(fdp/ft*100,1);fap=round(max(3,100-fhp-fdp),1)
    gpt_sc=gpt_r.get("ai_score","-") if isinstance(gpt_r,dict) else "-"
    gpt_an=gpt_r.get("analysis","N/A") if isinstance(gpt_r,dict) else "N/A"
    grok_sc=grok_r.get("ai_score","-") if isinstance(grok_r,dict) else "-"
    grok_an=grok_r.get("analysis","N/A") if isinstance(grok_r,dict) else "N/A"
    gem_sc=gemini_r.get("ai_score","-") if isinstance(gemini_r,dict) else "-"
    gem_an=gemini_r.get("analysis","N/A") if isinstance(gemini_r,dict) else "N/A"
    cl_sc=claude_r.get("ai_score","-") if isinstance(claude_r,dict) else engine_score
    cl_an=claude_r.get("analysis","N/A") if isinstance(claude_r,dict) else engine_result.get("reason","odds engine")
    return {
        "predicted_score":final_score,"home_win_pct":fhp,"draw_pct":fdp,"away_win_pct":fap,
        "confidence":cf,"risk_level":risk,
        "over_under_2_5":"大" if engine_result.get("over_25",50)>55 else "小",
        "both_score":"是" if engine_result.get("btts",45)>50 else "否",
        "gpt_score":gpt_sc,"gpt_analysis":gpt_an,
        "grok_score":grok_sc,"grok_analysis":grok_an,
        "gemini_score":gem_sc,"gemini_analysis":gem_an,
        "claude_score":cl_sc,"claude_analysis":cl_an,
        "ai_avg_confidence":round(avg_ai_conf,1),"value_kill_count":value_kills,
        "model_agreement":len(set(ai_scores))<=1 and len(ai_scores)>=2,
        "poisson":stats.get("poisson",{}),"refined_poisson":stats.get("refined_poisson",{}),
        "extreme_warning":engine_result.get("scissors_gap_signal",""),
        "smart_money_signal":" | ".join(stats.get("smart_signals",[])),
        "smart_signals":stats.get("smart_signals",[]),
        "model_consensus":stats.get("model_consensus",0),"total_models":stats.get("total_models",11),
        "expected_total_goals":engine_result.get("expected_goals",2.5),
        "over_2_5":engine_result.get("over_25",50),"btts":engine_result.get("btts",45),
        "top_scores":stats.get("refined_poisson",{}).get("top_scores",[]),
        "elo":stats.get("elo",{}),"random_forest":stats.get("random_forest",{}),
        "gradient_boost":stats.get("gradient_boost",{}),"neural_net":stats.get("neural_net",{}),
        "logistic":stats.get("logistic",{}),"svm":stats.get("svm",{}),"knn":stats.get("knn",{}),
        "dixon_coles":stats.get("dixon_coles",{}),"bradley_terry":stats.get("bradley_terry",{}),
        "home_form":stats.get("home_form",{}),"away_form":stats.get("away_form",{}),
        "handicap_signal":stats.get("handicap_signal",""),
        "odds_movement":stats.get("odds_movement",{}),"vote_analysis":stats.get("vote_analysis",{}),
        "h2h_blood":stats.get("h2h_blood",{}),"crs_analysis":stats.get("crs_analysis",{}),
        "ttg_analysis":stats.get("ttg_analysis",{}),"halftime":stats.get("halftime",{}),
        "pace_rating":stats.get("pace_rating",""),
        "kelly_home":stats.get("kelly_home",{}),"kelly_away":stats.get("kelly_away",{}),
        "odds":stats.get("odds",{}),
        "experience_analysis":stats.get("experience_analysis",{}),
        "pro_odds":stats.get("pro_odds",{}),
        "bivariate_poisson":stats.get("bivariate_poisson",{}),
        "asian_handicap_probs":stats.get("asian_handicap_probs",{}),
        "bookmaker_implied_home_xg":engine_result.get("bookmaker_implied_home_xg","?"),
        "bookmaker_implied_away_xg":engine_result.get("bookmaker_implied_away_xg","?"),
        "result":"主胜",  # placeholder, will be overwritten
    }

def select_top4(preds):
    for p in preds:
        pr=p.get("prediction",{});s=pr.get("confidence",0)*0.4
        mx=max(pr.get("home_win_pct",33),pr.get("away_win_pct",33),pr.get("draw_pct",33))
        s+=(mx-33)*0.2+pr.get("model_consensus",0)*2
        if pr.get("risk_level")=="低":s+=12
        elif pr.get("risk_level")=="高":s-=5
        if pr.get("model_agreement"):s+=10
        exp=pr.get("experience_analysis",{})
        if exp.get("total_score",0)>=15 and pr.get("result")=="平局" and exp.get("draw_rules",0)>=3:s+=12
        elif exp.get("total_score",0)>=10:s+=5
        if exp.get("recommendation","").startswith("⚠️"):s-=3
        sm=str(pr.get("smart_money_signal",""));d=pr.get("result","")
        if "Sharp" in sm:
            if ("客胜" in sm and d=="主胜") or ("主胜" in sm and d=="客胜"):s-=30
        # 可预测性评分加成
        ps=pr.get("predictability_score",50)
        if ps>=70:s+=8
        elif ps<35:s-=10
        # 价值投注加成
        vk=pr.get("value_kill_count",0)
        s+=vk*5
        p["recommend_score"]=round(s,2)
    preds.sort(key=lambda x:x.get("recommend_score",0),reverse=True)
    return preds[:4]

def extract_num(ms):
    wm={"一":1000,"二":2000,"三":3000,"四":4000,"五":5000,"六":6000,"日":7000,"天":7000}
    base=next((v for k,v in wm.items() if k in str(ms)),0)
    nums=re.findall(r"\d+",str(ms))
    return base+int(nums[0]) if nums else 9999

def run_predictions(raw, use_ai=True):
    ms=raw.get("matches",[])
    print("\n"+"="*80)
    print(f"  [QUANT ENGINE vMAX] {len(ms)} matches | Full Pipeline")
    print("="*80)
    match_analyses=[]
    for i,m in enumerate(ms):
        eng=predict_match(m)
        league_info,_,_,_=build_league_intelligence(m)
        sp=ensemble.predict(m,{})
        exp_result=exp_engine.analyze(m)
        match_analyses.append({"match":m,"engine":eng,"league_info":league_info,"stats":sp,"index":i+1,"experience":exp_result})

    all_ai={"claude":{},"gemini":{},"gpt":{},"grok":{}}
    if use_ai and match_analyses:
        prompt=build_batch_prompt(match_analyses)
        print(f"  [PROMPT] {len(prompt):,} chars | Calling AI Matrix...")
        t0=time.time()
        all_ai=asyncio.run(run_ai_matrix(prompt,len(match_analyses)))
        print(f"  [AI DONE] {time.time()-t0:.1f}s")

    res=[]
    for i,ma in enumerate(match_analyses):
        m=ma["match"];idx=i+1
        mg=merge_result(ma["engine"],all_ai["gpt"].get(idx,{}),all_ai["grok"].get(idx,{}),all_ai["gemini"].get(idx,{}),all_ai["claude"].get(idx,{}),ma["stats"],m)

        # ============ 4层增强管线（每层独立try/except防崩溃） ============
        try:
            mg=apply_experience_to_prediction(m,mg,exp_engine)
            print(f"    → experience_rules 已注入")
        except Exception as e:
            print(f"    ⚠️ experience_rules跳过: {e}")

        try:
            mg=apply_odds_history(m,mg)
            print(f"    → odds_history 已注入")
        except Exception as e:
            print(f"    ⚠️ odds_history跳过: {e}")

        try:
            mg=apply_quant_edge(m,mg)
            print(f"    → quant_edge 已注入")
        except Exception as e:
            print(f"    ⚠️ quant_edge跳过: {e}")

        try:
            mg=upgrade_ensemble_predict(m,mg)
            print(f"    → advanced_models 已注入")
        except Exception as e:
            print(f"    ⚠️ advanced_models跳过: {e}")
        # ================================================================

        pcts={"主胜":mg.get("home_win_pct",33),"平局":mg.get("draw_pct",33),"客胜":mg.get("away_win_pct",34)}
        mg["result"]=max(pcts,key=pcts.get)
        res.append({**m,"prediction":mg})
        print(f"  [{idx}] {m.get('home_team','?')} vs {m.get('away_team','?')} => {mg['result']} ({mg['predicted_score']}) CF:{mg['confidence']}%")

    t4=select_top4(res)
    t4ids=[t["id"] for t in t4]
    for r in res:r["is_recommended"]=r["id"] in t4ids
    res.sort(key=lambda x:extract_num(x.get("match_num","")))

    diary=load_ai_diary()
    diary["yesterday_win_rate"]=f"{len([r for r in res if r['prediction']['confidence']>70])}/{max(1,len(res))}"
    diary["reflection"]="AI矩阵+历史匹配+量化边缘+经验规则+高级模型全链路运行"
    save_ai_diary(diary)
    return res,t4