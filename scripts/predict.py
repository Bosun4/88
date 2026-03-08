import json,requests,time
from config import *
from models import EnsemblePredictor

ensemble=EnsemblePredictor()

def build_prompt(m,stats_pred):
    h=m["home_team"];a=m["away_team"];lg=m.get("league","")
    hs=m.get("home_stats",{});ast=m.get("away_stats",{});h2h=m.get("h2h",[])
    sp=stats_pred;poi=sp.get("poisson",{});elo=sp.get("elo",{});mc=sp.get("monte_carlo",{})
    rf=sp.get("random_forest",{});gb=sp.get("gradient_boost",{});nn=sp.get("neural_net",{})
    dc=sp.get("dixon_coles",{});bay=sp.get("bayesian",{})
    p="You are a top football analyst. Here are 9 statistical model predictions. Give final prediction in Chinese.\n\n"
    p+="Match: %s %s vs %s\n"%(lg,h,a)
    if hs:p+="Home: P%s W%s D%s L%s GF%s GA%s AvgGF%s AvgGA%s CS%s Form:%s\n"%(hs.get("played","?"),hs.get("wins","?"),hs.get("draws","?"),hs.get("losses","?"),hs.get("goals_for","?"),hs.get("goals_against","?"),hs.get("avg_goals_for","?"),hs.get("avg_goals_against","?"),hs.get("clean_sheets","?"),hs.get("form","?"))
    if ast:p+="Away: P%s W%s D%s L%s GF%s GA%s AvgGF%s AvgGA%s CS%s Form:%s\n"%(ast.get("played","?"),ast.get("wins","?"),ast.get("draws","?"),ast.get("losses","?"),ast.get("goals_for","?"),ast.get("goals_against","?"),ast.get("avg_goals_for","?"),ast.get("avg_goals_against","?"),ast.get("clean_sheets","?"),ast.get("form","?"))
    if h2h:
        p+="H2H:\n"
        for x in h2h[:5]:p+="%s %s %s %s\n"%(x["date"],x["home"],x["score"],x["away"])
    p+="\n9 Model Results:\n"
    p+="Poisson: H%.1f%% D%.1f%% A%.1f%% Score:%s\n"%(poi.get("home_win",33),poi.get("draw",33),poi.get("away_win",33),poi.get("predicted_score","?"))
    p+="Dixon-Coles: H%.1f%% D%.1f%% A%.1f%%\n"%(dc.get("home_win",33),dc.get("draw",33),dc.get("away_win",33))
    p+="ELO: H%.1f%% D%.1f%% A%.1f%% diff:%.0f\n"%(elo.get("home_win",33),elo.get("draw",33),elo.get("away_win",33),elo.get("elo_diff",0))
    p+="MonteCarlo(10k): H%.1f%% D%.1f%% A%.1f%%\n"%(mc.get("home_win",33),mc.get("draw",33),mc.get("away_win",33))
    p+="RandomForest: H%.1f%% D%.1f%% A%.1f%%\n"%(rf.get("home_win",33),rf.get("draw",33),rf.get("away_win",33))
    p+="GradientBoost: H%.1f%% D%.1f%% A%.1f%%\n"%(gb.get("home_win",33),gb.get("draw",33),gb.get("away_win",33))
    p+="NeuralNet: H%.1f%% D%.1f%% A%.1f%%\n"%(nn.get("home_win",33),nn.get("draw",33),nn.get("away_win",33))
    p+="Bayesian: H%.1f%% D%.1f%% A%.1f%%\n"%(bay.get("home_win",33),bay.get("draw",33),bay.get("away_win",33))
    p+="Ensemble: H%.1f%% D%.1f%% A%.1f%% Consensus:%d/10\n"%(sp["home_win_pct"],sp["draw_pct"],sp["away_win_pct"],sp.get("model_consensus",0))
    p+="Over2.5:%.1f%% BTTS:%.1f%%\n"%(sp.get("over_2_5",50),sp.get("btts",50))
    p+="\nGive final prediction as JSON only:\n"
    p+='{"predicted_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"\u4e3b\u80dc","over_under_2_5":"\u5927","both_score":"\u662f","risk_level":"\u4e2d","analysis":"200 char Chinese analysis","key_factors":["f1","f2","f3"]}'
    return p

def call_model(prompt,url,key,model):
    try:
        r=requests.post(url,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},json={"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0.3,"max_tokens":800},timeout=120)
        print("    status:%d"%r.status_code)
        resp=r.json()
        if "choices" in resp:
            t=resp["choices"][0]["message"]["content"].strip()
            if "```" in t:
                for part in t.split("```"):
                    part=part.strip()
                    if part.startswith("json"):part=part[4:].strip()
                    if part.startswith("{"):
                        try:return json.loads(part)
                        except:continue
            start=t.find("{");end=t.rfind("}")+1
            if start>=0 and end>start:
                return json.loads(t[start:end])
        else:print("    resp:%s"%str(resp)[:200])
    except Exception as e:print("    err:%s"%str(e)[:100])
    return None

def call_gpt(p):
    print("    GPT(%s)..."%GPT_MODEL)
    return call_model(p,GPT_API_URL,GPT_API_KEY,GPT_MODEL)
def call_gemini(p):
    print("    Gemini(%s)..."%GEMINI_MODEL)
    return call_model(p,GEMINI_API_URL,GEMINI_API_KEY,GEMINI_MODEL)

def merge_all(gpt,gemini,stats):
    ai_preds=[x for x in[gpt,gemini]if x]
    if ai_preds:
        ai_h=sum(x.get("home_win_pct",33)for x in ai_preds)/len(ai_preds)
        ai_d=sum(x.get("draw_pct",33)for x in ai_preds)/len(ai_preds)
        ai_a=sum(x.get("away_win_pct",33)for x in ai_preds)/len(ai_preds)
        ai_cf=sum(x.get("confidence",50)for x in ai_preds)/len(ai_preds)
        hp=ai_h*0.45+stats["home_win_pct"]*0.55
        dp=ai_d*0.45+stats["draw_pct"]*0.55
        ap=ai_a*0.45+stats["away_win_pct"]*0.55
        cf=ai_cf*0.5+stats["confidence"]*0.5
    else:
        hp=stats["home_win_pct"];dp=stats["draw_pct"];ap=stats["away_win_pct"];cf=stats["confidence"]
    t=hp+dp+ap
    if t>0:hp=round(hp/t*100,1);dp=round(dp/t*100,1);ap=round(100-hp-dp,1)
    cf=round(min(95,max(25,cf)),1)
    pcts={"\u4e3b\u80dc":hp,"\u5e73\u5c40":dp,"\u5ba2\u80dc":ap};result=max(pcts,key=pcts.get)
    score=stats["predicted_score"]
    if ai_preds:
        ais=[x.get("predicted_score","")for x in ai_preds if x.get("predicted_score")]
        if ais:score=ais[0]
        if len(ais)==2 and ais[0]==ais[1]:cf=min(cf+5,95)
    agree=True
    if gpt and gemini:agree=gpt.get("result","")==gemini.get("result","")
    if agree:cf=min(cf+3,95)
    o25=stats.get("over_2_5",50);bt=stats.get("btts",50)
    risk="\u4f4e" if cf>=70 else("\u4e2d" if cf>=50 else "\u9ad8")
    return{"predicted_score":score,"home_win_pct":hp,"draw_pct":dp,"away_win_pct":ap,"confidence":cf,"result":result,"over_under_2_5":"\u5927" if o25>55 else "\u5c0f","both_score":"\u662f" if bt>50 else "\u5426","risk_level":risk,"gpt_analysis":gpt.get("analysis","") if gpt else "N/A","gemini_analysis":gemini.get("analysis","") if gemini else "N/A","analysis":"","key_factors":list(set((gpt.get("key_factors",[]) if gpt else [])+(gemini.get("key_factors",[]) if gemini else [])))[:6],"gpt_score":gpt.get("predicted_score","?") if gpt else "?","gemini_score":gemini.get("predicted_score","?") if gemini else "?","model_agreement":agree,"poisson":stats.get("poisson",{}),"dixon_coles":stats.get("dixon_coles",{}),"elo":stats.get("elo",{}),"bradley_terry":stats.get("bradley_terry",{}),"monte_carlo":stats.get("monte_carlo",{}),"bayesian":stats.get("bayesian",{}),"random_forest":stats.get("random_forest",{}),"gradient_boost":stats.get("gradient_boost",{}),"neural_net":stats.get("neural_net",{}),"logistic":stats.get("logistic",{}),"home_form":stats.get("home_form",{}),"away_form":stats.get("away_form",{}),"over_2_5_pct":o25,"btts_pct":bt,"top_scores":stats.get("top_scores",[]),"odds_analysis":stats.get("odds",{}),"model_consensus":stats.get("model_consensus",0),"total_models":stats.get("total_models",10)}

def select_top4(preds):
    for p in preds:
        pr=p.get("prediction",{});s=pr.get("confidence",0)*0.4
        if pr.get("model_agreement"):s+=12
        mx=max(pr.get("home_win_pct",33),pr.get("away_win_pct",33),pr.get("draw_pct",33))
        s+=(mx-33)*0.3;con=pr.get("model_consensus",0);s+=con*2
        if pr.get("risk_level") in ["\u4f4e","low"]:s+=8
        elif pr.get("risk_level") in ["\u9ad8","high"]:s-=5
        p["recommend_score"]=round(s,2)
    preds.sort(key=lambda x:x.get("recommend_score",0),reverse=True)
    return preds[:4]

def run_predictions(raw):
    ms=raw.get("matches",[]);od=raw.get("odds",{})
    print("\n=== 11 Model Ensemble: %d matches ==="%len(ms))
    res=[]
    for i,m in enumerate(ms):
        print("\n[%d/%d] %s vs %s"%(i+1,len(ms),m["home_team"],m["away_team"]))
        odds_key=m["home_team"]+"_"+m["away_team"];match_odds=od.get(odds_key,{})
        print("  Stats(9 models)...")
        sp=ensemble.predict(m,match_odds)
        print("    Poi:%s RF:H%.0f%% GB:H%.0f%% NN:H%.0f%% Con:%d"%(sp["poisson"]["predicted_score"],sp["random_forest"]["home_win"],sp["gradient_boost"]["home_win"],sp["neural_net"]["home_win"],sp.get("model_consensus",0)))
        print("  AI models...")
        prompt=build_prompt(m,sp)
        gp=call_gpt(prompt);time.sleep(1);gm=call_gemini(prompt);time.sleep(1)
        print("  Merge(11)...")
        mg=merge_all(gp,gm,sp)
        print("  => %s (%s) %.1f%%"%(mg["result"],mg["predicted_score"],mg["confidence"]))
        res.append({"match_id":m.get("id",i+1),"league":m.get("league",""),"league_logo":m.get("league_logo",""),"home_team":m["home_team"],"away_team":m["away_team"],"home_logo":m.get("home_logo",""),"away_logo":m.get("away_logo",""),"match_time":m.get("date",""),"home_stats":m.get("home_stats",{}),"away_stats":m.get("away_stats",{}),"h2h":m.get("h2h",[])[:5],"prediction":mg,"match_num":m.get("match_num","")})
    t4=select_top4(res);t4ids=[t["match_id"]for t in t4]
    for r in res:r["is_recommended"]=r["match_id"]in t4ids
    return res,t4
