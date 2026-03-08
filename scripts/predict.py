import json,requests,time
from config import *
from models import EnsemblePredictor

ensemble=EnsemblePredictor()

def build_prompt(m,stats_pred):
    h=m["home_team"];a=m["away_team"];lg=m.get("league","")
    hs=m.get("home_stats",{});ast=m.get("away_stats",{});h2h=m.get("h2h",[])
    p="你是顶级足球竞彩分析师。结合以下数据给出精准预测。\n\n"
    p+="【比赛】%s %s vs %s\n"%(lg,h,a)
    if hs:
        p+="【主队】%s场 %s胜%s平%s负 进%s失%s 场均进%s失%s 零封%s 近况:%s\n"%(hs.get("played","?"),hs.get("wins","?"),hs.get("draws","?"),hs.get("losses","?"),hs.get("goals_for","?"),hs.get("goals_against","?"),hs.get("avg_goals_for","?"),hs.get("avg_goals_against","?"),hs.get("clean_sheets","?"),hs.get("form","?"))
    if ast:
        p+="【客队】%s场 %s胜%s平%s负 进%s失%s 场均进%s失%s 零封%s 近况:%s\n"%(ast.get("played","?"),ast.get("wins","?"),ast.get("draws","?"),ast.get("losses","?"),ast.get("goals_for","?"),ast.get("goals_against","?"),ast.get("avg_goals_for","?"),ast.get("avg_goals_against","?"),ast.get("clean_sheets","?"),ast.get("form","?"))
    if h2h:
        p+="【交锋】\n"
        for x in h2h[:5]:p+="%s %s %s %s\n"%(x["date"],x["home"],x["score"],x["away"])
    sp=stats_pred
    p+="\n【统计模型参考】\n"
    p+="泊松模型: 主胜%.1f%% 平%.1f%% 客胜%.1f%% 预测比分%s\n"%(sp["poisson"]["home_win"],sp["poisson"]["draw"],sp["poisson"]["away_win"],sp["poisson"]["predicted_score"])
    p+="ELO模型: 主胜%.1f%% 平%.1f%% 客胜%.1f%% ELO差:%.0f\n"%(sp["elo"]["home_win"],sp["elo"]["draw"],sp["elo"]["away_win"],sp["elo"]["elo_diff"])
    p+="主队状态:%s(%.0f分) 客队状态:%s(%.0f分)\n"%(sp["home_form"]["trend"],sp["home_form"]["score"],sp["away_form"]["trend"],sp["away_form"]["score"])
    p+="大2.5球概率:%.1f%% 双方进球:%.1f%%\n"%(sp["over_2_5"],sp["btts"])
    if sp.get("odds"):
        od=sp["odds"]
        p+="赔率共识:%s 平均赔率:主%.2f 平%.2f 客%.2f 隐含概率:主%.1f%%平%.1f%%客%.1f%%\n"%(od.get("consensus","?"),od.get("avg_home_odds",0),od.get("avg_draw_odds",0),od.get("avg_away_odds",0),od.get("implied_home",33),od.get("implied_draw",33),od.get("implied_away",33))
    p+="\n结合统计模型和你的专业判断，给出最终预测。只返回JSON:\n"
    p+='{"predicted_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","over_under_2_5":"大","both_score":"是","risk_level":"中","analysis":"详细分析理由200字","key_factors":["因素1","因素2","因素3"]}'
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
        ai_home=sum(x.get("home_win_pct",33)for x in ai_preds)/len(ai_preds)
        ai_draw=sum(x.get("draw_pct",33)for x in ai_preds)/len(ai_preds)
        ai_away=sum(x.get("away_win_pct",33)for x in ai_preds)/len(ai_preds)
        ai_conf=sum(x.get("confidence",50)for x in ai_preds)/len(ai_preds)
        hp=ai_home*0.5+stats["home_win_pct"]*0.5
        dp=ai_draw*0.5+stats["draw_pct"]*0.5
        ap=ai_away*0.5+stats["away_win_pct"]*0.5
        cf=(ai_conf*0.6+stats["confidence"]*0.4)
    else:
        hp=stats["home_win_pct"];dp=stats["draw_pct"];ap=stats["away_win_pct"]
        cf=stats["confidence"]*0.7
    total=hp+dp+ap
    if total>0:hp=round(hp/total*100,1);dp=round(dp/total*100,1);ap=round(100-hp-dp,1)
    cf=round(min(95,max(25,cf)),1)
    pcts={"主胜":hp,"平局":dp,"客胜":ap};result=max(pcts,key=pcts.get)
    score=stats["predicted_score"]
    if ai_preds:
        ai_scores=[x.get("predicted_score","")for x in ai_preds if x.get("predicted_score")]
        if ai_scores:score=ai_scores[0]
        if len(ai_scores)==2 and ai_scores[0]==ai_scores[1]:cf=min(cf+5,95)
    agreement=True
    if gpt and gemini:
        agreement=gpt.get("result","")==gemini.get("result","")
        if agreement:cf=min(cf+5,95)
    over25=stats.get("over_2_5",50)
    btts=stats.get("btts",50)
    risk="低" if cf>=70 else("中" if cf>=50 else "高")
    return{
        "predicted_score":score,"home_win_pct":hp,"draw_pct":dp,"away_win_pct":ap,
        "confidence":cf,"result":result,
        "over_under_2_5":"大" if over25>55 else "小",
        "both_score":"是" if btts>50 else "否",
        "risk_level":risk,
        "gpt_analysis":gpt.get("analysis","") if gpt else "未响应",
        "gemini_analysis":gemini.get("analysis","") if gemini else "未响应",
        "analysis":"",
        "key_factors":(gpt.get("key_factors",[]) if gpt else [])+(gemini.get("key_factors",[]) if gemini else []),
        "gpt_score":gpt.get("predicted_score","?") if gpt else "?",
        "gemini_score":gemini.get("predicted_score","?") if gemini else "?",
        "model_agreement":agreement,
        "poisson":stats.get("poisson",{}),
        "elo":stats.get("elo",{}),
        "home_form":stats.get("home_form",{}),
        "away_form":stats.get("away_form",{}),
        "over_2_5_pct":over25,"btts_pct":btts,
        "top_scores":stats.get("top_scores",[]),
        "odds_analysis":stats.get("odds",{}),
    }

def select_top4(preds):
    for p in preds:
        pr=p.get("prediction",{});s=pr.get("confidence",0)*0.4
        if pr.get("model_agreement"):s+=15
        mx=max(pr.get("home_win_pct",33),pr.get("away_win_pct",33),pr.get("draw_pct",33))
        s+=(mx-33)*0.3
        if pr.get("risk_level")=="低":s+=10
        elif pr.get("risk_level")=="高":s-=5
        p["recommend_score"]=round(s,2)
    preds.sort(key=lambda x:x.get("recommend_score",0),reverse=True)
    return preds[:4]

def run_predictions(raw):
    ms=raw.get("matches",[]);od=raw.get("odds",{})
    print("\n=== AI + Stats Ensemble: %d matches ==="%len(ms))
    res=[]
    for i,m in enumerate(ms):
        print("\n[%d/%d] %s vs %s"%(i+1,len(ms),m["home_team"],m["away_team"]))
        odds_key=m["home_team"]+"_"+m["away_team"]
        match_odds=od.get(odds_key,{})
        print("  1.Stats models...")
        stats_pred=ensemble.predict(m,match_odds)
        print("    Poisson:%s ELO diff:%.0f Over2.5:%.1f%%"%(stats_pred["predicted_score"],stats_pred["elo"]["elo_diff"],stats_pred["over_2_5"]))
        print("  2.AI models...")
        prompt=build_prompt(m,stats_pred)
        gp=call_gpt(prompt);time.sleep(1)
        gm=call_gemini(prompt);time.sleep(1)
        print("  3.Merge...")
        mg=merge_all(gp,gm,stats_pred)
        print("  => %s (%s) %.1f%%"%(mg["result"],mg["predicted_score"],mg["confidence"]))
        res.append({"match_id":m.get("id",i+1),"league":m.get("league",""),"league_logo":m.get("league_logo",""),"home_team":m["home_team"],"away_team":m["away_team"],"home_logo":m.get("home_logo",""),"away_logo":m.get("away_logo",""),"match_time":m.get("date",""),"home_stats":m.get("home_stats",{}),"away_stats":m.get("away_stats",{}),"h2h":m.get("h2h",[])[:5],"prediction":mg,"match_num":m.get("match_num","")})
    t4=select_top4(res);t4ids=[t["match_id"]for t in t4]
    for r in res:r["is_recommended"]=r["match_id"]in t4ids
    return res,t4
