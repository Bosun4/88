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
    p="你是顶级足球竞彩分析师。以下是9个统计/ML模型的预测结果，请综合分析给出最终判断。\n\n"
    p+="【比赛】%s %s vs %s\n"%(lg,h,a)
    if hs:p+="【主队】%s场 %s胜%s平%s负 进%s失%s 均进%s均失%s 零封%s 近况:%s\n"%(hs.get("played","?"),hs.get("wins","?"),hs.get("draws","?"),hs.get("losses","?"),hs.get("goals_for","?"),hs.get("goals_against","?"),hs.get("avg_goals_for","?"),hs.get("avg_goals_against","?"),hs.get("clean_sheets","?"),hs.get("form","?"))
    if ast:p+="【客队】%s场 %s胜%s平%s负 进%s失%s 均进%s均失%s 零封%s 近况:%s\n"%(ast.get("played","?"),ast.get("wins","?"),ast.get("draws","?"),ast.get("losses","?"),ast.get("goals_for","?"),ast.get("goals_against","?"),ast.get("avg_goals_for","?"),ast.get("avg_goals_against","?"),ast.get("clean_sheets","?"),ast.get("form","?"))
    if h2h:
        p+="【交锋】\n"
        for x in h2h[:5]:p+="%s %s %s %s\n"%(x["date"],x["home"],x["score"],x["away"])
    p+="\n【9大模型预测汇总】\n"
    p+="泊松分布: 主%.1f%% 平%.1f%% 客%.1f%% 比分%s xG:%.1f-%.1f\n"%(poi.get("home_win",33),poi.get("draw",33),poi.get("away_win",33),poi.get("predicted_score","?"),poi.get("home_xg",1.3),poi.get("away_xg",1.0))
    p+="Dixon-Coles: 主%.1f%% 平%.1f%% 客%.1f%%\n"%(dc.get("home_win",33),dc.get("draw",33),dc.get("away_win",33))
    p+="ELO评分: 主%.1f%% 平%.1f%% 客%.1f%% 差值:%.0f\n"%(elo.get("home_win",33),elo.get("draw",33),elo.get("away_win",33),elo.get("elo_diff",0))
    p+="蒙特卡洛(1万次): 主%.1f%% 平%.1f%% 客%.1f%% 均总球:%.1f\n"%(mc.get("home_win",33),mc.get("draw",33),mc.get("away_win",33),mc.get("avg_total_goals",2.5))
    p+="随机森林: 主%.1f%% 平%.1f%% 客%.1f%%\n"%(rf.get("home_win",33),rf.get("draw",33),rf.get("away_win",33))
    p+="梯度提升: 主%.1f%% 平%.1f%% 客%.1f%%\n"%(gb.get("home_win",33),gb.get("draw",33),gb.get("away_win",33))
    p+="神经网络: 主%.1f%% 平%.1f%% 客%.1f%%\n"%(nn.get("home_win",33),nn.get("draw",33),nn.get("away_win",33))
    p+="贝叶斯: 主%.1f%% 平%.1f%% 客%.1f%%\n"%(bay.get("home_win",33),bay.get("draw",33),bay.get("away_win",33))
    p+="模型融合: 主%.1f%% 平%.1f%% 客%.1f%% 共识:%d/10\n"%(sp["home_win_pct"],sp["draw_pct"],sp["away_win_pct"],sp.get("model_consensus",0))
    p+="大2.5球:%.1f%% 双方进球:%.1f%%\n"%(sp.get("over_2_5",50),sp.get("btts",50))
    if sp.get("odds"):
        od=sp["odds"]
        p+="赔率: 主%.2f 平%.2f 客%.2f 隐含:主%.1f%%平%.1f%%客%.1f%%\n"%(od.get("avg_home_odds",0),od.get("avg_draw_odds",0),od.get("avg_away_odds",0),od.get("implied_home",33),od.get("implied_draw",33),od.get("implied_away",33))
    p+="\n综合所有模型数据和你的专业判断，给出最终预测。只返回JSON:\n"
    p+='{"predicted_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","over_under_2_5":"大","both_score":"是","risk_level":"中","analysis":"详细200字分析","key_factors":["因素1","因素2","因素3"]}'
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
        hp=stats["home_win_pct"];dp=stats["draw_pct"];ap=stats["away_win_pct"]
        cf=stats["confidence"]
    t=hp+dp+ap
    if t>0:hp=round(hp/t*100,1);dp=round(dp/t*100,1);ap=round(100-hp-dp,1)
    cf=round(min(95,max(25,cf)),1)
    pcts={"主胜":hp,"平局":dp,"客胜":ap};result=max(pcts,key=pcts.get)
    score=stats["predicted_score"]
    if ai_preds:
        ais=[x.get("predicted_score","")for x in ai_preds if x.get("predicted_score")]
        if ais:score=ais[0]
        if len(ais)==2 and ais[0]==ais[1]:cf=min(cf+5,95)
    agree=True
    if gpt and gemini:
        agree=gpt.get("result","")==gemini.get("result","")
        if agree:cf=min(cf+3,95)
    o25=stats.get("over_2_5",50);bt=stats.get("btts",50)
    risk="低" if cf>=70 else("中" if cf>=50 else "高")
    return{
        "predicted_score":score,"home_win_pct":hp,"draw_pct":dp,"away_win_pct":ap,
        "confidence":cf,"result":result,
        "over_under_2_5":"大" if o25>55 else "小","both_score":"是" if bt>50 else "否","risk_level":risk,
        "gpt_analysis":gpt.get("analysis","") if gpt else "未响应",
        "gemini_analysis":gemini.get("analysis","") if gemini else "未响应",
        "analysis":"","key_factors":list(set((gpt.get("key_factors",[]) if gpt else [])+(gemini.get("key_factors",[]) if gemini else [])))[:6],
        "gpt_score":gpt.get("predicted_score","?") if gpt else "?",
        "gemini_score":gemini.get("predicted_score","?") if gemini else "?",
        "model_agreement":agree,
        "poisson":stats.get("poisson",{}),"dixon_coles":stats.get("dixon_coles",{}),
        "elo":stats.get("elo",{}),"bradley_terry":stats.get("bradley_terry",{}),
        "monte_carlo":stats.get("monte_carlo",{}),"bayesian":stats.get("bayesian",{}),
        "random_forest":stats.get("random_forest",{}),"gradient_boost":stats.get("gradient_boost",{}),
        "neural_net":stats.get("neural_net",{}),"logistic":stats.get("logistic",{}),
        "home_form":stats.get("home_form",{}),"away_form":stats.get("away_form",{}),
        "over_2_5_pct":o25,"btts_pct":bt,
        "top_scores":stats.get("top_scores",[]),
        "odds_analysis":stats.get("odds",{}),
        "model_consensus":stats.get("model_consensus",0),"total_models":stats.get("total_models",10),
    }

def select_top4(preds):
    for p in preds:
        pr=p.get("prediction",{});s=pr.get("confidence",0)*0.4
        if pr.get("model_agreement"):s+=12
        mx=max(pr.get("home_win_pct",33),pr.get("away_win_pct",33),pr.get("draw_pct",33))
        s+=(mx-33)*0.3
        con=pr.get("model_consensus",0);s+=con*2
        if pr.get("risk_level")=="低":s+=8
        elif pr.get("risk_level")=="高":s-=5
        p["recommend_score"]=round(s,2)
    preds.sort(key=lambda x:x.get("recommend_score",0),reverse=True)
    return preds[:4]

def run_predictions(raw):
    ms=raw.get("matches",[]);od=raw.get("odds",{})
    print("\n=== 9 Stats + 2 AI = 11 Model Ensemble: %d matches ==="%len(ms))
    res=[]
    for i,m in enumerate(ms):
        print("\n[%d/%d] %s vs %s"%(i+1,len(ms),m["home_team"],m["away_team"]))
        odds_key=m["home_team"]+"_"+m["away_team"]
        match_odds=od.get(odds_key,{})
        print("  1.Stats(9 models)...")
        sp=ensemble.predict(m,match_odds)
        print("    Poisson:%s MC:%s RF:H%.0f%% GB:H%.0f%% NN:H%.0f%% Consensus:%d/%d"%(
            sp["poisson"]["predicted_score"],sp["monte_carlo"].get("top_scores",[{}])[0].get("score","?"),
            sp["random_forest"]["home_win"],sp["gradient_boost"]["home_win"],sp["neural_net"]["home_win"],
            sp.get("model_consensus",0),sp.get("total_models",10)))
        print("  2.AI models...")
        prompt=build_prompt(m,sp)
        gp=call_gpt(prompt);time.sleep(1)
        gm=call_gemini(prompt);time.sleep(1)
        print("  3.Final merge (11 models)...")
        mg=merge_all(gp,gm,sp)
        print("  => %s (%s) %.1f%% [consensus:%d]"%(mg["result"],mg["predicted_score"],mg["confidence"],mg.get("model_consensus",0)))
        res.append({"match_id":m.get("id",i+1),"league":m.get("league",""),"league_logo":m.get("league_logo",""),
            "home_team":m["home_team"],"away_team":m["away_team"],"home_logo":m.get("home_logo",""),
            "away_logo":m.get("away_logo",""),"match_time":m.get("date",""),
            "home_stats":m.get("home_stats",{}),"away_stats":m.get("away_stats",{}),
            "h2h":m.get("h2h",[])[:5],"prediction":mg,"match_num":m.get("match_num","")})
    t4=select_top4(res);t4ids=[t["match_id"]for t in t4]
    for r in res:r["is_recommended"]=r["match_id"]in t4ids
    return res,t4