import json,requests,time
from config import *
def build_prompt(m,odds,st):
    h=m["home_team"];a=m["away_team"];lg=m.get("league","");hs=m.get("home_stats",{});ast=m.get("away_stats",{});h2h=m.get("h2h",[])
    p="你是一位顶级足球数据分析师和体彩竞彩专家。请分析这场比赛。\n\n"
    p+="联赛:%s 主队:%s 客队:%s\n"%(lg,h,a)
    if hs:p+="主队数据: %s场 %s胜%s平%s负 进%s失%s 近况:%s\n"%(hs.get("played","?"),hs.get("wins","?"),hs.get("draws","?"),hs.get("losses","?"),hs.get("goals_for","?"),hs.get("goals_against","?"),hs.get("form","?"))
    if ast:p+="客队数据: %s场 %s胜%s平%s负 进%s失%s 近况:%s\n"%(ast.get("played","?"),ast.get("wins","?"),ast.get("draws","?"),ast.get("losses","?"),ast.get("goals_for","?"),ast.get("goals_against","?"),ast.get("form","?"))
    if h2h:
        p+="交锋记录:\n"
        for x in h2h[:5]:p+="%s %s %s %s\n"%(x["date"],x["home"],x["score"],x["away"])
    p+="\n请只返回以下JSON格式，不要任何其他文字:\n"
    p+='{"predicted_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","over_under_2_5":"大","both_score":"是","risk_level":"中","analysis":"分析理由","key_factors":["因素1","因素2"]}'
    return p
def call_model(prompt,url,key,model):
    try:
        r=requests.post(url,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},json={"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0.3,"max_tokens":800},timeout=120)
        print("    status:%d"%r.status_code)
        resp=r.json()
        if "choices" in resp:
            t=resp["choices"][0]["message"]["content"].strip()
            if "```" in t:
                parts=t.split("```")
                for part in parts:
                    part=part.strip()
                    if part.startswith("json"):part=part[4:].strip()
                    if part.startswith("{"):
                        try:return json.loads(part)
                        except:continue
            if t.startswith("{"):
                return json.loads(t)
            start=t.find("{");end=t.rfind("}")+1
            if start>=0 and end>start:
                return json.loads(t[start:end])
        else:
            print("    no choices:%s"%str(resp)[:200])
    except Exception as e:
        print("    error:%s"%str(e)[:100])
    return None
def call_gpt(p):
    print("    GPT(%s)..."%GPT_MODEL)
    return call_model(p,GPT_API_URL,GPT_API_KEY,GPT_MODEL)
def call_gemini(p):
    print("    Gemini(%s)..."%GEMINI_MODEL)
    return call_model(p,GEMINI_API_URL,GEMINI_API_KEY,GEMINI_MODEL)
def merge(g,m):
    if g and m:
        hp=round((g.get("home_win_pct",33)+m.get("home_win_pct",33))/2,1);dp=round((g.get("draw_pct",33)+m.get("draw_pct",33))/2,1);ap=round((g.get("away_win_pct",33)+m.get("away_win_pct",33))/2,1)
        t=hp+dp+ap
        if t>0:hp=round(hp/t*100,1);dp=round(dp/t*100,1);ap=round(100-hp-dp,1)
        cf=round((g.get("confidence",50)+m.get("confidence",50))/2,1)
        pcts={"主胜":hp,"平局":dp,"客胜":ap};rs=max(pcts,key=pcts.get)
        sc=g.get("predicted_score","1-1")
        if g.get("predicted_score")==m.get("predicted_score"):sc=g["predicted_score"];cf=min(cf+5,99)
        return{"predicted_score":sc,"home_win_pct":hp,"draw_pct":dp,"away_win_pct":ap,"confidence":cf,"result":rs,"over_under_2_5":g.get("over_under_2_5",m.get("over_under_2_5","")),"both_score":g.get("both_score",m.get("both_score","")),"risk_level":g.get("risk_level","中"),"gpt_analysis":g.get("analysis",""),"gemini_analysis":m.get("analysis",""),"analysis":"GPT:"+g.get("analysis","")+" Gemini:"+m.get("analysis",""),"key_factors":list(set(g.get("key_factors",[])+m.get("key_factors",[])))[:6],"gpt_score":g.get("predicted_score","?"),"gemini_score":m.get("predicted_score","?"),"model_agreement":g.get("result","")==m.get("result","")}
    elif g:g["model_agreement"]=False;g["gpt_analysis"]=g.get("analysis","");g["gemini_analysis"]="N/A";return g
    elif m:m["model_agreement"]=False;m["gemini_analysis"]=m.get("analysis","");m["gpt_analysis"]="N/A";return m
    return{"predicted_score":"?-?","home_win_pct":33.3,"draw_pct":33.3,"away_win_pct":33.3,"confidence":0,"result":"无","analysis":"AI未响应","model_agreement":False}
def select_top4(preds):
    for p in preds:
        pr=p.get("prediction",{});s=pr.get("confidence",0)*0.4
        if pr.get("model_agreement"):s+=15
        mx=max(pr.get("home_win_pct",33),pr.get("away_win_pct",33),pr.get("draw_pct",33));s+=(mx-33)*0.3
        p["recommend_score"]=round(s,2)
    preds.sort(key=lambda x:x.get("recommend_score",0),reverse=True);return preds[:4]
def run_predictions(raw):
    ms=raw.get("matches",[]);st=raw.get("standings",{});od=raw.get("odds",{})
    print("\nAI analyzing %d jczq matches..."%len(ms));res=[]
    for i,m in enumerate(ms):
        print("\n  [%d/%d] %s vs %s"%(i+1,len(ms),m["home_team"],m["away_team"]))
        pr=build_prompt(m,od,st)
        gp=call_gpt(pr);time.sleep(1)
        gm=call_gemini(pr);time.sleep(1)
        mg=merge(gp,gm)
        print("    => %s (%s) %s%%"%(mg.get("result","?"),mg.get("predicted_score","?"),mg.get("confidence",0)))
        res.append({"match_id":m.get("id",i+1),"league":m.get("league",""),"league_logo":m.get("league_logo",""),"home_team":m["home_team"],"away_team":m["away_team"],"home_logo":m.get("home_logo",""),"away_logo":m.get("away_logo",""),"match_time":m.get("date",""),"home_stats":m.get("home_stats",{}),"away_stats":m.get("away_stats",{}),"h2h":m.get("h2h",[])[:5],"prediction":mg,"match_num":m.get("match_num","")})
    t4=select_top4(res);t4ids=[t["match_id"]for t in t4]
    for r in res:r["is_recommended"]=r["match_id"]in t4ids
    return res,t4
