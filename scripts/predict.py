import json,requests,time
from config import *
def build_prompt(m,odds,st):
    h=m["home_team"];a=m["away_team"];lg=m["league"];hs=m.get("home_stats",{});ast=m.get("away_stats",{});h2h=m.get("h2h",[])
    p="Analyze this football match and respond ONLY with a JSON object.\n"
    p+="League:%s Home:%s Away:%s\n"%(lg,h,a)
    p+="Home stats: P%s W%s D%s L%s GF%s GA%s Form:%s\n"%(hs.get("played","?"),hs.get("wins","?"),hs.get("draws","?"),hs.get("losses","?"),hs.get("goals_for","?"),hs.get("goals_against","?"),hs.get("form","?"))
    p+="Away stats: P%s W%s D%s L%s GF%s GA%s Form:%s\n"%(ast.get("played","?"),ast.get("wins","?"),ast.get("draws","?"),ast.get("losses","?"),ast.get("goals_for","?"),ast.get("goals_against","?"),ast.get("form","?"))
    p+="H2H:"
    for x in h2h[:5]:p+="\n%s %s %s %s"%(x["date"],x["home"],x["score"],x["away"])
    if not h2h:p+="\nNone"
    p+='\nRespond with ONLY this JSON format, nothing else:\n{"predicted_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","over_under_2_5":"大","both_score":"是","risk_level":"中","analysis":"中文分析理由","key_factors":["factor1","factor2"]}'
    return p
def call_model(prompt,url,key,model,retries=1):
    for i in range(retries+1):
        try:
            headers={"Content-Type":"application/json"}
            if key.startswith("sk-"):
                headers["Authorization"]="Bearer "+key
            else:
                headers["Authorization"]="Bearer "+key
            body={"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0.3,"max_tokens":800}
            r=requests.post(url,headers=headers,json=body,timeout=90)
            resp=r.json()
            if "choices" in resp:
                t=resp["choices"][0]["message"]["content"].strip()
                if "```" in t:
                    parts=t.split("```")
                    t=parts[1] if len(parts)>1 else parts[0]
                if t.startswith("json"):t=t[4:]
                t=t.strip()
                return json.loads(t)
            elif "error" in resp:
                print("    API error:%s"%str(resp["error"])[:100])
            else:
                print("    Unknown resp:%s"%str(resp)[:150])
        except json.JSONDecodeError:
            print("    JSON parse fail, raw:%s"%t[:100] if 't' in dir() else "no text")
        except Exception as e:
            print("    err%d:%s"%(i+1,str(e)[:80]))
        if i<retries:time.sleep(2)
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
        pcts={"\u4e3b\u80dc":hp,"\u5e73\u5c40":dp,"\u5ba2\u80dc":ap};rs=max(pcts,key=pcts.get)
        sc=g.get("predicted_score","1-1")
        if g.get("predicted_score")==m.get("predicted_score"):sc=g["predicted_score"];cf=min(cf+5,99)
        return{"predicted_score":sc,"home_win_pct":hp,"draw_pct":dp,"away_win_pct":ap,"confidence":cf,"result":rs,"over_under_2_5":g.get("over_under_2_5",m.get("over_under_2_5","")),"both_score":g.get("both_score",m.get("both_score","")),"risk_level":g.get("risk_level","\u4e2d"),"gpt_analysis":g.get("analysis",""),"gemini_analysis":m.get("analysis",""),"analysis":"GPT:"+g.get("analysis","")+" Gemini:"+m.get("analysis",""),"key_factors":list(set(g.get("key_factors",[])+m.get("key_factors",[])))[:6],"gpt_score":g.get("predicted_score","?"),"gemini_score":m.get("predicted_score","?"),"model_agreement":g.get("result","")==m.get("result","")}
    elif g:g["model_agreement"]=False;g["gpt_analysis"]=g.get("analysis","");g["gemini_analysis"]="N/A";return g
    elif m:m["model_agreement"]=False;m["gemini_analysis"]=m.get("analysis","");m["gpt_analysis"]="N/A";return m
    return{"predicted_score":"?-?","home_win_pct":33.3,"draw_pct":33.3,"away_win_pct":33.3,"confidence":0,"result":"N/A","analysis":"AI not available","model_agreement":False}
def select_top4(preds):
    for p in preds:
        pr=p.get("prediction",{});s=pr.get("confidence",0)*0.4
        if pr.get("model_agreement"):s+=15
        mx=max(pr.get("home_win_pct",33),pr.get("away_win_pct",33),pr.get("draw_pct",33));s+=(mx-33)*0.3
        p["recommend_score"]=round(s,2)
    preds.sort(key=lambda x:x.get("recommend_score",0),reverse=True);return preds[:4]
def run_predictions(raw):
    ms=raw.get("matches",[]);st=raw.get("standings",{});od=raw.get("odds",{});print("AI predict %d..."%len(ms));res=[]
    for i,m in enumerate(ms):
        print("  [%d/%d]%s v %s"%(i+1,len(ms),m["home_team"],m["away_team"]))
        pr=build_prompt(m,od,st)
        gp=call_gpt(pr);time.sleep(1)
        gm=call_gemini(pr);time.sleep(1)
        mg=merge(gp,gm);print("    =>%s(%s)%s%%"%(mg.get("result","?"),mg.get("predicted_score","?"),mg.get("confidence",0)))
        res.append({"match_id":m["id"],"league":m["league"],"league_logo":m.get("league_logo",""),"home_team":m["home_team"],"away_team":m["away_team"],"home_logo":m.get("home_logo",""),"away_logo":m.get("away_logo",""),"match_time":m.get("date",""),"home_stats":m.get("home_stats",{}),"away_stats":m.get("away_stats",{}),"h2h":m.get("h2h",[])[:5],"prediction":mg})
    t4=select_top4(res);t4ids=[t["match_id"]for t in t4]
    for r in res:r["is_recommended"]=r["match_id"]in t4ids
    return res,t4
