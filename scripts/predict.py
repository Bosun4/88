#!/usr/bin/env python3
"""
predict.py vMAX — 极致压榨AI版
================================
核心架构升级:
1. Claude = StatBrain (输出完整概率+比分+O/U+BTTS，作为第15个模型融合)
2. GPT/Grok/Gemini = Scout (只输出比分+置信度，省50%token)
3. Claude概率直接融合到ensemble(20%权重)，不再只是投票
4. Claude一票否决制 + 方向一致加分 + 方向冲突惩罚
5. 4层增强管线每层独立try/except防崩溃
"""
import json,os,re,time,asyncio,aiohttp,numpy as np
from datetime import datetime
from config import *
from models import EnsemblePredictor
from odds_engine import predict_match
from league_intel import build_league_intelligence
from experience_rules import ExperienceEngine,apply_experience_to_prediction
from advanced_models import upgrade_ensemble_predict
try:
    from odds_history import apply_odds_history
except:
    def apply_odds_history(m,mg):return mg
try:
    from quant_edge import apply_quant_edge
except:
    def apply_quant_edge(m,mg):return mg

ensemble=EnsemblePredictor()
exp_engine=ExperienceEngine()

def calculate_value_bet(prob_pct,odds):
    if not odds or odds<=1.05:return{"ev":0.0,"kelly":0.0,"is_value":False}
    prob=prob_pct/100.0;ev=(prob*odds)-1.0;b=odds-1.0;q=1.0-prob
    if b<=0:return{"ev":round(ev*100,2),"kelly":0.0,"is_value":False}
    kelly=((b*prob)-q)/b
    return{"ev":round(ev*100,2),"kelly":round(max(0.0,kelly*0.25)*100,2),"is_value":ev>0.05}

def load_ai_diary():
    if os.path.exists("data/ai_diary.json"):
        try:
            with open("data/ai_diary.json","r",encoding="utf-8") as f:return json.load(f)
        except:pass
    return{"yesterday_win_rate":"N/A","reflection":"","streak":0}

def save_ai_diary(diary):
    os.makedirs("data",exist_ok=True)
    with open("data/ai_diary.json","w",encoding="utf-8") as f:json.dump(diary,f,ensure_ascii=False,indent=2)

# ====================================================================
#  PROMPT ARCHITECTURE: Claude=StatBrain, Others=Scout
# ====================================================================

def build_claude_prompt(match_analyses):
    """
    Claude专属重型prompt — 让Claude做完整概率推理
    关键: 要求输出 H_prob/D_prob/A_prob，这样Claude变成第15个模型
    """
    diary=load_ai_diary()
    p="[IDENTITY] You are the world's #1 quantitative football analyst. You combine Poisson regression, Elo ratings, Dixon-Coles corrections, and market microstructure into precise probabilistic predictions.\n\n"
    p+="[CRITICAL INSTRUCTIONS]\n"
    p+="1. Output ONLY a raw JSON array. ZERO markdown, ZERO explanation.\n"
    p+="2. Each object MUST have these fields:\n"
    p+="   match(int), score(str like '1-0'), h_prob(int 0-100), d_prob(int 0-100), a_prob(int 0-100),\n"
    p+="   over25(bool), btts(bool), confidence(int 0-100), reason(80-120 chars statistical analysis)\n"
    p+="3. h_prob + d_prob + a_prob MUST sum to 100.\n"
    p+="4. Your reason must reference specific numbers: xG divergence, Elo gap, form delta, odds-probability gap.\n"
    p+="5. When Poisson and Elo disagree, explain WHY and which you trust more for THIS match.\n\n"
    if diary.get("reflection"):
        p+="[EVOLUTION] Previous: %s | Lesson: %s\n\n"%(diary.get("yesterday_win_rate","?"),diary["reflection"][:80])
    p+="[MATCH DATA]\n"
    for i,ma in enumerate(match_analyses):
        m=ma["match"];eng=ma["engine"];stats=ma.get("stats",{});exp=ma.get("experience",{})
        h=m.get("home_team",m.get("home","H"));a=m.get("away_team",m.get("guest","A"))
        lg=m.get("league",m.get("cup",""))
        sp_h=float(m.get("sp_home",m.get("win",0)) or 0)
        sp_d=float(m.get("sp_draw",m.get("same",0)) or 0)
        sp_a=float(m.get("sp_away",m.get("lose",0)) or 0)
        poi=stats.get("poisson",{});elo=stats.get("elo",{});dc=stats.get("dixon_coles",{})
        hs=m.get("home_stats",{});ast=m.get("away_stats",{})
        vote=m.get("vote",{})
        info=m.get("intelligence",m.get("information",{}))
        bad_h=str(info.get("home_bad_news",""))[:80]
        bad_a=str(info.get("guest_bad_news",""))[:80]
        p+="[%d] %s vs %s | %s | HC:%s\n"%(i+1,h,a,lg,m.get("give_ball",0))
        p+="  Odds: H%.2f D%.2f A%.2f | OddsProb: H%.1f D%.1f A%.1f\n"%(sp_h,sp_d,sp_a,eng.get("home_prob",33),eng.get("draw_prob",33),eng.get("away_prob",34))
        p+="  ImpliedXG: H%.3f A%.3f (gap:%s)\n"%(eng.get("bookmaker_implied_home_xg",1.3),eng.get("bookmaker_implied_away_xg",1.1),eng.get("scissors_gap_signal","none")[:30])
        p+="  Poisson: H%.1f D%.1f A%.1f | xG:%.2f-%.2f | Best:%s\n"%(poi.get("home_win",33),poi.get("draw",33),poi.get("away_win",34),poi.get("home_xg",1.3),poi.get("away_xg",1.1),poi.get("predicted_score","?"))
        p+="  Elo: H%.1f D%.1f A%.1f (diff:%s) | DC: H%.1f D%.1f A%.1f\n"%(elo.get("home_win",33),elo.get("draw",33),elo.get("away_win",34),elo.get("elo_diff","?"),dc.get("home_win",33),dc.get("draw",33),dc.get("away_win",34))
        p+="  Home: %sW%sD%sL gf/ga:%s/%s form:%s | Away: %sW%sD%sL gf/ga:%s/%s form:%s\n"%(hs.get("wins","?"),hs.get("draws","?"),hs.get("losses","?"),hs.get("avg_goals_for","?"),hs.get("avg_goals_against","?"),str(hs.get("form","?"))[-5:],ast.get("wins","?"),ast.get("draws","?"),ast.get("losses","?"),ast.get("avg_goals_for","?"),ast.get("avg_goals_against","?"),str(ast.get("form","?"))[-5:])
        p+="  Vote: H%s D%s A%s"%(vote.get("win",33),vote.get("same",33),vote.get("lose",34))
        if bad_h:p+=" | H_risk:%s"%bad_h[:60]
        if bad_a:p+=" | A_risk:%s"%bad_a[:60]
        smart=stats.get("smart_signals",[])
        if smart:p+=" | Sig:%s"%(", ".join(smart[:2]))
        if exp.get("triggered_count",0)>0:
            p+=" | Exp:%s"%(",".join([t["name"] for t in exp.get("triggered",[])[:2]]))
        p+="\n  TopScores: %s\n\n"%(", ".join(eng.get("top3_scores",["1-1","0-0"])))
    p+="[OUTPUT %d objects with h_prob+d_prob+a_prob=100]\n"%len(match_analyses)
    return p

def build_light_prompt(match_analyses):
    """GPT/Grok/Gemini轻量prompt — 只要比分和置信度"""
    p="Predict football scores. Output ONLY JSON array.\n"
    p+="Fields: match(int), score(str), confidence(int 0-100), reason(50 chars max)\n\n"
    for i,ma in enumerate(match_analyses):
        m=ma["match"];eng=ma["engine"]
        h=m.get("home_team",m.get("home","H"));a=m.get("away_team",m.get("guest","A"))
        sp_h=float(m.get("sp_home",m.get("win",0)) or 0)
        sp_a=float(m.get("sp_away",m.get("lose",0)) or 0)
        p+="[%d] %s vs %s H%.2f A%.2f xG:%.1f-%.1f P:H%.0f%%D%.0f%%A%.0f%% Top:%s\n"%(
            i+1,h,a,sp_h,sp_a,eng.get("bookmaker_implied_home_xg",1.3),eng.get("bookmaker_implied_away_xg",1.1),
            eng.get("home_prob",33),eng.get("draw_prob",33),eng.get("away_prob",34),eng.get("top3_scores",["1-1"])[0])
    p+="[%d objects]\n"%len(match_analyses)
    return p

# ====================================================================
#  AI调用引擎
# ====================================================================
FALLBACK_URLS=[None,"https://api520.pro/v1","https://www.api520.pro/v1","https://api521.pro/v1","https://www.api521.pro/v1","https://api522.pro/v1","https://www.api522.pro/v1","https://69.63.213.33:666/v1"]

def _env(name,default=""):
    v=os.environ.get(name,globals().get(name,default));v=str(v).strip(" \t\n\r\"'")
    match=re.search(r"(https?://[a-zA-Z0-9._:/-]+)",v)
    return match.group(1) if match else v

def _key(name):return str(os.environ.get(name,globals().get(name,""))).strip(" \t\n\r\"'")

async def _call_ai(session,prompt,url_env,key_env,models_list,n_matches,ai_name,parse_probs=False):
    key=_key(key_env)
    if not key:return ai_name,{},"no_key"
    primary=_env(url_env)
    urls=[primary]+[u for u in FALLBACK_URLS if u and u!=primary]
    for attempt in range(2):
        for mn in models_list:
            for base_url in urls:
                if not base_url:continue
                is_gem="generateContent" in base_url
                url=base_url.rstrip("/")
                if not is_gem and "chat/completions" not in url:url+="/chat/completions"
                headers={"Content-Type":"application/json"}
                if is_gem:
                    headers["x-goog-api-key"]=key
                    payload={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.08,"topP":0.92}}
                else:
                    headers["Authorization"]="Bearer "+key
                    sys_msg="You are a quantitative football analyst. Output ONLY valid JSON array." if parse_probs else "Output ONLY valid JSON array."
                    payload={"model":mn,"messages":[{"role":"system","content":sys_msg},{"role":"user","content":prompt}],"temperature":0.08}
                print("  [AI] %s: %s @ %s r%d"%(ai_name.upper(),mn[:22],url.split("/v1")[0][:30],attempt+1))
                try:
                    async with session.post(url,headers=headers,json=payload,timeout=aiohttp.ClientTimeout(total=60)) as r:
                        if r.status==200:
                            data=await r.json()
                            raw=data["candidates"][0]["content"]["parts"][0]["text"].strip() if is_gem else data["choices"][0]["message"]["content"].strip()
                            clean=re.sub(r"```[\w]*","",raw).strip()
                            s=clean.find("[");e=clean.rfind("]")+1
                            results={}
                            if s!=-1 and e>s:
                                try:
                                    arr=json.loads(clean[s:e])
                                    if isinstance(arr,list):
                                        for item in arr:
                                            mid=item.get("match")
                                            sc=item.get("score")
                                            if not mid or not sc:continue
                                            rd={"ai_score":sc,"analysis":str(item.get("reason",""))[:200],"ai_confidence":int(item.get("confidence",item.get("ai_confidence",60)))}
                                            if parse_probs:
                                                hp=item.get("h_prob",0);dp=item.get("d_prob",0);ap=item.get("a_prob",0)
                                                if hp+dp+ap>=90:
                                                    t=hp+dp+ap
                                                    rd["h_prob"]=round(hp/t*100,1)
                                                    rd["d_prob"]=round(dp/t*100,1)
                                                    rd["a_prob"]=round(ap/t*100,1)
                                                rd["over25"]=bool(item.get("over25",False))
                                                rd["btts"]=bool(item.get("btts",False))
                                            results[mid]=rd
                                except:pass
                            if len(results)>=max(1,n_matches*0.4):
                                print("    ✅ %s: %d/%d (%s)"%(ai_name.upper(),len(results),n_matches,mn[:20]))
                                return ai_name,results,mn
                        elif r.status==429:
                            await asyncio.sleep(2**attempt*4);continue
                        else:print("    ⚠️ HTTP %d"%r.status)
                except asyncio.TimeoutError:print("    ⏰ timeout")
                except Exception as ex:print("    ⚠️ %s"%str(ex)[:35])
                await asyncio.sleep(0.3)
        await asyncio.sleep(1)
    return ai_name,{},"failed"

async def run_ai_matrix(claude_prompt,light_prompt,n_matches):
    configs=[
        ("claude","CLAUDE_API_URL","CLAUDE_API_KEY",[
            "熊猫-按量-满血copilot-claude-opus-4.6-thinking",
            "熊猫-按量-顶级特供-官max-claude-opus-4.6",
            "熊猫-特供-按量-Q-claude-opus-4.6",
            "熊猫-按量-特供顶级-官方正向满血-claude-sonnet-4.6-thinking",
        ],claude_prompt,True),
        ("grok","GROK_API_URL","GROK_API_KEY",[
            "熊猫-A-6-grok-4.2-thinking",
            "熊猫-A-7-grok-4.2-多智能体讨论",
        ],light_prompt,False),
        ("gpt","GPT_API_URL","GPT_API_KEY",[
            "熊猫-A-7-gpt-5.4",
            "熊猫-按量-gpt-5.3-codex-满血",
            "熊猫-A-10-gpt-5.3-codex",
        ],light_prompt,False),
        ("gemini","GEMINI_API_URL","GEMINI_API_KEY",[
            "熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking",
            "熊猫-特供-X-12-gemini-3.1-pro-preview-thinking",
            "熊猫-顶级特供-X-17-gemini-3.1-pro-preview",
        ],light_prompt,False),
    ]
    all_r={"gpt":{},"grok":{},"claude":{},"gemini":{}}
    async with aiohttp.ClientSession() as session:
        tasks=[_call_ai(session,prompt,ue,ke,ml,n_matches,an,pp) for an,ue,ke,ml,prompt,pp in configs]
        results=await asyncio.gather(*tasks,return_exceptions=True)
    for res in results:
        if isinstance(res,tuple):an,parsed,_=res;all_r[an]=parsed
    return all_r

# ====================================================================
#  MERGE: Claude概率融合 + 一票否决 + 方向冲突检测
# ====================================================================
def _direction(hp,dp,ap):
    mx=max(hp,dp,ap)
    if mx==hp:return"home"
    if mx==dp:return"draw"
    return"away"

def merge_result(eng,gpt_r,grok_r,gemini_r,claude_r,stats,m):
    sp_h=float(m.get("sp_home",0) or 0);sp_d=float(m.get("sp_draw",0) or 0);sp_a=float(m.get("sp_away",0) or 0)
    e_score=eng.get("primary_score","1-1");e_conf=eng.get("confidence",50)

    # 基础概率融合 (引擎75% + 统计模型25%)
    hp=eng.get("home_prob",33);dp=eng.get("draw_prob",33);ap=eng.get("away_prob",34)
    shp=stats.get("home_win_pct",33);sdp=stats.get("draw_pct",33);sap=stats.get("away_win_pct",34)
    fhp=hp*0.75+shp*0.25;fdp=dp*0.75+sdp*0.25;fap=ap*0.75+sap*0.25

    # ========== Claude概率融合(第15个模型, 权重20%) ==========
    cl_fused=False
    if isinstance(claude_r,dict) and "h_prob" in claude_r:
        chp=claude_r["h_prob"];cdp=claude_r["d_prob"];cap=claude_r["a_prob"]
        if chp+cdp+cap>=90:
            w=0.20  # Claude概率融合权重
            fhp=fhp*(1-w)+chp*w
            fdp=fdp*(1-w)+cdp*w
            fap=fap*(1-w)+cap*w
            cl_fused=True
            print("    🧠 Claude概率融合: H%.1f D%.1f A%.1f (w=%.0f%%)"%(chp,cdp,cap,w*100))

    fhp=max(3,fhp);fdp=max(3,fdp);fap=max(3,fap);ft=fhp+fdp+fap
    if ft>0:fhp=round(fhp/ft*100,1);fdp=round(fdp/ft*100,1);fap=round(max(3,100-fhp-fdp),1)

    # ========== AI比分投票 ==========
    ai_all={"gpt":gpt_r,"grok":grok_r,"gemini":gemini_r,"claude":claude_r}
    ai_scores=[];ai_conf_sum=0;ai_conf_count=0
    weights={"claude":2.0,"grok":1.3,"gpt":1.1,"gemini":1.0}
    for name,r in ai_all.items():
        if not isinstance(r,dict):continue
        sc=r.get("ai_score","-")
        if sc and sc not in ["-","?",""]:
            ai_scores.append(sc)
            conf=r.get("ai_confidence",60)
            ai_conf_sum+=conf*weights.get(name,1.0)
            ai_conf_count+=weights.get(name,1.0)
    vote={}
    for sc in ai_scores:vote[sc]=vote.get(sc,0)+1

    # ========== 比分决定: Claude一票否决制 ==========
    final_score=e_score
    cl_score=claude_r.get("ai_score","") if isinstance(claude_r,dict) else ""
    if cl_score and cl_score in eng.get("top3_scores",[]):
        final_score=cl_score  # Claude在top3中→直接采用
    elif vote:
        best=max(vote,key=vote.get)
        if best in eng.get("top3_scores",[]) and vote[best]>=2:final_score=best

    # ========== 置信度计算 ==========
    avg_ai_conf=(ai_conf_sum/ai_conf_count) if ai_conf_count>0 else 60
    cf=e_conf
    cf=min(95,cf+int((avg_ai_conf-60)*0.3))

    # Claude与引擎方向一致→大幅加分
    eng_dir=_direction(eng.get("home_prob",33),eng.get("draw_prob",33),eng.get("away_prob",34))
    if isinstance(claude_r,dict) and "h_prob" in claude_r:
        cl_dir=_direction(claude_r.get("h_prob",33),claude_r.get("d_prob",33),claude_r.get("a_prob",34))
        if cl_dir==eng_dir:cf=min(95,cf+12)
        else:cf=max(35,cf-8)  # 方向冲突→惩罚
    elif cl_score==e_score:cf=min(95,cf+10)

    # 警告信号惩罚
    has_warn=any("🚨" in str(s) for s in stats.get("smart_signals",[]))
    if has_warn:cf=max(35,cf-8)
    risk="低" if cf>=75 else ("中" if cf>=55 else "高")

    # ========== 构建输出 ==========
    gpt_sc=gpt_r.get("ai_score","-") if isinstance(gpt_r,dict) else "-"
    gpt_an=gpt_r.get("analysis","N/A") if isinstance(gpt_r,dict) else "N/A"
    grok_sc=grok_r.get("ai_score","-") if isinstance(grok_r,dict) else "-"
    grok_an=grok_r.get("analysis","N/A") if isinstance(grok_r,dict) else "N/A"
    gem_sc=gemini_r.get("ai_score","-") if isinstance(gemini_r,dict) else "-"
    gem_an=gemini_r.get("analysis","N/A") if isinstance(gemini_r,dict) else "N/A"
    cl_sc_d=claude_r.get("ai_score",e_score) if isinstance(claude_r,dict) else e_score
    cl_an=claude_r.get("analysis","Engine") if isinstance(claude_r,dict) else eng.get("reason","")

    # Over/Under和BTTS: 优先用Claude的判断
    o25=eng.get("over_25",50)
    btts_v=eng.get("btts",45)
    if isinstance(claude_r,dict):
        if claude_r.get("over25") is not None:
            if claude_r["over25"]:o25=max(o25,60)
            else:o25=min(o25,45)
        if claude_r.get("btts") is not None:
            if claude_r["btts"]:btts_v=max(btts_v,55)
            else:btts_v=min(btts_v,40)

    return{
        "predicted_score":final_score,"home_win_pct":fhp,"draw_pct":fdp,"away_win_pct":fap,
        "confidence":cf,"risk_level":risk,
        "over_under_2_5":"大" if o25>55 else "小",
        "both_score":"是" if btts_v>50 else "否",
        "gpt_score":gpt_sc,"gpt_analysis":gpt_an,
        "grok_score":grok_sc,"grok_analysis":grok_an,
        "gemini_score":gem_sc,"gemini_analysis":gem_an,
        "claude_score":cl_sc_d,"claude_analysis":cl_an,
        "ai_avg_confidence":round(avg_ai_conf,1),
        "value_kill_count":sum(1 for _,r in ai_all.items() if isinstance(r,dict) and r.get("value_kill")),
        "model_agreement":len(set(ai_scores))<=1 and len(ai_scores)>=2,
        "claude_prob_fused":cl_fused,
        "poisson":stats.get("poisson",{}),"refined_poisson":stats.get("refined_poisson",{}),
        "extreme_warning":eng.get("scissors_gap_signal",""),
        "smart_money_signal":" | ".join(stats.get("smart_signals",[])),
        "smart_signals":stats.get("smart_signals",[]),
        "model_consensus":stats.get("model_consensus",0),"total_models":stats.get("total_models",11),
        "expected_total_goals":eng.get("expected_goals",2.5),
        "over_2_5":o25,"btts":btts_v,
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
        "bookmaker_implied_home_xg":eng.get("bookmaker_implied_home_xg","?"),
        "bookmaker_implied_away_xg":eng.get("bookmaker_implied_away_xg","?"),
        "result":"主胜",
    }

# ====================================================================
def select_top4(preds):
    for p in preds:
        pr=p.get("prediction",{});s=pr.get("confidence",0)*0.4
        mx=max(pr.get("home_win_pct",33),pr.get("away_win_pct",33),pr.get("draw_pct",33))
        s+=(mx-33)*0.2+pr.get("model_consensus",0)*2
        if pr.get("risk_level")=="低":s+=12
        elif pr.get("risk_level")=="高":s-=5
        if pr.get("model_agreement"):s+=10
        # Claude概率融合成功时加分
        if pr.get("claude_prob_fused"):s+=6
        exp=pr.get("experience_analysis",{})
        if exp.get("total_score",0)>=15 and pr.get("result")=="平局" and exp.get("draw_rules",0)>=3:s+=12
        elif exp.get("total_score",0)>=10:s+=5
        if exp.get("recommendation","").startswith("⚠️"):s-=3
        sm=str(pr.get("smart_money_signal",""));d=pr.get("result","")
        if "Sharp" in sm:
            if("客胜" in sm and d=="主胜") or ("主胜" in sm and d=="客胜"):s-=30
        ps=pr.get("predictability_score",50)
        if ps>=70:s+=8
        elif ps<35:s-=10
        # CLV加分
        sigs=pr.get("smart_signals",[])
        for sig in sigs:
            if "CLV+" in str(sig) and "🔥" in str(sig):s+=7;break
        p["recommend_score"]=round(s,2)
    preds.sort(key=lambda x:x.get("recommend_score",0),reverse=True)
    return preds[:4]

def extract_num(ms):
    wm={"一":1000,"二":2000,"三":3000,"四":4000,"五":5000,"六":6000,"日":7000,"天":7000}
    base=next((v for k,v in wm.items() if k in str(ms)),0)
    nums=re.findall(r"\d+",str(ms))
    return base+int(nums[0]) if nums else 9999

# ====================================================================
def run_predictions(raw,use_ai=True):
    ms=raw.get("matches",[])
    print("\n"+"="*80)
    print("  [QUANT ENGINE vMAX] %d matches | Claude=StatBrain(2.0) + 3xScout"%len(ms))
    print("="*80)
    match_analyses=[]
    for i,m in enumerate(ms):
        try:
            eng=predict_match(m)
            lg_info,_,_,_=build_league_intelligence(m)
            sp=ensemble.predict(m,{})
            exp_result=exp_engine.analyze(m)
        except Exception as e:
            print("  ⚠️ Match %d init error: %s"%(i+1,e))
            eng={"primary_score":"1-1","confidence":40,"home_prob":33,"draw_prob":34,"away_prob":33,"top3_scores":["1-1","0-0","1-0"],"bookmaker_implied_home_xg":1.3,"bookmaker_implied_away_xg":1.1}
            sp={};exp_result={};lg_info=""
        match_analyses.append({"match":m,"engine":eng,"league_info":lg_info,"stats":sp,"index":i+1,"experience":exp_result})

    all_ai={"claude":{},"gemini":{},"gpt":{},"grok":{}}
    if use_ai and match_analyses:
        claude_p=build_claude_prompt(match_analyses)
        light_p=build_light_prompt(match_analyses)
        save_pct=round((1-len(light_p)/max(1,len(claude_p)))*100)
        print("  [PROMPT] Claude:%d chars | Scout:%d chars | Save %d%%"%(len(claude_p),len(light_p),save_pct))
        t0=time.time()
        all_ai=asyncio.run(run_ai_matrix(claude_p,light_p,len(match_analyses)))
        print("  [AI DONE] %.1fs"%(time.time()-t0))

    res=[]
    for i,ma in enumerate(match_analyses):
        m=ma["match"];idx=i+1
        mg=merge_result(ma["engine"],all_ai["gpt"].get(idx,{}),all_ai["grok"].get(idx,{}),all_ai["gemini"].get(idx,{}),all_ai["claude"].get(idx,{}),ma["stats"],m)

        # 4层增强管线（独立防崩溃）
        try:mg=apply_experience_to_prediction(m,mg,exp_engine)
        except Exception as e:print("    ⚠️ exp:%s"%e)
        try:mg=apply_odds_history(m,mg)
        except Exception as e:print("    ⚠️ hist:%s"%e)
        try:mg=apply_quant_edge(m,mg)
        except Exception as e:print("    ⚠️ qe:%s"%e)
        try:mg=upgrade_ensemble_predict(m,mg)
        except Exception as e:print("    ⚠️ adv:%s"%e)

        pcts={"主胜":mg.get("home_win_pct",33),"平局":mg.get("draw_pct",33),"客胜":mg.get("away_win_pct",34)}
        mg["result"]=max(pcts,key=pcts.get)
        res.append({**m,"prediction":mg})
        fused_tag=" [🧠FUSED]" if mg.get("claude_prob_fused") else ""
        print("  [%d] %s vs %s => %s (%s) CF:%s%%%s"%(idx,m.get("home_team","?"),m.get("away_team","?"),mg["result"],mg["predicted_score"],mg["confidence"],fused_tag))

    t4=select_top4(res)
    t4ids=[t["id"] for t in t4]
    for r in res:r["is_recommended"]=r["id"] in t4ids
    res.sort(key=lambda x:extract_num(x.get("match_num","")))
    diary=load_ai_diary()
    diary["reflection"]="Claude=StatBrain概率融合(20%%w)+一票否决+Scout省%d%%token"%save_pct if use_ai else "no_ai_mode"
    save_ai_diary(diary)
    return res,t4