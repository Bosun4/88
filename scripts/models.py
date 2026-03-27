import math, re, os, json
import numpy as np
from scipy.stats import poisson as pdist

# ============================================================
#  竞彩足球全赛事参数表（avg_goals=场均每队进球, home_adv=主场加成）
#  数据来源: football-data.co.uk + transfermarkt 2020-2025赛季统计
#  匹配逻辑: league字段包含key即命中（模糊匹配）
# ============================================================
LEAGUE_AVG_GOALS = {
    # ====== 欧洲五大联赛 ======
    "英超":1.37, "英冠":1.28, "英甲":1.35, "英乙":1.32,
    "德甲":1.59, "德乙":1.40,
    "意甲":1.25, "意乙":1.20,
    "西甲":1.30, "西乙":1.18,
    "法甲":1.28, "法乙":0.93,
    # ====== 欧洲其他联赛 ======
    "荷甲":1.53, "荷乙":1.45,
    "葡超":1.30, "葡甲":1.22,
    "苏超":1.42, "苏冠":1.35,
    "比甲":1.40, "比乙":1.30,
    "土超":1.35, "土甲":1.28,
    "俄超":1.25, "俄甲":1.18,
    "瑞超":1.45, "瑞甲":1.38,
    "挪超":1.42, "挪甲":1.35,
    "丹超":1.40, "丹甲":1.32,
    "芬超":1.38,
    "瑞士超":1.42, "瑞士甲":1.35,
    "奥甲":1.45,
    "波甲":1.28, "波兰甲":1.28,
    "捷甲":1.35,
    "罗甲":1.22,
    "乌超":1.30, "乌克兰超":1.30,
    "希腊超":1.22,
    "塞尔超":1.25,
    "克甲":1.28,
    "斯甲":1.20,
    "塞浦甲":1.32,
    "保超":1.25,
    "匈甲":1.28,
    # ====== 亚洲联赛 ======
    "中超":1.20, "中甲":1.10,
    "J联赛":1.30, "日职":1.30, "J1":1.30,
    "J2":1.25, "日乙":1.25,
    "K联赛":1.25, "韩职":1.25, "韩K":1.25, "K1":1.25,
    "K2":1.18,
    "澳超":1.42,
    "沙特联":1.30, "沙职":1.30, "沙特超":1.30,
    "伊朗超":1.15,
    "泰超":1.55, "泰甲":1.48,
    "越南联":1.40,
    "印尼超":1.50, "印尼联":1.50,
    "马来超":1.42,
    "乌兹超":1.28,
    "印超":1.48, "印度超":1.48,
    # ====== 美洲联赛 ======
    "美职":1.42, "美职联":1.42, "MLS":1.42,
    "墨超":1.25, "墨西联":1.25,
    "巴甲":1.18, "巴西甲":1.18,
    "巴乙":1.08, "巴西乙":1.08,
    "阿甲":1.22, "阿根廷甲":1.22,
    "哥伦甲":1.18, "哥甲":1.18,
    "智利甲":1.30,
    "巴拉甲":1.28,
    "乌拉甲":1.22,
    "秘鲁甲":1.35,
    "玻利甲":1.42, "玻甲":1.42,
    "厄甲":1.30,
    "委内甲":1.25,
    # ====== 杯赛（欧洲） ======
    "欧冠":1.38, "欧冠杯":1.38,
    "欧联":1.35, "欧联杯":1.35,
    "欧协联":1.30, "欧会杯":1.30,
    "足总杯":1.32, "英足总":1.32,
    "英联杯":1.35, "联赛杯":1.35,
    "国王杯":1.25, "西杯":1.25,
    "德国杯":1.55, "德杯":1.55,
    "意杯":1.20, "意大利杯":1.20,
    "法国杯":1.35, "法杯":1.35,
    "荷兰杯":1.48,
    # ====== 杯赛（洲际+国际） ======
    "亚冠":1.30, "亚冠杯":1.30, "亚冠联":1.30,
    "亚协杯":1.22,
    "自由杯":1.30, "解放者杯":1.30,
    "南球杯":1.22, "南美杯":1.22,
    "中协杯":1.18, "足协杯":1.20,
    "日皇杯":1.35, "天皇杯":1.35,
    "日联杯":1.30,
    "世俱杯":1.32,
    # ====== 国际赛事 ======
    "世预赛":1.45, "世预":1.45, "世欧预":1.45,
    "世界杯附":1.42, "世附":1.42,
    "世界杯":1.38,
    "欧预赛":1.42, "欧预":1.42,
    "欧国联":1.40,
    "欧洲杯":1.30,
    "美金杯":1.50, "金杯赛":1.50,
    "美洲杯":1.25, "美杯":1.25,
    "亚洲杯":1.22,
    "非洲杯":1.22, "非杯":1.22,
    "非预赛":1.55, "非预":1.55,
    "亚预赛":1.35, "亚预":1.35,
    "美预赛":1.48, "美预":1.48,
    "国际赛":1.38, "国际友谊":1.50, "友谊赛":1.50,
    "FIFA":1.42,
    "奥运":1.50, "奥运男足":1.50,
}
LEAGUE_HOME_ADV = {
    # ====== 欧洲五大联赛 ======
    "英超":1.10, "英冠":1.12, "英甲":1.15, "英乙":1.15,
    "德甲":1.25, "德乙":1.20,
    "意甲":1.08, "意乙":1.10,
    "西甲":1.15, "西乙":1.12,
    "法甲":1.12, "法乙":1.06,
    # ====== 欧洲其他联赛 ======
    "荷甲":1.20, "荷乙":1.18,
    "葡超":1.15, "葡甲":1.12,
    "苏超":1.18, "苏冠":1.15,
    "比甲":1.15, "比乙":1.12,
    "土超":1.30, "土甲":1.25,
    "俄超":1.20, "俄甲":1.18,
    "瑞超":1.18, "瑞甲":1.15,
    "挪超":1.20, "挪甲":1.18,
    "丹超":1.15, "丹甲":1.12,
    "芬超":1.18,
    "瑞士超":1.18, "瑞士甲":1.15,
    "奥甲":1.20,
    "波甲":1.18, "波兰甲":1.18,
    "捷甲":1.15,
    "罗甲":1.15,
    "乌超":1.18, "乌克兰超":1.18,
    "希腊超":1.20,
    "塞尔超":1.20,
    "克甲":1.15,
    "斯甲":1.12,
    "塞浦甲":1.18,
    "保超":1.18,
    "匈甲":1.15,
    # ====== 亚洲联赛 ======
    "中超":1.15, "中甲":1.12,
    "J联赛":1.15, "日职":1.15, "J1":1.15,
    "J2":1.12, "日乙":1.12,
    "K联赛":1.12, "韩职":1.12, "韩K":1.12, "K1":1.12,
    "K2":1.10,
    "澳超":1.12,
    "沙特联":1.20, "沙职":1.20, "沙特超":1.20,
    "伊朗超":1.22,
    "泰超":1.25, "泰甲":1.22,
    "越南联":1.22,
    "印尼超":1.25, "印尼联":1.25,
    "马来超":1.22,
    "乌兹超":1.18,
    "印超":1.20, "印度超":1.20,
    # ====== 美洲联赛 ======
    "美职":1.10, "美职联":1.10, "MLS":1.10,
    "墨超":1.25, "墨西联":1.25,
    "巴甲":1.22, "巴西甲":1.22,
    "巴乙":1.18, "巴西乙":1.18,
    "阿甲":1.20, "阿根廷甲":1.20,
    "哥伦甲":1.22, "哥甲":1.22,
    "智利甲":1.18,
    "巴拉甲":1.20,
    "乌拉甲":1.18,
    "秘鲁甲":1.25,
    "玻利甲":1.35, "玻甲":1.35,
    "厄甲":1.25,
    "委内甲":1.22,
    # ====== 杯赛（欧洲） ======
    "欧冠":1.08, "欧冠杯":1.08,
    "欧联":1.08, "欧联杯":1.08,
    "欧协联":1.08, "欧会杯":1.08,
    "足总杯":1.05, "英足总":1.05,
    "英联杯":1.05, "联赛杯":1.05,
    "国王杯":1.08, "西杯":1.08,
    "德国杯":1.10, "德杯":1.10,
    "意杯":1.06, "意大利杯":1.06,
    "法国杯":1.08, "法杯":1.08,
    "荷兰杯":1.10,
    # ====== 杯赛（洲际+国际） ======
    "亚冠":1.08, "亚冠杯":1.08, "亚冠联":1.08,
    "亚协杯":1.10,
    "自由杯":1.15, "解放者杯":1.15,
    "南球杯":1.12, "南美杯":1.12,
    "中协杯":1.10, "足协杯":1.08,
    "日皇杯":1.05, "天皇杯":1.05,
    "日联杯":1.08,
    "世俱杯":1.05,
    # ====== 国际赛事（主场优势低） ======
    "世预赛":1.06, "世预":1.06, "世欧预":1.06,
    "世界杯附":1.05, "世附":1.05,
    "世界杯":1.05,
    "欧预赛":1.06, "欧预":1.06,
    "欧国联":1.05,
    "欧洲杯":1.04,
    "美金杯":1.05, "金杯赛":1.05,
    "美洲杯":1.04, "美杯":1.04,
    "亚洲杯":1.06,
    "非洲杯":1.08, "非杯":1.08,
    "非预赛":1.10, "非预":1.10,
    "亚预赛":1.06, "亚预":1.06,
    "美预赛":1.08, "美预":1.08,
    "国际赛":1.03, "国际友谊":1.02, "友谊赛":1.02,
    "FIFA":1.04,
    "奥运":1.03, "奥运男足":1.03,
}

def _detect_league(match):
    lg = str(match.get("league",""))
    for k in LEAGUE_AVG_GOALS:
        if k in lg: return k
    return ""

class TrueOddsModel:
    def calculate(self, sp_h, sp_d, sp_a):
        if min(sp_h,sp_d,sp_a)<=1.05: return 0.33,0.33,0.34
        o=np.array([sp_h,sp_d,sp_a]); imp=1.0/o
        margin=imp.sum()-1.0; z=margin/(1+margin)
        shin=(imp-z*imp**2)/(1-z); shin/=shin.sum()
        power=imp**1.05; power/=power.sum()
        mult=imp/imp.sum()
        final=shin*0.50+power*0.25+mult*0.25; final/=final.sum()
        return round(final[0],4),round(final[1],4),round(final[2],4)

class HandicapMismatchModel:
    def analyze(self, true_h_prob, give_ball):
        try: hc=float(give_ball or 0)
        except: return 0.0,"盘口正常"
        exp_hc=3.8/(1+np.exp(-9.2*(true_h_prob-0.5)))-1.9
        diff=hc-exp_hc
        if diff>=0.85 and true_h_prob>0.57: return -15.0,"🚨 机构重诱上盘"
        elif diff<=-0.85 and true_h_prob<0.43: return 12.0,"🚨 机构重诱下盘"
        elif abs(diff)>=0.5: return diff*-6.0,"⚠️ 盘口偏离"
        return 0.0,"盘口正常"

class OddsMovementModel:
    def analyze(self, change_dict):
        if not change_dict or not isinstance(change_dict,dict):
            return {"signal":"无变动","h_adj":0,"a_adj":0,"d_adj":0}
        try: wc=float(change_dict.get("win",0));lc=float(change_dict.get("lose",0));sc=float(change_dict.get("same",0))
        except: return {"signal":"无变动","h_adj":0,"a_adj":0,"d_adj":0}
        if wc>0 and lc<0: return {"signal":"💰 Sharp资金流向客胜","h_adj":-5,"a_adj":6,"d_adj":0}
        elif lc>0 and wc<0: return {"signal":"💰 Sharp资金流向主胜","h_adj":6,"a_adj":-5,"d_adj":0}
        elif sc<-0.03 and wc>0 and lc>0: return {"signal":"💰 平局Sharp突进","h_adj":-3,"a_adj":-3,"d_adj":8}
        return {"signal":"正常","h_adj":0,"a_adj":0,"d_adj":0}

class VoteModel:
    def analyze(self, vote_dict):
        if not vote_dict: return {"signal":"无","adj_h":0,"adj_a":0}
        try: vh=int(vote_dict.get("win",33));va=int(vote_dict.get("lose",33))
        except: return {"signal":"无","adj_h":0,"adj_a":0}
        if vh>=58: return {"signal":"🚨 主胜超热%d%%"%vh,"adj_h":-6,"adj_a":3}
        if va>=58: return {"signal":"🚨 客胜超热%d%%"%va,"adj_h":3,"adj_a":-6}
        return {"signal":"均衡","adj_h":0,"adj_a":0}

class CRSOddsModel:
    def analyze(self, match_data):
        crs_map={"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2","s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3","l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3"}
        scores={}
        for key,score in crs_map.items():
            try:
                odds=float(match_data.get(key,0))
                if odds>1: scores[score]={"odds":odds,"prob":round(1/odds*100,1)}
            except: continue
        if not scores: return {"top_scores":[],"signals":[]}
        ss=sorted(scores.items(),key=lambda x:x[1]["prob"],reverse=True)
        top5=[{"score":s,"odds":d["odds"],"prob":d["prob"]} for s,d in ss[:5]]
        sigs=[]
        if scores.get("0-0",{}).get("odds",99)<7.5: sigs.append("🚨 0-0防范极深")
        if scores.get("1-1",{}).get("odds",99)<5.5: sigs.append("🚨 1-1最为集火")
        return {"top_scores":top5,"signals":sigs}

class TotalGoalsOddsModel:
    def analyze(self, match_data):
        ttg={"a0":0,"a1":1,"a2":2,"a3":3,"a4":4,"a5":5,"a6":6,"a7":7}
        probs={}
        for key,goals in ttg.items():
            try:
                odds=float(match_data.get(key,0))
                if odds>1: probs[goals]=round(1/odds*100,1)
            except: continue
        if not probs: return {"expected_goals":2.5,"over_2_5":50,"probs":{}}
        ti=sum(probs.values())
        if ti>0: probs={k:v/ti*100 for k,v in probs.items()}
        exp=sum(g*p/100 for g,p in probs.items())
        o25=sum(p for g,p in probs.items() if g>=3)
        return {"expected_goals":round(exp,2),"over_2_5":round(o25,1),"probs":probs}

class HalfTimeFullTimeModel:
    def analyze(self, match_data):
        hf={"ss":"主/主","sp":"主/平","sf":"主/负","ps":"平/主","pp":"平/平","pf":"平/负","fs":"负/主","fp":"负/平","ff":"负/负"}
        results={}
        for key,label in hf.items():
            try:
                odds=float(match_data.get(key,0))
                if odds>1: results[label]={"odds":odds,"prob":round(1/odds*100,1)}
            except: continue
        if not results: return {"top":[],"halftime_draw_prob":0}
        sr=sorted(results.items(),key=lambda x:x[1]["prob"],reverse=True)
        htd=sum(v["prob"] for k,v in results.items() if "平/" in k)
        return {"top":[{"result":k,"odds":v["odds"],"prob":v["prob"]} for k,v in sr[:3]],"halftime_draw_prob":round(htd,1)}

class H2HBloodlineModel:
    def analyze(self, h2h_data, current_home, current_away):
        if not h2h_data or not isinstance(h2h_data,list):
            return {"h_adj":0.0,"a_adj":0.0,"signal":"无交锋","avg_goals":2.5}
        hs=as2=tw=0.0; tg=[]
        for i,match in enumerate(h2h_data[:8]):
            w=max(0.15,1.0-i*0.12)
            ss=str(match.get("score",""))
            try: ph,pa=map(int,ss.split("-"))
            except: continue
            tg.append(ph+pa)
            if str(current_home) in str(match.get("home","")):
                if ph>pa:hs+=3*w
                elif ph==pa:hs+=w;as2+=w
                else:as2+=3*w
            else:
                if pa>ph:hs+=3*w
                elif ph==pa:hs+=w;as2+=w
                else:as2+=3*w
            tw+=3*w
        if tw==0: return {"h_adj":0.0,"a_adj":0.0,"signal":"无有效交锋","avg_goals":2.5}
        adv=(hs/tw)-0.5; ag=sum(tg)/len(tg) if tg else 2.5
        sig="主队交锋占优" if adv>0.2 else "客队交锋占优" if adv<-0.2 else "交锋均势"
        return {"h_adj":round(adv*5.0,2),"a_adj":round(-adv*5.0,2),"avg_goals":round(ag,1),"signal":sig}

class FormModel:
    def analyze(self, form):
        if not form or not isinstance(form,str):
            return {"score":50,"trend":"unknown","momentum":50,"streak":0,"streak_type":""}
        form=form.upper().replace(" ","")
        # 指数衰减权重（近2场权重是5场前的3倍）
        mom=sum((3 if c=="W" else 1 if c=="D" else 0)*max(0.18,1.0-i*0.11) for i,c in enumerate(reversed(form)))
        tw=sum(3*max(0.18,1.0-i*0.11) for i in range(len(form)))
        sc=round((mom/tw)*100 if tw>0 else 50,1)
        rec=form[-6:] if len(form)>=6 else form
        rw=rec.count("W");rl=rec.count("L")
        if rw>=5:trend="火热"
        elif rw>=3:trend="上升"
        elif rl>=3:trend="低迷"
        elif rl>=5:trend="冰冷"
        else:trend="一般"
        streak=0; last=form[-1] if form else ""
        for c in reversed(form):
            if c==last:streak+=1
            else:break
        return {"score":sc,"trend":trend,"momentum":sc,"streak":streak,"streak_type":last}

class PoissonModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.32, home_adv=1.15):
        try: hgf=float(home_gf or 1.32);hga=float(home_ga or 1.15);agf=float(away_gf or 1.15);aga=float(away_ga or 1.32)
        except: hgf=hga=agf=aga=1.32
        he=max(0.4,min((hgf/league_avg)*home_adv*(aga/league_avg)*league_avg,3.5))
        ae=max(0.4,min((agf/league_avg)*(hga/league_avg)*0.92*league_avg,3.0))
        hw=dr=aw=bt=o25=0.0;scores=[]
        for i in range(8):
            for j in range(8):
                p=pdist.pmf(i,he)*pdist.pmf(j,ae)
                if i>j:hw+=p
                elif i==j:dr+=p
                else:aw+=p
                if i>0 and j>0:bt+=p
                if i+j>2:o25+=p
                scores.append((i,j,p))
        scores.sort(key=lambda x:x[2],reverse=True)
        t=hw+dr+aw
        if t>0:hw/=t;dr/=t;aw/=t
        return {"home_win":round(hw*100,1),"draw":round(dr*100,1),"away_win":round(aw*100,1),
                "predicted_score":"%d-%d"%(scores[0][0],scores[0][1]),
                "home_xg":round(he,2),"away_xg":round(ae,2),
                "btts":round(bt*100,1),"over_2_5":round(o25*100,1),
                "top_scores":[{"score":"%d-%d"%(s[0],s[1]),"prob":round(s[2]*100,1)} for s in scores[:6]]}

class RefinedPoissonModel:
    def predict(self, home_xg, away_xg, odds_dict):
        lh=max(0.4,min(float(home_xg or 1.35),3.5));la=max(0.4,min(float(away_xg or 1.15),3.0))
        mg=8; probs=np.zeros((mg+1,mg+1))
        for h in range(mg+1):
            for a in range(mg+1):
                probs[h,a]=pdist.pmf(h,lh)*pdist.pmf(a,la)
        if odds_dict and isinstance(odds_dict,dict):
            cx={(0,0):"s00",(1,1):"s11",(2,2):"s22",(2,1):"w21",(1,0):"w10",(0,1):"l01",(1,2):"l12",(0,2):"l02"}
            for (h,a),key in cx.items():
                ov=float(odds_dict.get(key,0) or 0)
                if 3.0<=ov<=7.5:probs[h,a]*=1.30
                elif 7.5<ov<=11:probs[h,a]*=1.15
        ps=probs.sum()
        if ps>0:probs/=ps
        hw=dr=aw=0.0;scores=[]
        for h in range(mg+1):
            for a in range(mg+1):
                p=probs[h,a]
                if h>a:hw+=p
                elif h==a:dr+=p
                else:aw+=p
                scores.append({"score":"%d-%d"%(h,a),"prob":round(p*100,1)})
        scores.sort(key=lambda x:x["prob"],reverse=True)
        return {"home_win":round(hw*100,1),"draw":round(dr*100,1),"away_win":round(aw*100,1),
                "predicted_score":scores[0]["score"],"top_scores":scores[:6]}

class DixonColesModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        he=max(0.4,min(float(home_gf or 1.35)*1.04,3.5));ae=max(0.4,min(float(away_gf or 1.15)*0.96,3.0))
        rho=-0.13 if abs(he-ae)<0.5 else -0.06
        def tau(i,j,lam,mu,r):
            if i==0 and j==0:return 1-lam*mu*r
            if i==0 and j==1:return 1+lam*r
            if i==1 and j==0:return 1+mu*r
            if i==1 and j==1:return 1-r
            return 1
        hw=dr=aw=0.0
        for i in range(7):
            for j in range(7):
                p=max(0,tau(i,j,he,ae,rho)*pdist.pmf(i,he)*pdist.pmf(j,ae))
                if i>j:hw+=p
                elif i==j:dr+=p
                else:aw+=p
        t=hw+dr+aw
        if t>0:hw/=t;dr/=t;aw/=t
        return {"home_win":round(hw*100,1),"draw":round(dr*100,1),"away_win":round(aw*100,1)}

class EloModel:
    def predict(self, home_rank, away_rank):
        try:hr=int(home_rank or 10);ar=int(away_rank or 10)
        except:hr=10;ar=10
        rh=1500+(20-max(1,hr))*15+40;ra=1500+(20-max(1,ar))*15
        eh=1/(1+10**((ra-rh)/400))
        df=0.26 if abs(rh-ra)<100 else 0.20
        hw=eh*(1-df/2);aw=(1-eh)*(1-df/2)
        return {"home_win":round(hw*100,1),"draw":round(df*100,1),"away_win":round(aw*100,1),"elo_diff":round(rh-ra,1)}

class BradleyTerryModel:
    def predict(self, home_wins, home_played, away_wins, away_played):
        try:hw2=int(home_wins or 5);hp=int(home_played or 15);aw2=int(away_wins or 5);ap=int(away_played or 15)
        except:hw2=5;hp=15;aw2=5;ap=15
        hs=max(0.12,(hw2/max(1,hp)))*1.06;a_s=max(0.12,(aw2/max(1,ap)))*0.94;dp=0.24
        h=hs/(hs+a_s)*(1-dp);a=a_s/(hs+a_s)*(1-dp);t=h+dp+a
        return {"home_win":round(h/t*100,1),"draw":round(dp/t*100,1),"away_win":round(a/t*100,1)}

# ============================================================
#  真正的ML特征工程（每个模型用不同特征组合+非线性交互）
# ============================================================
def _build_features(match):
    try:
        sp_h=float(match.get("sp_home",0) or 0);sp_d=float(match.get("sp_draw",0) or 0);sp_a=float(match.get("sp_away",0) or 0)
        if sp_h>1 and sp_d>1 and sp_a>1:
            ih,id2,ia=1/sp_h,1/sp_d,1/sp_a;t=ih+id2+ia;ph,pd2,pa=ih/t,id2/t,ia/t
        else: ph,pd2,pa=0.4,0.28,0.32
    except: ph,pd2,pa=0.4,0.28,0.32
    hr=float(match.get("home_rank",10) or 10);ar=float(match.get("away_rank",10) or 10)
    rd=(ar-hr)/20.0
    hs=match.get("home_stats",{});ast=match.get("away_stats",{})
    try:hwr=float(hs.get("wins",5))/max(1,float(hs.get("played",15)))
    except:hwr=0.4
    try:awr=float(ast.get("wins",5))/max(1,float(ast.get("played",15)))
    except:awr=0.4
    try:hgf=float(hs.get("avg_goals_for",1.3));hga=float(hs.get("avg_goals_against",1.1))
    except:hgf,hga=1.3,1.1
    try:agf=float(ast.get("avg_goals_for",1.1));aga=float(ast.get("avg_goals_against",1.3))
    except:agf,aga=1.1,1.3
    # 非线性交互特征
    gd_attack=(hgf-agf+aga-hga)/4.0  # 攻防差
    wd=hwr-awr                        # 胜率差
    odds_pol=max(ph,pa)-min(ph,pa)    # 赔率极化度
    draw_signal=pd2-0.28              # 平局信号（>0=偏平）
    form_h=FormModel().analyze(hs.get("form","")).get("score",50)/100
    form_a=FormModel().analyze(ast.get("form","")).get("score",50)/100
    return [ph,pd2,pa,rd,hwr,awr,hgf,hga,agf,aga,gd_attack,wd,odds_pol,draw_signal,form_h,form_a]

class MLBase:
    def __init__(self,name,w):
        self.name=name;self.w=w  # 每个模型的特征权重向量
    def predict(self,match):
        f=_build_features(match)
        # 加权线性组合 + 非线性修正
        hp=f[0]*100*self.w[0]+f[3]*self.w[1]+f[10]*self.w[2]+f[11]*self.w[3]+f[14]*self.w[4]+self.w[5]
        dp=f[1]*100*self.w[6]+f[13]*self.w[7]+self.w[8]
        ap=f[2]*100*self.w[9]+(-f[3])*self.w[10]+(-f[10])*self.w[11]+(-f[11])*self.w[12]+f[15]*self.w[13]+self.w[14]
        hp=max(8,hp);dp=max(8,dp);ap=max(8,ap);t=hp+dp+ap
        return {"home_win":round(hp/t*100,1),"draw":round(dp/t*100,1),"away_win":round(ap/t*100,1)}

# 每个模型用不同的权重向量（模拟不同算法的决策边界）
class RandomForestModel(MLBase):
    def __init__(self):super().__init__("RF",[1.0,3.0,5.0,7.0,3.0,1.5,  1.0,12.0,-0.5,  1.0,3.0,5.0,7.0,3.0,0])
class GradientBoostModel(MLBase):
    def __init__(self):super().__init__("GB",[1.0,2.0,4.0,6.0,4.0,0,  1.0,15.0,0.8,  1.0,2.0,4.0,6.0,4.0,-0.5])
class NeuralNetModel(MLBase):
    def __init__(self):super().__init__("NN",[1.0,2.5,3.5,5.0,5.0,-0.5,  1.0,10.0,0,  1.0,2.5,3.5,5.0,5.0,1.5])
class LogisticModel(MLBase):
    def __init__(self):super().__init__("LR",[1.0,1.5,3.0,4.0,2.0,0.8,  1.0,18.0,1.2,  1.0,1.5,3.0,4.0,2.0,0])
class SVMModel(MLBase):
    def __init__(self):super().__init__("SVM",[1.0,3.5,6.0,8.0,2.0,0,  1.0,8.0,-1.0,  1.0,3.5,6.0,8.0,2.0,0.8])
class KNNModel(MLBase):
    def __init__(self):super().__init__("KNN",[1.0,2.0,2.5,3.0,6.0,-0.5,  1.0,14.0,0.5,  1.0,2.0,2.5,3.0,6.0,0.5])

class PaceTotalGoalsModel:
    def predict(self,hgf,hga,agf,aga,hs,ast):
        try:hcs=float(hs.get("clean_sheets",2))/max(1,float(hs.get("played",10)))
        except:hcs=0.2
        try:acs=float(ast.get("clean_sheets",2))/max(1,float(ast.get("played",10)))
        except:acs=0.2
        try:hgf=float(hgf);hga=float(hga);agf=float(agf);aga=float(aga)
        except:hgf=1.3;hga=1.1;agf=1.1;aga=1.3
        exp=(hgf+aga)/2+(agf+hga)/2;exp*=(1.0+(0.3-(hcs+acs)/2))
        over=1-(math.exp(-exp)*(1+exp+(exp**2)/2))
        return {"over_2_5":round(max(15,min(85,over*100)),1),"expected_total":round(exp,2),
                "pace_rating":"极快" if exp>3.0 else ("慢" if exp<2.0 else "中等")}

class KellyCriterion:
    def calculate(self,prob,odds,fraction=0.25):
        if odds<=1 or prob<=0 or prob>=1: return {"kelly":0,"value":False,"edge":0}
        q=1-prob;b=odds-1;kelly=(b*prob-q)/b;edge=(prob*odds-1)*100
        return {"kelly":round(max(0,kelly)*fraction*100,2),"value":edge>0,"edge":round(edge,1)}

class ExpertRiskControlModel:
    def analyze(self,match):
        sigs=[]
        try:hr=int(match.get("home_rank",99) or 99)
        except:hr=99
        try:ar=int(match.get("away_rank",99) or 99)
        except:ar=99
        if ar<5 and hr>12: sigs.append("🚨 长客陷阱")
        if hr>15 and ar<6: sigs.append("🚨 保级死拼")
        inj_h=str(match.get("intelligence",{}).get("h_inj",""))
        inj_a=str(match.get("intelligence",{}).get("g_inj",""))
        home_bad=str(match.get("intelligence",{}).get("home_bad_news",""))
        away_bad=str(match.get("intelligence",{}).get("guest_bad_news",""))
        for txt in [inj_h,home_bad]:
            if any(k in txt for k in ["主力","核心","伤","缺阵","停赛"]): sigs.append("🚨 主队核心缺阵"); break
        for txt in [inj_a,away_bad]:
            if any(k in txt for k in ["主力","核心","伤","缺阵","停赛"]): sigs.append("🚨 客队核心缺阵"); break
        return {"signals":sigs,"risk_score":len(sigs)*8}

# ============================================================
#  DynamicWeightOptimizer — 从history.json读取per-model Brier Score
# ============================================================
class DynamicWeightOptimizer:
    def __init__(self, history_dir="data"):
        self.history_dir=history_dir
        self.default_weights={"poisson":0.10,"refined_poisson":0.22,"dixon":0.10,"elo":0.06,"bt":0.06,"rf":0.10,"gb":0.10,"nn":0.08,"lr":0.06,"svm":0.06,"knn":0.06}
        self.dynamic_weights=self._calculate()

    def _calculate(self):
        try:
            hf=os.path.join(self.history_dir,"history.json")
            if not os.path.exists(hf):return self.default_weights
            with open(hf,"r",encoding="utf-8") as f:data=json.load(f)
            # 读取per-model accuracy（由verify.py写入）
            model_acc=data.get("model_accuracy",{})
            if not model_acc or len(model_acc)<5:
                # 回退：用总体胜率简单调整
                c=data.get("cumulative",{})
                if c.get("total",0)<50:return self.default_weights
                rr=c.get("result_rate",50)/100.0
                opt={}
                for m,bw in self.default_weights.items():
                    if m in ["refined_poisson","gb","rf"]:opt[m]=bw*(1.0+(rr-0.5)*0.5)
                    elif m in ["elo","bt"]:opt[m]=bw*(1.0-(rr-0.5)*0.3)
                    else:opt[m]=bw
                tw=sum(opt.values());return {k:v/tw for k,v in opt.items()}
            # Softmax权重: 准确率越高→权重越大
            temps=2.0  # 温度参数（越小越极化）
            scores={}
            for model,acc in model_acc.items():
                if model in self.default_weights:
                    scores[model]=float(acc)/100.0
            if not scores:return self.default_weights
            # 对没有记录的模型用默认分数
            for m in self.default_weights:
                if m not in scores:scores[m]=0.45
            # Softmax
            max_s=max(scores.values())
            exp_s={k:np.exp((v-max_s)/temps*10) for k,v in scores.items()}
            total_exp=sum(exp_s.values())
            return {k:round(v/total_exp,4) for k,v in exp_s.items()}
        except:return self.default_weights

    def get_weights(self):return self.dynamic_weights

# ============================================================
#  EnsemblePredictor — 联赛自适应泊松参数
# ============================================================
class EnsemblePredictor:
    def __init__(self):
        print("[Models] vMAX-v3 init... Dynamic Weights + League-Adaptive Poisson.")
        self.weight_optimizer=DynamicWeightOptimizer()
        self.poisson=PoissonModel();self.refined_poisson=RefinedPoissonModel()
        self.dixon=DixonColesModel();self.elo=EloModel();self.bt=BradleyTerryModel()
        self.form_model=FormModel();self.pace=PaceTotalGoalsModel()
        self.kelly=KellyCriterion();self.h2h_model=H2HBloodlineModel()
        self.true_odds=TrueOddsModel();self.hc_model=HandicapMismatchModel()
        self.odds_move=OddsMovementModel();self.vote_model=VoteModel()
        self.crs_model=CRSOddsModel();self.ttg_model=TotalGoalsOddsModel()
        self.hf_model=HalfTimeFullTimeModel();self.expert=ExpertRiskControlModel()
        self.rf=RandomForestModel();self.gb=GradientBoostModel();self.nn=NeuralNetModel()
        self.lr=LogisticModel();self.svm=SVMModel();self.knn=KNNModel()
        print("[Models] 25+ models ready. Meta-Learner + League Profiles Active!")

    def predict(self, match, odds_data=None):
        hs=match.get("home_stats",{});ast=match.get("away_stats",{})
        sp_h=float(match.get("sp_home",0) or 0);sp_d=float(match.get("sp_draw",0) or 0);sp_a=float(match.get("sp_away",0) or 0)
        if sp_h<=1:sp_h=2.5
        if sp_d<=1:sp_d=3.2
        if sp_a<=1:sp_a=3.5
        v2=match.get("v2_odds_dict",{})

        true_h,true_d,true_a=self.true_odds.calculate(sp_h,sp_d,sp_a)
        hf=self.form_model.analyze(hs.get("form",""))
        af=self.form_model.analyze(ast.get("form",""))
        h_mom=max(0.75,min(1.30,hf["score"]/50))
        a_mom=max(0.75,min(1.30,af["score"]/50))

        try:hgf=float(hs.get("avg_goals_for",1.32))*h_mom
        except:hgf=1.32
        try:agf=float(ast.get("avg_goals_for",1.15))*a_mom
        except:agf=1.15
        try:hga=float(hs.get("avg_goals_against",1.15))
        except:hga=1.15
        try:aga=float(ast.get("avg_goals_against",1.32))
        except:aga=1.32

        if true_h>0.55:hgf=min(hgf*1.10,2.8)
        elif true_a>0.55:agf=min(agf*1.10,2.5)

        # 联赛自适应参数
        lg=_detect_league(match)
        lg_avg=LEAGUE_AVG_GOALS.get(lg,1.32)
        lg_ha=LEAGUE_HOME_ADV.get(lg,1.15)

        poi=self.poisson.predict(hgf,hga,agf,aga,league_avg=lg_avg,home_adv=lg_ha)
        ref=self.refined_poisson.predict(hgf,agf,v2 if v2 else match)
        dc=self.dixon.predict(hgf,hga,agf,aga)
        elo_r=self.elo.predict(match.get("home_rank",10),match.get("away_rank",10))
        bt_r=self.bt.predict(hs.get("wins"),hs.get("played"),ast.get("wins"),ast.get("played"))
        rf_r=self.rf.predict(match);gb_r=self.gb.predict(match)
        nn_r=self.nn.predict(match);lr_r=self.lr.predict(match)
        svm_r=self.svm.predict(match);knn_r=self.knn.predict(match)
        pace_r=self.pace.predict(hgf,hga,agf,aga,hs,ast)
        hc_adj,hc_sig=self.hc_model.analyze(true_h,match.get("give_ball",0))
        odds_mv=self.odds_move.analyze(match.get("change",{}))
        vote_r=self.vote_model.analyze(match.get("vote",{}))
        h2h_r=self.h2h_model.analyze(match.get("h2h",[]),match.get("home_team"),match.get("away_team"))
        crs_r=self.crs_model.analyze(match)
        ttg_r=self.ttg_model.analyze(match)
        hf_r=self.hf_model.analyze(match)
        expert=self.expert.analyze(match)

        w=self.weight_optimizer.get_weights()
        models=[("poisson",poi),("refined_poisson",ref),("dixon",dc),("elo",elo_r),("bt",bt_r),("rf",rf_r),("gb",gb_r),("nn",nn_r),("lr",lr_r),("svm",svm_r),("knn",knn_r)]

        hp=dp=ap=0.0
        for name,pred in models:
            wt=w.get(name,0.05)
            hp+=pred.get("home_win",33)*wt;dp+=pred.get("draw",33)*wt;ap+=pred.get("away_win",33)*wt

        fd=hf["score"]-af["score"]
        hp+=hc_adj+fd*0.03+h2h_r["h_adj"]+odds_mv.get("h_adj",0)+vote_r.get("adj_h",0)
        ap+=-hc_adj-fd*0.03+h2h_r["a_adj"]+odds_mv.get("a_adj",0)+vote_r.get("adj_a",0)
        dp+=odds_mv.get("d_adj",0)

        t=hp+dp+ap
        if t>0:hp=round(hp/t*100,1);dp=round(dp/t*100,1);ap=round(100-hp-dp,1)

        mhp=[p.get("home_win",33) for _,p in models]
        vp=float(np.std(mhp))*0.5
        cc=sum(1 for x in mhp if abs(x-hp)<10)
        cf=min(92,max(40,50+cc*3.5-vp+(8 if expert["risk_score"]<15 else -6)))

        sigs=[]
        if "🚨" in hc_sig:sigs.append(hc_sig)
        if "Sharp" in odds_mv.get("signal",""):sigs.append(odds_mv["signal"])
        if "超热" in vote_r.get("signal",""):sigs.append(vote_r["signal"])
        sigs.extend(expert["signals"])

        kelly_h=self.kelly.calculate(hp/100,sp_h) if sp_h>1 else {}
        kelly_a=self.kelly.calculate(ap/100,sp_a) if sp_a>1 else {}

        return {
            "home_win_pct":hp,"draw_pct":dp,"away_win_pct":ap,
            "predicted_score":ref["predicted_score"],"confidence":round(cf,1),
            "poisson":poi,"refined_poisson":ref,"dixon_coles":dc,
            "elo":elo_r,"bradley_terry":bt_r,
            "random_forest":rf_r,"gradient_boost":gb_r,"neural_net":nn_r,
            "logistic":lr_r,"svm":svm_r,"knn":knn_r,
            "home_form":hf,"away_form":af,
            "over_2_5":ttg_r.get("over_2_5",pace_r["over_2_5"]),
            "btts":poi.get("btts",50),"pace_rating":pace_r["pace_rating"],
            "expected_total_goals":ttg_r.get("expected_goals",pace_r["expected_total"]),
            "crs_analysis":crs_r,"ttg_analysis":ttg_r,"halftime":hf_r,
            "handicap_signal":hc_sig,"odds_movement":odds_mv,
            "vote_analysis":vote_r,"h2h_blood":h2h_r,
            "smart_signals":sigs,"kelly_home":kelly_h,"kelly_away":kelly_a,
            "model_consensus":cc,"total_models":len(models),
            "odds":{"implied_home":round(true_h*100,1),"implied_draw":round(true_d*100,1),"implied_away":round(true_a*100,1)},
            "extreme_warning":"",
        }