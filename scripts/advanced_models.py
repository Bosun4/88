#!/usr/bin/env python3
"""advanced_models.py v3.0 — 胜率核弹级升级 (5个致命弱点修复)"""
import numpy as np, os, json
from scipy.stats import poisson as pdist
from collections import Counter

class ZeroInflatedBivariatePoisson:
    def predict(self, home_xg, away_xg, correlation=0.10, p0_inflate=0.03, max_goals=7):
        try: hxg=max(0.15,float(home_xg or 1.3)); axg=max(0.15,float(away_xg or 1.1))
        except: hxg,axg=1.3,1.1
        l1=max(0.05,hxg-correlation); l2=max(0.05,axg-correlation); l3=max(0.0,min(correlation,min(hxg,axg)*0.5))
        total_xg=hxg+axg
        if total_xg<2.2: p0_inflate=0.05
        elif total_xg<2.6: p0_inflate=0.03
        else: p0_inflate=0.01
        mg=max_goals+1; probs=np.zeros((mg,mg))
        for x in range(mg):
            for y in range(mg):
                for k in range(min(x,y)+1):
                    probs[x,y]+=pdist.pmf(x-k,l1)*pdist.pmf(y-k,l2)*pdist.pmf(k,l3)
        probs[0,0]+=p0_inflate
        rho=-0.08 if abs(hxg-axg)<0.4 else -0.04
        probs[0,0]*=(1-l1*l2*rho); probs[1,0]*=(1+l2*rho); probs[0,1]*=(1+l1*rho); probs[1,1]*=(1-rho)
        ps=probs.sum()
        if ps>0: probs/=ps
        hw=dr=aw=bt=o25=0.0; scores=[]
        for x in range(mg):
            for y in range(mg):
                p=probs[x,y]
                if x>y:hw+=p
                elif x==y:dr+=p
                else:aw+=p
                if x>0 and y>0:bt+=p
                if x+y>2:o25+=p
                scores.append({"score":f"{x}-{y}","prob":round(p*100,2)})
        scores.sort(key=lambda x:x["prob"],reverse=True)
        return {"home_win":round(hw*100,1),"draw":round(dr*100,1),"away_win":round(aw*100,1),"predicted_score":scores[0]["score"],"btts":round(bt*100,1),"over_2_5":round(o25*100,1),"correlation":round(l3,3),"p0_inflate":round(p0_inflate,3),"top_scores":scores[:6]}

class SmartFilter:
    @staticmethod
    def score_predictability(match, prediction):
        score=50.0
        hp=prediction.get("home_win_pct",33); dp=prediction.get("draw_pct",33); ap=prediction.get("away_win_pct",34)
        mx=max(hp,dp,ap)
        if mx>=60:score+=15
        elif mx>=50:score+=8
        elif mx<38:score-=15
        mc=prediction.get("model_consensus",0); score+=mc*3
        if dp>32:score-=(dp-32)*1.5
        if dp>38:score-=10
        gap=str(prediction.get("scissors_gap_signal",prediction.get("extreme_warning","")))
        if gap and gap!="无" and "🚨" in gap:score-=12
        hs=match.get("home_stats",{})
        if str(hs.get("played","?"))=="?" or str(hs.get("form","?"))=="?":score-=20
        smart=str(prediction.get("smart_money_signal",""));result=prediction.get("result","")
        if "客胜" in smart and result=="主胜":score-=15
        elif "主胜" in smart and result=="客胜":score-=15
        sp_h=float(match.get("sp_home",0) or 0);sp_a=float(match.get("sp_away",0) or 0)
        if sp_h>0 and sp_a>0:
            if sp_h>8 or sp_a>8:score-=10
            if abs(sp_h-sp_a)<0.3:score-=8
        return max(0,min(100,round(score)))
    @staticmethod
    def should_skip(s):
        if s<30:return True,"🚫 强烈建议跳过"
        if s<40:return True,"⚠️ 建议跳过"
        return False,""

class AdaptiveCalibrator:
    def __init__(self, history_dir="data"):
        self.calibrated=False;self.shrink=0.92;self.draw_boost=0.0;self.confidence_penalty=0.0
        try:
            hf=os.path.join(history_dir,"history.json")
            if not os.path.exists(hf):return
            with open(hf,"r",encoding="utf-8") as f:data=json.load(f)
            c=data.get("cumulative",{});total=c.get("total",0)
            if total<20:return
            rr=c.get("result_rate",50)
            if rr<42:self.shrink=0.85;self.draw_boost=4.0;self.confidence_penalty=8.0
            elif rr<48:self.shrink=0.88;self.draw_boost=2.5;self.confidence_penalty=4.0
            elif rr<54:self.shrink=0.92;self.draw_boost=1.0;self.confidence_penalty=0.0
            elif rr<60:self.shrink=0.95;self.draw_boost=0.0;self.confidence_penalty=0.0
            else:self.shrink=0.97;self.draw_boost=-1.0;self.confidence_penalty=-3.0
            self.calibrated=True
        except:pass
    def calibrate(self,hp,dp,ap):
        if not self.calibrated:return hp,dp,ap
        mean=(hp+dp+ap)/3
        hc=max(5,hp*self.shrink+mean*(1-self.shrink));dc=max(5,dp*self.shrink+mean*(1-self.shrink)+self.draw_boost);ac=max(5,ap*self.shrink+mean*(1-self.shrink))
        t=hc+dc+ac;return round(hc/t*100,1),round(dc/t*100,1),round(100-hc/t*100-dc/t*100,1)
    def adjust_confidence(self,conf):return max(25,min(92,conf-self.confidence_penalty))

class FavouriteLongshotBias:
    @staticmethod
    def correct(hp,dp,ap,sp_h,sp_d,sp_a):
        if min(sp_h,sp_d,sp_a)<=1.0:return hp,dp,ap
        corrections=[]
        for prob,odds in [(hp,sp_h),(dp,sp_d),(ap,sp_a)]:
            if odds>5.0:f=0.88
            elif odds>3.5:f=0.94
            elif odds<1.50:f=1.04
            elif odds<2.0:f=1.02
            else:f=1.0
            corrections.append(prob*f)
        hc,dc,ac=corrections
        if dc<25 and sp_d<3.8:dc*=1.06
        hc=max(3,hc);dc=max(3,dc);ac=max(3,ac);t=hc+dc+ac
        return round(hc/t*100,1),round(dc/t*100,1),round(100-hc/t*100-dc/t*100,1)

class DynamicFusionWeight:
    @staticmethod
    def get_weights(match,prediction):
        sp_h=float(match.get("sp_home",0) or 0);sp_d=float(match.get("sp_draw",0) or 0);sp_a=float(match.get("sp_away",0) or 0)
        mw=0.75
        if sp_h>1 and sp_a>1:
            r=max(sp_h,sp_d,sp_a)-min(sp_h,sp_d,sp_a)
            if r<0.5:mw=0.60
            elif r>2.0:mw=0.82
        exp=prediction.get("experience_analysis",{})
        if exp.get("total_score",0)>=20:mw-=0.08
        if "Sharp" in str(prediction.get("smart_money_signal","")):mw-=0.05
        mw=max(0.55,min(0.85,mw));return round(mw,2),round(1-mw,2)

class ProOverroundRemoval:
    def calculate(self,sp_h,sp_d,sp_a):
        if min(sp_h,sp_d,sp_a)<=1.05:return 0.33,0.33,0.34
        odds=np.array([sp_h,sp_d,sp_a]);imp=1.0/odds;margin=imp.sum()-1.0;z=margin/(1+margin)
        shin=(imp-z*imp**2)/(1-z);shin/=shin.sum();power=imp**1.05;power/=power.sum();mult=imp/imp.sum()
        final=shin*0.50+power*0.25+mult*0.25;final/=final.sum()
        return round(float(final[0]),4),round(float(final[1]),4),round(float(final[2]),4)

class AsianHandicapConverter:
    @staticmethod
    def from_xg(hxg,axg,mg=8):
        probs=np.zeros((mg,mg))
        for i in range(mg):
            for j in range(mg):probs[i,j]=pdist.pmf(i,max(0.2,hxg))*pdist.pmf(j,max(0.2,axg))
        return probs/probs.sum()
    @classmethod
    def ah(cls,probs,hc):
        mg=probs.shape[0];hcover=push=acover=0.0
        for i in range(mg):
            for j in range(mg):
                m=(i-j)-hc
                if m>0:hcover+=probs[i,j]
                elif m==0:push+=probs[i,j]
                else:acover+=probs[i,j]
        t=hcover+push+acover
        if t>0:hcover/=t;push/=t;acover/=t
        return [round(hcover,4),round(push,4),round(acover,4)]
    @classmethod
    def ou(cls,probs,line):
        mg=probs.shape[0];over=under=push=0.0
        for i in range(mg):
            for j in range(mg):
                total=i+j
                if total>line:over+=probs[i,j]
                elif total==line:push+=probs[i,j]
                else:under+=probs[i,j]
        t=over+push+under
        if t>0:over/=t;push/=t;under/=t
        return [round(over,4),round(push,4),round(under,4)]
    @classmethod
    def btts(cls,probs):
        mg=probs.shape[0];return round(sum(probs[i,j] for i in range(1,mg) for j in range(1,mg)),4)
    @classmethod
    def full_analysis(cls,hxg,axg):
        p=cls.from_xg(hxg,axg)
        return {"ah_0":cls.ah(p,0),"ah_0.5":cls.ah(p,0.5),"ah_1.0":cls.ah(p,1.0),"ou_1.5":cls.ou(p,1.5),"ou_2.5":cls.ou(p,2.5),"ou_3.5":cls.ou(p,3.5),"btts":cls.btts(p)}

class CLVDetector:
    @staticmethod
    def detect(model_prob,odds,direction="home"):
        if not odds or odds<=1.0:return {"has_clv":False,"clv_pct":0,"edge":0,"signal":""}
        mp=model_prob/100.0;fair_p=(1.0/odds)*0.97;clv=(mp-fair_p)/max(fair_p,0.01)*100;edge=(mp*odds-1)*100
        if clv>5 and edge>3:return {"has_clv":True,"clv_pct":round(clv,1),"edge":round(edge,1),"signal":f"🔥 CLV+{clv:.1f}% Edge+{edge:.1f}% [{direction}]"}
        elif clv>2:return {"has_clv":True,"clv_pct":round(clv,1),"edge":round(edge,1),"signal":f"📈 CLV+{clv:.1f}% [{direction}]"}
        elif clv<-8:return {"has_clv":False,"clv_pct":round(clv,1),"edge":round(edge,1),"signal":f"⚠️ 负CLV{clv:.1f}% [{direction}]"}
        return {"has_clv":False,"clv_pct":round(clv,1),"edge":round(edge,1),"signal":""}

_bvp=ZeroInflatedBivariatePoisson();_ov=ProOverroundRemoval();_cal=AdaptiveCalibrator();_flb=FavouriteLongshotBias();_sf=SmartFilter();_clv=CLVDetector()

def upgrade_ensemble_predict(match,prediction,odds_data=None):
    sp_h=float(match.get("sp_home",0) or 0);sp_d=float(match.get("sp_draw",0) or 0);sp_a=float(match.get("sp_away",0) or 0)
    hxg=prediction.get("bookmaker_implied_home_xg");axg=prediction.get("bookmaker_implied_away_xg")
    if not hxg or hxg=="?" or not axg or axg=="?":
        hs=match.get("home_stats",{});ast=match.get("away_stats",{})
        try:hxg=float(hs.get("avg_goals_for",1.3))
        except:hxg=1.3
        try:axg=float(ast.get("avg_goals_for",1.1))
        except:axg=1.1
    else:hxg=float(hxg);axg=float(axg)
    hp=prediction.get("home_win_pct",33);dp=prediction.get("draw_pct",33);ap=prediction.get("away_win_pct",34)
    hp,dp,ap=_flb.correct(hp,dp,ap,sp_h,sp_d,sp_a)
    hp,dp,ap=_cal.calibrate(hp,dp,ap)
    bvp_result=_bvp.predict(hxg,axg);prediction["bivariate_poisson"]=bvp_result
    if sp_h>1 and sp_d>1 and sp_a>1:
        th,td,ta=_ov.calculate(sp_h,sp_d,sp_a);prediction["pro_odds"]={"true_home":round(th*100,1),"true_draw":round(td*100,1),"true_away":round(ta*100,1)}
    mw,mdw=DynamicFusionWeight.get_weights(match,prediction)
    bs=0.15;hp_f=hp*(1-bs)+bvp_result["home_win"]*bs;dp_f=dp*(1-bs)+bvp_result["draw"]*bs;ap_f=ap*(1-bs)+bvp_result["away_win"]*bs
    if "pro_odds" in prediction:
        po=prediction["pro_odds"];ps=0.08;hp_f=hp_f*(1-ps)+po["true_home"]*ps;dp_f=dp_f*(1-ps)+po["true_draw"]*ps;ap_f=ap_f*(1-ps)+po["true_away"]*ps
    t=hp_f+dp_f+ap_f
    if t>0:prediction["home_win_pct"]=round(hp_f/t*100,1);prediction["draw_pct"]=round(dp_f/t*100,1);prediction["away_win_pct"]=round(100-prediction["home_win_pct"]-prediction["draw_pct"],1)
    sigs=prediction.get("smart_signals",[])
    for prob,odds,name in [(prediction["home_win_pct"],sp_h,"主胜"),(prediction["draw_pct"],sp_d,"平局"),(prediction["away_win_pct"],sp_a,"客胜")]:
        c=_clv.detect(prob,odds,name)
        if c["signal"] and c["signal"] not in sigs:sigs.append(c["signal"])
    prediction["smart_signals"]=sigs
    pred_score=_sf.score_predictability(match,prediction);skip,skip_reason=_sf.should_skip(pred_score)
    prediction["predictability_score"]=pred_score
    if skip:prediction["skip_warning"]=skip_reason;sigs.append(skip_reason);prediction["confidence"]=max(20,prediction.get("confidence",50)-20)
    ahc=AsianHandicapConverter.full_analysis(hxg,axg);prediction["asian_handicap_probs"]=ahc
    o25_old=prediction.get("over_2_5",50);prediction["over_2_5"]=round(o25_old*0.55+bvp_result["over_2_5"]*0.25+ahc["ou_2.5"][0]*100*0.20,1)
    btts_old=prediction.get("btts",50);prediction["btts"]=round(btts_old*0.55+bvp_result["btts"]*0.25+ahc["btts"]*100*0.20,1)
    pcts={"主胜":prediction["home_win_pct"],"平局":prediction["draw_pct"],"客胜":prediction["away_win_pct"]};prediction["result"]=max(pcts,key=pcts.get)
    prediction["confidence"]=_cal.adjust_confidence(prediction.get("confidence",50))
    prediction["total_models"]=prediction.get("total_models",11)+4;prediction["fusion_weights"]={"market":mw,"model":mdw}
    return prediction