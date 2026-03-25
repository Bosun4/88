#!/usr/bin/env python3
"""
odds_history.py v1.0 — 历史赔率匹配引擎
灵感: czl0325/football_frontend (264★, 欧洲杯60万战绩)
升级: 欧赔匹配+距离加权+盘口太便宜+连续升盘+赔率形态
"""
import json,os,numpy as np
from collections import defaultdict

class OddsHistoryMatcher:
    def __init__(self,history_file="historical_matches.json",min_matches=8):
        self.history=[];self.min_matches=min_matches;self.loaded=False
        for p in [history_file,os.path.join("data",history_file),os.path.join(os.path.dirname(__file__),history_file),os.path.join(os.path.dirname(__file__),"data",history_file)]:
            if os.path.exists(p):
                try:
                    with open(p,"r",encoding="utf-8") as f:data=json.load(f)
                    raw=data.get("matches",data if isinstance(data,list) else [])
                    for m in raw:
                        sh=float(m.get("sp_home",0) or 0);sd=float(m.get("sp_draw",0) or 0);sa=float(m.get("sp_away",0) or 0);ar=m.get("actual_result","")
                        if sh>1 and sd>1 and sa>1 and ar in ["home","draw","away"]:
                            self.history.append({"sp_h":sh,"sp_d":sd,"sp_a":sa,"result":ar,"league":m.get("league",""),"score":m.get("actual_score","")})
                    self.loaded=len(self.history)>50
                    if self.loaded:print(f"[OddsHistory] {len(self.history)}场历史数据已加载")
                    return
                except Exception as e:print(f"[OddsHistory] 加载失败:{e}")
        print("[OddsHistory] 未找到历史数据，请先运行grabber.py")

    def match(self,sp_h,sp_d,sp_a,league="",tolerance=0.15,league_only=False):
        if not self.loaded or sp_h<=1:return None
        matched=[]
        for h in self.history:
            dh=abs(h["sp_h"]-sp_h);dd=abs(h["sp_d"]-sp_d);da=abs(h["sp_a"]-sp_a)
            if dh>tolerance or dd>tolerance or da>tolerance:continue
            if league_only and league and h["league"]!=league:continue
            dist=np.sqrt(dh**2+dd**2+da**2)
            matched.append({"result":h["result"],"weight":max(0.1,1.0-dist/(tolerance*1.73)),"score":h.get("score",""),"dist":dist})
        if len(matched)<self.min_matches:
            if tolerance<0.30:return self.match(sp_h,sp_d,sp_a,league,tolerance*1.5,league_only)
            return None
        matched.sort(key=lambda x:x["dist"]);matched=matched[:200]
        tw=sum(m["weight"] for m in matched)
        hr=sum(m["weight"] for m in matched if m["result"]=="home")/tw*100
        dr=sum(m["weight"] for m in matched if m["result"]=="draw")/tw*100
        ar=100-hr-dr
        sc=defaultdict(int)
        for m in matched:
            if m["score"]:sc[m["score"]]+=1
        top=sorted(sc.items(),key=lambda x:x[1],reverse=True)[:5]
        avg_d=np.mean([m["dist"] for m in matched])
        conf=min(90,30+len(matched)*0.3+(1.0-avg_d)*30)
        mx=max(hr,dr,ar);sig=""
        if mx>=70:
            d="主胜" if hr>=70 else ("平局" if dr>=70 else "客胜")
            sig=f"📊 历史{len(matched)}场→{d}{mx:.0f}%"
        elif mx>=55:
            d="主胜" if hr==mx else ("平局" if dr==mx else "客胜")
            sig=f"📊 历史{len(matched)}场偏向{d}{mx:.0f}%"
        return {"matched_count":len(matched),"home_rate":round(hr,1),"draw_rate":round(dr,1),"away_rate":round(ar,1),"confidence":round(conf,1),"top_scores":[{"score":s,"count":c} for s,c in top],"signal":sig}

class CheapHandicapDetector:
    @staticmethod
    def detect(match):
        hr=int(match.get("home_rank",0) or 0);ar=int(match.get("away_rank",0) or 0)
        give=float(match.get("give_ball",0) or 0);sp_h=float(match.get("sp_home",0) or 0);sp_a=float(match.get("sp_away",0) or 0)
        if hr<=0 or ar<=0 or sp_h<=1:return {"is_cheap":False,"signal":"","adj_h":0,"adj_a":0}
        rd=ar-hr
        if rd>=10 and abs(give)<=0.25:return {"is_cheap":True,"signal":f"🚨 盘口太便宜!排名差{rd}但仅让{give}球","adj_h":-8,"adj_a":5}
        if rd<=-10 and give>=0:return {"is_cheap":True,"signal":f"🚨 客队盘口太便宜!排名差{abs(rd)}但不让球","adj_h":5,"adj_a":-8}
        if abs(rd)>=8 and sp_h>0 and sp_a>0 and abs(sp_h-sp_a)<0.3:
            w="主队" if rd<0 else "客队"
            return {"is_cheap":True,"signal":f"⚠️ 排名差{abs(rd)}但赔率相同,{w}被高估","adj_h":3 if rd>0 else -3,"adj_a":-3 if rd>0 else 3}
        return {"is_cheap":False,"signal":"","adj_h":0,"adj_a":0}

class LineMovementDetector:
    @staticmethod
    def detect(match):
        change=match.get("change",{})
        if not change or not isinstance(change,dict):return {"has_anomaly":False,"signal":"","direction":""}
        try:wc=float(change.get("win",0));lc=float(change.get("lose",0));sc=float(change.get("same",0))
        except:return {"has_anomaly":False,"signal":"","direction":""}
        vote=match.get("vote",{});vh=int(vote.get("win",33) if vote else 33);va=int(vote.get("lose",33) if vote else 33)
        if wc<-0.08 and vh>=55:return {"has_anomaly":True,"signal":f"🚨 主胜降水{wc:.2f}+热度{vh}%=造热主队","direction":"upset_away"}
        if lc<-0.08 and va>=55:return {"has_anomaly":True,"signal":f"🚨 客胜降水{lc:.2f}+热度{va}%=造热客队","direction":"upset_home"}
        if sc<-0.06 and wc>=0 and lc>=0:return {"has_anomaly":True,"signal":"💰 平赔独降!Sharp资金进平局","direction":"draw"}
        if wc>0.03 and lc>0.03:return {"has_anomaly":True,"signal":"⚠️ 主客赔同升→平局概率上升","direction":"draw"}
        if max(abs(wc),abs(lc),abs(sc))>0.15:return {"has_anomaly":True,"signal":f"🔥 赔率剧变(幅度{max(abs(wc),abs(lc),abs(sc)):.2f})","direction":"volatile"}
        return {"has_anomaly":False,"signal":"","direction":""}

class OddsPatternAnalyzer:
    @staticmethod
    def analyze_pattern(sp_h,sp_d,sp_a):
        if sp_h<=1 or sp_d<=1 or sp_a<=1:return {"pattern":"无效","adj_h":0,"adj_d":0,"adj_a":0,"signal":""}
        if sp_h<1.25:return {"pattern":"超级大热","adj_h":-8,"adj_d":6,"adj_a":2,"signal":"🚨 超级大热(<1.25)历史翻车率28%"}
        if sp_h<1.40:return {"pattern":"大热","adj_h":-5,"adj_d":4,"adj_a":1,"signal":"⚠️ 大热(<1.40)平局率+12%"}
        if 1.80<=sp_h<=2.20 and 2.80<=sp_d<=3.50 and 1.80<=sp_a<=2.20:return {"pattern":"232均势","adj_h":-2,"adj_d":5,"adj_a":-2,"signal":"📊 232均势，平局率38-45%"}
        if abs(sp_h-sp_a)<0.15:return {"pattern":"极端均势","adj_h":-3,"adj_d":6,"adj_a":-3,"signal":"📊 赔率极端均势，平局最高"}
        if sp_a>5.0:return {"pattern":"大冷门客","adj_h":2,"adj_d":1,"adj_a":-3,"signal":"📊 客队冷门(>5.0)实际胜率低于隐含8%"}
        if sp_h>5.0:return {"pattern":"大冷门主","adj_h":-3,"adj_d":1,"adj_a":2,"signal":"📊 主队冷门(>5.0)实际胜率低于隐含8%"}
        return {"pattern":"正常","adj_h":0,"adj_d":0,"adj_a":0,"signal":""}

_matcher=None
def get_matcher():
    global _matcher
    if _matcher is None:_matcher=OddsHistoryMatcher()
    return _matcher

def apply_odds_history(match,prediction):
    sp_h=float(match.get("sp_home",0) or 0);sp_d=float(match.get("sp_draw",0) or 0);sp_a=float(match.get("sp_away",0) or 0)
    sigs=prediction.get("smart_signals",[])
    # 1.历史匹配
    matcher=get_matcher()
    if matcher.loaded:
        hist=matcher.match(sp_h,sp_d,sp_a,match.get("league",""))
        if hist and hist["matched_count"]>=8:
            prediction["odds_history"]=hist
            w=min(0.12,0.05+hist["matched_count"]*0.0005)
            hp=prediction.get("home_win_pct",33);dp=prediction.get("draw_pct",33);ap=prediction.get("away_win_pct",34)
            hp=hp*(1-w)+hist["home_rate"]*w;dp=dp*(1-w)+hist["draw_rate"]*w;ap=ap*(1-w)+hist["away_rate"]*w
            t=hp+dp+ap
            if t>0:prediction["home_win_pct"]=round(hp/t*100,1);prediction["draw_pct"]=round(dp/t*100,1);prediction["away_win_pct"]=round(100-prediction["home_win_pct"]-prediction["draw_pct"],1)
            if hist["signal"]:sigs.append(hist["signal"])
    # 2.便宜盘
    cheap=CheapHandicapDetector.detect(match)
    if cheap["is_cheap"]:
        sigs.append(cheap["signal"])
        hp=prediction.get("home_win_pct",33)+cheap["adj_h"]*0.3;ap=prediction.get("away_win_pct",34)+cheap["adj_a"]*0.3;dp=100-hp-ap
        hp=max(5,hp);dp=max(5,dp);ap=max(5,ap);t=hp+dp+ap
        prediction["home_win_pct"]=round(hp/t*100,1);prediction["draw_pct"]=round(dp/t*100,1);prediction["away_win_pct"]=round(100-prediction["home_win_pct"]-prediction["draw_pct"],1)
    # 3.赔率变动
    line=LineMovementDetector.detect(match)
    if line["has_anomaly"]:sigs.append(line["signal"]);prediction["line_movement_anomaly"]=line
    # 4.赔率形态
    pat=OddsPatternAnalyzer.analyze_pattern(sp_h,sp_d,sp_a)
    if pat.get("signal"):
        sigs.append(pat["signal"])
        hp=prediction.get("home_win_pct",33)+pat["adj_h"]*0.15;dp=prediction.get("draw_pct",33)+pat["adj_d"]*0.15;ap=prediction.get("away_win_pct",34)+pat["adj_a"]*0.15
        hp=max(5,hp);dp=max(5,dp);ap=max(5,ap);t=hp+dp+ap
        prediction["home_win_pct"]=round(hp/t*100,1);prediction["draw_pct"]=round(dp/t*100,1);prediction["away_win_pct"]=round(100-prediction["home_win_pct"]-prediction["draw_pct"],1)
    prediction["smart_signals"]=sigs;prediction["odds_pattern"]=pat.get("pattern","正常")
    pcts={"主胜":prediction["home_win_pct"],"平局":prediction["draw_pct"],"客胜":prediction["away_win_pct"]}
    prediction["result"]=max(pcts,key=pcts.get)
    return prediction