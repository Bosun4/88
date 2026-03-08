import math
from collections import defaultdict

class PoissonModel:
    """泊松分布预测比分概率"""
    def predict(self,home_avg_gf,home_avg_ga,away_avg_gf,away_avg_ga,league_avg=1.35):
        try:
            home_avg_gf=float(home_avg_gf or 1.3);home_avg_ga=float(home_avg_ga or 1.1)
            away_avg_gf=float(away_avg_gf or 1.1);away_avg_ga=float(away_avg_ga or 1.3)
        except:home_avg_gf=1.3;home_avg_ga=1.1;away_avg_gf=1.1;away_avg_ga=1.3
        home_strength=home_avg_gf/league_avg if league_avg else 1
        away_strength=away_avg_gf/league_avg if league_avg else 1
        home_weak=home_avg_ga/league_avg if league_avg else 1
        away_weak=away_avg_ga/league_avg if league_avg else 1
        home_exp=home_strength*away_weak*league_avg*1.05
        away_exp=away_strength*home_weak*league_avg*0.95
        home_exp=max(0.3,min(home_exp,4.5))
        away_exp=max(0.2,min(away_exp,4.0))
        def poisson_pmf(k,lam):
            return(lam**k)*math.exp(-lam)/math.factorial(k)
        home_win=0;draw=0;away_win=0;over25=0;btts=0
        scores=[]
        for i in range(7):
            for j in range(7):
                p=poisson_pmf(i,home_exp)*poisson_pmf(j,away_exp)
                if i>j:home_win+=p
                elif i==j:draw+=p
                else:away_win+=p
                if i+j>2:over25+=p
                if i>0 and j>0:btts+=p
                scores.append((i,j,p))
        scores.sort(key=lambda x:x[2],reverse=True)
        best=scores[0]
        total=home_win+draw+away_win
        if total>0:home_win/=total;draw/=total;away_win/=total
        return{"home_win":round(home_win*100,1),"draw":round(draw*100,1),"away_win":round(away_win*100,1),"predicted_score":"%d-%d"%(best[0],best[1]),"home_expected_goals":round(home_exp,2),"away_expected_goals":round(away_exp,2),"over_2_5":round(over25*100,1),"btts":round(btts*100,1),"top_scores":[{"score":"%d-%d"%(s[0],s[1]),"prob":round(s[2]*100,1)} for s in scores[:5]]}

class EloModel:
    """ELO评分系统"""
    def __init__(self):
        self.ratings=defaultdict(lambda:1500)
        self.k=32
    def update(self,home,away,home_goals,away_goals):
        rh=self.ratings[home];ra=self.ratings[away]
        eh=1/(1+10**((ra-rh)/400));ea=1-eh
        if home_goals>away_goals:sh=1;sa=0
        elif home_goals==away_goals:sh=0.5;sa=0.5
        else:sh=0;sa=1
        self.ratings[home]=rh+self.k*(sh-eh)
        self.ratings[away]=ra+self.k*(sa-ea)
    def predict(self,home,away):
        rh=self.ratings[home]+65
        ra=self.ratings[away]
        eh=1/(1+10**((ra-rh)/400))
        draw_factor=0.26
        home_win=eh*(1-draw_factor/2)
        away_win=(1-eh)*(1-draw_factor/2)
        dr=draw_factor
        return{"home_win":round(home_win*100,1),"draw":round(dr*100,1),"away_win":round(away_win*100,1),"home_elo":round(rh,1),"away_elo":round(ra,1),"elo_diff":round(rh-ra,1)}
    def load_h2h(self,h2h_records):
        for rec in reversed(h2h_records):
            try:
                parts=rec["score"].split("-")
                hg=int(parts[0]);ag=int(parts[1])
                self.update(rec["home"],rec["away"],hg,ag)
            except:pass

class FormModel:
    """近期状态分析"""
    def analyze(self,form_str):
        if not form_str:return{"score":50,"trend":"unknown","wins":0,"draws":0,"losses":0}
        w=form_str.count("W");d=form_str.count("D");l=form_str.count("L")
        total=w+d+l
        if total==0:return{"score":50,"trend":"unknown","wins":0,"draws":0,"losses":0}
        score=(w*3+d*1)/(total*3)*100
        recent=form_str[-5:] if len(form_str)>=5 else form_str
        rw=recent.count("W");rl=recent.count("L")
        if rw>=3:trend="hot"
        elif rl>=3:trend="cold"
        elif rw>rl:trend="good"
        elif rl>rw:trend="poor"
        else:trend="mixed"
        return{"score":round(score,1),"trend":trend,"wins":w,"draws":d,"losses":l,"recent":recent}

class KellyCriterion:
    """凯利公式计算价值投注"""
    def calculate(self,prob,odds,fraction=0.25):
        if odds<=1 or prob<=0 or prob>=1:return{"kelly":0,"value":False,"edge":0}
        q=1-prob
        b=odds-1
        kelly=(b*prob-q)/b
        edge=(prob*odds-1)*100
        return{"kelly":round(max(0,kelly)*fraction*100,2),"value":edge>0,"edge":round(edge,1),"recommended_stake":round(max(0,kelly)*fraction*100,1)}

class OddsAnalyzer:
    """赔率深度分析"""
    def implied_probability(self,odds):
        if odds<=0:return 0
        return round(1/odds*100,1)
    def analyze_market(self,bookmakers):
        if not bookmakers:return{}
        home_odds=[];draw_odds=[];away_odds=[]
        for bk in bookmakers:
            h2h=bk.get("markets",{}).get("h2h",{})
            if "Home" in h2h:home_odds.append(h2h["Home"])
            if "Draw" in h2h:draw_odds.append(h2h["Draw"])
            if "Away" in h2h:away_odds.append(h2h["Away"])
        if not home_odds:return{}
        avg_h=sum(home_odds)/len(home_odds)
        avg_d=sum(draw_odds)/len(draw_odds) if draw_odds else 3.3
        avg_a=sum(away_odds)/len(away_odds) if away_odds else 3.0
        margin=1/avg_h+1/avg_d+1/avg_a-1
        hp=self.implied_probability(avg_h)/(1+margin) if margin>-1 else 33
        dp=self.implied_probability(avg_d)/(1+margin) if margin>-1 else 33
        ap=100-hp-dp
        spread_h=max(home_odds)-min(home_odds) if len(home_odds)>1 else 0
        return{"avg_home_odds":round(avg_h,2),"avg_draw_odds":round(avg_d,2),"avg_away_odds":round(avg_a,2),"implied_home":round(hp,1),"implied_draw":round(dp,1),"implied_away":round(ap,1),"margin":round(margin*100,1),"consensus":"home" if hp>dp and hp>ap else("away" if ap>hp and ap>dp else "draw"),"odds_spread":round(spread_h,2),"bookmaker_count":len(home_odds)}

class EnsemblePredictor:
    """多模型融合预测器"""
    def __init__(self):
        self.poisson=PoissonModel()
        self.elo=EloModel()
        self.form=FormModel()
        self.kelly=KellyCriterion()
        self.odds_analyzer=OddsAnalyzer()
    def predict(self,match,odds_data=None):
        hs=match.get("home_stats",{})
        ast=match.get("away_stats",{})
        h2h=match.get("h2h",[])
        home=match["home_team"]
        away=match["away_team"]
        poisson_result=self.poisson.predict(hs.get("avg_goals_for"),hs.get("avg_goals_against"),ast.get("avg_goals_for"),ast.get("avg_goals_against"))
        if h2h:self.elo.load_h2h(h2h)
        elo_result=self.elo.predict(home,away)
        home_form=self.form.analyze(hs.get("form",""))
        away_form=self.form.analyze(ast.get("form",""))
        form_diff=home_form["score"]-away_form["score"]
        form_home_adj=form_diff*0.15
        weights={"poisson":0.35,"elo":0.25,"form":0.15,"odds":0.25}
        home_pct=poisson_result["home_win"]*weights["poisson"]+elo_result["home_win"]*weights["elo"]+(50+form_home_adj)*weights["form"]
        draw_pct=poisson_result["draw"]*weights["poisson"]+elo_result["draw"]*weights["elo"]+25*weights["form"]
        away_pct=poisson_result["away_win"]*weights["poisson"]+elo_result["away_win"]*weights["elo"]+(50-form_home_adj)*weights["form"]
        odds_result={}
        if odds_data:
            bks=odds_data.get("bookmakers",[])
            if bks:
                odds_result=self.odds_analyzer.analyze_market(bks)
                if odds_result:
                    home_pct+=odds_result.get("implied_home",33)*weights["odds"]
                    draw_pct+=odds_result.get("implied_draw",33)*weights["odds"]
                    away_pct+=odds_result.get("implied_away",33)*weights["odds"]
                else:
                    home_pct+=33*weights["odds"]
                    draw_pct+=33*weights["odds"]
                    away_pct+=33*weights["odds"]
            else:
                home_pct+=33*weights["odds"];draw_pct+=33*weights["odds"];away_pct+=33*weights["odds"]
        else:
            home_pct+=33*weights["odds"];draw_pct+=33*weights["odds"];away_pct+=33*weights["odds"]
        total=home_pct+draw_pct+away_pct
        if total>0:home_pct=round(home_pct/total*100,1);draw_pct=round(draw_pct/total*100,1);away_pct=round(100-home_pct-draw_pct,1)
        agreement=0
        models_home=[poisson_result["home_win"]>poisson_result["draw"] and poisson_result["home_win"]>poisson_result["away_win"],elo_result["home_win"]>elo_result["draw"] and elo_result["home_win"]>elo_result["away_win"]]
        models_away=[poisson_result["away_win"]>poisson_result["home_win"],elo_result["away_win"]>elo_result["home_win"]]
        if all(models_home):agreement=2
        elif all(models_away):agreement=2
        confidence=50+agreement*8
        max_pct=max(home_pct,draw_pct,away_pct)
        if max_pct>60:confidence+=10
        elif max_pct>50:confidence+=5
        if home_form["trend"]in["hot","good"]and form_diff>15:confidence+=5
        confidence=min(95,max(30,confidence))
        return{"home_win_pct":home_pct,"draw_pct":draw_pct,"away_win_pct":away_pct,"predicted_score":poisson_result["predicted_score"],"confidence":confidence,"poisson":poisson_result,"elo":elo_result,"home_form":home_form,"away_form":away_form,"odds":odds_result,"over_2_5":poisson_result["over_2_5"],"btts":poisson_result["btts"],"top_scores":poisson_result.get("top_scores",[]),"model_weights":weights}
