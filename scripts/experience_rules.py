#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experience_rules.py v5.0 — 足球全维度经验规则引擎（修复6大BUG + 新增12条规则）
BUG1修复: G08从match本体读a0（旧版从v2空字典读=永远99）
BUG2修复: 新增CRS 0-0赔率检测
BUG3修复: 新增双闷队0-0检测
BUG4修复: 新增a7大球封顶/突破检测
BUG5修复: 新增排名差vs盘口矛盾（冷门）检测
BUG6修复: apply权重从0.20→0.45
"""

import re
from league_intel import detect_league_key, detect_derby, LEAGUE_PROFILES

RULES_DATABASE = {
"draw": [
    {"id":"D01","name":"大热必死","category":"平局","sub":"市场热度","weight":8,"confidence":"高"},
    {"id":"D02","name":"中下游逼平强队后客场低迷","category":"平局","sub":"连续走势","weight":6,"confidence":"中高"},
    {"id":"D03","name":"下游两连胜无三连","category":"平局","sub":"连胜规律","weight":7,"confidence":"高"},
    {"id":"D04","name":"中游三连胜无四连","category":"平局","sub":"连胜规律","weight":7,"confidence":"高"},
    {"id":"D05","name":"强队三连胜易六连胜难","category":"平局","sub":"连胜规律","weight":6,"confidence":"中高"},
    {"id":"D06","name":"强队怕克星与德比","category":"平局","sub":"对阵关系","weight":8,"confidence":"高"},
    {"id":"D07","name":"多线作战强队易平","category":"平局","sub":"赛程因素","weight":7,"confidence":"高"},
    {"id":"D08","name":"强强对话平局率超30%","category":"平局","sub":"对阵关系","weight":8,"confidence":"高"},
    {"id":"D09","name":"九配走平口","category":"平局","sub":"赔率信号","weight":7,"confidence":"高"},
    {"id":"D10","name":"平手盘水位不动易平","category":"平局","sub":"盘口信号","weight":8,"confidence":"高"},
    {"id":"D11","name":"半球盘高水诱下盘","category":"平局","sub":"盘口信号","weight":6,"confidence":"中高"},
    {"id":"D12","name":"232指数体系平局多","category":"平局","sub":"赔率信号","weight":8,"confidence":"高"},
    {"id":"D13","name":"攻防数据接近必防平","category":"平局","sub":"数据指标","weight":7,"confidence":"高"},
    {"id":"D14","name":"积分差4分内平局率高","category":"平局","sub":"数据指标","weight":8,"confidence":"高"},
    {"id":"D15","name":"杯赛首轮与淘汰赛易平","category":"平局","sub":"赛事特征","weight":6,"confidence":"中高"},
    {"id":"D16","name":"中游无欲无求易平","category":"平局","sub":"动机","weight":6,"confidence":"高"},
    {"id":"D17","name":"裁判黄牌大户出平","category":"平局","sub":"裁判因素","weight":4,"confidence":"中"},
    {"id":"D18","name":"主场连败止血易平","category":"平局","sub":"连续走势","weight":5,"confidence":"中高"},
    {"id":"D19","name":"国际赛均势闷平","category":"平局","sub":"赛事特征","weight":7,"confidence":"高"},
],
"upset": [
    {"id":"U01","name":"豪门同时开赛冷门必出","category":"冷门","sub":"庄家做盘","weight":7,"confidence":"高"},
    {"id":"U02","name":"大盘临场升盘出下盘","category":"冷门","sub":"盘口信号","weight":7,"confidence":"高"},
    {"id":"U03","name":"降盘升水正诱盘","category":"冷门","sub":"盘口信号","weight":6,"confidence":"中高"},
    {"id":"U04","name":"受注比例一边倒反向操作","category":"冷门","sub":"市场热度","weight":8,"confidence":"高"},
    {"id":"U05","name":"平局交易量突增","category":"冷门","sub":"市场热度","weight":7,"confidence":"高"},
    {"id":"U06","name":"强队欧冠大胜后联赛翻车","category":"冷门","sub":"人气指数","weight":5,"confidence":"中"},
    {"id":"U07","name":"换帅新官效应","category":"冷门","sub":"基本面","weight":5,"confidence":"中高"},
    {"id":"U08","name":"升班马黑马效应","category":"冷门","sub":"基本面","weight":5,"confidence":"中"},
    {"id":"U09","name":"排名差大但盘口便宜","category":"冷门","sub":"盘口信号","weight":8,"confidence":"高"},
    {"id":"U10","name":"赔率剧烈变动","category":"冷门","sub":"临场信号","weight":7,"confidence":"高"},
],
"goals": [
    {"id":"G01","name":"让球盘与大小球盘矛盾出小","category":"大小球","sub":"盘口对比","weight":7,"confidence":"高"},
    {"id":"G02","name":"平半盘配2.5以上大球盘看大","category":"大小球","sub":"盘口对比","weight":6,"confidence":"中高"},
    {"id":"G03","name":"深盘大小球降盘防冷","category":"大小球","sub":"盘口信号","weight":8,"confidence":"高"},
    {"id":"G04","name":"大球联赛开浅盘防小","category":"大小球","sub":"联赛特征","weight":6,"confidence":"高"},
    {"id":"G05","name":"小球联赛开深盘看大","category":"大小球","sub":"联赛特征","weight":6,"confidence":"中高"},
    {"id":"G06","name":"初盘2.25球低水出小","category":"大小球","sub":"赔率信号","weight":6,"confidence":"高"},
    {"id":"G07","name":"德比大战多进球","category":"大小球","sub":"对阵关系","weight":5,"confidence":"中高"},
    {"id":"G08","name":"0球赔率极低信号","category":"大小球","sub":"波胆信号","weight":8,"confidence":"高"},
    {"id":"G09","name":"7+球赔率极低信号","category":"大小球","sub":"波胆信号","weight":7,"confidence":"高"},
    {"id":"G10","name":"CRS 0-0赔率极低","category":"大小球","sub":"波胆信号","weight":8,"confidence":"高"},
    {"id":"G11","name":"双闷队0-0高危","category":"大小球","sub":"球队风格","weight":7,"confidence":"高"},
    {"id":"G12","name":"双攻队大球高危","category":"大小球","sub":"球队风格","weight":7,"confidence":"高"},
],
"bookmaker": [
    {"id":"B01","name":"死水盘出超低水方","category":"盘口","sub":"水位信号","weight":8,"confidence":"极高"},
    {"id":"B02","name":"浅盘持续降水诱上","category":"盘口","sub":"造热手法","weight":7,"confidence":"高"},
    {"id":"B03","name":"多公司协同热捧一方","category":"盘口","sub":"造热手法","weight":7,"confidence":"高"},
    {"id":"B04","name":"欧赔亚盘方向矛盾","category":"盘口","sub":"异常信号","weight":6,"confidence":"高"},
    {"id":"B05","name":"平手盘临场不变水位下调方不败","category":"盘口","sub":"水位信号","weight":6,"confidence":"高"},
    {"id":"B06","name":"半球盘诡盘三结果","category":"盘口","sub":"盘口特征","weight":4,"confidence":"中"},
],
"motivation": [
    {"id":"M01","name":"保级队赛季末激战","category":"动机","sub":"赛季末","weight":7,"confidence":"高"},
    {"id":"M02","name":"夺冠锁定后强队放水","category":"动机","sub":"赛季末","weight":7,"confidence":"高"},
    {"id":"M03","name":"赛季末中游无动力","category":"动机","sub":"赛季末","weight":5,"confidence":"中高"},
    {"id":"M04","name":"欧冠资格生死战","category":"动机","sub":"排名争夺","weight":6,"confidence":"中高"},
    {"id":"M05","name":"意甲保级财务灾难","category":"动机","sub":"联赛特色","weight":7,"confidence":"高"},
    {"id":"M06","name":"法甲TV崩溃每分必争","category":"动机","sub":"联赛特色","weight":5,"confidence":"中高"},
    {"id":"M07","name":"杯赛两回合次回合试探","category":"动机","sub":"杯赛","weight":5,"confidence":"中高"},
],
"form": [
    {"id":"F01","name":"连败止血主场反弹","category":"走势","sub":"连败","weight":6,"confidence":"高"},
    {"id":"F02","name":"客场三连败防翻车","category":"走势","sub":"连败","weight":5,"confidence":"中高"},
    {"id":"F03","name":"强队联赛失利后反弹","category":"走势","sub":"人气","weight":6,"confidence":"中高"},
    {"id":"F04","name":"净胜2球是强弱心理分界线","category":"走势","sub":"人气","weight":4,"confidence":"中"},
    {"id":"F05","name":"一周双赛体能折扣","category":"走势","sub":"体能","weight":5,"confidence":"中高"},
],
"league_specific": [
    {"id":"L01","name":"英超主场优势下降","category":"联赛","sub":"英超","weight":3,"confidence":"高"},
    {"id":"L02","name":"意甲防守优先平局30%","category":"联赛","sub":"意甲","weight":5,"confidence":"高"},
    {"id":"L03","name":"德甲最高进球联赛","category":"联赛","sub":"德甲","weight":4,"confidence":"高"},
    {"id":"L04","name":"法乙超级小球联赛","category":"联赛","sub":"法乙","weight":5,"confidence":"高"},
    {"id":"L05","name":"土超主场情绪化","category":"联赛","sub":"土超","weight":4,"confidence":"中"},
    {"id":"L06","name":"荷甲进攻型如德甲","category":"联赛","sub":"荷甲","weight":4,"confidence":"高"},
],
}


class ExperienceEngine:
    def __init__(self):
        self.rules = RULES_DATABASE
        total = sum(len(v) for v in self.rules.values())
        print(f"[ExperienceEngine] v5.0 loaded: {total} rules")

    def _tier(self, rank, total=20):
        if not rank or rank <= 0: return "未知"
        r = rank / total
        if r <= 0.25: return "强队"
        elif r <= 0.60: return "中游"
        elif r <= 0.75: return "中下游"
        else: return "下游"

    def _sf(self, val, d=0.0):
        try: return float(val) if val is not None else d
        except: return d

    def _si(self, val, d=0):
        try: return int(val) if val is not None else d
        except: return d

    def analyze(self, match_data: dict) -> dict:
        triggered = []
        draw_boost = home_adj = away_adj = over_adj = 0.0
        zero_zero_boost = 0.0
        risk_signals = []

        sp_h = self._sf(match_data.get("sp_home", match_data.get("win")), 2.5)
        sp_d = self._sf(match_data.get("sp_draw", match_data.get("same")), 3.2)
        sp_a = self._sf(match_data.get("sp_away", match_data.get("lose")), 3.5)
        hr = self._si(match_data.get("home_rank"), 10)
        ar = self._si(match_data.get("away_rank"), 10)
        hs = match_data.get("home_stats", {})
        ast = match_data.get("away_stats", {})
        change = match_data.get("change", {})
        vote = match_data.get("vote", {})
        league = str(match_data.get("league", match_data.get("cup", "")))
        give_ball = self._sf(match_data.get("give_ball"), 0)
        baseface = str(match_data.get("baseface", ""))
        h_form = str(hs.get("form", "")).upper()
        a_form = str(ast.get("form", "")).upper()
        h_tier = self._tier(hr)
        a_tier = self._tier(ar)
        lk = detect_league_key(league)

        # ★ BUG1修复：从match本体读赔率，不从v2空字典读
        a0_val = self._sf(match_data.get("a0"), 99)
        a1_val = self._sf(match_data.get("a1"), 99)
        a2_val = self._sf(match_data.get("a2"), 99)
        a7_val = self._sf(match_data.get("a7"), 99)
        s00_val = self._sf(match_data.get("s00"), 99)
        s11_val = self._sf(match_data.get("s11"), 99)

        try:
            hgf = float(hs.get("avg_goals_for", 1.2))
            agf = float(ast.get("avg_goals_for", 1.0))
            hga = float(hs.get("avg_goals_against", 1.1))
            aga = float(ast.get("avg_goals_against", 1.2))
        except:
            hgf = 1.2; agf = 1.0; hga = 1.1; aga = 1.2

        def add(rid, name, cat, w, reason, direction):
            triggered.append({"id":rid,"name":name,"category":cat,"weight":w,"reason":reason,"direction":direction})

        # ========== 平局规则 ==========
        vh = self._si(vote.get("win"), 33)
        va = self._si(vote.get("lose"), 33)

        # D01: 大热必死（双向检测）
        if sp_h < 1.40 and vh >= 55:
            add("D01","大热必死","平局",8,f"主赔{sp_h}极低+受注{vh}%","draw"); draw_boost += 5; risk_signals.append("🚨 大热必死")
        elif sp_h < 1.30:
            add("D01","大热必死","平局",6,f"主赔{sp_h}超低","draw"); draw_boost += 3
        elif sp_a < 1.40 and va >= 55:
            add("D01","大热必死","平局",8,f"客赔{sp_a}极低+受注{va}%","draw"); draw_boost += 5; risk_signals.append("🚨 大热必死")

        # D03/D04/D05: 连胜规律
        for side, tier, form in [("主队",h_tier,h_form),("客队",a_tier,a_form)]:
            if tier == "下游" and form.endswith("WW") and not form.endswith("WWW"):
                add("D03","下游两连胜无三连","平局",7,f"{side}(下游)已两连胜","draw"); draw_boost += 4
            if tier == "中游" and form.endswith("WWW") and not form.endswith("WWWW"):
                add("D04","中游三连胜无四连","平局",7,f"{side}(中游)已三连胜","draw"); draw_boost += 4
            if tier == "强队" and len(form) >= 5 and form[-5:] == "WWWWW":
                add("D05","强队六连胜难","平局",6,f"{side}已5+连胜","draw"); draw_boost += 3

        # D06: 德比
        derby = detect_derby(match_data.get("home_team",""), match_data.get("away_team",""), lk)
        if derby:
            add("D06","德比/克星效应","平局",8,f"检测到:{derby}","draw"); draw_boost += 4

        # D08: 强强对话
        if h_tier == "强队" and a_tier == "强队":
            add("D08","强强对话平局率超30%","平局",8,f"排名{hr}vs{ar}均为强队","draw"); draw_boost += 5

        # D09: 九配走平口
        if sp_d > 0 and sp_d < sp_h and sp_d < sp_a and sp_d < 3.30:
            add("D09","九配走平口","平局",7,f"平赔{sp_d}是三项最低","draw"); draw_boost += 4

        # D10: 平手盘水位不动
        wc = self._sf(change.get("win"), 0)
        lc = self._sf(change.get("lose"), 0)
        sc = self._sf(change.get("same"), 0)
        if abs(give_ball) < 0.1 and abs(wc) < 0.02 and abs(lc) < 0.02:
            add("D10","平手盘水位不动易平","平局",8,"平手盘临场水位几乎不变","draw"); draw_boost += 5

        # D11: 半球盘高水
        if abs(abs(give_ball) - 0.5) < 0.1 and sp_h > 1.90:
            add("D11","半球盘高水诱下盘","平局",6,f"半球盘主赔{sp_h}偏高","draw"); draw_boost += 3

        # D12: 232
        if 1.8 <= sp_h <= 2.2 and 2.8 <= sp_d <= 3.5 and 1.8 <= sp_a <= 2.2:
            add("D12","232指数体系平局多","平局",8,f"{sp_h:.2f}-{sp_d:.2f}-{sp_a:.2f}呈232","draw"); draw_boost += 5

        # D13: 攻防接近（阈值收紧）
        if hgf > 0 and agf > 0 and abs(hgf-agf) <= 0.35 and abs(hga-aga) <= 0.35:
            add("D13","攻防数据接近必防平","平局",7,f"进球差{abs(hgf-agf):.2f}失球差{abs(hga-aga):.2f}","draw"); draw_boost += 4

        # D15: 杯赛
        if any(k in league for k in ["杯","cup","Cup"]) or "首回合" in baseface:
            add("D15","杯赛首轮易平","平局",6,"杯赛试探性打法","draw"); draw_boost += 3

        # D16: 中游无欲
        if 8 <= hr <= 14 and 8 <= ar <= 14:
            add("D16","中游无欲无求易平","平局",6,f"排名{hr}vs{ar}","draw"); draw_boost += 3

        # D18: 主场连败
        if len(h_form) >= 3 and h_form[-3:] == "LLL":
            add("D18","主场连败止血易平","平局",5,"主队三连败止血靠平","draw"); draw_boost += 2

        # D19: 国际赛均势闷平（★新增）
        is_intl = any(k in league for k in ["国际","友谊","FIFA","世预","欧预","欧国联","亚预","非预","美预"])
        if is_intl and abs(sp_h - sp_a) < 1.0 and abs(give_ball) <= 0.5:
            add("D19","国际赛均势闷平","平局",7,f"国际赛+赔差{abs(sp_h-sp_a):.1f}","draw"); draw_boost += 4

        # ========== 冷门规则 ==========
        # U04: 受注一边倒（阈值从70降到65）
        if vh >= 65:
            add("U04","受注一边倒反向","冷门",8,f"主胜受注{vh}%过热","upset_away")
            home_adj -= 5; risk_signals.append(f"🚨 主胜超热{vh}%")
        elif va >= 65:
            add("U04","受注一边倒反向","冷门",8,f"客胜受注{va}%过热","upset_home")
            away_adj -= 5; risk_signals.append(f"🚨 客胜超热{va}%")

        # U09: 排名差vs盘口矛盾（★新增BUG5修复）
        rank_gap = abs(hr - ar)
        if rank_gap >= 8 and abs(give_ball) <= 0.5:
            cheap = "客队" if hr < ar else "主队"
            add("U09",f"{cheap}盘口太便宜!排名差{rank_gap}但不让球","冷门",8,
                f"排名差{rank_gap}应让球但仅{give_ball}","upset")
            risk_signals.append(f"🚨 {cheap}盘口太便宜!排名差{rank_gap}但不让球")
        elif rank_gap >= 5 and abs(give_ball) <= 0.25:
            add("U09","盘口偏浅","冷门",6,f"排名差{rank_gap}但平手盘","upset")

        # U10: 赔率剧变（★新增）
        max_change = max(abs(wc), abs(lc), abs(sc)) if change else 0
        if max_change >= 0.15:
            dir_str = "主胜降水" if wc < -0.1 else ("客胜降水" if lc < -0.1 else "平赔降水")
            add("U10","赔率剧变","冷门",7,f"最大变动{max_change:.2f}→{dir_str}","upset")
            risk_signals.append(f"🔥 赔率剧变(幅度{max_change:.2f})")

        # ========== 大小球/0-0规则 ==========

        # G08: 0球赔率（★BUG1修复：从match_data读）
        if a0_val < 7.5:
            add("G08","0球赔率极低","大小球",8,f"0球@{a0_val}","under")
            over_adj -= 8; zero_zero_boost += 6; risk_signals.append(f"🚨 0球@{a0_val}")
        elif a0_val < 9.0:
            add("G08","0球赔率偏低","大小球",6,f"0球@{a0_val}","under")
            over_adj -= 4; zero_zero_boost += 3

        # G09: 7+球赔率（★新增BUG4修复）
        if a7_val < 15.0:
            add("G09","7+球赔率极低=大球","大小球",7,f"7+球@{a7_val}","over")
            over_adj += 6; risk_signals.append(f"🔥 7+球@{a7_val}大球信号")
        elif a7_val > 30.0 and a0_val < 12:
            over_adj -= 3

        # G10: CRS 0-0赔率（★新增BUG2修复）
        if s00_val < 8.0:
            add("G10","CRS 0-0赔率极低","大小球",8,f"0-0@{s00_val}","zero_zero")
            zero_zero_boost += 6; over_adj -= 5; risk_signals.append(f"🚨 CRS 0-0@{s00_val}")
        elif s00_val < 10.0 and a0_val < 10.0:
            add("G10","CRS 0-0+总0球双低","大小球",6,f"0-0@{s00_val}+0球@{a0_val}","zero_zero")
            zero_zero_boost += 4; over_adj -= 3

        # G11: 双闷队0-0（★新增BUG3修复）
        if hgf < 1.1 and agf < 1.1:
            add("G11","双闷队0-0高危","大小球",7,f"主均进{hgf:.1f}+客均进{agf:.1f}","zero_zero")
            zero_zero_boost += 5; over_adj -= 4
        elif hgf < 1.2 and agf < 1.2 and hga < 1.0 and aga < 1.0:
            add("G11","双防守队闷平","大小球",6,f"双方进球<1.2失球<1.0","zero_zero")
            zero_zero_boost += 3; over_adj -= 2

        # G12: 双攻队大球（★新增）
        if hgf > 1.6 and agf > 1.6:
            add("G12","双攻队大球高危","大小球",7,f"主均进{hgf:.1f}+客均进{agf:.1f}","over")
            over_adj += 5
        elif hgf > 1.5 and agf > 1.5 and hga > 1.3 and aga > 1.3:
            add("G12","双漏队大球","大小球",6,"进攻强+防守漏","over"); over_adj += 4

        # G03: 深盘矛盾
        if abs(give_ball) >= 1.5 and a1_val < 5.0:
            add("G03","深盘大小球矛盾","大小球",8,f"让{give_ball}深但1球@{a1_val}低","under_upset")
            over_adj -= 5; risk_signals.append("⚠️ 深盘小球矛盾")

        # ========== 盘口/Sharp ==========
        if sc < -0.05 and wc > 0 and lc > 0:
            add("B_SHARP","平局Sharp资金突进","盘口",7,f"平赔降{sc:.2f}","draw")
            draw_boost += 5; risk_signals.append("💰 平局Sharp突进")

        # Steam检测（★新增）
        if wc < -0.10 and vh < 50:
            add("B_STEAM","主胜Steam","盘口",7,f"主赔降{wc:.2f}散户仅{vh}%","steam_home")
            risk_signals.append(f"🔥 主胜Steam! 降水{wc:.2f}但散户未跟({vh}%)")
        elif lc < -0.10 and va < 50:
            add("B_STEAM","客胜Steam","盘口",7,f"客赔降{lc:.2f}散户仅{va}%","steam_away")
            risk_signals.append(f"🔥 客胜Steam! 降水{lc:.2f}但散户未跟({va}%)")

        # ========== 联赛特色 ==========
        league_rules = {
            "ita_top": ("L02","意甲防守优先平局30%","draw",5,2,0),
            "fra2": ("L04","法乙超级小球联赛","under",5,0,-4),
            "eng_top": ("L01","英超主场优势下降","eng_upset",3,0,0),
            "ger_top": ("L03","德甲最高进球联赛","over",4,0,3),
            "tur_top": ("L05","土超主场情绪化","tur_home",4,0,0),
            "ned_top": ("L06","荷甲进攻型如德甲","over",4,0,3),
        }
        if lk in league_rules:
            rid, name, direction, w, db, oa = league_rules[lk]
            profile = LEAGUE_PROFILES.get(lk, LEAGUE_PROFILES.get("default"))
            desc = profile[6] if profile and len(profile) > 6 else ""
            add(rid, name, "联赛", w, desc[:60], direction)
            draw_boost += db; over_adj += oa

        # ========== 动机 ==========
        if hr >= 16 or ar >= 16:
            rel_side = "主队" if hr >= 16 else "客队"
            rel_rank = hr if hr >= 16 else ar
            add("M01","保级队激战","动机",7,f"{rel_side}(#{rel_rank})保级","relegation")
            if hr >= 16: home_adj += 3
            else: away_adj += 3
            if lk == "ita_top":
                add("M05","意甲保级财务灾难","动机",7,"意甲降级=破产","relegation")
                if hr >= 16: home_adj += 2
                else: away_adj += 2

        # ========== 走势 ==========
        if h_form.endswith("LLL"):
            add("F01","连败止血主场反弹","走势",6,"主队三连败反弹","home_bounce"); home_adj += 3
        if a_form.endswith("LLL"):
            add("F02","客场三连败继续输","走势",5,"客队三连败低迷","away_sink"); away_adj -= 2
        if h_tier == "强队" and h_form.endswith("L") and a_tier in ["中下游","下游"]:
            add("F03","强队失利后反弹","走势",6,"强队面对弱旅反弹","home_bounce"); home_adj += 4

        # ========== 综合评估 ==========
        total_score = sum(t["weight"] for t in triggered)
        draw_rules = [t for t in triggered if t.get("direction") == "draw"]
        zero_rules = [t for t in triggered if t.get("direction") == "zero_zero"]

        if total_score >= 25 or len(draw_rules) >= 4:
            rec = "⚠️ 多规则叠加，需高度警惕"
        elif total_score >= 15 or len(draw_rules) >= 3:
            rec = "🔶 经验信号较强，建议重点关注"
        elif total_score >= 8 or len(draw_rules) >= 2:
            rec = "📌 存在经验信号"
        elif len(triggered) >= 1:
            rec = "ℹ️ 单一信号，结合其他因素判断"
        else:
            rec = "✅ 无明显经验信号"

        return {
            "triggered": triggered, "triggered_count": len(triggered),
            "draw_boost": round(draw_boost, 2), "home_adj": round(home_adj, 2),
            "away_adj": round(away_adj, 2), "over_adj": round(over_adj, 2),
            "zero_zero_boost": round(zero_zero_boost, 2),
            "risk_signals": risk_signals, "total_score": total_score,
            "recommendation": rec, "draw_rules_count": len(draw_rules),
            "zero_zero_rules_count": len(zero_rules),
        }


def apply_experience_to_prediction(match_data, prediction, engine=None):
    if engine is None: engine = ExperienceEngine()
    exp = engine.analyze(match_data)

    hp = prediction.get("home_win_pct", 33)
    dp = prediction.get("draw_pct", 33)
    ap = prediction.get("away_win_pct", 34)

    # ★ BUG6修复：权重0.20→0.45
    w = 0.45
    hp += exp["home_adj"] * w
    dp += exp["draw_boost"] * w
    ap += exp["away_adj"] * w
    hp = max(3, hp); dp = max(3, dp); ap = max(3, ap)
    t = hp + dp + ap
    if t > 0:
        prediction["home_win_pct"] = round(hp/t*100, 1)
        prediction["draw_pct"] = round(dp/t*100, 1)
        prediction["away_win_pct"] = round(100-prediction["home_win_pct"]-prediction["draw_pct"], 1)

    sigs = prediction.get("smart_signals", [])
    for s in exp["risk_signals"]:
        if s not in sigs: sigs.append(s)
    prediction["smart_signals"] = sigs

    prediction["experience_analysis"] = {
        "triggered_count": exp["triggered_count"], "total_score": exp["total_score"],
        "draw_rules": exp["draw_rules_count"], "recommendation": exp["recommendation"],
        "over_adj": exp["over_adj"], "zero_zero_boost": exp["zero_zero_boost"],
        "rules": [{"id":t["id"],"name":t["name"],"weight":t["weight"]} for t in exp["triggered"]],
    }

    if exp["over_adj"] != 0:
        cur = prediction.get("over_2_5", 50)
        prediction["over_2_5"] = round(max(10, min(90, cur + exp["over_adj"] * 0.6)), 1)

    if exp["zero_zero_boost"] >= 10:
        prediction["zero_zero_alert"] = True
        if "🚨 0-0强信号" not in sigs:
            sigs.append("🚨 0-0强信号")

    return prediction