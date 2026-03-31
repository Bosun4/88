#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
koudai_intel.py
机构内部绝密情报抓取与结构化引擎 (针对真实抓包结构优化版)
"""

import requests
import json
import time
import random
import string
import os
from datetime import datetime

class KoudaiSpider:
    def __init__(self):
        self.list_url = "https://apic.91bixin.net/api/match/getReportBriefData"
        self.detail_url = "https://apic.91bixin.net/api/match/reportList"
        self.fixed_sign = "2b9b53ce7573dc0f29c05d57bd1761d364e49503016de3d2c6a4b4fedfe1aa5a"
        self.fixed_token = "1is2oo21h54kkf1d8i6ta19c8l06sdtqiu1e5mgha1mzed3n"
        self.station_user_id = "50168030"

    @staticmethod
    def get_random_user_agent():
        os_versions = ["16_5", "17_0", "17_1", "17_4_1", "18_0", "18_7"]
        safari_versions = ["604.1", "605.1.15", "606.1"]
        os_ver = random.choice(os_versions)
        safari_ver = random.choice(safari_versions)
        return f"Mozilla/5.0 (iPhone; CPU iPhone OS {os_ver} like Mac OS X) AppleWebKit/{safari_ver} (KHTML, like Gecko) Version/26.3 Mobile/15E148 Safari/604.1"

    @staticmethod
    def get_random_nonce(length=16):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    @staticmethod
    def get_current_timestamp():
        return str(int(time.time()))

    @staticmethod
    def get_current_t_param():
        return str(int(time.time() * 10000000))

    def get_match_list(self):
        params = {
            "platform": "koudai_mobile",
            "_prt": "https",
            "ver": "20180101000000",
            "t": self.get_current_t_param()
        }
        headers = {
            "User-Agent": self.get_random_user_agent(),
            "Accept": "application/json",
            "Origin": "https://koudai.17itou.com",
            "Referer": "https://koudai.17itou.com/",
            "Connection": "keep-alive"
        }
        
        print("⏳ [KoudaiSpider] 正在潜入后台拉取今日绝密赛事列表...")
        try:
            response = requests.get(self.list_url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                result = response.json()
                if result.get("errcode") == 0:
                    matches = result.get("data", {}).get("info", [])
                    print(f"✅ [KoudaiSpider] 成功锁定 {len(matches)} 场高价值比赛情报！\n")
                    return matches
        except Exception as e:
            print(f"❌ [KoudaiSpider] 列表请求遭遇拦截: {e}")
        return []

    def get_detail_intel(self, match_id):
        params = {
            "platform": "koudai_mobile",
            "_prt": "https",
            "ver": "20180101000000"
        }
        headers = {
            "User-Agent": self.get_random_user_agent(),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        data = {
            "sign": self.fixed_sign,
            "token": self.fixed_token,
            "station_user_id": self.station_user_id,
            "match_id2": match_id,
            "lottery_id": "90",
            "enc": "0",
            "ts": self.get_current_timestamp(),
            "nonce": self.get_random_nonce()
        }

        try:
            response = requests.post(self.detail_url, params=params, headers=headers, data=data, timeout=15)
            if response.status_code == 200:
                result_json = response.json()
                if "data" in result_json and isinstance(result_json["data"], dict):
                    return result_json["data"].get("trace_analysis", [])
                elif "trace_analysis" in result_json:
                     return result_json.get("trace_analysis", [])
        except Exception as e:
            print(f"  ⚠️ [KoudaiSpider] 详情抓取超时或异常: {e}")
        return None

    def classify_traces(self, traces, home_team, away_team):
        intel_dict = {
            "home_good_news": "",
            "home_bad_news": "",
            "guest_good_news": "",
            "guest_bad_news": "",
            "neutral_news": ""
        }
        
        if not traces:
            return intel_dict
            
        for trace in traces:
            title = trace.get("title", "")
            tag_name = trace.get("tag_info", {}).get("name", "")
            content = trace.get("content", "").replace('\n', ' ')
            belong_team = str(trace.get("belong_team", "0"))
            
            full_text = f"[{tag_name}] {title}。{content}\n"
            
            is_home_related = (belong_team == "1") or (home_team in full_text) or ("主队" in full_text) or ("主场" in full_text)
            is_guest_related = (belong_team == "2") or (away_team in full_text) or ("客队" in full_text) or ("客场" in full_text)
            
            if not is_home_related and not is_guest_related:
                intel_dict["neutral_news"] += full_text
                continue
            
            if tag_name in ["有利", "利好", "大名单齐整", "战意", "核心复出"]:
                if is_home_related and not is_guest_related:
                    intel_dict["home_good_news"] += full_text
                elif is_guest_related and not is_home_related:
                    intel_dict["guest_good_news"] += full_text
                else:
                    intel_dict["neutral_news"] += full_text
                    
            elif tag_name in ["不利", "伤停", "缺席", "内讧", "连败", "体能"]:
                if is_home_related and not is_guest_related:
                    intel_dict["home_bad_news"] += full_text
                elif is_guest_related and not is_home_related:
                    intel_dict["guest_bad_news"] += full_text
                else:
                    intel_dict["neutral_news"] += full_text
            else:
                intel_dict["neutral_news"] += full_text
                
        return intel_dict

    def run_all_intel(self):
        final_intel_map = {}
        raw_debug_data = {}
        
        match_list = self.get_match_list()
        if not match_list:
            return final_intel_map

        for match in match_list:
            m_id = match.get("match_id2")
            home_team = match.get('home_team_name', '').strip()
            away_team = match.get('away_team_name', '').strip()
            match_key = f"{home_team}_{away_team}"
            
            print(f"🕵️‍♂️ 正在窃取情报: {home_team} VS {away_team}")
            
            traces = self.get_detail_intel(m_id)
            raw_debug_data[match_key] = traces
            
            if traces:
                structured_intel = self.classify_traces(traces, home_team, away_team)
                final_intel_map[match_key] = structured_intel
                bad_h = len(structured_intel['home_bad_news']) > 0
                bad_g = len(structured_intel['guest_bad_news']) > 0
                print(f"   ┖─ 提取成功! ⚠️主队隐患:{bad_h} | ⚠️客队隐患:{bad_g}")
            else:
                final_intel_map[match_key] = self.classify_traces([], home_team, away_team)
                print(f"   ┖─ ⚠️ 暂无深度情报或抓取失败")
            
            time.sleep(random.uniform(1.5, 3.5))
            
        try:
            os.makedirs("data", exist_ok=True)
            debug_path = os.path.join("data", "koudai_debug_latest.json")
            
            output_content = {
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_matches": len(match_list),
                "raw_data_from_api": raw_debug_data,
                "parsed_intel_for_ai": final_intel_map
            }
            
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(output_content, f, ensure_ascii=False, indent=2)
                
            print(f"\n📁 [KoudaiSpider] 【数据已落盘】赶紧去查看: {debug_path}")
        except Exception as e:
            print(f"\n⚠️ [KoudaiSpider] 数据保存到 data/ 失败: {e}")
            
        print("\n🎉 [KoudaiSpider] 所有可用绝密情报提取完毕！可以移交 AI 矩阵！")
        return final_intel_map

if __name__ == "__main__":
    spider = KoudaiSpider()
    all_intel = spider.run_all_intel()