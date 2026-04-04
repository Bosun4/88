#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
koudai_intel_v2.py
机构内部绝密情报抓取与结构化引擎 (高鲁棒性重构版)
修复：增加 Session 会话、强化错误日志、分离核心凭证配置
"""

import requests
import json
import time
import random
import os
from datetime import datetime

class KoudaiSpider:
    def __init__(self):
        self.list_url = "https://apic.91bixin.net/api/match/getReportBriefData"
        self.detail_url = "https://apic.91bixin.net/api/match/reportList"
        
        # 使用 Session 保持底层 TCP 连接，提高连续请求速度
        self.session = requests.Session()
        
        # ==========================================
        # ⚠️ 【核心通行证配置区】 ⚠️
        # 每次抓包后，仅需更新这里的字典即可
        # ==========================================
        self.auth_payload = {
            "sign": "2b9b53ce7573dc0f29c05d57bd1761d364e49503016de3d2c6a4b4fedfe1aa5a", # 需替换最新
            "token": "1is2oo21h54kkf1d8i6ta19c8l06sdtqiu1e5mgha1mzed3n",               # 需替换最新
            "station_user_id": "50168030",
            "ts": "1774700632",                                                       # 需替换最新
            "nonce": "hyew3dzzo1puccl1",                                              # 需替换最新
            "lottery_id": "90",
            "enc": "0"
        }

    @staticmethod
    def get_random_user_agent():
        os_versions = ["16_5", "17_0", "17_1", "17_4_1", "18_0"]
        safari_versions = ["604.1", "605.1.15", "606.1"]
        return f"Mozilla/5.0 (iPhone; CPU iPhone OS {random.choice(os_versions)} like Mac OS X) AppleWebKit/{random.choice(safari_versions)} (KHTML, like Gecko) Version/26.3 Mobile/15E148 Safari/604.1"

    def get_match_list(self):
        params = {
            "platform": "koudai_mobile",
            "_prt": "https",
            "ver": "20180101000000",
            "t": str(int(time.time() * 10000000))
        }
        headers = {
            "User-Agent": self.get_random_user_agent(),
            "Accept": "application/json",
            "Origin": "https://koudai.17itou.com",
            "Referer": "https://koudai.17itou.com/",
        }
        
        print("⏳ [KoudaiSpider] 正在请求列表接口...")
        try:
            response = self.session.get(self.list_url, params=params, headers=headers, timeout=10)
            
            # 如果 HTTP 状态码不是 200，直接抛出异常
            response.raise_for_status() 
            result = response.json()
            
            if result.get("errcode") == 0:
                matches = result.get("data", {}).get("info", [])
                print(f"✅ [KoudaiSpider] 成功获取 {len(matches)} 场比赛！\n")
                return matches
            else:
                # 核心改进：打印出服务器拒绝的具体原因
                print(f"❌ [KoudaiSpider] 列表接口业务报错: {result.get('errmsg', '未知错误')} (errcode: {result.get('errcode')})")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ [KoudaiSpider] 列表请求网络异常: {e}")
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
        
        # 组装完整的 POST 数据
        data = self.auth_payload.copy()
        data["match_id2"] = match_id

        try:
            response = self.session.post(self.detail_url, params=params, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            result_json = response.json()
            
            if result_json.get("errcode") == 0:
                if "data" in result_json and isinstance(result_json["data"], dict):
                    return result_json["data"].get("trace_analysis", [])
                elif "trace_analysis" in result_json:
                     return result_json.get("trace_analysis", [])
            else:
                # 核心改进：详细暴露验签失败的原因
                print(f"  ⚠️ [详情接口拦截] 比赛ID {match_id} 拉取失败。服务器响应: {result_json}")
                
        except Exception as e:
            print(f"  ⚠️ [KoudaiSpider] 详情抓取网络异常: {e}")
        return None

    # ...(此处保留原有的 classify_traces 逻辑，未做大幅修改)...
    def classify_traces(self, traces, home_team, away_team):
        intel_dict = {
            "home_good_news": "", "home_bad_news": "",
            "guest_good_news": "", "guest_bad_news": "", "neutral_news": ""
        }
        if not traces: return intel_dict
            
        for trace in traces:
            title = trace.get("title", "")
            tag_name = trace.get("tag_info", {}).get("name", "爆料") 
            content = trace.get("content", "").replace('\n', ' ')
            belong_team = str(trace.get("belong_team", "0"))
            
            full_text = f"[{tag_name}] {title}。{content}\n"
            is_home_related = (belong_team == "1") or (home_team in full_text) or ("主队" in full_text) or ("主场" in full_text)
            is_guest_related = (belong_team == "2") or (away_team in full_text) or ("客队" in full_text) or ("客场" in full_text)
            
            if not is_home_related and not is_guest_related:
                intel_dict["neutral_news"] += full_text
                continue
            
            if tag_name in ["有利", "利好", "大名单齐整", "战意", "核心复出"]:
                if is_home_related and not is_guest_related: intel_dict["home_good_news"] += full_text
                elif is_guest_related and not is_home_related: intel_dict["guest_good_news"] += full_text
                else: intel_dict["neutral_news"] += full_text
            elif tag_name in ["不利", "伤停", "缺席", "内讧", "连败", "体能", "爆料", "情报"]: 
                if is_home_related and not is_guest_related: intel_dict["home_bad_news"] += full_text
                elif is_guest_related and not is_home_related: intel_dict["guest_bad_news"] += full_text
                else: intel_dict["neutral_news"] += full_text
            else:
                intel_dict["neutral_news"] += full_text
        return intel_dict

    def run_all_intel(self):
        final_intel_map = {}
        raw_debug_data = {}
        
        match_list = self.get_match_list()
        if not match_list:
            print("🚨 致命错误：未能获取比赛列表，进程终止。请检查网络或配置！")
            return final_intel_map

        for match in match_list:
            m_id = match.get("match_id2")
            home_team = match.get('home_team_name', '').strip()
            away_team = match.get('away_team_name', '').strip()
            match_key = f"{home_team}_{away_team}"
            
            print(f"\n{'='*50}")
            print(f"🕵️‍♂️ 正在窃取情报: {home_team} VS {away_team}")
            
            traces = self.get_detail_intel(m_id)
            raw_debug_data[match_key] = traces
            
            if traces:
                structured_intel = self.classify_traces(traces, home_team, away_team)
                final_intel_map[match_key] = structured_intel
                print(f"   ┖─ 提取成功! ")
            else:
                brief_title = match.get('title', '')
                brief_content = match.get('content', '').replace('\n', '')
                final_intel_map[match_key] = {"neutral_news": f"{brief_title} {brief_content}"}
                print(f"   ┖─ ⚠️ 高级情报为空，使用基础摘要")
            
            # 引入随机休眠，避免被速率限制 (Rate Limiting) 封禁
            time.sleep(random.uniform(1.0, 2.5))
            
        try:
            os.makedirs("data", exist_ok=True)
            debug_path = os.path.join("data", "koudai_debug_latest.json")
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump({"update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": final_intel_map}, f, ensure_ascii=False, indent=2)
            print(f"\n📁 [KoudaiSpider] 数据已保存至: {debug_path}")
        except Exception as e:
            print(f"\n⚠️ 保存失败: {e}")
            
        return final_intel_map

if __name__ == "__main__":
    spider = KoudaiSpider()
    all_intel = spider.run_all_intel()
