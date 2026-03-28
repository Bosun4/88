#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
koudai_intel.py
机构内部绝密情报抓取与结构化引擎 (针对真实抓包结构优化版)
作用：抓取比赛的突发伤停、利好利空情报，并自动转化为 AI 矩阵需要的标准格式。
"""

import requests
import json
import time
import random
import string
import os

class KoudaiSpider:
    def __init__(self):
        self.list_url = "https://apic.91bixin.net/api/match/getReportBriefData"
        self.detail_url = "https://apic.91bixin.net/api/match/reportList"
        # 你的核心账号 Token 和签名（注意：如果过期需要重新抓包替换）
        self.fixed_sign = "2b9b53ce7573dc0f29c05d57bd1761d364e49503016de3d2c6a4b4fedfe1aa5a"
        self.fixed_token = "1is2oo21h54kkf1d8i6ta19c8l06sdtqiu1e5mgha1mzed3n"
        self.station_user_id = "50168030"

    # --- 随机伪装工具箱 ---
    @staticmethod
    def get_random_user_agent():
        """随机生成不同 iOS 版本的 iPhone User-Agent，极致防封"""
        os_versions = ["16_5", "17_0", "17_1", "17_4_1", "18_0", "18_7"]
        safari_versions = ["604.1", "605.1.15", "606.1"]
        os_ver = random.choice(os_versions)
        safari_ver = random.choice(safari_versions)
        return f"Mozilla/5.0 (iPhone; CPU iPhone OS {os_ver} like Mac OS X) AppleWebKit/{safari_ver} (KHTML, like Gecko) Version/26.3 Mobile/15E148 Safari/604.1"

    @staticmethod
    def get_random_nonce(length=16):
        """随机生成 16 位小写字母+数字"""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    @staticmethod
    def get_current_timestamp():
        """获取当前的 10 位秒级时间戳"""
        return str(int(time.time()))

    @staticmethod
    def get_current_t_param():
        """获取列表接口需要的 17 位微秒级时间戳模拟值"""
        return str(int(time.time() * 10000000))

    # --- 核心拉取逻辑 ---
    def get_match_list(self):
        """拉取赛事列表 (加入随机请求头)"""
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
        """拉取单场详细情报 (针对真实抓包结构优化)"""
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
                # 兼容不同的返回层级结构
                if "data" in result_json and isinstance(result_json["data"], dict):
                    return result_json["data"].get("trace_analysis", [])
                elif "trace_analysis" in result_json:
                     return result_json.get("trace_analysis", [])
        except Exception as e:
            print(f"  ⚠️ [KoudaiSpider] 详情抓取超时或异常: {e}")
        return None

    # --- 结构化情报分类器 (核心枢纽) ---
    def classify_traces(self, traces, home_team, away_team):
        """
        将杂乱的 Tag 和 Title 精准归类，生成 predict.py 需要的 information 结构。
        """
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
            
            # 合并标题和内容，提供给 AI 最丰满的数据
            full_text = f"[{tag_name}] {title}。{content}\n"
            
            # 增强归属判断：优先使用 belong_team，辅以文本模糊匹配
            # 假设 1 代表主队，2 代表客队
            is_home_related = (belong_team == "1") or (home_team in full_text) or ("主队" in full_text) or ("主场" in full_text)
            is_guest_related = (belong_team == "2") or (away_team in full_text) or ("客队" in full_text) or ("客场" in full_text)
            
            # 如果文本里既没提到主也没提到客，且 belong_team 为 0，默认放到中立新闻
            if not is_home_related and not is_guest_related:
                intel_dict["neutral_news"] += full_text
                continue
            
            # 语义情感判断
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
                # 裁判、中立场地、数据统计等
                intel_dict["neutral_news"] += full_text
                
        return intel_dict

    # --- 全自动一键拉取封装 ---
    def run_all_intel(self):
        """
        一键执行：获取列表 -> 遍历详情 -> 结构化分类 -> 返回字典。
        返回格式: {"主队名_客队名": {"home_bad_news": "...", ...}}
        """
        final_intel_map = {}
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
            if traces:
                # 结构化分类
                structured_intel = self.classify_traces(traces, home_team, away_team)
                final_intel_map[match_key] = structured_intel
                
                # 终端华丽展示
                bad_h = len(structured_intel['home_bad_news']) > 0
                bad_g = len(structured_intel['guest_bad_news']) > 0
                print(f"   ┖─ 提取成功! ⚠️主队隐患:{bad_h} | ⚠️客队隐患:{bad_g}")
            else:
                final_intel_map[match_key] = self.classify_traces([], home_team, away_team)
                print(f"   ┖─ ⚠️ 暂无深度情报或抓取失败")
            
            # 随机停顿 1.5 到 3.5 秒，模仿人类真实节奏，极其重要！
            time.sleep(random.uniform(1.5, 3.5))
            
        print("\n🎉 [KoudaiSpider] 所有可用绝密情报提取完毕！可以移交 AI 矩阵！")
        return final_intel_map

# ====================================================================
# 独立测试入口 (直接运行此文件可测试抓取效果)
# ====================================================================
if __name__ == "__main__":
    spider = KoudaiSpider()
    all_intel = spider.run_all_intel()
    
    # 随便打印一场看看效果
    if all_intel:
        first_match = list(all_intel.keys())[0]
        print(f"\n--- 测试输出 [{first_match}] 的结构化情报 ---")
        print(json.dumps(all_intel[first_match], indent=2, ensure_ascii=False))


