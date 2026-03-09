import requests
import time
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from config import API_FOOTBALL_KEY, API_FOOTBALL_BASE

# ==========================================
# 1. 核心纠错字典 (防止机器翻错黑话)
# ==========================================
FIXED_MAPPING = {
    "马竞": "Atletico Madrid", "国米": "Inter", "皇马": "Real Madrid",
    "巴萨": "Barcelona", "曼联": "Manchester United", "曼城": "Manchester City",
    "热刺": "Tottenham", "切尔西": "Chelsea", "阿森纳": "Arsenal",
    "尤文": "Juventus", "拜仁": "Bayern Munich", "中国女": "China PR W"
}

# ==========================================
# 2. 自动化翻译引擎 (中 -> 英)
# ==========================================
def translate_team_name(chinese_name):
    """先查字典，再走Google自动直译，确保匹配率"""
    if chinese_name in FIXED_MAPPING:
        return FIXED_MAPPING[chinese_name]
    try:
        # 去掉干扰词，增加直译成功率
        clean_name = chinese_name.replace("女足", " Women").replace("联", " United")
        translated = GoogleTranslator(source='zh-CN', target='en').translate(clean_name)
        return translated.replace("FC", "").replace("'", "").strip()
    except Exception as e:
        print(f"    ⚠️ 翻译引擎故障: {e}")
        return chinese_name

# ==========================================
# 3. 核心 API 匹配函数 (换取 ID)
# ==========================================
def search_team_api(name):
    """自动化翻译并换取 API-Football 唯一 ID"""
    english_name = translate_team_name(name)
    print(f"    🔍 转换匹配: [{name}] -> [{english_name}]")
    
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        # 第一次尝试：全名搜索
        r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=headers, params={"search": english_name}, timeout=15)
        data = r.json()
        if data.get("response"):
            t = data["response"][0]["team"]
            return {"id": t["id"], "name": t["name"], "logo": t.get("logo", "")}
            
        # 第二次尝试：模糊搜索首词
        if " " in english_name:
            short = english_name.split()[0]
            print(f"    ⚠️ 未命中，尝试模糊搜索短名: {short}")
            r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=headers, params={"search": short}, timeout=15)
            data = r.json()
            if data.get("response"):
                t = data["response"][0]["team"]
                return {"id": t["id"], "name": t["name"], "logo": t.get("logo", "")}
    except Exception as e:
        print(f"    ❌ API搜索异常: {e}")
    return None

# ==========================================
# 4. 实时数据抓取 (为 14 模型供能)
# ==========================================
def get_team_stats(team_id):
    """抓取场均进球、失球、近况，防止模型输出 1-0"""
    if not team_id: return {}
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        # 获取本年度该队统计 (默认 2025/2026 赛季)
        params = {"team": team_id, "season": 2025}
        r = requests.get(f"{API_FOOTBALL_BASE}/teams/statistics", headers=headers, params=params, timeout=15)
        d = r.json().get("response", {})
        
        # 核心字段：这是 14 个模型最需要的“食材”
        return {
            "avg_goals_for": d.get("goals", {}).get("for", {}).get("average", {}).get("total", 1.3),
            "avg_goals_against": d.get("goals", {}).get("against", {}).get("average", {}).get("total", 1.1),
            "clean_sheets": d.get("clean_sheet", {}).get("total", 0),
            "played": d.get("fixtures", {}).get("played", {}).get("total", 10),
            "form": d.get("form", "WWDDD") # 近期走势：胜平负
        }
    except:
        return {"avg_goals_for": 1.3, "avg_goals_against": 1.1, "form": "DDDDD", "played": 10}

def get_h2h_data(id1, id2):
    """抓取两队历史交锋数据"""
    if not id1 or not id2: return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead", headers=headers, params={"h2h": f"{id1}-{id2}"}, timeout=15)
        return r.json().get("response", [])
    except:
        return []

# ==========================================
# 5. 500.com 爬虫入口 (保持原有逻辑)
# ==========================================
def get_500_matches(date_str):
    """
    这里保留你原来的 500.com 爬虫代码，
    抓取当日比赛的中文名、赔率、联赛信息。
    """
    # ... 原有爬虫代码 ...
    # 示例返回: [{"home": "曼联", "away": "利物浦", "league": "英超", "odds": {...}}]
    pass

# ==========================================
# 6. 核心指挥部：数据收集闭环
# ==========================================
def collect_all(date_str):
    raw_list = get_500_matches(date_str)
    final_matches = []
    
    print(f"🚀 开始为 {len(raw_list)} 场比赛装载实时数据...")
    for m in raw_list:
        # 1. 中文队名 -> 英文名 -> API ID
        h_info = search_team_api(m['home'])
        a_info = search_team_api(m['away'])
        
        if h_info and a_info:
            # 2. 拿着 ID 抓取真实的进球/防守/H2H
            h_stats = get_team_stats(h_info['id'])
            a_stats = get_team_stats(a_info['id'])
            h2h = get_h2h_data(h_info['id'], a_info['id'])
            
            # 3. 完美封装，喂给 predict.py
            m.update({
                "home_stats": h_stats,
                "away_stats": a_stats,
                "h2h": h2h,
                "home_logo": h_info['logo'],
                "away_logo": a_info['logo']
            })
        else:
            # 搜不到则打标记，让模型进入保底模式
            m["fallback"] = True
            
        final_matches.append(m)
        time.sleep(1) # 防屏蔽
        
    return {"matches": final_matches, "update_time": date_str}
