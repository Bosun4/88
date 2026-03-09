import requests
import time
import re
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from config import API_FOOTBALL_KEY, API_FOOTBALL_BASE

# ==========================================
# 1. 核心纠错字典 (黑话简称强制映射)
# ==========================================
FIXED_MAPPING = {
    "马竞": "Atletico Madrid", "国米": "Inter", "皇马": "Real Madrid",
    "巴萨": "Barcelona", "曼联": "Manchester United", "曼城": "Manchester City",
    "热刺": "Tottenham", "切尔西": "Chelsea", "阿森纳": "Arsenal",
    "尤文": "Juventus", "拜仁": "Bayern Munich", "中国女": "China PR W",
    "蔚山现代": "Ulsan Hyundai", "全北现代": "Jeonbuk Motors"
}

# ==========================================
# 2. 自动化工具库 (翻译与匹配)
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
        print(f"    ⚠️ 翻译失败: {chinese_name} -> {e}")
        return chinese_name

def search_team_api(name):
    """通过翻译后的英文名，去 API-Football 换取唯一 ID 和 Logo"""
    english_name = translate_team_name(name)
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        # 1. 全名搜索
        r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=headers, params={"search": english_name}, timeout=15)
        data = r.json()
        if data.get("response") and len(data["response"]) > 0:
            t = data["response"][0]["team"]
            return {"id": t["id"], "name": t["name"], "logo": t.get("logo", "")}
            
        # 2. 模糊搜索首词 (防冗余后缀)
        if " " in english_name:
            short = english_name.split()[0]
            r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=headers, params={"search": short}, timeout=15)
            data = r.json()
            if data.get("response") and len(data["response"]) > 0:
                t = data["response"][0]["team"]
                return {"id": t["id"], "name": t["name"], "logo": t.get("logo", "")}
    except Exception as e:
        print(f"    ❌ API搜索异常: {e}")
    return None

# ==========================================
# 3. 实时统计抓取 (为 14 模型供能)
# ==========================================
def get_team_stats(team_id):
    """抓取场均进球、失球、近况，这是防止 1-0 的关键数据"""
    if not team_id: return {}
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        # 获取本年度统计 (2025/2026)
        params = {"team": team_id, "season": 2025}
        r = requests.get(f"{API_FOOTBALL_BASE}/teams/statistics", headers=headers, params=params, timeout=15)
        d = r.json().get("response", {})
        
        # 提取 14 模型最核心的 5 个维度食材
        return {
            "avg_goals_for": d.get("goals", {}).get("for", {}).get("average", {}).get("total", 1.3),
            "avg_goals_against": d.get("goals", {}).get("against", {}).get("average", {}).get("total", 1.1),
            "clean_sheets": d.get("clean_sheet", {}).get("total", 0),
            "played": d.get("fixtures", {}).get("played", {}).get("total", 10),
            "form": d.get("form", "WWDDD"),
            "wins": d.get("fixtures", {}).get("wins", {}).get("total", 4),
            "draws": d.get("fixtures", {}).get("draws", {}).get("total", 3),
            "losses": d.get("fixtures", {}).get("losses", {}).get("total", 3)
        }
    except:
        return {"avg_goals_for": 1.3, "avg_goals_against": 1.1, "form": "DDDDD", "played": 10}

def get_h2h_data(id1, id2):
    """抓取两队历史交锋"""
    if not id1 or not id2: return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead", headers=headers, params={"h2h": f"{id1}-{id2}"}, timeout=15)
        raw_h2h = r.json().get("response", [])
        clean_h2h = []
        for x in raw_h2h[:5]:
            f = x.get("fixture", {})
            clean_h2h.append({
                "date": f.get("date", "")[:10],
                "home": x.get("teams", {}).get("home", {}).get("name", ""),
                "away": x.get("teams", {}).get("away", {}).get("name", ""),
                "score": f"{x.get('goals',{}).get('home','?')}-{x.get('goals',{}).get('away','?')}"
            })
        return clean_h2h
    except:
        return []

# ==========================================
# 4. 500.com 竞彩解析引擎 (全代码还原)
# ==========================================
def get_500_matches(date_str):
    """
    全量解析 500 彩票网竞彩页面，提取场次、队伍、赔率
    """
    url = f"https://predict.500.com/static/public/ssc/xml/expect/{date_str}.xml" # 竞彩底层数据接口
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    print(f"    🌐 正在连接 500.com 数据源...")
    try:
        r = requests.get(f"https://trade.500.com/jczq/?date={date_str}", headers=headers, timeout=20)
        r.encoding = 'gbk' # 500网使用GBK编码
        soup = BeautifulSoup(r.text, 'html.parser')
        
        matches = []
        rows = soup.select("tr.tr_off") + soup.select("tr.tr_on")
        
        for row in rows:
            try:
                # 提取场次信息 (如 周一001)
                m_num = row.select_one("td.td_id").text.strip() if row.select_one("td.td_id") else ""
                # 提取联赛
                lg = row.select_one("td.td_ls").text.strip() if row.select_one("td.td_ls") else ""
                # 提取队伍
                home = row.select_one("td.td_home a").text.strip()
                away = row.select_one("td.td_away a").text.strip()
                
                # 提取初赔 (胜平负)
                odds_td = row.select("td.td_pl")
                odds_list = [span.text.strip() for span in odds_td[0].select("span.pk_pl")] if odds_td else []
                
                if home and away:
                    matches.append({
                        "match_num": m_num,
                        "league": lg,
                        "home": home,
                        "away": away,
                        "odds": {
                            "avg_home_odds": float(odds_list[0]) if len(odds_list)>0 else 0,
                            "avg_draw_odds": float(odds_list[1]) if len(odds_list)>1 else 0,
                            "avg_away_odds": float(odds_list[2]) if len(odds_list)>2 else 0
                        }
                    })
            except:
                continue
        print(f"    ✅ 成功解析 {len(matches)} 场竞彩场次")
        return matches
    except Exception as e:
        print(f"    ❌ 网页解析失败: {e}")
        return []

# ==========================================
# 5. 指挥部：全数据生命周期闭环
# ==========================================
def collect_all(date_str):
    # 步骤 A: 抓取 500 网基础信息
    raw_list = get_500_matches(date_str)
    final_matches = []
    
    print(f"\n🚀 启动【全自动翻译+数据补全】引擎 (共 {len(raw_list)} 场)")
    
    for i, m in enumerate(raw_list):
        print(f"--- [{i+1}/{len(raw_list)}] 处理: {m['home']} vs {m['away']} ---")
        
        # 步骤 B: 中文 -> 英文 -> 换取 API 唯一 ID
        h_info = search_team_api(m['home'])
        a_info = search_team_api(m['away'])
        
        if h_info and a_info:
            # 步骤 C: 拿着 ID 抓取实时状态 (这是 14 个模型真正需要的数据)
            h_stats = get_team_stats(h_info['id'])
            a_stats = get_team_stats(a_info['id'])
            h2h = get_h2h_data(h_info['id'], a_info['id'])
            
            # 封装完整数据包
            m.update({
                "id": h_info['id'],
                "home_stats": h_stats,
                "away_stats": a_stats,
                "h2h": h2h,
                "home_logo": h_info['logo'],
                "away_logo": a_info['logo'],
                "fallback": False
            })
            print(f"    📊 成功挂载实时数据 (xG进球: {h_stats['avg_goals_for']})")
        else:
            print(f"    ⚠️ 无法匹配 ID，该场比赛进入 ELO 保底模式")
            m["fallback"] = True
            
        final_matches.append(m)
        time.sleep(1.2) # 严格控制频率，防止被 API 或 Google 封锁
        
    return {"matches": final_matches, "update_time": date_str, "total_matches": len(final_matches)}
