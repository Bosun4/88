import requests
import json
import time
import random
import string

# --- 随机伪装工具箱 ---

def get_random_user_agent():
    """随机生成不同 iOS 版本的 iPhone User-Agent"""
    os_versions = ["16_5", "17_0", "17_1", "17_4_1", "18_0", "18_7"]
    safari_versions = ["604.1", "605.1.15", "606.1"]
    
    os_ver = random.choice(os_versions)
    safari_ver = random.choice(safari_versions)
    
    return f"Mozilla/5.0 (iPhone; CPU iPhone OS {os_ver} like Mac OS X) AppleWebKit/{safari_ver} (KHTML, like Gecko) Version/26.3 Mobile/15E148 Safari/604.1"

def get_random_nonce(length=16):
    """随机生成类似 'hyew3dzzo1puccl1' 的 16 位小写字母+数字"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def get_current_timestamp():
    """获取当前的 10 位秒级时间戳"""
    return str(int(time.time()))

def get_current_t_param():
    """获取列表接口需要的 17 位微秒级时间戳模拟值"""
    return str(int(time.time() * 10000000))

# --- 核心拉取逻辑 ---

def get_match_list():
    """拉取赛事列表 (加入随机请求头)"""
    url = "https://apic.91bixin.net/api/match/getReportBriefData"
    
    # 动态生成 URL 参数里的时间戳
    params = {
        "platform": "koudai_mobile",
        "_prt": "https",
        "ver": "20180101000000",
        "t": get_current_t_param()  # 随机时间戳
    }
    
    # 动态生成 User-Agent
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "application/json",
        "Origin": "https://koudai.17itou.com",
        "Referer": "https://koudai.17itou.com/",
        "Connection": "keep-alive"
    }
    
    print("⏳ 正在拉取今日赛事列表...")
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if result.get("errcode") == 0:
                matches = result.get("data", {}).get("info", [])
                print(f"✅ 成功拉取到 {len(matches)} 场比赛！\n")
                return matches
    except Exception as e:
        print(f"❌ 列表请求报错: {e}")
    return []

def get_detail_intel(match_id):
    """拉取详细情报 (加入动态参数)"""
    url = "https://apic.91bixin.net/api/match/reportList"
    
    params = {
        "platform": "koudai_mobile",
        "_prt": "https",
        "ver": "20180101000000"
    }
    
    headers = {
        "User-Agent": get_random_user_agent(), # 每次请求都换一台“新手机”
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    
    # 动态生成表单数据
    current_ts = get_current_timestamp()
    current_nonce = get_random_nonce()
    
    data = {
        "sign": "2b9b53ce7573dc0f29c05d57bd1761d364e49503016de3d2c6a4b4fedfe1aa5a", # 签名仍需固定
        "token": "1is2oo21h54kkf1d8i6ta19c8l06sdtqiu1e5mgha1mzed3n",               # Token 代表你的账号，必须固定
        "station_user_id": "50168030",                                          # 用户ID必须固定
        "match_id2": match_id,
        "lottery_id": "90",
        "enc": "0",
        "ts": current_ts,       # 替换为当前真实时间
        "nonce": current_nonce  # 替换为全新随机数
    }

    try:
        response = requests.post(url, params=params, headers=headers, data=data)
        if response.status_code == 200:
            return response.json().get("data", {}).get("trace_analysis", [])
    except:
        pass
    return None

# --- 主程序 ---
if __name__ == "__main__":
    match_list = get_match_list()
    if not match_list:
        exit()

    for match in match_list:
        m_id = match.get("match_id2")
        print(f"🏆 {match.get('league_name')} | {match.get('home_team_name')} VS {match.get('away_team_name')}")
        
        traces = get_detail_intel(m_id)
        if traces:
            for trace in traces:
                print(f" 🔴 [{trace.get('tag_info', {}).get('name', '')}] {trace.get('title')}")
        else:
            print(f" ⚠️ 高级情报拉取失败，返回基础摘要: {match.get('title')}")
        
        print("-" * 30)
        # 随机停顿 1.5 到 3.5 秒，模仿人类阅读和点击的节奏！极其重要！
        time.sleep(random.uniform(1.5, 3.5)) 
