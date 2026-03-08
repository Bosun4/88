import json,os,requests,time
from datetime import datetime,timedelta
from config import *

def get_yesterday():
    from zoneinfo import ZoneInfo
    return (datetime.now(ZoneInfo(TIMEZONE))-timedelta(days=1)).strftime("%Y-%m-%d")

def fetch_actual_results(date):
    """从API获取昨日实际比分"""
    h={"x-apisports-key":API_FOOTBALL_KEY}
    results={}
    try:
        r=requests.get(API_FOOTBALL_BASE+"/fixtures",headers=h,params={"date":date},timeout=15)
        d=r.json()
        if d.get("response"):
            for m in d["response"]:
                if m["fixture"]["status"]["short"] in ["FT","AET","PEN"]:
                    key=m["teams"]["home"]["name"]+"_"+m["teams"]["away"]["name"]
                    results[key]={"home_goals":m["goals"]["home"],"away_goals":m["goals"]["away"],"status":"FT"}
    except Exception as e:
        print("Fetch results error:%s"%e)
    return results

def verify_predictions(pred_file,date):
    """对比预测vs实际"""
    if not os.path.exists(pred_file):
        print("No prediction file for %s"%date)
        return None
    with open(pred_file,"r",encoding="utf-8") as f:
        data=json.load(f)
    actual=fetch_actual_results(date)
    print("Found %d actual results"%len(actual))
    verified=[]
    total=0;correct_result=0;correct_score=0;correct_ou=0
    for match in data.get("results",[]):
        home=match["home_team"];away=match["away_team"]
        pred=match.get("prediction",{})
        # 尝试匹配
        found=None
        for key,res in actual.items():
            if home.lower() in key.lower() or away.lower() in key.lower():
                found=res;break
        if not found:continue
        total+=1
        hg=found["home_goals"];ag=found["away_goals"]
        actual_result="主胜" if hg>ag else("平局" if hg==ag else "客胜")
        actual_score="%d-%d"%(hg,ag)
        actual_ou="大" if hg+ag>2 else "小"
        actual_btts="是" if hg>0 and ag>0 else "否"
        pred_correct=pred.get("result","")==actual_result
        score_correct=pred.get("predicted_score","")==actual_score
        ou_correct=pred.get("over_under_2_5","")==actual_ou
        if pred_correct:correct_result+=1
        if score_correct:correct_score+=1
        if ou_correct:correct_ou+=1
        verified.append({
            "home_team":home,"away_team":away,"league":match.get("league",""),
            "predicted_result":pred.get("result","?"),
            "actual_result":actual_result,
            "predicted_score":pred.get("predicted_score","?"),
            "actual_score":actual_score,
            "confidence":pred.get("confidence",0),
            "result_correct":pred_correct,
            "score_correct":score_correct,
            "ou_correct":ou_correct,
            "was_recommended":match.get("is_recommended",False),
            "model_consensus":pred.get("model_consensus",0),
        })
    stats={
        "date":date,
        "total_verified":total,
        "result_correct":correct_result,
        "score_correct":correct_score,
        "ou_correct":correct_ou,
        "result_rate":round(correct_result/total*100,1) if total else 0,
        "score_rate":round(correct_score/total*100,1) if total else 0,
        "ou_rate":round(correct_ou/total*100,1) if total else 0,
        "matches":verified,
    }
    # 推荐场次单独统计
    rec=[v for v in verified if v["was_recommended"]]
    rec_correct=sum(1 for v in rec if v["result_correct"])
    stats["recommended_total"]=len(rec)
    stats["recommended_correct"]=rec_correct
    stats["recommended_rate"]=round(rec_correct/len(rec)*100,1) if rec else 0
    # 高置信度统计
    high=[v for v in verified if v["confidence"]>=65]
    high_correct=sum(1 for v in high if v["result_correct"])
    stats["high_conf_total"]=len(high)
    stats["high_conf_correct"]=high_correct
    stats["high_conf_rate"]=round(high_correct/len(high)*100,1) if high else 0
    return stats

def load_history():
    """加载历史记录"""
    hfile="data/history.json"
    if os.path.exists(hfile):
        with open(hfile,"r",encoding="utf-8") as f:
            return json.load(f)
    return{"days":[],"cumulative":{"total":0,"result_correct":0,"score_correct":0,"ou_correct":0,"rec_total":0,"rec_correct":0}}

def save_history(history):
    with open("data/history.json","w",encoding="utf-8") as f:
        json.dump(history,f,ensure_ascii=False,indent=2)

def run_verify():
    yesterday=get_yesterday()
    print("Verifying %s..."%yesterday)
    pred_file="data/predictions_%s.json"%yesterday
    if not os.path.exists(pred_file):
        pred_file="data/predictions.json"
    stats=verify_predictions(pred_file,yesterday)
    if not stats or stats["total_verified"]==0:
        print("No matches to verify")
        return
    # 保存当日验证
    os.makedirs("data",exist_ok=True)
    with open("data/verify_%s.json"%yesterday,"w",encoding="utf-8") as f:
        json.dump(stats,f,ensure_ascii=False,indent=2)
    # 更新累计历史
    history=load_history()
    # 检查是否已验证过
    if any(d.get("date")==yesterday for d in history["days"]):
        print("Already verified %s"%yesterday)
        return
    day_summary={"date":yesterday,"total":stats["total_verified"],"result_correct":stats["result_correct"],"result_rate":stats["result_rate"],"score_correct":stats["score_correct"],"ou_correct":stats["ou_correct"],"rec_total":stats["recommended_total"],"rec_correct":stats["recommended_correct"],"rec_rate":stats["recommended_rate"]}
    history["days"].append(day_summary)
    history["days"]=history["days"][-30:]
    c=history["cumulative"]
    c["total"]+=stats["total_verified"]
    c["result_correct"]+=stats["result_correct"]
    c["score_correct"]+=stats["score_correct"]
    c["ou_correct"]+=stats["ou_correct"]
    c["rec_total"]+=stats["recommended_total"]
    c["rec_correct"]+=stats["recommended_correct"]
    c["result_rate"]=round(c["result_correct"]/c["total"]*100,1) if c["total"] else 0
    c["score_rate"]=round(c["score_correct"]/c["total"]*100,1) if c["total"] else 0
    c["rec_rate"]=round(c["rec_correct"]/c["rec_total"]*100,1) if c["rec_total"] else 0
    save_history(history)
    print("Verified! Result rate:%.1f%% (%d/%d) Rec rate:%.1f%%"%(stats["result_rate"],stats["result_correct"],stats["total_verified"],stats["recommended_rate"]))

if __name__=="__main__":
    run_verify()
