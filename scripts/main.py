import json,os,sys
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fetch_data import collect_all,get_today
from predict import run_predictions
def main():
    date=get_today();now=datetime.now(ZoneInfo("Asia/Shanghai"));session="morning"if now.hour<15 else"evening"
    print("Football AI|%s|%s"%(date,session))
    raw=collect_all(date);os.makedirs("data",exist_ok=True)
    with open("data/raw_data.json","w",encoding="utf-8")as f:json.dump(raw,f,ensure_ascii=False,indent=2)
    if not raw["matches"]:
        print("No matches");out={"date":date,"session":session,"update_time":now.strftime("%Y-%m-%d %H:%M:%S"),"total_matches":0,"results":[],"top4":[]}
    else:
        res,t4=run_predictions(raw);out={"date":date,"session":session,"update_time":now.strftime("%Y-%m-%d %H:%M:%S"),"total_matches":len(res),"results":res,"top4":[dict(rank=i+1,**t)for i,t in enumerate(t4)]}
    with open("data/predictions.json","w",encoding="utf-8")as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print("Done!%d matches,%d picks"%(out["total_matches"],len(out["top4"])))
if __name__=="__main__":main()
