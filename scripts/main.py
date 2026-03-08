import json,os,sys
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fetch_data import collect_all,get_today
from predict import run_predictions

def main():
    date=get_today()
    now=datetime.now(ZoneInfo("Asia/Shanghai"))
    session="morning" if now.hour<15 else "evening"
    print("=== Football AI Pro | %s | %s ==="%(date,"AM" if session=="morning" else "PM"))

    # Verify yesterday
    try:
        from verify import run_verify
        print("\n[Step 1] Verify yesterday...")
        run_verify()
    except Exception as e:
        print("Verify skip: %s"%e)

    # Fetch
    print("\n[Step 2] Fetch data...")
    raw=collect_all(date)

    # Save raw (in scripts/data for workflow copy)
    os.makedirs("data",exist_ok=True)
    # Also save to parent data/ directly
    parent_data=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data")
    os.makedirs(parent_data,exist_ok=True)

    with open("data/raw_data.json","w",encoding="utf-8") as f:
        json.dump(raw,f,ensure_ascii=False,indent=2)

    if not raw["matches"]:
        print("No matches today")
        output={"date":date,"session":session,"update_time":now.strftime("%Y-%m-%d %H:%M:%S"),"total_matches":0,"results":[],"top4":[],"history":{}}
    else:
        print("\n[Step 3] Predict %d matches..."%len(raw["matches"]))
        results,top4=run_predictions(raw)
        history={}
        try:
            hf=os.path.join(parent_data,"history.json")
            if os.path.exists(hf):
                with open(hf,"r",encoding="utf-8") as f:
                    h=json.load(f)
                    history=h.get("cumulative",{})
                    history["recent_days"]=h.get("days",[])[-7:]
            elif os.path.exists("data/history.json"):
                with open("data/history.json","r",encoding="utf-8") as f:
                    h=json.load(f)
                    history=h.get("cumulative",{})
                    history["recent_days"]=h.get("days",[])[-7:]
        except:pass
        output={"date":date,"session":session,"update_time":now.strftime("%Y-%m-%d %H:%M:%S"),
            "total_matches":len(results),"results":results,
            "top4":[dict(rank=i+1,**t) for i,t in enumerate(top4)],
            "history":history}

    # Save to BOTH locations
    for d in ["data",parent_data]:
        with open(os.path.join(d,"predictions.json"),"w",encoding="utf-8") as f:
            json.dump(output,f,ensure_ascii=False,indent=2)
        with open(os.path.join(d,"predictions_%s.json"%date),"w",encoding="utf-8") as f:
            json.dump(output,f,ensure_ascii=False,indent=2)

    print("\n=== Done! %d matches, %d picks ==="%(output["total_matches"],len(output.get("top4",[]))))

if __name__=="__main__":
    main()