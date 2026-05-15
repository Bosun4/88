import argparse
import sys
from forward_ledger.ledger import create_ledger_from_prediction
from forward_ledger.scoring import score_ledger_with_actuals

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    cmd_lock = sub.add_parser("lock")
    cmd_lock.add_argument("--pred", required=True)
    cmd_lock.add_argument("--out", default="reports/forward_ledger/ledger_latest.jsonl")
    
    cmd_score = sub.add_parser("score")
    cmd_score.add_argument("--ledger", required=True)
    cmd_score.add_argument("--actual", required=True)
    cmd_score.add_argument("--out_csv", default="reports/forward_ledger/scored_latest.csv")
    cmd_score.add_argument("--out_md", default="reports/forward_ledger/scored_latest.md")
    
    args = parser.parse_args()
    
    if args.cmd == "lock":
        cnt = create_ledger_from_prediction(args.pred, args.out)
        print(f"Locked {cnt} matches into {args.out}")
        
    elif args.cmd == "score":
        scored = score_ledger_with_actuals(args.ledger, args.actual, args.out_csv, args.out_md)
        print(f"Scored {len(scored)} entries. Results in {args.out_csv}")

if __name__ == "__main__":
    main()
