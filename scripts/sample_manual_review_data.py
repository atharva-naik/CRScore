import random
import pandas as pd
from src.datautils import read_jsonl

# main 
if __name__ == "__main__":
    random.seed(42)
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    preds = read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")
    for item, pred_and_label in zip(data, preds):
        item["pred"] = pred_and_label["pred"]
        del item["oldf"]
    df = pd.DataFrame(data)
    print(df.columns)
    df.to_csv("./manual_gold_labels_and_preds_analysis.csv", index=False)