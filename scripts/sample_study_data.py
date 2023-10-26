import os
import json
import random
import pandas as pd
from tqdm import tqdm
from src.datautils import read_jsonl

def load_infer_data(exp_path):
    preds = []
    preds_path = os.path.join(exp_path, "preds.txt")
    with open(preds_path, "r") as f:
        for line in tqdm(f):
            line = line.strip()
            preds.append(line)

    return preds

# main 
if __name__ == "__main__":
    random.seed(2023)
    gold_data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    rand_data = load_infer_data("./ckpts/gen_study_rand/checkpoints-1800-5.39")
    inf_data = load_infer_data("./ckpts/gen_study_inf/checkpoints-1800-5.72")
    rel_data = load_infer_data("./ckpts/gen_study_rel/checkpoints-1800-5.62")
    sys_pool = [rand_data, inf_data, rel_data]
    sys_names = ["rand", "inf", "rel"]
    sys_key = []
    all_data = []
    for i in range(len(gold_data)):
        sampled_sys_ids = random.sample(range(3), k=3)
        sys_key.append({
            "sys1": sys_names[sampled_sys_ids[0]], 
            "sys2": sys_names[sampled_sys_ids[1]], 
            "sys3": sys_names[sampled_sys_ids[2]]
        })
        all_data.append({
            "id": i,
            "code_diff": gold_data[i]["patch"],
            "gold_review": gold_data[i]["msg"],
            "sys1 review": sys_pool[sampled_sys_ids[0]][i],
            "sys2 review": sys_pool[sampled_sys_ids[1]][i],
            "sys3 review": sys_pool[sampled_sys_ids[2]][i],
            "choice": ""
        })
    idx = random.sample(range(len(all_data)), k=200)
    sampled_data, sampled_sys_keys = [], []
    for i in idx:
        sampled_data.append(all_data[i])
        sampled_sys_keys.append(sys_key[i])
    df = pd.DataFrame(sampled_data)
    # print(df.columns)
    df.to_csv("./manual_comparison_data_filt_study.csv", index=False)
    with open("./manual_comparison_data_filt_sys_key.json", "w") as f:
        json.dump(sampled_sys_keys, f, indent=4)