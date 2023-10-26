# do qualitative analysis by randomly sampling some instances

import random
import pandas as pd

pred_paths = {
    "inf": "./ckpts/gen_inf/checkpoints-1800-5.64/preds.txt",        
    "rand": "./ckpts/gen_rand/checkpoints-3600-5.48/preds.txt",
    "rel": "./ckpts/gen_rel/checkpoints-1800-5.7/preds.txt"
}
gold_path = "ckpts/gen_rel/checkpoints-1800-5.7/golds.txt"

def load_data(path: str):
    return open(path).readlines()

gold_data = load_data(gold_path)
preds = {split: load_data(pred_path) for split, pred_path in pred_paths.items()}
comparison_data = []

for i, gold in enumerate(gold_data):
    rec = {"id": i+1}
    for split, pred_data in preds.items():
        rec[split] = pred_data[i]
    rec["gold"] = gold
    comparison_data.append(rec)

comparison_data = random.sample(comparison_data, k=100)

pd.DataFrame(comparison_data).to_csv("./qual_analysis_rel_inf.csv", index=False)