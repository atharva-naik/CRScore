# analyze hard cases for retrieval.
import json
import pandas as pd
from collections import defaultdict
from src.datautils import read_jsonl, write_jsonl

# main
if __name__ == "__main__":
    test_indices = json.load(open("./ckpts/crr_rcr_ccr_0.005/test_ids.json"))
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    count = 0
    hard_cases = []
    recall_at_5 = []
    recall_at_5_by_pl = defaultdict(lambda: [])
    recall_at_10_by_pl = defaultdict(lambda: [])
    recall_at_25_by_pl = defaultdict(lambda: [])
    recall_at_100_by_pl = defaultdict(lambda: [])
    recall_at_250_by_pl = defaultdict(lambda: [])
    recall_at_1000_by_pl = defaultdict(lambda: [])
    # retrievable_instances = []
    for true_label, id_list in enumerate(test_indices):
        # check if true instance is recalled within top 500.
        if true_label in id_list[:5]:
            with open("retrievable_instances.jsonl", "a") as f:
                f.write(json.dumps({"msg": data[true_label]['msg']})+"\n")
        if true_label not in id_list[:2500]:
            count += 1
            # print(data[true_label])
            # print(true_label)
            rank_assigned = id_list.index(true_label)
            del data[true_label]['oldf']
            data[true_label]["rank_assigned"] = rank_assigned
            hard_cases.append(data[true_label])
        recall_at_5.append(int(true_label in id_list[:5]))
        recall_at_5_by_pl[data[true_label]['lang']].append(int(true_label in id_list[:5]))
        recall_at_10_by_pl[data[true_label]['lang']].append(int(true_label in id_list[:10]))
        recall_at_25_by_pl[data[true_label]['lang']].append(int(true_label in id_list[:25]))
        recall_at_100_by_pl[data[true_label]['lang']].append(int(true_label in id_list[:100]))
        recall_at_250_by_pl[data[true_label]['lang']].append(int(true_label in id_list[:250]))
        recall_at_1000_by_pl[data[true_label]['lang']].append(int(true_label in id_list[:1000]))
    print(len(hard_cases))
    recall_at_5 = round(100*sum(recall_at_5)/len(recall_at_5),2)
    for pl in recall_at_5_by_pl:
        recall_at_5_by_pl[pl] = round(100*sum(recall_at_5_by_pl[pl])/len(recall_at_5_by_pl[pl]), 2)
    for pl in recall_at_10_by_pl:
        recall_at_10_by_pl[pl] = round(100*sum(recall_at_10_by_pl[pl])/len(recall_at_10_by_pl[pl]), 2)
    for pl in recall_at_25_by_pl:
        recall_at_25_by_pl[pl] = round(100*sum(recall_at_25_by_pl[pl])/len(recall_at_25_by_pl[pl]), 2)
    for pl in recall_at_100_by_pl:
        recall_at_100_by_pl[pl] = round(100*sum(recall_at_100_by_pl[pl])/len(recall_at_100_by_pl[pl]), 2)
    for pl in recall_at_250_by_pl:
        recall_at_250_by_pl[pl] = round(100*sum(recall_at_250_by_pl[pl])/len(recall_at_250_by_pl[pl]), 2)
    for pl in recall_at_1000_by_pl:
        recall_at_1000_by_pl[pl] = round(100*sum(recall_at_1000_by_pl[pl])/len(recall_at_1000_by_pl[pl]), 2)
    print("R@5:", recall_at_5)
    for pl, r_at_k in recall_at_5_by_pl.items():
        print(f"R@5 for {pl}:", r_at_k)
    for pl, r_at_k in recall_at_10_by_pl.items():
        print(f"R@10 for {pl}:", r_at_k)
    for pl, r_at_k in recall_at_25_by_pl.items():
        print(f"R@25 for {pl}:", r_at_k)
    for pl, r_at_k in recall_at_100_by_pl.items():
        print(f"R@100 for {pl}:", r_at_k)
    for pl, r_at_k in recall_at_250_by_pl.items():
        print(f"R@250 for {pl}:", r_at_k)
    for pl, r_at_k in recall_at_1000_by_pl.items():
        print(f"R@1000 for {pl}:", r_at_k)
    # pd.DataFrame(hard_cases).to_csv("./ckpts/crr_rcr_ccr_0.005/hard_cases.csv", index=False)