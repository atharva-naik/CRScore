# script to sample instances from the CodeReview comment generation test set for human annotations of review quality.

import os
import json
import random
import pandas as pd
from collections import defaultdict
from src.datautils import read_jsonl

def process_lstm_output(review: str):
    if review.startswith("<msg>"): review = review[len("<msg>"):].strip()
        
    return review

def process_magicoder_output(review: str):
    review = review.split("@@ Code Change")[0].strip("\n")
    if "The code change is as follows:" in review and "The review is as follows:" in review:
        review = review.split("The code change is as follows:")[0].strip("\n")
    # remove repeated lines:
    review_lines = review.split("\n")
    seen_lines = set()
    dedup_lines = []
    for line in review_lines:
        if line in seen_lines: continue
        seen_lines.add(line)
        dedup_lines.append(line)
    review = "\n".join(dedup_lines)

    return review

# main
if __name__ == "__main__":
    random.seed(42)
    save_path = "human_study_data.csv"
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    codereviewer_preds = [rec['pred'] for rec in read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")]
    #[rec['gen_review'] for rec in json.load(open("./experiments/MS_CR_ZeroShot.json"))]
    magicoder_preds = [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./data/Comment_Generation/llm_outputs/Magicoder-S-DS-6.7B.jsonl')]
    knn_retriever_preds = [r for r,_ in json.load(open("./experiments/knn_retriever_preds.json"))]
    lstm_preds = [r['pred'] for r in read_jsonl("./ckpts/lstm_reviewer_1_layer/preds.jsonl")]
    lang_buckets = defaultdict(lambda: [])
    index = 0
    long_inst_len = 1000
    long_inst_ctr = 0
    for rec in data:
        rec["index"] = index
        rec['codereviewer_pred'] = codereviewer_preds[index]
        rec['magicoder_pred'] = magicoder_preds[index]
        rec['lstm_pred'] = process_lstm_output(lstm_preds[index])
        rec['knn_pred'] = knn_retriever_preds[index]
        if len(rec['patch']) > long_inst_len:
            long_inst_ctr += 1
            continue
        lang_buckets[rec['lang']].append(rec)
        index += 1
    print(f"{(100*long_inst_ctr/len(data)):.2f}% code changes > {long_inst_len} characters")
    # get the distribution of languages
    data_for_annotation = []
    for lang in lang_buckets:
        if lang == "go": k = 12
        else: k = 11
        print(lang, len(lang_buckets[lang]))
        instances = random.sample(lang_buckets[lang], k=k)
        data_for_annotation += instances
    annot_df = pd.DataFrame(data_for_annotation).drop(columns=['oldf', 'idx', 'id', 'y'])
    print(annot_df.columns)
    print(len(annot_df))
    annot_df.to_csv(save_path, index=False)
    pre_shuffled_data = []
    pre_model_names = []
    for rec in annot_df.to_dict("records"):
        pre_shuffled_data.append({"index": rec['index'], 'code_change': rec['patch'], 'review': rec['msg'], 'relevance': None, 'informativeness': None, 'correctness': None})
        pre_model_names.append('ground_truth')
        pre_shuffled_data.append({"index": rec['index'], 'code_change': rec['patch'], 'review': rec['magicoder_pred'], 'relevance': None, 'informativeness': None, 'correctness': None})
        pre_model_names.append('magicoder')
        pre_shuffled_data.append({"index": rec['index'], 'code_change': rec['patch'], 'review': rec['codereviewer_pred'], 'relevance': None, 'informativeness': None, 'correctness': None})
        pre_model_names.append('codereviewer')
        pre_shuffled_data.append({"index": rec['index'], 'code_change': rec['patch'], 'review': rec['lstm_pred'], 'relevance': None, 'informativeness': None, 'correctness': None})
        pre_model_names.append('lstm')
        pre_shuffled_data.append({"index": rec['index'], 'code_change': rec['patch'], 'review': rec['knn_pred'], 'relevance': None, 'informativeness': None, 'correctness': None})
        pre_model_names.append('knn')
    shuffled_index_order = random.sample(range(len(pre_shuffled_data)), k=len(pre_shuffled_data))
    shuffled_data = []
    shuffled_model_names = []
    for i in shuffled_index_order:
        shuffled_data.append(pre_shuffled_data[i])
        shuffled_model_names.append(pre_model_names[i])
    with open("./model_names_key.json", "w") as f:
        json.dump(shuffled_model_names, f, indent=4)
    pd.DataFrame(shuffled_data).to_csv("human_study_shuffled_annot_sheet.csv", index=False)
    print(f"{len(shuffled_data)} rows (3 annotations per row) to be annotated!")
    annotation_effort = 3 * 3 * len(shuffled_data)
    print(f"{annotation_effort} uniques annotations required") 