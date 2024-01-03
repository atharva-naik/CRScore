# use a kNN search model to find most relevant review from train+val set for any given code change from test set.
import os
import json
import pyserini
from typing import *
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from src.datautils import read_jsonl, remove_patch_header
from CodeBERT.CodeReviewer.code.evaluator.smooth_bleu import bleu_fromstr

# def process_patch(code: str):
#     lines = code.split("\n")
#     lines[0]
data_folder = "./data/Comment_Generation/"
def create_code_data_for_indexing(train_path):
    data = read_jsonl(train_path)
    write_folder = os.path.join(data_folder, "codepatch_data_for_index")
    index_folder = os.path.join(data_folder, "codepatch_index")
    os.makedirs(write_folder, exist_ok=True)
    os.makedirs(index_folder, exist_ok=True)
    write_path = os.path.join(write_folder, "train.jsonl")
    open(write_path, "w")
    i = 1
    for rec in tqdm(data, desc="writing processed codes"):
        contents = remove_patch_header(rec['patch'])
        idx = f"doc{i}"
        with open(write_path, "a") as f:
            f.write(json.dumps({"id": idx, "contents": contents}) +"\n")
        i += 1

def knn_review_gen(queries: List[str], data_path: str, index_path: str):
    data = read_jsonl(data_path)
    searcher = LuceneSearcher(index_path)
    results = []
    for q in tqdm(queries, desc="1NN review gen"):
        hits = searcher.search(q)
        i = int(hits[0].docid.replace("doc",""))
        results.append((data[i]['msg'], hits[0].score))

    return results

# main
if __name__ == "__main__":
    # create_code_data_for_indexing("./data/Comment_Generation/msg-train.jsonl")
    test_data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    queries = [r['patch'] for r in test_data]
    golds = [r['msg'] for r in test_data]
    results = knn_review_gen(
        queries, data_path="./data/Comment_Generation/msg-train.jsonl",
        index_path="./data/Comment_Generation/codepatch_index",
    )
    with open("./experiments/knn_retriever_preds.json", "w") as f: json.dump(results, f, indent=4) 
    preds = [r for r,_ in results]
    bleu_with_stop = bleu_fromstr(predictions=preds, golds=golds, rmstop=False)
    print(f"bleu_with_stop = {bleu_with_stop}")
    bleu_without_stop = bleu_fromstr(predictions=preds, golds=golds, rmstop=True)
    print(f"bleu_without_stop = {bleu_without_stop}")
