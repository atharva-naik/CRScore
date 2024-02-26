import os
import json
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from src.datautils import read_jsonl
from sentence_transformers import SentenceTransformer, util

class RelevanceScorer:
    def __init__(self, model_path):
        self.sbert = SentenceTransformer(model_path)

    def compute(self, change_summs: List[str], reviews: List[str]):
        scores =  []
        for change_summ, review in tqdm(zip(change_summs, reviews), total=len(reviews)):
            scores.append(self.compute_inst(change_summ, review))
        
        return scores, np.mean(scores)

    def compute_inst(self, change_summ: str, review: str):
        change_claims = []
        for line in change_summ.split("\n"):
            if line.strip() != "":
                for chunk in line.split(". "):
                    change_claims.append(chunk)
        # add blank claim if none are found
        if len(change_claims) == 0: change_claims.append("") 
        review_claims = []
        for line in review.split("\n"):
            if line.strip() != "":
                for chunk in line.split(". "):
                    review_claims.append(chunk)
        # add blank claim if none are found
        if len(review_claims) == 0: review_claims.append("")
        change_enc = self.sbert.encode(change_claims, show_progress_bar=False)
        review_enc = self.sbert.encode(review_claims, show_progress_bar=False)

        return (util.cos_sim(change_enc, review_enc).max(axis=0).values>0.3).float().mean().item()

# main
if __name__ == "__main__":
    # codereviewer model generated preds.
    # modelgen_code_reviews = [rec['pred'] for rec in read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")]
    model_preds = pd.read_csv("human_study_data.csv")
    all_code_change_summ = {rec['code_change']: rec["change_summary"] for rec in read_jsonl("./experiments/code_change_summ_v2/Magicoder-S-DS-6.7B.jsonl")}
    code_change_summ = [all_code_change_summ[patch] for patch in model_preds["patch"]]
    rel_scorer = RelevanceScorer("all-roberta-large-v1")
    rel_score_human_data = [{'index': index} for index in model_preds['index']]
    for model in ["codereviewer", "magicoder", "lstm", "knn", "ground_truth"]:
        #read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")
        if model == "ground_truth": reviews = model_preds['msg'].tolist()
        else: reviews = model_preds[f"{model}_pred"].tolist()
        inst_rel_scores, rel_score = rel_scorer.compute(code_change_summ, reviews)
        print(model, rel_score)
        for i,val in enumerate(inst_rel_scores):
            rel_score_human_data[i][model] = val
    with open("./human_study_info_scores.json", "w") as f:
        json.dump(rel_score_human_data, f, indent=4)