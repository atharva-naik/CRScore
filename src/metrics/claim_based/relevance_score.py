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
        P = []
        R = []
        for change_summ, review in tqdm(zip(change_summs, reviews), total=len(reviews)):
            p, r = self.compute_inst(change_summ, review)
            P.append(p)
            R.append(r)
        p_score = np.mean(P)
        r_score = np.mean(R)
        f_score = (2*p_score*r_score)/(p_score+r_score)

        return P, R, p_score, r_score, f_score 

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
        sem_similarity_matrix = util.cos_sim(change_enc, review_enc)
        # precision/conciseness
        P = sem_similarity_matrix.max(dim=0).values.mean().item()
        # recall/comprehensiveness
        R = sem_similarity_matrix.max(dim=1).values.mean().item()

        return P, R

def human_study_results():
    model_preds = pd.read_csv("human_study_data.csv")
    all_code_change_summ = {rec['code_change']: rec["change_summary"] for rec in read_jsonl("./experiments/code_change_summ_v2/Magicoder-S-DS-6.7B.jsonl")}
    code_change_summ = [all_code_change_summ[patch] for patch in model_preds["patch"]]
    rel_scorer = RelevanceScorer("all-roberta-large-v1")
    rel_score_human_data = [{'index': index} for index in model_preds['index']]
    for model in ["codereviewer", "magicoder", "lstm", "knn", "ground_truth"]:
        #read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")
        if model == "ground_truth": reviews = model_preds['msg'].tolist()
        else: reviews = model_preds[f"{model}_pred"].tolist()
        mean_review_length = np.mean([len(r.split()) for r in reviews])
        inst_rel_P_scores, inst_rel_R_scores, rel_P_score, rel_R_score, rel_F_score = rel_scorer.compute(code_change_summ, reviews)
        print(model, f"P={100*rel_P_score:.2f} R={100*rel_R_score:.2f} F={100*rel_F_score:.2f} RL={mean_review_length:.2f}")
        for i,val in enumerate(inst_rel_P_scores):
            rel_score_human_data[i][model] = val
    with open("./human_study_relevance_scores.json", "w") as f:
        json.dump(rel_score_human_data, f, indent=4)

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

def all_model_all_data_results():
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    model_preds = {
        "codereviewer": [rec['pred'] for rec in read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")],
        "magicoder": [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./data/Comment_Generation/llm_outputs/Magicoder-S-DS-6.7B.jsonl')],
        "lstm": [r['pred'] for r in read_jsonl("./ckpts/lstm_reviewer_1_layer/preds.jsonl")],
        "knn": [r for r,_ in json.load(open("./experiments/knn_retriever_preds.json"))],
        "ground_truth": [i['msg'] for i in data],
    }
    all_code_change_summ = {rec['code_change']: rec["change_summary"] for rec in read_jsonl("./experiments/code_change_summ_v2/Magicoder-S-DS-6.7B.jsonl")}
    code_change_summ = [all_code_change_summ[i['patch']] for i in data]
    rel_scorer = RelevanceScorer("all-roberta-large-v1")
    # rel_score_human_data = [{'index': index} for index in model_preds['index']]
    rel_scores = {}
    for model in ["codereviewer", "magicoder", "lstm", "knn", "ground_truth"]:
        #read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")
        reviews = model_preds[model]
        review_lengths = [len(r.split()) for r in reviews]
        mean_review_length = np.mean(review_lengths)
        inst_rel_P_scores, inst_rel_R_scores, rel_P_score, rel_R_score, rel_F_score = rel_scorer.compute(code_change_summ, reviews)
        print(model, f"P={100*rel_P_score:.2f} R={100*rel_R_score:.2f} F={100*rel_F_score:.2f} RL={mean_review_length:.2f}")
        inst_rel_F_scores = [(2*p*r)/(p+r) for p,r in zip(inst_rel_P_scores, inst_rel_R_scores)]
        rel_scores[model] = {
            "P": inst_rel_P_scores,
            "R": inst_rel_R_scores,
            "F": inst_rel_F_scores,
            "RL": review_lengths,
        }
    with open("./all_model_rel_scores.json", "w") as f:
        json.dump(rel_scores, f, indent=4)

# main
if __name__ == "__main__":
    # codereviewer model generated preds.
    # modelgen_code_reviews = [rec['pred'] for rec in read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")]
    # human_study_results()
    all_model_all_data_results()
