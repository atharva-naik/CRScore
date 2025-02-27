import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr

# skip certain systems to compute what-if correlations if they weren't a part of the annotations.
SKIP_SYSTEMS = []#["llama3", "codellama_13b"]
collapse_4_and_5 = False
map_system_name_to_annot_sheet_name = {
    "BM-25 kNN": "knn_pred",
    "LSTM": "lstm_pred",
    "CodeReviewer": "codereviewer_pred",
    "CodeLLaMA-Instruct-7B": "codellama_7b_pred", # not present in the sheet.
    "CodeLLaMA-Instruct-13B": "codellama_13b_pred",
    "Magicoder-S-DS-6.7B": "magicoder_pred",
    "Stable-Code-Instruct-3B": "stable_code_pred",
    "DeepSeekCoder-Instruct-6.7B": "deepseekcoder_pred",
    "GPT-3.5-Turbo": "gpt3.5_pred",
    "Llama-3-8B-Instruct": "llama3_pred"
}
map_annot_sheet_name_to_system_name = {
    v: k for k,v in map_system_name_to_annot_sheet_name.items()
}

def load_metric_scores_with_thresh(matrices_folder: str, thresh: float):
    import torch
    from src.metrics.claim_based.relevance_score import compute_scores_from_sts_sim_matrix
    scores = {}
    for model_matfile in tqdm(os.listdir(matrices_folder)):
        # extract the name of the model from the file.
        model_name = model_matfile.removesuffix("_sts_matrix.npz") 
        if model_name in SKIP_SYSTEMS: continue
        scores[model_name] = {} 
        sts_matrices = np.load(os.path.join(matrices_folder, model_matfile))
        scores[model_name]["P"] = []
        scores[model_name]["R"] = []
        scores[model_name]["F"] = []
        # compute scores using the threshold from the STS matrices corresponding to each instance.
        for sts_mat in tqdm(sts_matrices.values(), disable=True):
            # convert to torch tensor.
            sts_mat = torch.as_tensor(sts_mat) 
            p_score, r_score, _ = compute_scores_from_sts_sim_matrix(sts_mat, thresh)
            f_score = (2*p_score*r_score)/(p_score+r_score) if (p_score + r_score) != 0 else 0
            scores[model_name]["P"].append(p_score)
            scores[model_name]["R"].append(r_score)
            scores[model_name]["F"].append(f_score)

    return scores

# main
if __name__ == "__main__":
    # scores for our metrics.
    # our_metric_scores = json.load(open("all_model_rel_scores_thresh_0.7.json"))
    threshold = float(sys.argv[1])
    our_metric_scores = load_metric_scores_with_thresh("./sts_matrices", thresh=threshold)
    # scores for baseline metrics.
    baseline_metric_scores = json.load(open("./baseline_metric_scores.json"))
    human_annot_rel_scores = {}
    # boundary points to switch between annotators while building the list of human annotations.
    marcus_annot_end_points = {"py": 501-2, "java": 501-2, "js": 505-2}
    index_to_lang = {}
    langs = list(marcus_annot_end_points.keys())
    dimensions = ["Con (P)", "Comp (R)", "Rel (F)"]
    
    # human likert score annotations.
    # it's a map where the key is lang::index::system.
    human_con_annot = {}
    human_comp_annot = {}
    human_rel_annot = {}

    # our metric's values where the key is lang::index::system.
    metric_con_values = {}
    metric_comp_values = {}
    metric_rel_values = {}
    count_annots = 0 # count the number of annotations collected.

    baseline_metric_values = {metric: {} for metric in baseline_metric_scores}

    for lang in langs:
        marcus_annot = pd.read_csv(f"human_study/phase2/{lang}_marcus_review_qual_final.csv").to_dict("records")
        atharva_annot = pd.read_csv(f"human_study/phase2/{lang}_atharva_review_qual_final.csv").to_dict("records")
        index = None
        for i, rec in enumerate(marcus_annot):
            # if beyond the boundary of Marcus' annotations then switch to Atharva's annotations.
            if i > marcus_annot_end_points[lang]: 
                rec = atharva_annot[i]
            if str(rec['index']) != "nan":
                index = int(rec["index"])
                index_to_lang[index] = lang
            
            # skip blank system/reviews:
            if isinstance(rec['system'], float): continue

            system = rec['system'].replace("_pred", "")
            key = f"{lang}::{index}::{system}"

            # skip certain systems for "what-if" correlation computations.
            if system in SKIP_SYSTEMS: continue

            # skip if annotations for any of the dimensions are missing:
            if str(rec["Con (P)"]) == "nan": continue
            if str(rec["Comp (R)"]) == "nan": continue
            if str(rec["Rel (F)"]) == "nan": continue

            count_annots += 1
            if system == "msg": continue # skip CodeReviewer ground truth/references among the evaluated systems, because we don't count it for the correlations as reference based metrics would default to 1 on them and disadvantage their correlation values.

            if collapse_4_and_5:
                con_p = rec["Con (P)"] if rec["Con (P)"] != 5 else 4
                comp_r = rec["Comp (R)"] if rec["Comp (R)"] != 5 else 4
                rel_f = rec["Rel (F)"] if rec["Rel (F)"] != 5 else 4
                human_con_annot[key] = 0.25*(con_p-1)
                human_comp_annot[key] = 0.25*(comp_r-1)
                human_rel_annot[key] = 0.25*(rel_f-1)
            else:
                human_con_annot[key] = 0.25*(rec["Con (P)"]-1)
                human_comp_annot[key] = 0.25*(rec["Comp (R)"]-1)
                human_rel_annot[key] = 0.25*(rec["Rel (F)"]-1)

            metric_con_values[key] = our_metric_scores[system]["P"][index]
            metric_comp_values[key] = our_metric_scores[system]["R"][index]
            metric_rel_values[key] = our_metric_scores[system]["F"][index]
            for metric in baseline_metric_scores:
                sys_key = map_annot_sheet_name_to_system_name[system+"_pred"]
                baseline_metric_values[metric][key] = baseline_metric_scores[metric][index][sys_key]

    human_annot = [human_con_annot, human_comp_annot, human_rel_annot]
    metric_values = [metric_con_values, metric_comp_values, metric_rel_values]
    print(f"threshold = {threshold}")
    for dim, Y_d, X_d in zip(dimensions, human_annot, metric_values):
        X = list(X_d.values())
        Y = list(Y_d.values())
        result = kendalltau(X, Y)
        result2 = spearmanr(X, Y)
        print(dim, "tau", "\x1b[32;1m" if result.pvalue < 0.05 else "\x1b[31;1m", round(result.statistic, 4), "\x1b[0m", "rho", "\x1b[32;1m" if result2.pvalue < 0.05 else "\x1b[31;1m", round(result2.statistic, 4), "\x1b[0m")

    print("labeled instances:", count_annots)
    for metric in baseline_metric_scores:
        metric_values = [
            baseline_metric_values[metric],
            baseline_metric_values[metric],
            baseline_metric_values[metric],
        ]
        for dim, Y_d, X_d in zip(dimensions, human_annot, metric_values):
            X = list(X_d.values())
            Y = list(Y_d.values())
            # print(len(X))
            result = kendalltau(X, Y)
            result2 = spearmanr(X, Y)
            print(metric, dim, "tau", "\x1b[32;1m" if result.pvalue < 0.05 else "\x1b[31;1m", round(result.statistic, 4), "\x1b[0m", "rho", "\x1b[32;1m" if result2.pvalue < 0.05 else "\x1b[31;1m", round(result2.statistic, 4), "\x1b[0m")