import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr

# main
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
if __name__ == "__main__":
    # ccr_metric_scores = 
    baseline_metric_scores = json.load(open("./baseline_metric_scores.json"))
    human_annot_rel_scores = {}
    marcus_annot_end_points = {"py": 501-2, "java": 501-2, "js": 505-2}
    index_to_lang = {}
    langs = list(marcus_annot_end_points.keys())
    dimensions = ["Con (P)", "Comp (R)", "Rel (F)"]
    system_scores_human_annot = defaultdict(lambda: {dim: [] for dim in dimensions})
    for lang in langs:
        human_annot_rel_scores[lang] = {}
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
            system = rec['system']
            for dimension in dimensions:
                if str(rec[dimension]) != "nan":
                    system_scores_human_annot[system][dimension].append(rec[dimension])
            if str(rec["Rel (F)"]) != "nan" and system != "msg": # don't include annotations over ground truth reviews within the correlation computation
                human_annot_rel_scores[lang][f"{index}::"+system] = rec["Rel (F)"]
    system_scores_human_annot = dict(system_scores_human_annot)
    for system, scores in system_scores_human_annot.items():
        print(f'{system}: Con (P): {(25*(np.mean(scores["Con (P)"])-1)):.2f} Comp (R): {(25*(np.mean(scores["Comp (R)"])-1)):.2f} Rel (F): {(25*(np.mean(scores["Rel (F)"])-1)):.2f}')
    # compute correlations with baseline metrics:
    
    for metric, system_wise_scores in baseline_metric_scores.items():
        print(metric, end=" ")
        for lang in langs:
            index_and_system_to_metric_val = {}
            for index, rec in enumerate(system_wise_scores):
                for system, score in rec.items():
                    system = map_system_name_to_annot_sheet_name[system]
                    index_and_system_to_metric_val[f"{index}::{system}"] = score
            Y = [float(x) for x in human_annot_rel_scores[lang].values()]
            X = [float(index_and_system_to_metric_val[index_and_system]) for index_and_system in human_annot_rel_scores[lang]]
            # if metric == "CHRF++": 
            #     print(X)
            #     print(Y)
            # print(len(X))
            # print(len(Y))
            result = kendalltau(X, Y)
            result2 = spearmanr(X, Y)
            print("tau", "\x1b[32;1m" if result.pvalue < 0.05 else "\x1b[31;1m", round(result.statistic, 4), "\x1b[0m", "rho", "\x1b[32;1m" if result2.pvalue < 0.05 else "\x1b[31;1m", round(result2.statistic, 4), "\x1b[0m", end="")
        print()
    # print(human_annot_rel_scores)