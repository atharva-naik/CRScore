import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr

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

# codereviewer
# magicoder
# lstm
# knn
# ground_truth
# deepseekcoder
# stable_code
# gpt_3.5
# llama3

def load_our_metric_files(path: str):
    data = json.load(open(path))
    models = list(data.keys())
    N = len(data[models[0]]["P"])
    P, R, F = [{} for _ in range(N)], [{} for _ in range(N)], [{} for _ in range(N)]
    for model in models:
        for i, score in enumerate(data[model]["P"]):
            P[i][model+"_pred"] = score
        for i, score in enumerate(data[model]["R"]):
            R[i][model+"_pred"] = score
        for i, score in enumerate(data[model]["F"]):
            F[i][model+"_pred"] = score

    return {"Con (P)": P, "Comp (R)": R, "Rel (F)": F}

def load_metric_scores_with_thresh(matrices_folder: str, thresh: float):
    import torch
    from src.metrics.claim_based.relevance_score import compute_scores_from_sts_sim_matrix
    N, P, R, F = -1, [], [], []
    for model_matfile in tqdm(os.listdir(matrices_folder)):
        # extract the name of the model from the file.
        model_name = model_matfile.removesuffix("_sts_matrix.npz") 
        sts_matrices = np.load(os.path.join(matrices_folder, model_matfile))
        if N == -1: 
            N = len(sts_matrices)
            P, R, F = [{} for _ in range(N)], [{} for _ in range(N)], [{} for _ in range(N)]
        # compute scores using the threshold from the STS matrices corresponding to each instance.
        for index, sts_mat in tqdm(enumerate(sts_matrices.values()), disable=True):
            # convert to torch tensor.
            sts_mat = torch.as_tensor(sts_mat) 
            p_score, r_score, _ = compute_scores_from_sts_sim_matrix(sts_mat, thresh)
            f_score = (2*p_score*r_score)/(p_score+r_score) if (p_score + r_score) != 0 else 0
            P[index][model_name+"_pred"] = p_score
            R[index][model_name+"_pred"] = r_score
            F[index][model_name+"_pred"] = f_score

    return {"Con (P)": P, "Comp (R)": R, "Rel (F)": F}


def hm(x, y):
    if (x+y) == 0: return 0
    return 2*x*y/(x+y)

# main
if __name__ == "__main__":
    # ccr_metric_scores = 
    # our_metric_scores = load_our_metric_files("all_model_rel_scores_thresh_0.7.json")
    threshold = float(sys.argv[1])
    our_metric_scores = load_metric_scores_with_thresh("./sts_matrices", thresh=threshold)
    baseline_metric_scores = json.load(open("./baseline_metric_scores.json"))
    human_annot_rel_scores = {}
    marcus_annot_end_points = {"py": 501-2, "java": 501-2, "js": 505-2}
    index_to_lang = {}
    langs = list(marcus_annot_end_points.keys())
    dimensions = ["Con (P)", "Comp (R)", "Rel (F)"]

    # human annotated score distribution per dimension of code review systems evalauted (include the CodeReviewer dataset references)
    tested_systems_annot_scores = defaultdict(lambda: {dim: [] for dim in dimensions}) 
    tested_systems_our_metric_scores = defaultdict(lambda: {dim: [] for dim in dimensions}) 
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
            if str(rec["Rel (F)"]) != "nan" and system != "msg": # skip CodeReviewer ground truth/references among the evaluated systems, because we don't count it for the correlations as reference based metrics would default to 1 on them and disadvantage their correlation values.
                human_annot_rel_scores[lang][f"{index}::"+system] = rec["Rel (F)"]
            # score_aggregator = codereviewer_data_ref_scores if system == "msg" else tested_systems_annot_scores
            
            # collect dimension wise annotations and our metric scores for each system. (Table 2)
            for dim in dimensions:
                if str(rec[dim]) != "nan": 
                    tested_systems_annot_scores[system][dim].append(rec[dim])
                    if system != 'msg': tested_systems_our_metric_scores[system][dim].append(our_metric_scores[dim][index][system])
                    else: tested_systems_our_metric_scores[system][dim].append(our_metric_scores[dim][index]["ground_truth_pred"])

    print(f'system evaluated    \tCon (P)\tComp (R)\tRel (F)\tCon (P)\tComp (R)\tRel (F)')
    for system in ["knn_pred", "lstm_pred", "codereviewer_pred", "stable_code_pred", "codellama_13b_pred", "deepseekcoder_pred", "magicoder_pred", "llama3_pred", "gpt3.5_pred", "msg"]:
        scores = tested_systems_annot_scores[system]
        scores2 = tested_systems_our_metric_scores[system]
        # if system == "msg":
        #     print(f'{system.ljust(20-len(system), " ")}\t{(0.25*(np.mean(scores["Con (P)"])-1)):.4f}\t{(0.25*(np.mean(scores["Comp (R)"])-1)):.4f}\t{(0.25*(np.mean(scores["Rel (F)"])-1)):.4f}')
        # else:
        print(f'{system.ljust(20-len(system), " ")}\t{(0.25*(np.mean(scores["Con (P)"])-1)):.4f}\t{(0.25*(np.mean(scores["Comp (R)"])-1)):.4f}\t{(0.25*(np.mean(scores["Rel (F)"])-1)):.4f}\t{(np.mean(scores2["Con (P)"])):.4f}\t{np.mean(scores2["Comp (R)"]):.4f}\t{(np.mean(scores2["Rel (F)"])):.4f}')

    # Table 3 extension Appendix
    # compute correlations with baseline metrics: 
    all_metric_scores = {}
    all_metric_scores.update(baseline_metric_scores)
    all_metric_scores.update({"Rel (F)": our_metric_scores["Rel (F)"]})
    for metric, system_wise_scores in all_metric_scores.items():
        print(metric, end=" ")
        for lang in langs:
            index_and_system_to_metric_val = {}
            for index, rec in enumerate(system_wise_scores):
                for system, score in rec.items():
                    system = map_system_name_to_annot_sheet_name.get(system, system)
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

    # Table 3
    print()
    for metric, system_wise_scores in all_metric_scores.items():
        print(metric, end=" ")
        # for lang in langs:
        index_and_system_to_metric_val = {}
        for index, rec in enumerate(system_wise_scores):
            for system, score in rec.items():
                system = map_system_name_to_annot_sheet_name.get(system, system)
                index_and_system_to_metric_val[f"{index}::{system}"] = score
        Y, X = [], []
        for lang in langs:
            Y += [float(x) for x in human_annot_rel_scores[lang].values()]
            X += [float(index_and_system_to_metric_val[index_and_system]) for index_and_system in human_annot_rel_scores[lang]]
        result = kendalltau(X, Y)
        result2 = spearmanr(X, Y)
        print("tau", "\x1b[32;1m" if result.pvalue < 0.05 else "\x1b[31;1m", round(result.statistic, 4), "\x1b[0m", "rho", "\x1b[32;1m" if result2.pvalue < 0.05 else "\x1b[31;1m", round(result2.statistic, 4), "\x1b[0m", end="")
        print()

    tested_systems_annot_scores = dict(tested_systems_annot_scores)
    # print(tested_systems_annot_scores.keys())
    print("CodeReviewer References:")
    for dim in dimensions:
        print(dim+":", round(np.mean(tested_systems_annot_scores["msg"][dim]), 2))

    print("Tested Systems:")
    for dim in dimensions:
        all_vals = []
        for system in tested_systems_annot_scores:
            if system == "msg": continue
            all_vals.extend(tested_systems_annot_scores[system][dim])
        print(dim+":", round(np.mean(all_vals), 2))

    print("GPT-3.5:")
    for dim in dimensions:
        print(dim+":", round(np.mean(tested_systems_annot_scores["gpt3.5_pred"][dim]), 2))
    # print(human_annot_rel_scores)

    # directly compare our metrics and human annotations for relevance.
    index_and_system_to_Rel = {}
    index_and_system_to_Con = {}
    index_and_system_to_Comp = {}
    
    for index, rec in enumerate(our_metric_scores["Rel (F)"]):
        for system, score in rec.items(): 
            index_and_system_to_Rel[f"{index}::{system}"] = score

    for index, rec in enumerate(our_metric_scores["Con (P)"]):
        for system, score in rec.items(): 
            index_and_system_to_Con[f"{index}::{system}"] = score

    for index, rec in enumerate(our_metric_scores["Comp (R)"]):
        for system, score in rec.items(): 
            index_and_system_to_Comp[f"{index}::{system}"] = score

    # P, R, F, humF = [], [], [], []
    # F_system = defaultdict(lambda: [])
    # humF_system = defaultdict(lambda: [])
    # for lang in ["py", "java", "js"]:
    #     P += [index_and_system_to_Con[index_and_system] for index_and_system in human_annot_rel_scores[lang]]
    #     F += [index_and_system_to_Rel[index_and_system] for index_and_system in human_annot_rel_scores[lang]]
    #     R += [index_and_system_to_Comp[index_and_system] for index_and_system in human_annot_rel_scores[lang]]
    #     for index_and_system, value in human_annot_rel_scores[lang].items():
    #         index, system = index_and_system.split("::")
    #         index = index.strip()
    #         system = system.strip()
    #         F_system[system].append(index_and_system_to_Rel[index_and_system])
    #         humF_system[system].append(0.25*(value-1))
    #         # print(value)
    #     humF += [0.25*(humf-1) for humf in human_annot_rel_scores[lang].values()]
    # assert len(humF) == len(F)
    # CTR = 0