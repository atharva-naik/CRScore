# analyze errors like overestimation and underestimation compared to human annotations.
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from src.datautils import read_jsonl
from scipy.stats import kendalltau, spearmanr
from src.metrics.claim_based.relevance_score import load_code_claims_and_issues, split_claims, split_claims_and_impl

# skip certain systems to compute what-if correlations if they weren't a part of the annotations.
SKIP_SYSTEMS = []#["llama3", "codellama_13b"]
SYSTEM_TO_BASELINE_KEY = {v: k for k,v in {
    "BM-25 kNN": "knn",
    "LSTM": "lstm",
    "CodeReviewer": "codereviewer",
    "CodeLLaMA-Instruct-7B": "codellama_7b", # not present in the sheet.
    "CodeLLaMA-Instruct-13B": "codellama_13b",
    "Magicoder-S-DS-6.7B": "magicoder",
    "Stable-Code-Instruct-3B": "stable_code",
    "DeepSeekCoder-Instruct-6.7B": "deepseekcoder",
    "GPT-3.5-Turbo": "gpt3.5",
    "Llama-3-8B-Instruct": "llama3"
}.items()}

def plot_multiple_models(model_data, save_path: str):
    """
    Plot multiple line plots on a single figure.
    
    Parameters:
    - model_data: List of tuples, where each tuple contains (x_model, y_model, model_name).
      x_model and y_model are lists/arrays of x and y data points, respectively.
      model_name is the name of the model.
    """
    
    # # Create a new figure and axis
    plt.figure(figsize=(8, 6))
    plt.clf()
    
    # Plot each model's data
    for metric_name, strata_values in model_data.items():
        x = list(strata_values.keys())
        y = list(strata_values.values())
        plt.plot(x, y, label=metric_name)
    
    # Add labels and a legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Monotonicity of metric values')
    # plt.legend(loc='best')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(range(1, 5+1), labels=[str(i) for i in range(1, 5+1)])
    plt.tight_layout()

    # # Show the plot
    # plt.grid(True)
    plt.ylabel("Metric values")
    plt.xlabel("Likert scale values")
    plt.savefig(save_path)

def load_metric_scores_with_thresh(matrices_folder: str, thresh: float):
    import torch
    from src.metrics.claim_based.relevance_score import compute_scores_from_sts_sim_matrix
    scores = {}
    specific_thresh_metric_data = json.load(open("all_model_rel_scores_thresh_0.7311.json"))
    all_review_lengths = []
    for system in specific_thresh_metric_data:
        all_review_lengths.extend(specific_thresh_metric_data[system]["RL"])
    print(f"\x1b[34;1maverage review length = {np.mean(all_review_lengths):.2f}\x1b[0m")
    for model_matfile in tqdm(os.listdir(matrices_folder)):
        # extract the name of the model from the file.
        model_name = model_matfile.removesuffix("_sts_matrix.npz") 
        if model_name in SKIP_SYSTEMS: continue
        scores[model_name] = {} 
        sts_matrices = np.load(os.path.join(matrices_folder, model_matfile))
        scores[model_name]["P"] = []
        scores[model_name]["R"] = []
        scores[model_name]["F"] = []
        scores[model_name]["RL"] = []
        scores[model_name]["scores"] = [] 
        # compute scores using the threshold from the STS matrices corresponding to each instance.
        index = 0
        for sts_mat in tqdm(sts_matrices.values(), disable=True):
            # convert to torch tensor.
            scores[model_name]["scores"].append(sts_mat)
            sts_mat = torch.as_tensor(sts_mat) 
            p_score, r_score, _ = compute_scores_from_sts_sim_matrix(sts_mat, thresh)
            f_score = (2*p_score*r_score)/(p_score+r_score) if (p_score + r_score) != 0 else 0
            scores[model_name]["P"].append(p_score)
            scores[model_name]["R"].append(r_score)
            scores[model_name]["F"].append(f_score)
            scores[model_name]["RL"].append(specific_thresh_metric_data[model_name]["RL"][index])
            index += 1

    return scores

def load_metric_scores_with_two_thresh(matrices_folder: str, thresh1: float, thresh2: Union[float, None]=None):
    import torch
    from src.metrics.claim_based.relevance_score import compute_scores_from_sts_sim_matrix, \
        compute_scores_from_sts_sim_matrix_two_thresh
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
        scores[model_name]["scores"] = [] 
        # compute scores using the threshold from the STS matrices corresponding to each instance.
        for sts_mat in tqdm(sts_matrices.values(), disable=True):
            # convert to torch tensor.
            scores[model_name]["scores"].append(sts_mat)
            sts_mat = torch.as_tensor(sts_mat) 
            if thresh2 is None: p_score, r_score, _ = compute_scores_from_sts_sim_matrix(sts_mat, thresh1)
            else: p_score, r_score, _ = compute_scores_from_sts_sim_matrix_two_thresh(sts_mat, thresh1, thresh2)
            f_score = (2*p_score*r_score)/(p_score+r_score) if (p_score + r_score) != 0 else 0
            scores[model_name]["P"].append(p_score)
            scores[model_name]["R"].append(r_score)
            scores[model_name]["F"].append(f_score)

    return scores

# main
if __name__ == "__main__":
    # scores for our metrics.
    # our_metric_scores = json.load(open("all_model_rel_scores_thresh_0.7.json"))
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    code_claims_path = "./experiments/code_change_summ_finetune_impl/Magicoder-S-DS-6.7B.jsonl"
    all_code_change_summ = load_code_claims_and_issues(
        data=data,
        claims_path=code_claims_path,
        issues_paths={
            "python": "./experiments/python_code_smells",
            "java": "./experiments/java_code_smells",
            "javascript": "./experiments/javascript_code_smells",
        },
        patch_ranges_path="./data/Comment_Generation/test_set_codepatch_ranges.json",
        split_function=split_claims_and_impl if "_impl" in code_claims_path else split_claims,
    ) 
    code_change_summ = [all_code_change_summ[i['patch']] for i in data]

    thresh1 = float(sys.argv[1])
    try: # pass this if you want to use two thresholds.
        thresh2 = float(sys.argv[2])
        our_metric_scores = load_metric_scores_with_two_thresh(
            "./sts_matrices", thresh1=thresh1, thresh2=thresh2,
        )
        print(f"using 2 thresholds: thresh1 = {thresh1:.4f} thresh2 = {thresh2:.4f}")
    except IndexError as e: 
        our_metric_scores = load_metric_scores_with_thresh("./sts_matrices", thresh=thresh1)
        print(f"using threshold = {thresh1:.4f}")
    baseline_metric_scores = json.load(open("./baseline_metric_scores.json"))
    
    # metrics that need to be normalized to 0 to 1 scale.
    for metric in ["BLEU", "BLEU_WITHOUT_STOP"]:
        for i in range(len(baseline_metric_scores[metric])):
            for k in baseline_metric_scores[metric][i]:
                baseline_metric_scores[metric][i][k] = baseline_metric_scores[metric][i][k]/100
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

    # store review length values.
    review_length_values = {}

    # baseline metric scores where the key is lang::index::system.
    baseline_metric_values = {baseline_metric: {} for baseline_metric in baseline_metric_scores} # these are compared with the relevance scores.

    # system rankings according to each metric.
    system_ranks_per_metric = {system: defaultdict(lambda: []) for system in ["Rel", "Human"]+list(baseline_metric_scores.keys())}

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
            if system == "msg": continue # skip CodeReviewer ground truth/references among the evaluated systems, because we don't count it for the correlations as reference based metrics would default to 1 on them and disadvantage their correlation values.

            # skip certain systems for "what-if" correlation computations.
            if system in SKIP_SYSTEMS: continue

            # skip if annotations for any of the dimensions are missing:
            if str(rec["Con (P)"]) == "nan": continue
            if str(rec["Comp (R)"]) == "nan": continue
            if str(rec["Rel (F)"]) == "nan": continue

            key = f"{lang}::{index}::{system}"
            human_con_annot[key] = rec["Con (P)"] # 0.25*(rec["Con (P)"]-1)
            human_comp_annot[key] = rec["Comp (R)"] # 0.25*(rec["Comp (R)"]-1)
            human_rel_annot[key] = rec["Rel (F)"] # 0.25*(rec["Rel (F)"]-1)

            metric_con_values[key] = our_metric_scores[system]["P"][index]
            metric_comp_values[key] = our_metric_scores[system]["R"][index]
            metric_rel_values[key] = our_metric_scores[system]["F"][index]
            review_length_values[key] = our_metric_scores[system]["RL"][index]

            # aggregate metric/annotation values per system.
            system_ranks_per_metric["Rel"][system].append(our_metric_scores[system]["F"][index])
            system_ranks_per_metric["Human"][system].append(rec["Rel (F)"])
            for baseline_metric in baseline_metric_scores:
                system_ranks_per_metric[baseline_metric][system].append(baseline_metric_scores[baseline_metric][index][SYSTEM_TO_BASELINE_KEY[system]])

            for baseline_metric in baseline_metric_scores:
                baseline_metric_values[baseline_metric][key] = \
                    baseline_metric_scores[baseline_metric][index][SYSTEM_TO_BASELINE_KEY[system]]
    
    metric_strata = {metric: {i: [] for i in range(1, 5+1)} for metric in ["Rel (ours)"]+list(baseline_metric_scores.keys())}
    metric_rel_mean = np.mean(list(metric_rel_values.values()))
    metric_rel_std = np.std(list(metric_rel_values.values()))
    metric_rel_min = np.min(list(metric_rel_values.values()))
    # print(metric_strata)
    for metric in system_ranks_per_metric:
        for system in system_ranks_per_metric[metric]:
            system_ranks_per_metric[metric][system] = np.mean(system_ranks_per_metric[metric][system])
        system_ranks_per_metric[metric] = [system for system, score in sorted(system_ranks_per_metric[metric].items(), reverse=True, key=lambda x: x[1])]
        # {system: score for system, score in sorted(system_ranks_per_metric[metric].items(), reverse=True, key=lambda x: x[1])}
        # print(metric+":", system_ranks_per_metric[metric])
    map_model_to_rank = {}
    for i, system in enumerate(system_ranks_per_metric["Human"]):
        map_model_to_rank[system] = i+1

    for metric in system_ranks_per_metric:
        system_ranks_per_metric[metric] = [map_model_to_rank[system] for system in system_ranks_per_metric[metric]]
    
    # system ranking vs human annotations - spearman rho correlations 
    for metric in system_ranks_per_metric:
        if metric == "Human": continue
        result = spearmanr(system_ranks_per_metric[metric], system_ranks_per_metric["Human"])
        val = result.statistic
        pval = round(result.pvalue, 3)
        if result.pvalue <= 0.05:
            print(metric+":", f"\x1b[32;1m{val:.4f}\x1b[0m", pval)
        else: print(metric+":", f"\x1b[31;1m{val:.4f}\x1b[0m", pval)
    
    # system ranking vs human annotations - kendall tau correlations 
    for metric in system_ranks_per_metric:
        if metric == "Human": continue
        result = kendalltau(system_ranks_per_metric[metric], system_ranks_per_metric["Human"])
        val = result.statistic
        pval = round(result.pvalue, 3)
        if result.pvalue <= 0.05:
            print(metric+":", f"\x1b[32;1m{val:.4f}\x1b[0m", pval)
        else: print(metric+":", f"\x1b[31;1m{val:.4f}\x1b[0m", pval)

    # print(metric_rel_mean - metric_rel_std, metric_rel_std, metric_rel_mean, metric_rel_min)
    sns.set()
    sns_plot = sns.displot(list(metric_rel_values.values()), binwidth=0.1)
    fig = sns_plot.figure
    fig.savefig("plots/rel_scores_dist.png")
    # q1, q2, q3, q4, q5 = np.percentile(list(metric_rel_values.values()), [16.67, 33.34, 50, 66.67, 83.34])
    q1, q2, q3, q4 = np.percentile(list(metric_rel_values.values()), [20, 40, 60, 80])
    # print(f"q1: {q1:.4f} q2: {q2:.4f} q3: {q3:.4f} q4: {q4:.4f} q5: {q5:.4f}")
    print(f"q1: {q1:.4f} q2: {q2:.4f} q3: {q3:.4f} q4: {q4:.4f}")
    
    # underestimate/overestimate errors.
    error_cases = {
        "underestimates": {
            "5 rated as 1/2": [],
            "5 rated as 3": [],
            "5 rated as 4": [],
        },
        "overestimates": {
            "1/2 rated as 5": [],
            "1/2 rated as 4": [],
            "1/2 rated as 3": [],
        }
    }
    
    # aggregate over and underestimate errors.
    raw_data = json.load(open("human_study/phase1/raw_data.json"))
    raw_data_by_index = {rec['index']: rec for rec in raw_data}
    print(len(metric_rel_values))
    
    X, Y, Xerr, Yerr = [], [], [], []
    under_review_lens = []
    over_review_lens = []
    under_review_sents = []
    over_review_sents = []
    all_review_sents = []
    all_review_lens = []
    under_claims = []
    over_claims = []
    all_claims = []
    under_has_code = []
    over_has_code = []
    all_has_code = []
    for key in human_rel_annot:
        index = int(key.split("::")[1])
        system = key.split("::")[2]
        lang = key.split("::")[0]
        human_rel = human_rel_annot[key]
        claims = code_change_summ[index]
        scores = our_metric_scores[system]["scores"][index].tolist()
        metric_scores = {}
        has_code = int("`" in raw_data_by_index[index][system+"_pred"])
        review_length = review_length_values[key]
        num_review_sentences = len(split_claims(raw_data_by_index[index][system+"_pred"]))
        for metric_ in ["P", "R", "F"]:
            metric_scores[metric_] = our_metric_scores[system][metric_][index]
        all_review_sents.append(num_review_sentences)
        all_review_lens.append(review_length)
        all_claims.append(len(claims))
        all_has_code.append(has_code)
        if human_rel_annot[key] == 5 and metric_rel_values[key] <= q1: # q1 = q2 = q3 = 0.
            error_cases["underestimates"]["5 rated as 1/2"].append({
                "code change": data[index]["patch"],
                "pred review": raw_data_by_index[index][system+"_pred"],
                "metric_scores": metric_scores,
                "claims": claims,
                "scores": scores,
                "human": human_rel,
                "has_code": has_code,
                "lang": lang,
                "system": system,
            })
            under_review_lens.append(review_length)
            under_review_sents.append(num_review_sentences)
            under_claims.append(len(claims))
            under_has_code.append(has_code)
            Xerr.append(metric_scores["R"])
            Yerr.append(human_rel)
        # elif human_rel_annot[key] == 5 and metric_rel_values[key] <= q4: # q4 = 0.4444 (rated as 3)
        #     error_cases["underestimates"]["5 rated as 3"].append(data[index])
        # elif metric_rel_values[key] <= q5: # q5 = 0.6667 (rated as 4)
        #     error_cases["underestimates"]["5 rated as 4"].append(data[index])
        elif human_rel_annot[key] == 1 and metric_rel_values[key] > q4:
            error_cases["overestimates"]["1/2 rated as 5"].append({
                "code change": data[index]["patch"],
                "pred review": raw_data_by_index[index][system+"_pred"],
                "metric_scores": metric_scores,
                "claims": claims,
                "scores": scores,
                "human": human_rel,
                "has_code": has_code,
                "lang": lang,
                "system": system,
            })
            over_review_lens.append(review_length)
            over_review_sents.append(num_review_sentences)
            over_claims.append(len(claims))
            over_has_code.append(has_code)
            Xerr.append(metric_scores["R"])
            Yerr.append(human_rel)
        # elif human_rel_annot[key] == 1 and metric_rel_values[key] > q4:
        #     error_cases["overestimates"]["1/2 rated as 4"].append(data[index])
        # elif metric_rel_values[key] > q3:
        #     error_cases["overestimates"]["1/2 rated as 3"].append(data[index])
        else:
            X.append(metric_scores["R"])
            Y.append(human_rel)
    
    print(f"mean review length: {np.mean(all_review_lens):.2f} underestimated review length: {np.mean(under_review_lens):.2f} overestimated review length: {np.mean(over_review_lens):.2f}")
    print(f"mean review sents: {np.mean(all_review_sents):.2f} underestimated review sents: {np.mean(under_review_sents):.2f} overestimated review sents: {np.mean(over_review_sents):.2f}")
    print(f"mean claims: {np.mean(all_claims):.2f} underestimated claims: {np.mean(under_claims):.2f} overestimated claims: {np.mean(over_claims):.2f}")
    print(f"mean has_code: {np.mean(all_has_code):.2f} underestimated has_code: {np.mean(under_has_code):.2f} overestimated has_code: {np.mean(over_has_code):.2f}")

    print(f"correlation without error cases:\n")
    result = kendalltau(X, Y)
    result2 = spearmanr(X, Y)
    print("tau", "\x1b[32;1m" if result.pvalue < 0.05 else "\x1b[31;1m", round(result.statistic, 4), "\x1b[0m", "rho", "\x1b[32;1m" if result2.pvalue < 0.05 else "\x1b[31;1m", round(result2.statistic, 4), "\x1b[0m")

    print(f"correlation for error cases:\n")
    result = kendalltau(Xerr, Yerr)
    result2 = spearmanr(Xerr, Yerr)
    print("tau", "\x1b[32;1m" if result.pvalue < 0.05 else "\x1b[31;1m", round(result.statistic, 4), "\x1b[0m", "rho", "\x1b[32;1m" if result2.pvalue < 0.05 else "\x1b[31;1m", round(result2.statistic, 4), "\x1b[0m")

    print("underestimates:", len(error_cases["underestimates"]["5 rated as 1/2"]))
    print("overestimates:", len(error_cases["overestimates"]["1/2 rated as 5"]))
    
    # save the error cases as JSON.
    with open("./error_cases_our_rel_score.json", "w") as f:
        json.dump(error_cases, f, indent=4)

    for key in human_rel_annot:
        stratum = human_rel_annot[key]
        metric_strata["Rel (ours)"][stratum].append(metric_rel_values[key])
        for metric in baseline_metric_values:
            metric_strata[metric][stratum].append(baseline_metric_values[metric][key])
    for metric in metric_strata:
        for stratum in metric_strata[metric]:
            metric_strata[metric][stratum] = np.mean(metric_strata[metric][stratum])
    # print(metric_strata)
    plot_multiple_models(metric_strata, "./plots/metric_spread_over_review_quality.png")

    human_annot = [human_con_annot, human_comp_annot, human_rel_annot]
    metric_values = [metric_con_values, metric_comp_values, metric_rel_values]    
    # diff_thresh = 0.1
    for dim, Y_d, X_d in zip(dimensions, human_annot, metric_values):
    #     total, overestimate, underestimate = 0, 0, 0
    #     for key in X_d:
    #         total += 1
    #         if X_d[key] > Y_d[key] + diff_thresh: # the metric overestimates
    #             overestimate += 1                
    #         elif X_d[key] + diff_thresh < Y_d[key]: # metric underestimates
    #             underestimate += 1
    #     print(f"{dim}: under: {(100*underestimate/total):.2f}% over: {(100*overestimate/total):.2f}%")
    #     if dim == "Rel (F)":
    #         for metric, Z_d in baseline_metric_values.items():
    #             total, overestimate, underestimate = 0, 0, 0
    #             for key in X_d:
    #                 total += 1
    #                 if X_d[key] > Z_d[key] + diff_thresh: # the metric overestimates
    #                     overestimate += 1                
    #                 elif X_d[key] + diff_thresh < Z_d[key]: # metric underestimates
    #                     underestimate += 1
    #             print(f"{dim}: {metric}: under: {(100*underestimate/total):.2f}% over: {(100*overestimate/total):.2f}%")
        X = list(X_d.values())
        Y = list(Y_d.values())
        result = kendalltau(X, Y)
        result2 = spearmanr(X, Y)
        print(dim, "tau", "\x1b[32;1m" if result.pvalue < 0.05 else "\x1b[31;1m", round(result.statistic, 4), "\x1b[0m", "rho", "\x1b[32;1m" if result2.pvalue < 0.05 else "\x1b[31;1m", round(result2.statistic, 4), "\x1b[0m")