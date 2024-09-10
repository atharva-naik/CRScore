import os
import copy
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datautils import read_jsonl
from src.metrics.claim_based.relevance_score import compute_scores_from_sts_sim_matrix, load_code_claims_and_issues, split_claims, split_claims_and_impl, RelevanceScorer

# main
if __name__ == '__main__':
    thresholds = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    # json.load(open("all_model_rel_scores_thresh_0.7.json"))["ground_truth"]
    sts_mat_ground_truth = np.load("sts_matrices/ground_truth_sts_matrix.npz")
    # the human annotated reviews are to be exlcuded as held out data.
    human_annot_review_indices = []
    for lang in ["java", 'py', 'js']:
        human_annot_review_indices += pd.read_csv(f"human_study/phase1/{lang}_claim_acc_annot.csv")["index"].tolist()

    i = 0
    indices_to_use = []
    # filter out reviews where not a single sentence has relatively high similarity to at least 1 claim.
    for sts_mat in sts_mat_ground_truth.values():
        if sts_mat.max().item() >= 0.8 and i not in human_annot_review_indices:
            indices_to_use.append(i)
        i += 1
    print(len(indices_to_use))

    # load CodeReviewer references and code change summaries.
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    codereviewer_references = [i['msg'] for i in data]
    code_claims_path = "./experiments/code_change_summ_finetune_impl/Magicoder-S-DS-6.7B.jsonl"
    all_code_change_summ = load_code_claims_and_issues(
        data=data,
        claims_path=code_claims_path,
        issues_paths={
            "python": "./experiments/python_code_smells",
            "java": "./experiments/java_code_smells",
            "javascript": "./experiments/python_code_smells",
        },
        patch_ranges_path="./data/Comment_Generation/test_set_codepatch_ranges.json",
        split_function=split_claims_and_impl if "_impl" in code_claims_path else split_claims,
    )
    rel_scorer = RelevanceScorer(model_path="mixedbread-ai/mxbai-embed-large-v1", hi_sim_thresh=0.65) # threshold use here doesn't really matter

    max_sts_per_ref = []
    for i in indices_to_use:
        ref = codereviewer_references[i]
        claims = all_code_change_summ[data[i]['patch']]
        _, _, sts_mat = rel_scorer.compute_inst(claims, ref)
        max_sts_per_ref.append(sts_mat.max().item())
        # classification_data.append((ref, claims, 1))
    #     j = copy.deepcopy(i)
    #     while j == i:
    #         j = random.sample(indices_to_use, k=1)[0]
    #         break
    #     neg_claims = all_code_change_summ[data[j]['patch']]
    #     classification_data.append((ref, neg_claims, 0))

    # acc_per_thresh = {}
    # for thresh in thresholds:
    #     accs = []
    #     ctr = 0
    #     for ref, claims, label in tqdm(classification_data, desc=f"computing acc. of {thresh}"):
    #         p_score, r_score, sts_mat = rel_scorer.compute_inst(claims, ref)
    #         f_score = (2*p_score*r_score)/(p_score+r_score) if (p_score + r_score) != 0 else 0
    #         pred = int(f_score >= thresh)
    #         accs.append(int(pred == label))
    #         # if ctr % 2500 == 0: 
    #             # print(f"{thresh}:", np.mean(accs))
    #         ctr += 1
    #     print(f"{thresh}:", np.mean(accs))
    #     acc_per_thresh[thresh] = np.mean(accs)
        # with open("sts_threshold_comparison.json", "w") as f:
        #     json.dump(acc_per_thresh, f, indent=4)
    # # To run, execute:
    # python -m scripts.compute_sts_thresh_v2