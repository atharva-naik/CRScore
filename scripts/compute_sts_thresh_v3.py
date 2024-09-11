import os
import copy
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datautils import read_jsonl
from src.metrics.claim_based.relevance_score import compute_scores_from_sts_sim_matrix, load_code_claims_and_issues, split_claims, split_claims_and_impl, RelevanceScorer, process_magicoder_output

# main
if __name__ == '__main__':
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
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
        # if sts_mat.max().item() >= 0.8 and i not in human_annot_review_indices:
        if i not in human_annot_review_indices: indices_to_use.append(i)
        i += 1
    print(len(indices_to_use))

    # load CodeReviewer ground truth references, other system reviews and code change summaries.
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    codereviewer_references = [i['msg'] for i in data]
    magicoder_reviews = [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./data/Comment_Generation/llm_outputs/Magicoder-S-DS-6.7B.jsonl')]
    stable_code_reviews = [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./experiments/llm_outputs/Stable-Code-Instruct-3b.jsonl')]
    codereviewer_pred_reviews = [rec['pred'] for rec in read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")]
    deepseekcoder_reviews = [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./experiments/llm_outputs/DeepSeekCoder-6.7B-Instruct.jsonl')]
    llama3_reviews = [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./experiments/llm_outputs/Llama-3-8B-Instruct.jsonl')]
    gpt35_reviews = [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./experiments/llm_outputs/GPT-3.5-Turbo.jsonl')]

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

    pos_claim_review_scores = []
    neg_claim_review_scores = []
    ctr = 0
    for index in tqdm(indices_to_use):
        review = codereviewer_references[index]
        pos_claims = all_code_change_summ[data[index]['patch']]
        # remaining_indices = indices_to_use[:index] + indices_to_use[index+1:]
        # neg_indices = random.sample(remaining_indices, k=250)
        # neg_claims = []
        # for j in neg_indices:
        #     neg_claims.extend(all_code_change_summ[data[j]['patch']])
        _, _, pos_claims_sts_mat = rel_scorer.compute_inst(change_summ=pos_claims, review=review)
        # _, _, neg_claims_sts_mat = rel_scorer.compute_inst(change_summ=neg_claims, review=review)
        pos_claims_sts_mat = torch.as_tensor(pos_claims_sts_mat)
        # neg_claims_sts_mat = torch.as_tensor(neg_claims_sts_mat)
        # for val in pos_claims_sts_mat.flatten():

        # iterate over the highest score obtained by the review sentence for any claim for the same code change.
        for val in pos_claims_sts_mat.max(dim=0).values:
            pos_claim_review_scores.append(val.item())
        # # iterate over all the scores obtained by the review sentences for claims from random code changes.
        # for val in neg_claims_sts_mat.flatten():
        #     neg_claim_review_scores.append(val.item())
        print("pos mean:", round(np.mean(pos_claim_review_scores), 4))
        print("pos std:", round(np.std(pos_claim_review_scores), 4))
        # print("neg mean:", round(np.mean(neg_claim_review_scores), 4))
        # print("neg std:", round(np.std(neg_claim_review_scores), 4))

    # # To run, execute:
    # python -m scripts.compute_sts_thresh_v3