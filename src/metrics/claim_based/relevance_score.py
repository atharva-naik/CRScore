import os
import re
import sys
import nltk
import json
import torch
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
import torch.nn.functional as F
from nltk.corpus import stopwords
from src.datautils import read_jsonl
from sentence_transformers import SentenceTransformer, util
from scripts.create_code_smell_analysis_data import generate_newf, remove_space_clean

nltk.download("english")

DEBUG = False
stop = set(stopwords.words("english")+[".",","])

def split_string_by_multiple_delimiters(s, delimiters):
    pattern = '|'.join(map(re.escape, delimiters))
    return re.split(pattern, s)

def split_claims(text):
    text_claims = []
    for line in text.split("\n"):
        if line.strip() != "":
            for chunk in split_string_by_multiple_delimiters(line, [". ", "? "]):
                text_claims.append(chunk)

    return text_claims

def split_claims_and_impl(text):
    text = text.replace("**Implications:**", "Implications")
    text = text.replace("### Implications:", "Implications")
    text = text.replace("**Summary of changes:**", "")
    text = text.replace("**Summary of Changes:**", "")
    text = text.replace("### Summary of changes:", "")
    text = text.replace("### Summary of Changes:", "")
    text = text.replace("Summary of Changes:", "")
    text = text.replace("Summary of changes:", "")
    text_claims = []
    text_impls = [] # implications
    add_impl = False
    ctr = 1
    for line in text.split("\n"):
        if line.strip().lower().startswith("implications"):
            add_impl = True
            ctr = 1
            continue
        elif line.strip() != "":
            line = line.strip()
            if add_impl:
                line = line.replace(f"{ctr}. ","").strip()
                if not line.endswith("."): break # print(line+"\x1b[34;1m#\x1b[0m")
                text_impls.append(line)
                ctr += 1
            else: 
                line = line.replace(f"{ctr}. ","")
                text_claims.append(line)
                ctr += 1
            # for chunk in split_string_by_multiple_delimiters(line, [". ", "? "]):
            #     text_claims.append(chunk)
    # print(text_claims)
    # print(text_impls)
    # exit()
    return text_claims+text_impls

def compute_scores_from_sts_sim_matrix(sts_sim_matrix, thresh: float):
    # semantic similarity of diff/change claim and review claim pairs.
    sem_similarity_thresh = (sts_sim_matrix > thresh)

    # mask out review claims with maximum similarity with any diff claim less than the threshold.
    prec_alignment = sts_sim_matrix.max(dim=0)
    prec_array = sts_sim_matrix.max(dim=0).values
    prec_mask = (prec_array > thresh).float()
    # prec_array = prec_mask * prec_array

    # counts up claims that are covered (STS greater than the threshold with any review sentence) by at least one review sentence.
    rec_array = (sem_similarity_thresh.sum(dim=1)>0)
    
    # precision/conciseness
    P = prec_mask.mean().item()

    # recall/comprehensiveness
    num_change_claims = sts_sim_matrix.shape[0]
    # normalization factor for recall/comprehensiveness.
    Z = 1 if num_change_claims == 0 else num_change_claims
    R = rec_array.sum().item()/Z

    return P, R, prec_alignment

def thresh_scaling_function(x, thresh1, thresh2):
    """
    thresh1 > thresh2, where thresh1 is the primary threshold
    """
    x1, y1 = thresh2, 0
    x2, y2 = thresh1, 1
    # Calculate the slope (m) for the linear transformation
    m = (y2 - y1) / (x2 - x1)
    # Apply the linear function element-wise for the tensor
    y = m * (x - x1) + y1
    # Clamp the output between 0 and 1
    return torch.clamp(y, min=0, max=1)

def compute_scores_from_sts_sim_matrix_two_thresh(sts_sim_matrix, thresh1: float, thresh2: float):
    """
    do score computation with primary and secondary overlap thresholds.
    - sts_sim_matrix: similarities between review sentences and code claims. Dimensions: claims x review_sentences
    - thresh1: primary threshold
    - thresh2: secondary threshold
    """
    # semantic similarity of diff/change claim and review claim pairs.
    matches_mat = thresh_scaling_function(sts_sim_matrix, thresh1, thresh2)

    # mask out review claims with maximum similarity with any diff claim less than the threshold.
    prec_alignment = sts_sim_matrix.max(dim=0)
    prec_array = sts_sim_matrix.max(dim=0).values
    prec_mask = (prec_array >= thresh1).float()

    # precision / Con
    num_review_sentences = sts_sim_matrix.shape[1] # normalize by the number of review sentences. 
    P = matches_mat.max(dim=0).values.sum().item() / num_review_sentences

    # recall / Comp
    num_claims = sts_sim_matrix.shape[0] # normalize by the number of claims.
    R = matches_mat.max(dim=1).values.sum().item() / num_claims

    return P, R, prec_alignment

class RelevanceScorer(object):
    def __init__(self, model_path, hi_sim_thresh: float=0.85):
        self.sbert = SentenceTransformer(model_path)
        self.hi_sim_thresh = hi_sim_thresh

    def compute(self, change_summs: List[str], reviews: List[str], debug: bool=False):
        P = []
        R = []
        STS = []
        for change_summ, review in tqdm(zip(change_summs, reviews), total=len(reviews)):
            p, r, sts_mat = self.compute_inst(change_summ, review, debug=debug)
            P.append(p)
            R.append(r)
            STS.append(sts_mat)
        p_score = np.mean(P)
        r_score = np.mean(R)
        f_score = (2*p_score*r_score)/(p_score+r_score) if (p_score + r_score) != 0 else 0

        return P, R, STS, p_score, r_score, f_score 

    def sbert_filt_encode(self, claims, show_progress_bar: bool=False):
        token_vecs = self.sbert.encode(claims, output_value="token_embeddings", show_progress_bar=show_progress_bar)
        filt_pooled_vecs = []
        for i,token_vecs in enumerate(token_vecs):
            claim = claims[0]
            # claim = claims[i]
            tokens = self.sbert.tokenizer.batch_decode(
                self.sbert.tokenize([claim])['input_ids'][0]
            )
            # print([token for token in tokens if token not in stop]) # TODO: DEBUG
            filt_token_vecs = torch.stack([vec for tok,vec in zip(tokens, token_vecs) if tok not in stop])
            pooled_filt_sent_vec = torch.sum(filt_token_vecs, 0) / len(filt_token_vecs)
            pooled_normed_filt_sent_vec = F.normalize(pooled_filt_sent_vec.unsqueeze(0), p=2, dim=1).squeeze()
            filt_pooled_vecs.append(pooled_normed_filt_sent_vec.tolist())

        return filt_pooled_vecs

    def compute_inst(self, change_summ: List[str], review: str, debug: bool=False):
        change_claims = change_summ
        # add blank claim if none are found
        if len(change_claims) == 0: change_claims.append("") 
        review_claims = split_claims(review)
        # add blank claim if none are found
        if len(review_claims) == 0: review_claims.append("")
        change_enc = self.sbert_filt_encode(change_claims, show_progress_bar=False)
        review_enc = self.sbert_filt_encode(review_claims, show_progress_bar=False)
        # semantic similarity of diff/change claim and review claim pairs.
        sem_similarity_matrix = util.cos_sim(change_enc, review_enc)

        P, R, prec_alignment = compute_scores_from_sts_sim_matrix(sem_similarity_matrix, self.hi_sim_thresh)
        
        if debug:
            # print associations:
            cc_per_rc = [(change_claims[j], review_claims[i]) for i,j in enumerate(prec_alignment.indices.tolist())]
            print("\x1b[34;1mMost Relevant Claims/Smells for Review Claims:\x1b[0m")
            print(f"P: {P:.3f}, R: {R:.3f}")
            print("All change claims:")
            print(change_claims)
            for i, (cc, rc) in enumerate(cc_per_rc):
                print(f"CC: {cc} RC: {rc} sim: {prec_alignment.values[i].item():.3} rec_array: {sem_similarity_matrix[:,i].tolist()}")
            print()

        return P, R, sem_similarity_matrix.cpu().numpy()

    # def compute_inst(self, change_summ: List[str], review: str, debug: bool=False):
    #     change_claims = change_summ
    #     # add blank claim if none are found
    #     if len(change_claims) == 0: change_claims.append("") 
    #     review_claims = split_claims(review)
    #     # add blank claim if none are found
    #     if len(review_claims) == 0: review_claims.append("")
    #     change_enc = self.sbert_filt_encode(change_claims, show_progress_bar=False)
    #     review_enc = self.sbert_filt_encode(review_claims, show_progress_bar=False)
    #     # semantic similarity of diff/change claim and review claim pairs.
    #     sem_similarity_matrix = util.cos_sim(change_enc, review_enc)
    #     sem_similarity_thresh = (sem_similarity_matrix > self.hi_sim_thresh)

    #     # mask out review claims with maximum similarity with any diff claim less than the threshold.
    #     prec_alignment = sem_similarity_matrix.max(dim=0)
    #     prec_array = sem_similarity_matrix.max(dim=0).values
    #     prec_mask = (prec_array > self.hi_sim_thresh).float()
    #     # prec_array = prec_mask * prec_array

    #     rec_array = (sem_similarity_thresh.sum(dim=1)>0)
        
    #     # precision/conciseness
    #     # P = prec_array.mean().item()
    #     P = prec_mask.mean().item()

    #     # recall/comprehensiveness
    #     num_change_claims = len(change_enc)
    #     # normalization factor for recall/comprehensiveness.
    #     Z = 1 if num_change_claims == 0 else num_change_claims
    #     R = rec_array.sum().item()/Z
        
    #     if debug:
    #         # print associations:
    #         cc_per_rc = [(change_claims[j], review_claims[i]) for i,j in enumerate(prec_alignment.indices.tolist())]
    #         print("\x1b[34;1mMost Relevant Claims/Smells for Review Claims:\x1b[0m")
    #         print(f"P: {P:.3f}, R: {R:.3f}")
    #         print("All change claims:")
    #         print(change_claims)
    #         for i, (cc, rc) in enumerate(cc_per_rc):
    #             print(f"CC: {cc} RC: {rc} sim: {prec_alignment.values[i].item():.3} rec_array: {sem_similarity_matrix[:,i].tolist()}")
    #         print()

    #     return P, R

def process_java_smells(path, file, add_line_numbers: bool=True):
    import re
    file, _ = os.path.splitext(file)
    smells = []
    with open(path) as f:
        for og_line in f:
            og_line = og_line.strip("\n")
            # print(PROJECT_FILE_DIR.format(file=file))
            PROJECT_FILE_LINE = "/home/arnaik/code-review-test-projects/java/{file}/{file}.java"
            if og_line.startswith(PROJECT_FILE_LINE.format(file=file)+":"):
                cleaned_line = og_line.replace(PROJECT_FILE_LINE.format(file=file), "")
                line_no = int(cleaned_line.split(":\t")[0].replace(":",""))
                cleaned_line = cleaned_line.removeprefix(f":{line_no}:\t")
                if add_line_numbers:
                    cleaned_line = f"line {line_no}, " + cleaned_line
                smells.append([cleaned_line, line_no])
            elif og_line.startswith(PROJECT_FILE_LINE.format(file=file)+"\t-\t"):
                cleaned_line = og_line.replace(PROJECT_FILE_LINE.format(file=file)+"\t-\t", "")
                line_number_pattern = r'line (\d+)'
                try: line_no = int(re.search(line_number_pattern, cleaned_line).group(1))
                except AttributeError: 
                    # print(og_line) s
                    line_no = -1
                smells.append([cleaned_line, line_no])
            else:
                # print(og_line) 
                smells[-1][0] += "\n" + og_line

    return smells

def is_error_count_line(line: str):
    return (line.strip().endswith("error") or line.strip().endswith("errors")) and line.strip().split()[0].strip().isnumeric()

def process_python_smells(path, file):
    return json.load(open(path))['smells']

def process_javascript_smells(path, file):
    import re
    file, _ = os.path.splitext(file)
    smells = []
    with open(path) as f:
        PROJECT_FILE_LINE = "/home/atharva/CMU/sem4/Directed Study/javascript/{file}/{file}.js"
        for og_line in f:
            og_line = og_line.strip("\n")
            if og_line.strip() == "": continue # blank line
            elif is_error_count_line(og_line): continue # error count line.
            cleaned_line = og_line.replace(PROJECT_FILE_LINE.format(file=file)+":", "").strip()
            line_number_pattern = r'line (\d+)'
            try: line_no = re.search(line_number_pattern, cleaned_line).group(1) 
            except AttributeError:
                # print(og_line)
                return []
            # print(PROJECT_FILE_DIR.format(file=file))
            # smells.append(line.replace(PROJECT_FILE_DIR.format(file=file), ""))
            smells.append([cleaned_line, int(line_no)])

    return smells

# def filter_by_changed_lines(smells, range_) -> List[str]:
#     filt_smells = []
#     for line, line_no in smells:
#         if range_[0] <= line_no <= range_[1]:
#             filt_smells.append(line)
#         elif line_no == -1:
#             filt_smells.append(line)

#     return filt_smells

def filter_by_changed_lines(smells, range_: List[int], new_file: str, diff: str) -> List[str]:
    """
    only consider the specific lines that are changed and not even the whole patch/diff
    """
    filt_smells = []
    changed_lines = [line[1:].strip() for line in diff.split("\n") if line[0] in ["+","-"]]
    # print(changed_lines)
    for line, line_no in smells:
        # relevant_lines = new_file.split("\n")[range_[0]-1:range_[1]]
        # new_file_line_1 = new_file.split("\n")[line_no]
        new_file_line = new_file.split("\n")[line_no-1].strip()
        if new_file_line in changed_lines: filt_smells.append(line)
        # print(new_file_line)
        # print(line)
        # print("line:", new_file_line)
        # print("line1:", new_file_line_1)
        # print("line2:", new_file_line_2)
        # if range_[0] <= line_no <= range_[1]:
        #     filt_smells.append(line)
        # elif line_no == -1:
        #     filt_smells.append(line)
    # print(filt_smells)
    # exit()
    return filt_smells

def load_code_claims_and_issues(
        data: List[dict],
        claims_path: str, issues_paths: Dict[str, str], 
        patch_ranges_path: str, split_function=split_claims,
    ):
    patch_ranges = json.load(open(patch_ranges_path))

    newfs = {f"test{i}": generate_newf(rec['oldf'], rec['patch'])[0] for i, rec in enumerate(data)}
    diffs = {f"test{i}": rec['patch'] for i, rec in enumerate(data)}
    java_smell_summaries = {file.split('.')[0].strip(): process_java_smells(os.path.join(issues_paths["java"], file), file) for file in os.listdir(issues_paths["java"])}
    print(f"{sum([len(s) for s in java_smell_summaries.values()])} java smells before filtering across {sum([len(s) > 0 for s in java_smell_summaries.values()])} instances")
    java_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k], newfs[k], diffs[k]) for k,v in java_smell_summaries.items()}
    print(f"{sum([len(s) for s in java_smell_summaries.values()])} java smells after filtering across {sum([len(s) > 0 for s in java_smell_summaries.values()])} instances")

    python_smell_summaries = {file.split('.')[0].strip(): process_python_smells(os.path.join(issues_paths["python"], file), file) for file in os.listdir(issues_paths["python"])}
    print(f"{sum([len(s) for s in python_smell_summaries.values()])} python smells before filtering across {sum([len(s) > 0 for s in python_smell_summaries.values()])} instances")
    python_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k], newfs[k], diffs[k]) for k,v in python_smell_summaries.items()}
    print(f"{sum([len(s) for s in python_smell_summaries.values()])} python smells after filtering across {sum([len(s) > 0 for s in python_smell_summaries.values()])} instances")

    javascript_smell_summaries = {file.split('.')[0].strip(): process_javascript_smells(os.path.join(issues_paths["javascript"], file), file) for file in os.listdir(issues_paths["javascript"])}
    print(f"{sum([len(s) for s in javascript_smell_summaries.values()])} javascript smells before filtering across {sum([len(s) > 0 for s in javascript_smell_summaries.values()])} instances")
    javascript_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k], newfs[k], diffs[k]) for k,v in javascript_smell_summaries.items()}
    print(f"{sum([len(s) for s in javascript_smell_summaries.values()])} javascript smells after filtering across {sum([len(s) > 0 for s in javascript_smell_summaries.values()])} instances")

    smell_claims = {}
    smell_claims.update(java_smell_summaries)
    smell_claims.update(python_smell_summaries)
    smell_claims.update(javascript_smell_summaries)
    llm_generated_claims = read_jsonl(claims_path)
    # llm_generated_claims = {f"test{i}": split_claims_and_impl(rec['response']) for i, rec in enumerate(read_jsonl(claims_path))}
    
    code_change_to_claims_and_issues = {}
    for i in range(len(llm_generated_claims)):
        claims = split_function(llm_generated_claims[i]['response'])
        # split_claims(llm_generated_claims[i]['response'])
        issues = smell_claims.get(f"test{i}", [])
        code_change = llm_generated_claims[i]["code_change"]
        code_change_to_claims_and_issues[code_change] = claims + issues

    return code_change_to_claims_and_issues

# outdated code - remove in future commits.
# def human_study_results(thresh: float):
#     model_preds = pd.read_csv("human_study_data.csv")
#     code_claims_path = "./experiments/code_change_summ_finetune_impl/Magicoder-S-DS-6.7B.jsonl"
#     all_code_change_summ = load_code_claims_and_issues(
#         data=data,
#         claims_path=code_claims_path,
#         issues_paths={
#             "python": "./experiments/python_code_smells",
#             "java": "./experiments/java_code_smells",
#             "javascript": "./experiments/javascript_code_smells",
#         },
#         patch_ranges_path="./data/Comment_Generation/test_set_codepatch_ranges.json",
#         split_function=split_claims_and_impl if "_impl" in code_claims_path else split_claims,
#     ) 
#     # {rec['code_change']: rec["change_summary"] for rec in read_jsonl("./experiments/code_change_summ_v2/Magicoder-S-DS-6.7B.jsonl")}
#     code_change_summ = [all_code_change_summ[patch] for patch in model_preds["patch"]]
#     # Snowflake/snowflake-arctic-embed-l
#     rel_scorer = RelevanceScorer(model_path="mixedbread-ai/mxbai-embed-large-v1", hi_sim_thresh=thresh)
#     rel_score_human_data = [{'index': index} for index in model_preds['index']]
#     for model in ["codereviewer", "magicoder", "lstm", "knn", "ground_truth"]:
#         #read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")
#         if model == "ground_truth": reviews = model_preds['msg'].tolist()
#         else: reviews = model_preds[f"{model}_pred"].tolist()
#         mean_review_length = np.mean([len(r.split()) for r in reviews])
#         inst_rel_P_scores, inst_rel_R_scores, rel_P_score, rel_R_score, rel_F_score = rel_scorer.compute(code_change_summ, reviews)
#         print(model, f"P={100*rel_P_score:.2f} R={100*rel_R_score:.2f} F={100*rel_F_score:.2f} RL={mean_review_length:.2f}")
#         for i,val in enumerate(inst_rel_P_scores):
#             rel_score_human_data[i][model] = val
#     with open("./human_study_relevance_scores.json", "w") as f:
#         json.dump(rel_score_human_data, f, indent=4)

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

def process_codellama_output(review: str):
    review_lines = review.split("\n")
    review_ = " ".join([line.strip() for line in review_lines[:-1]])
    if review_lines[-1].strip().endswith("."): 
        review_ += " "+review_lines[-1].strip()

    return review_

def safe_division(a, b):
    if b == 0: return 0
    return a/b

def process_llama3_output(text):
    return text.split("\n\n")[0].strip("\n")

def process_lstm_output(review: str):
    if review.startswith("<msg>"): review = review[len("<msg>"):].strip()
        
    return review

def all_model_all_data_results(thresh: float):
    """
    Compute the Con, Comp and Rel scores for all the models, using STS scores with a threshold of 0.85.
    """
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    model_preds = {
        # CodeReviewer
        "codereviewer": [rec['pred'] for rec in read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")],
        # Magicoder
        "magicoder": [process_magicoder_output(process_magicoder_output(rec['pred_review'])) for rec in read_jsonl('./data/Comment_Generation/llm_outputs/Magicoder-S-DS-6.7B.jsonl')],
        # LSTM
        "lstm": [process_lstm_output(r['pred']) for r in read_jsonl("./ckpts/lstm_reviewer_1_layer/preds.jsonl")],
        # KNN
        "knn": [r for r,_ in json.load(open("./experiments/knn_retriever_preds.json"))],
        # Ground Truth
        "ground_truth": [i['msg'] for i in data],
        # DeepSeekCoder
        "deepseekcoder": [process_magicoder_output(process_magicoder_output(rec['pred_review'])) for rec in read_jsonl('./experiments/llm_outputs/DeepSeekCoder-6.7B-Instruct.jsonl')],
        # Stable Code
        "stable_code": [process_magicoder_output(process_magicoder_output(rec['pred_review'])) for rec in read_jsonl('./experiments/llm_outputs/Stable-Code-Instruct-3b.jsonl')], 
        # GPT-3.5
        "gpt3.5": [process_magicoder_output(process_magicoder_output(rec['pred_review'])) for rec in read_jsonl('./experiments/llm_outputs/GPT-3.5-Turbo.jsonl')],
        # LLaMA-3
        "llama3": [process_llama3_output(process_magicoder_output(process_magicoder_output(rec['pred_review']))) for rec in read_jsonl('./experiments/llm_outputs/Llama-3-8B-Instruct.jsonl')],
        # Code Llama models
        "codellama_7b": [process_codellama_output( process_codellama_output(rec['pred'])) for rec in read_jsonl("./experiments/codellama_codellama_7b_instruct_hf_zero_shot/preds.jsonl")],
        "codellama_13b": [process_codellama_output(process_codellama_output(rec['pred'])) for rec in read_jsonl("./experiments/CodeLLaMA_Prompting/preds.jsonl")],
        
    }
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
    rel_scorer = RelevanceScorer(model_path="mixedbread-ai/mxbai-embed-large-v1", hi_sim_thresh=thresh)
    # rel_score_human_data = [{'index': index} for index in model_preds['index']]
    rel_scores = {}
    # ["codereviewer", "magicoder", "deepseekcoder", "stable_code", "lstm", "knn", "ground_truth"]:
    for model in model_preds:
        #read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")
        reviews = model_preds[model]
        review_lengths = [len(r.split()) for r in reviews]
        mean_review_length = np.mean(review_lengths)
        inst_rel_P_scores, inst_rel_R_scores, sts_matrices, rel_P_score, rel_R_score, rel_F_score = rel_scorer.compute(
            code_change_summ, reviews, debug=DEBUG
        )
        print(model, f"P={100*rel_P_score:.2f} R={100*rel_R_score:.2f} F={100*rel_F_score:.2f} RL={mean_review_length:.2f}")
        # F-score or relevance computed with instance level averaging.
        inst_rel_F_scores = [safe_division(2*p*r, p+r) for p,r in zip(inst_rel_P_scores, inst_rel_R_scores)]
        rel_scores[model] = {
            "P": inst_rel_P_scores,
            "R": inst_rel_R_scores,
            "F": inst_rel_F_scores,
            "RL": review_lengths,
        }
        sts_mat_save_path = f"sts_matrices/{model}_sts_matrix.npz"
        os.makedirs("sts_matrices", exist_ok=True)
        np.savez_compressed(sts_mat_save_path, *sts_matrices)
    with open(f"./all_model_rel_scores_thresh_{thresh}.json", "w") as f:
        json.dump(rel_scores, f, indent=4)
    
def load_human_eval_samples():
    human_annot_datapoints = []
    marcus_annot_end_points = {"py": 501-2, "java": 501-2, "js": 505-2}
    index_to_lang = {}
    langs = list(marcus_annot_end_points.keys())
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
                diff = rec['diff']
                index_to_lang[index] = lang
            system = rec['system']
            if str(rec["Rel (F)"]) != "nan" and system != "msg": # skip CodeReviewer ground truth/references among the evaluated systems, because we don't count it for the correlations as reference based metrics would default to 1 on them and disadvantage their correlation values.
                human_annot_datapoints.append({"index": index, "lang": lang, "system": system, "diff": diff, "review": rec["review"], "rel": rec["Rel (F)"]})
            
    return human_annot_datapoints

def human_eval_data_results(thresh: float):
    """
    Compute the Con, Comp and Rel scores for all the models, using STS scores with a threshold of 0.85.
    """
    from scipy.stats import spearmanr, kendalltau

    # load only human eval data here?
    # data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    human_eval_data = load_human_eval_samples()
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    model_preds = {
        # CodeReviewer
        "codereviewer": [rec['review'] for rec in human_eval_data if rec['system'] == 'codereviewer_pred'],
        # Magicoder
        "magicoder": [rec['review'] for rec in human_eval_data if rec['system'] == 'magicoder_pred'],
        # LSTM
        "lstm": [rec['review'] for rec in human_eval_data if rec['system'] == 'lstm_pred'],
        # KNN
        "knn": [rec['review'] for rec in human_eval_data if rec['system'] == 'knn_pred'],
        # DeepSeekCoder
        "deepseekcoder": [rec['review'] for rec in human_eval_data if rec['system'] == 'deepseekcoder_pred'],
        # Stable Code
        "stable_code": [rec['review'] for rec in human_eval_data if rec['system'] == 'stable_code_pred'], 
        # GPT-3.5
        "gpt3.5": [rec['review'] for rec in human_eval_data if rec['system'] == 'gpt3.5_pred'],
        # LLaMA-3
        "llama3": [rec['review'] for rec in human_eval_data if rec['system'] == 'llama3_pred'],
        # Code Llama models
        "codellama_13b": [rec['review'] for rec in human_eval_data if rec['system'] == 'codellama_13b_pred'],        
    }
    model_to_GT_rel = {}
    for model in model_preds:
        model_to_GT_rel[model] = [rec['rel'] for rec in human_eval_data if rec['system'] == model+"_pred"]

    code_claims_path = "./experiments/code_change_summ_finetune_impl/gpt-4o.jsonl"
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
    
    # print(code_change_summ[0])
    # exit()
    rel_scorer = RelevanceScorer(model_path="mixedbread-ai/mxbai-embed-large-v1", hi_sim_thresh=thresh)
    # rel_score_human_data = [{'index': index} for index in model_preds['index']]
    rel_scores = {}
    # ["codereviewer", "magicoder", "deepseekcoder", "stable_code", "lstm", "knn", "ground_truth"]:
    X, Y = [], []
    for model in model_preds:
        #read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")
        reviews = [r if isinstance(r, str) else "" for r in model_preds[model]]
        code_change_summ = [all_code_change_summ[rec['diff']] for rec in human_eval_data if rec['system'] == f"{model}_pred"]
        review_lengths = [len(r.split()) for r in reviews]
        mean_review_length = np.mean(review_lengths)
        inst_rel_P_scores, inst_rel_R_scores, sts_matrices, rel_P_score, rel_R_score, rel_F_score = rel_scorer.compute(
            code_change_summ, reviews, debug=DEBUG
        )
        print(model, f"P={100*rel_P_score:.2f} R={100*rel_R_score:.2f} F={100*rel_F_score:.2f} RL={mean_review_length:.2f}")
        # F-score or relevance computed with instance level averaging.
        inst_rel_F_scores = [safe_division(2*p*r, p+r) for p,r in zip(inst_rel_P_scores, inst_rel_R_scores)]
        rel_scores[model] = {
            "P": inst_rel_P_scores,
            "R": inst_rel_R_scores,
            "F": inst_rel_F_scores,
            "RL": review_lengths,
        }
        X += model_to_GT_rel[model]
        Y += inst_rel_R_scores
        sts_mat_save_path = f"sts_matrices/gpt_4o_humaneval_{model}_sts_matrix.npz"
        os.makedirs("sts_matrices", exist_ok=True)
        np.savez_compressed(sts_mat_save_path, *sts_matrices)
    print(spearmanr(X, Y))
    print(kendalltau(X, Y))
    with open(f"./human_eval_rel_scores_thresh_{thresh}.json", "w") as f:
        json.dump(rel_scores, f, indent=4)

# main
if __name__ == "__main__":
    # codereviewer model generated preds.
    # modelgen_code_reviews = [rec['pred'] for rec in read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")]
    # human_study_results()
    
    # # take the STS similarity threshold as an argument.
    # thresh = float(sys.argv[1])
    # print(f"using threshold: {thresh}")
    # all_model_all_data_results(thresh=thresh)

    # take the STS similarity threshold as an argument.
    thresh = float(sys.argv[1])
    print(f"using threshold: {thresh}")
    human_eval_data_results(thresh=thresh)
