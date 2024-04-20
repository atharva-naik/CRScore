import os
import re
import json
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from src.datautils import read_jsonl
from sentence_transformers import SentenceTransformer, util

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

    def compute_inst(self, change_summ: List[str], review: str, debug: bool=False):
        change_claims = change_summ
        # add blank claim if none are found
        if len(change_claims) == 0: change_claims.append("") 
        review_claims = split_claims(review)
        # add blank claim if none are found
        if len(review_claims) == 0: review_claims.append("")
        change_enc = self.sbert.encode(change_claims, show_progress_bar=False)
        review_enc = self.sbert.encode(review_claims, show_progress_bar=False)
        sem_similarity_matrix = util.cos_sim(change_enc, review_enc)
        prec_array = sem_similarity_matrix.max(dim=0)
        rec_array = sem_similarity_matrix.max(dim=1)
        # precision/conciseness
        P = prec_array.values.mean().item()
        # recall/comprehensiveness
        R = rec_array.values.mean().item()
        if debug:
            # print associations:
            cc_per_rc = [(change_claims[j], review_claims[i]) for i,j in enumerate(prec_array.indices.tolist())]
            print("\x1b[34;1mMost Relevant Claims/Smells for Review Claims:\x1b[0m")
            print("All change claims:")
            print(change_claims)
            for i, (cc, rc) in enumerate(cc_per_rc):
                print(f"CC: {cc} RC: {rc} sim: {prec_array.values[i].item():.3f} R: {R:.3f}")
            print()

        return P, R

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

def filter_by_changed_lines(smells, range_) -> List[str]:
    filt_smells = []
    for line, line_no in smells:
        if range_[0] <= line_no <= range_[1]:
            filt_smells.append(line)
        elif line_no == -1:
            filt_smells.append(line)

    return filt_smells

def load_code_claims_and_issues(claims_path: str, issues_paths: Dict[str, str], patch_ranges_path: str):
    patch_ranges = json.load(open(patch_ranges_path))
    java_smell_summaries = {file.split('.')[0].strip(): process_java_smells(os.path.join(issues_paths['java'], file), file) for file in os.listdir(issues_paths['java'])}
    java_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k]) for k,v in java_smell_summaries.items()}

    python_smell_summaries = {file.split('.')[0].strip(): process_python_smells(os.path.join(issues_paths['python'], file), file) for file in os.listdir(issues_paths['python'])}
    python_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k]) for k,v in python_smell_summaries.items()}

    javascript_smell_summaries = {file.split('.')[0].strip(): process_javascript_smells(os.path.join(issues_paths['javascript'], file), file) for file in os.listdir(issues_paths['javascript'])}
    javascript_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k]) for k,v in javascript_smell_summaries.items()}

    smell_claims = {}
    smell_claims.update(java_smell_summaries)
    smell_claims.update(python_smell_summaries)
    smell_claims.update(javascript_smell_summaries)
    llm_generated_claims = read_jsonl("./experiments/code_change_summ_finetune/Magicoder-S-DS-6.7B.jsonl")
    
    code_change_to_claims_and_issues = {}
    for i in range(len(llm_generated_claims)):
        claims = split_claims(llm_generated_claims[i]['response'])
        issues = smell_claims.get(f"test{i}", [])
        code_change = llm_generated_claims[i]["code_change"]
        code_change_to_claims_and_issues[code_change] = claims + issues

    return code_change_to_claims_and_issues

def human_study_results():
    model_preds = pd.read_csv("human_study_data.csv")
    all_code_change_summ = load_code_claims_and_issues(
        claims_path="./experiments/code_change_summ_finetune/Magicoder-S-DS-6.7B.jsonl",
        issues_paths={
            "python": "./experiments/python_code_smells",
            "java": "./experiments/java_code_smells",
            "javascript": "./experiments/javascript_code_smells",
        },
        patch_ranges_path="./data/Comment_Generation/test_set_codepatch_ranges.json",
    ) 
    # {rec['code_change']: rec["change_summary"] for rec in read_jsonl("./experiments/code_change_summ_v2/Magicoder-S-DS-6.7B.jsonl")}
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
        "deepseekcoder": [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./experiments/llm_outputs/DeepSeekCoder-6.7B-Instruct.jsonl')],
        "stable_code": [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./experiments/llm_outputs/Stable-Code-Instruct-3b.jsonl')], 
    }
    all_code_change_summ = load_code_claims_and_issues(
        claims_path="./experiments/code_change_summ_finetune/Magicoder-S-DS-6.7B.jsonl",
        issues_paths={
            "python": "./experiments/python_code_smells",
            "java": "./experiments/java_code_smells",
            "javascript": "./experiments/python_code_smells",
        },
        patch_ranges_path="./data/Comment_Generation/test_set_codepatch_ranges.json",
    ) 
    code_change_summ = [all_code_change_summ[i['patch']] for i in data]
    rel_scorer = RelevanceScorer("all-roberta-large-v1")
    # rel_score_human_data = [{'index': index} for index in model_preds['index']]
    rel_scores = {}
    for model in ["codereviewer", "magicoder", "deepseekcoder", "stable_code", "lstm", "knn", "ground_truth"]:
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
