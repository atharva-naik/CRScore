# script to sample instances from the CodeReview comment generation test set for human annotations of review quality.

import os
import re
import json
import random
import pandas as pd
from typing import *
from collections import defaultdict
from src.datautils import read_jsonl

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

def process_lstm_output(review: str):
    if review.startswith("<msg>"): review = review[len("<msg>"):].strip()
        
    return review

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
                print(og_line)
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

# main
if __name__ == "__main__":
    random.seed(42)
    os.makedirs("human_study/pilot2", exist_ok=True)
    code_claims_data_path = "human_study/pilot2/code_claim_accuracy.csv"
    review_quality_data_path = "human_study/pilot2/review_quality_data.csv"
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    codereviewer_preds = [rec['pred'] for rec in read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")]
    magicoder_preds = [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./experiments/llm_outputs/Magicoder-S-DS-6.7B.jsonl')]
    stable_code_preds = [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./experiments/llm_outputs/Stable-Code-Instruct-3b.jsonl')]
    deepseekcoder_preds = [process_magicoder_output(rec['pred_review']) for rec in read_jsonl('./experiments/llm_outputs/DeepSeekCoder-6.7B-Instruct.jsonl')] 
    # lstm_preds = [r['pred'] for r in read_jsonl("./ckpts/lstm_reviewer_1_layer/preds.jsonl")]
    # knn_retriever_preds = [r for r,_ in json.load(open("./experiments/knn_retriever_preds.json"))]
    patch_ranges = json.load(open("./data/Comment_Generation/test_set_codepatch_ranges.json"))

    java_smell_summaries = {file.split('.')[0].strip(): process_java_smells(os.path.join("./experiments/java_code_smells", file), file) for file in os.listdir("./experiments/java_code_smells")}
    print(f"{sum([len(s) for s in java_smell_summaries.values()])} java smells before filtering across {sum([len(s) > 0 for s in java_smell_summaries.values()])} instances")
    java_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k]) for k,v in java_smell_summaries.items()}
    print(f"{sum([len(s) for s in java_smell_summaries.values()])} java smells after filtering across {sum([len(s) > 0 for s in java_smell_summaries.values()])} instances")

    python_smell_summaries = {file.split('.')[0].strip(): process_python_smells(os.path.join("./experiments/python_code_smells", file), file) for file in os.listdir("./experiments/python_code_smells")}
    print(f"{sum([len(s) for s in python_smell_summaries.values()])} python smells before filtering across {sum([len(s) > 0 for s in python_smell_summaries.values()])} instances")
    python_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k]) for k,v in python_smell_summaries.items()}
    print(f"{sum([len(s) for s in python_smell_summaries.values()])} python smells after filtering across {sum([len(s) > 0 for s in python_smell_summaries.values()])} instances")

    javascript_smell_summaries = {file.split('.')[0].strip(): process_javascript_smells(os.path.join("./experiments/javascript_code_smells", file), file) for file in os.listdir("./experiments/javascript_code_smells")}
    print(f"{sum([len(s) for s in javascript_smell_summaries.values()])} javascript smells before filtering across {sum([len(s) > 0 for s in javascript_smell_summaries.values()])} instances")
    javascript_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k]) for k,v in javascript_smell_summaries.items()}
    print(f"{sum([len(s) for s in javascript_smell_summaries.values()])} javascript smells after filtering across {sum([len(s) > 0 for s in javascript_smell_summaries.values()])} instances")

    smell_claims = {}
    smell_claims.update(java_smell_summaries)
    smell_claims.update(python_smell_summaries)
    smell_claims.update(javascript_smell_summaries)
    llm_generated_claims = {f"test{i}": split_claims(rec['response']) for i, rec in enumerate(read_jsonl("./experiments/code_change_summ_finetune/Magicoder-S-DS-6.7B.jsonl"))}
    
    lang_buckets = defaultdict(lambda: [])
    index = 0
    long_inst_len = 1000
    long_inst_ctr = 0
    has_smells = set()
    for index, rec in enumerate(data):
        if rec['lang'] not in ['py', 'java', 'js']: continue
        ck = f"test{index}" # claim key
        rec["index"] = index
        smells = smell_claims.get(ck,[])
        if len(smells) > 0: has_smells.add(index)
        rec['claims'] = [("claim",claim) for claim in llm_generated_claims[ck]] + [("issue",smell) for smell in smells]
        rec['codereviewer_pred'] = codereviewer_preds[index]
        rec['magicoder_pred'] = process_magicoder_output(magicoder_preds[index])
        rec['deepseekcoder_pred'] = process_magicoder_output(deepseekcoder_preds[index])
        rec['stable_code_pred'] = process_magicoder_output(stable_code_preds[index])
        if len(rec['patch']) > long_inst_len:
            long_inst_ctr += 1
            continue
        lang_buckets[rec['lang']].append(rec)
    print(f"{(100*long_inst_ctr/len(data)):.2f}% code changes > {long_inst_len} characters")
    # get the distribution of languages
    data_for_annotation = []
    for lang in lang_buckets:
        k = 100
        # if lang == "go": k = 12
        # else: k = 11
        print(lang, len(lang_buckets[lang]))
        instances = random.sample(lang_buckets[lang], k=k)
        data_for_annotation += instances
    inst_with_smells = 0
    for rec in data_for_annotation:
        if rec["index"] in has_smells: 
            rec["has_smell"] = True
            inst_with_smells += 1
    print(f"human study data has \x1b[31;1m{inst_with_smells}\x1b[0m instances with smells")
    with open("./human_study/pilot2/raw_data.json", "w") as f:
        json.dump(data_for_annotation, f, indent=4)
    # annot_df = pd.DataFrame(data_for_annotation).drop(columns=['idx', 'id'])
    # print(annot_df.columns)
    # print(len(annot_df))
    # annot_df.to_csv("human_study/pilot2/raw_data.csv", index=False)
    # pre_shuffled_data = []
    # pre_model_names = []
    # for rec in annot_df.to_dict("records"):
    #     pre_shuffled_data.append({"index": rec['index'], 'code_change': rec['patch'], 'review': rec['msg'], 'relevance': None, 'informativeness': None, 'correctness': None})
    #     pre_model_names.append('ground_truth')
    #     pre_shuffled_data.append({"index": rec['index'], 'code_change': rec['patch'], 'review': rec['magicoder_pred'], 'relevance': None, 'informativeness': None, 'correctness': None})
    #     pre_model_names.append('magicoder')
    #     pre_shuffled_data.append({"index": rec['index'], 'code_change': rec['patch'], 'review': rec['codereviewer_pred'], 'relevance': None, 'informativeness': None, 'correctness': None})
    #     pre_model_names.append('codereviewer')
    #     pre_shuffled_data.append({"index": rec['index'], 'code_change': rec['patch'], 'review': rec['lstm_pred'], 'relevance': None, 'informativeness': None, 'correctness': None})
    #     pre_model_names.append('lstm')
    #     pre_shuffled_data.append({"index": rec['index'], 'code_change': rec['patch'], 'review': rec['knn_pred'], 'relevance': None, 'informativeness': None, 'correctness': None})
    #     pre_model_names.append('knn')
    # shuffled_index_order = random.sample(range(len(pre_shuffled_data)), k=len(pre_shuffled_data))
    # shuffled_data = []
    # shuffled_model_names = []
    # for i in shuffled_index_order:
    #     shuffled_data.append(pre_shuffled_data[i])
    #     shuffled_model_names.append(pre_model_names[i])
    # with open("./model_names_key.json", "w") as f:
    #     json.dump(shuffled_model_names, f, indent=4)
    # pd.DataFrame(shuffled_data).to_csv("human_study_shuffled_annot_sheet.csv", index=False)
    # print(f"{len(shuffled_data)} rows (3 annotations per row) to be annotated!")
    # annotation_effort = 3 * 3 * len(shuffled_data)
    # print(f"{annotation_effort} uniques annotations required") 