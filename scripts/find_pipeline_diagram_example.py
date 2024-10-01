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
from src.metrics.claim_based.relevance_score import *
from scripts.create_code_smell_analysis_data import generate_newf, remove_space_clean

def load_code_claims_and_issues(
        data: List[dict],
        claims_path: str, issues_paths: Dict[str, str], 
        patch_ranges_path: str, split_function=split_claims,
        issue_filter: bool=True # only return instances with at least one issue.
    ):
    patch_ranges = json.load(open(patch_ranges_path))

    newfs = {f"test{i}": generate_newf(rec['oldf'], rec['patch'])[0] for i, rec in enumerate(data)}
    diffs = {f"test{i}": rec['patch'] for i, rec in enumerate(data)}
    # java_smell_summaries = {file.split('.')[0].strip(): process_java_smells(os.path.join(issues_paths["java"], file), file) for file in os.listdir(issues_paths["java"])}
    # print(f"{sum([len(s) for s in java_smell_summaries.values()])} java smells before filtering across {sum([len(s) > 0 for s in java_smell_summaries.values()])} instances")
    # java_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k], newfs[k], diffs[k]) for k,v in java_smell_summaries.items()}
    # print(f"{sum([len(s) for s in java_smell_summaries.values()])} java smells after filtering across {sum([len(s) > 0 for s in java_smell_summaries.values()])} instances")

    python_smell_summaries = {file.split('.')[0].strip(): process_python_smells(os.path.join(issues_paths["python"], file), file) for file in os.listdir(issues_paths["python"])}
    print(f"{sum([len(s) for s in python_smell_summaries.values()])} python smells before filtering across {sum([len(s) > 0 for s in python_smell_summaries.values()])} instances")
    python_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k], newfs[k], diffs[k]) for k,v in python_smell_summaries.items()}
    print(f"{sum([len(s) for s in python_smell_summaries.values()])} python smells after filtering across {sum([len(s) > 0 for s in python_smell_summaries.values()])} instances")

    # javascript_smell_summaries = {file.split('.')[0].strip(): process_javascript_smells(os.path.join(issues_paths["javascript"], file), file) for file in os.listdir(issues_paths["javascript"])}
    # print(f"{sum([len(s) for s in javascript_smell_summaries.values()])} javascript smells before filtering across {sum([len(s) > 0 for s in javascript_smell_summaries.values()])} instances")
    # javascript_smell_summaries = {k: filter_by_changed_lines(v, patch_ranges[k], newfs[k], diffs[k]) for k,v in javascript_smell_summaries.items()}
    # print(f"{sum([len(s) for s in javascript_smell_summaries.values()])} javascript smells after filtering across {sum([len(s) > 0 for s in javascript_smell_summaries.values()])} instances")

    smell_claims = {}
    # smell_claims.update(java_smell_summaries)
    smell_claims.update(python_smell_summaries)
    # smell_claims.update(javascript_smell_summaries)
    llm_generated_claims = read_jsonl(claims_path)
    # llm_generated_claims = {f"test{i}": split_claims_and_impl(rec['response']) for i, rec in enumerate(read_jsonl(claims_path))}
    
    claims_and_issues_and_code_changes = []
    for i in range(len(llm_generated_claims)):
        claims = list(split_function(llm_generated_claims[i]['response']))
        # split_claims(llm_generated_claims[i]['response'])
        issues = list(smell_claims.get(f"test{i}", []))
        if issue_filter and len(issues) == 0: continue
        code_change = llm_generated_claims[i]["code_change"]
        claims_and_issues_and_code_changes.append({
            "code_change": code_change, 
            "index": i, "msg": data[i]["msg"],
            "claims_and_issues": [(claim, "claim") for claim in claims]+[(issue, "issue") for issue in issues],
        })

    return claims_and_issues_and_code_changes

# main
if __name__ == "__main__":
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    code_claims_path = "./experiments/code_change_summ_finetune_impl/Magicoder-S-DS-6.7B.jsonl"
    claims_and_issues = load_code_claims_and_issues(
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
    print(len(claims_and_issues))
    print(claims_and_issues[6]["index"])