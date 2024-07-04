# corpus level metrics
# 1. identify generic reviews.
# 2. identify hallucinations.
# 3. identify nitpicks.

import re
from tqdm import tqdm
from src.datautils import read_jsonl

# main
if __name__ == "__main__":
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    preds = read_jsonl("./experiments/MS_CR_ZeroShot/preds.jsonl")
    tot = len(preds)
    pattern = r'`([^`]+)`'
    nit_ctr = 0
    union_ctr = 0 # union of all possible systemic issues.
    hallucinations_ctr = 0 # count hallucinations in reviews.
    generic_review_ctr = 0 # generic reviews (totally uncontextualized)
    GENERIC_REVIEWS = [
        "why do we need this?", 
        "why is this needed?",
        "why this change?",
        "why was this removed?", 
        "why do we need this change?",
        "why did you change this?",
        "why is this change needed?",
        "why do we need to change this?",
        "why did you remove this?",
        "why is this removed?",
        'why do we need this method?',
        'why is this being removed?',
        'why do we need this interface?',
        'why do we need this?',
        'why did this change?',
        'why do we need this check?',
        'why do we need this import?',
        'why do we need this class?']
    for pred, item in tqdm(zip(preds, data)):
        label = pred["gold"]
        pred = pred['pred']
        patch = item["patch"]
        oldf = item["oldf"]
        has_issue = False
        if pred.lower().startswith("nit: "): 
            has_issue = True
            nit_ctr += 1
        if pred.lower() in GENERIC_REVIEWS: 
            has_issue = True
            generic_review_ctr += 1
        for match in re.findall(pattern, pred):
            if match not in oldf+"\n"+patch:
                hallucinations_ctr += 1
                has_issue = True
        if has_issue: union_ctr += 1
    nit_rate = 100*nit_ctr/tot
    overall_rate = 100*union_ctr/tot 
    hallucinations_rate = 100*hallucinations_ctr/tot
    generic_review_rate = 100*generic_review_ctr/tot
    print(f"NIT rate: {nit_rate:.2f}%")
    print(f"Hallucination rate: {hallucinations_rate:.2f}%")
    print(f"Generic Review rate: {generic_review_rate:.2f}%")
    print(f"Overall rate: {overall_rate:.2f}%")
    print(f"METRIC: {(100-overall_rate):.2f}%")
    # print(f"TOTAL rate: {(generic_review_rate+nit_rate+hallucinations_rate):.2f}%")