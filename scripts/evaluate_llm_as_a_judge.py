import os
import sys
import json
import openai
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict
from src.datautils import read_jsonl
from scipy.stats import spearmanr, kendalltau

def load_human_eval_samples():
    human_annot_datapoints = {}
    marcus_annot_end_points = {"py": 501-2, "java": 501-2, "js": 505-2}
    index_to_lang = {}
    langs = list(marcus_annot_end_points.keys())
    for lang in langs:
        human_annot_datapoints[lang] = []
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
                human_annot_datapoints[lang].append({"index": index, "system": system, "diff": diff, "review": rec["review"], "rel": rec["Rel (F)"]})
            
    return human_annot_datapoints

if __name__ == "__main__":
    # llm_as_a_judge_data = read_jsonl("Magicoder-as-a-judge_metric_scores.jsonl")
    llm_as_a_judge_data = read_jsonl("GPT-4o-as-a-judge_metric_scores.jsonl")
    human_eval_data = load_human_eval_samples()
    X =  [rec['score'] for rec in llm_as_a_judge_data]
    Y = []
    for lang in human_eval_data:
        Y += [rec["rel"] for rec in human_eval_data[lang]]
    assert len(X) == len(Y)
    result = kendalltau(X, Y)
    print(round(result.statistic, 4), result.pvalue)
    result = spearmanr(X, Y)
    print(round(result.statistic, 4), result.pvalue)
