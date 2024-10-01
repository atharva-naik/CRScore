import os
import json
import evaluate
import numpy as np
import pandas as pd
from typing import *
import statsmodels.api as sm
from scipy.stats import spearmanr
from collections import defaultdict
from src.datautils import read_jsonl
from rouge_score import rouge_scorer
from scipy.stats import pointbiserialr
from statsmodels.formula.api import ols
from Levenshtein import distance as lev
from src.metrics.cr_score import CRScorer
from CodeBERT.CodeReviewer.code.evaluator.smooth_bleu import bleu_fromstr

BASELINE_TO_SYSTEM_KEY = {
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
}

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

# main
if __name__ == "__main__":
    trues = [rec['msg'] for rec in read_jsonl("./data/Comment_Generation/msg-test.jsonl")]
    bleu_score = evaluate.load("bleu")
    # BERT score.
    bert_score = evaluate.load("bertscore")
    # character level F-score (chrF) and chrF++.
    chrf = evaluate.load("chrf")
    # ROUGE-L score.
    rouge_score_ = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    output_path = "baseline_metric_scores.json"

    all_baseline_metrics = ["BLEU", "BLEU_WITHOUT_STOP", "BERTSCORE", "EDIT_DISTANCE", "ROUGE_L", "CHRF", "CHRF++"]
    models_and_pred_paths = {
        "BM-25 kNN": "./experiments/knn_retriever_preds.json",
        "LSTM": "./ckpts/lstm_reviewer_1_layer/preds.jsonl",
        "CodeReviewer": "./experiments/MS_CR_ZeroShot/preds.jsonl",
        "CodeLLaMA-Instruct-7B": "./experiments/codellama_codellama_7b_instruct_hf_zero_shot/preds.jsonl",
        "CodeLLaMA-Instruct-13B": "./experiments/CodeLLaMA_Prompting/preds.jsonl",
        "Magicoder-S-DS-6.7B": "./experiments/llm_outputs/Magicoder-S-DS-6.7B.jsonl",
        "Stable-Code-Instruct-3B": "./experiments/llm_outputs/Stable-Code-Instruct-3b.jsonl",
        "DeepSeekCoder-Instruct-6.7B": "./experiments/llm_outputs/DeepSeekCoder-6.7B-Instruct.jsonl",
        "GPT-3.5-Turbo": "./experiments/llm_outputs/GPT-3.5-Turbo.jsonl",
        "Llama-3-8B-Instruct": "./experiments/llm_outputs/Llama-3-8B-Instruct.jsonl",
    }

    if os.path.exists(output_path):
        overwrite = input("Overwrite results (y/N)?").strip().lower() in ["yes", "y"]
        if not overwrite: 
            our_metric_scores_path = "./all_model_rel_scores_thresh_0.7311.json"
            our_metric_scores = None
            if os.path.exists(our_metric_scores_path):
                our_metric_scores = json.load(open(our_metric_scores_path))
            data = json.load(open(output_path))
            baseline_metric_values = json.load(open(output_path))
            print("\t".join([metric for metric in all_baseline_metrics]))
            for model in models_and_pred_paths:
                print(model, end="\t")
                for metric in all_baseline_metrics:
                    Z = 100 if metric in ["BLEU", "BLEU_WITHOUT_STOP"] else 1
                    print(round(np.mean([rec[model]/Z for rec in baseline_metric_values[metric]]), 3), end="\t")
                if our_metric_scores is not None:
                    for metric in ["P", "R", "F"]:
                        model1 = BASELINE_TO_SYSTEM_KEY[model]
                        print(round(np.mean([rec for rec in our_metric_scores[model1][metric]]), 3), end="\t")
                print()
            exit()

    all_baseline_scores = {metric: [{model: 0 for model in models_and_pred_paths} for _ in trues] for metric in all_baseline_metrics} 

    for model, preds_path in models_and_pred_paths.items():
        if preds_path.endswith(".json"): preds = json.load(open(preds_path))
        elif preds_path.endswith(".jsonl"): preds = read_jsonl(preds_path)
        if model == "BM-25 kNN": preds = [rec[0] for rec in preds]
        elif model in ["LSTM", "CodeReviewer", 
                       "CodeLLaMA-Instruct-7B", 
                       "CodeLLaMA-Instruct-13B"]:
            preds = [rec['pred'] for rec in preds]
        else: preds = [process_magicoder_output(rec['pred_review']) for rec in preds]

        BLEUS = [bleu_fromstr(predictions=[p], golds=[t], rmstop=False) for p,t in zip(preds, trues)]
        BLEU = bleu_fromstr(predictions=preds, golds=trues, rmstop=False)
        assert BLEU == round(np.mean(BLEUS), 2), "some error in BLEU aggregation"

        BLEU_WITHOUT_STOPS = [bleu_fromstr(predictions=[p], golds=[t], rmstop=True) for p,t in zip(preds, trues)]
        BLEU_WITHOUT_STOP = bleu_fromstr(predictions=preds, golds=trues, rmstop=True)
        assert BLEU_WITHOUT_STOP == round(np.mean(BLEU_WITHOUT_STOPS), 2), "some error in BLEU aggregation"

        CHRFS = [chrf.compute(predictions=[p], references=[t])['score']/100 for p,t in zip(preds, trues)]
        CHRF = np.mean(CHRFS)
        CHRF_PPS = [chrf.compute(predictions=[p], references=[t], word_order=2)['score']/100 for p,t in zip(preds, trues)]
        CHRF_PP = np.mean(CHRF_PPS)

        BERTSCORES = bert_score.compute(predictions=preds, references=trues, model_type="distilbert-base-uncased")['f1']
        BERTSCORE = np.mean(BERTSCORES)

        EDIT_DISTANCES = [lev(p, r)/max(len(p), len(r)) for p,r in zip(preds, trues)]
        EDIT_DISTANCE = np.mean(EDIT_DISTANCES)

        ROUGE_LS = [rouge_score_.score(p,r)['rougeL'].fmeasure for p,r in zip(preds, trues)]
        ROUGE_L = np.mean(ROUGE_LS)

        print(f"\x1b[34;1m{model}\x1b[0m: BLEU: {BLEU:.2f} BLEU without stop: {BLEU_WITHOUT_STOP:.2f} Edit Distance: {EDIT_DISTANCE:.3f} BERTScore: {BERTSCORE:.3f} ROUGE-L: {ROUGE_L:.3f} CHRF: {CHRF:.3f} CHRF++: {CHRF_PP:.3f}")
        for i, score in enumerate(BLEUS):
            all_baseline_scores["BLEU"][i][model] = score
        for i, score in enumerate(BLEU_WITHOUT_STOPS):
            all_baseline_scores["BLEU_WITHOUT_STOP"][i][model] = score
        for i, score in enumerate(CHRFS):
            all_baseline_scores["CHRF"][i][model] = score
        for i, score in enumerate(CHRF_PPS):
            all_baseline_scores["CHRF++"][i][model] = score
        for i, score in enumerate(BERTSCORES):
            all_baseline_scores["BERTSCORE"][i][model] = score
        for i, score in enumerate(EDIT_DISTANCES):
            all_baseline_scores["EDIT_DISTANCE"][i][model] = score
        for i, score in enumerate(ROUGE_LS):
            all_baseline_scores["ROUGE_L"][i][model] = score
        # bleu_score.compute(predictions=preds, references=trues) 
        with open(output_path, "w") as f:
            json.dump(all_baseline_scores, f, indent=4)