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
    # ROUGE-L score.
    rouge_score_ = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for model, preds_path in {
        "BM-25 kNN": "./experiments/knn_retriever_preds.json",
        "LSTM": "./ckpts/lstm_reviewer_1_layer/preds.jsonl",
        "CodeReviewer": "./experiments/MS_CR_ZeroShot/preds.jsonl",
        "CodeLLaMA-Instruct-7B": "./experiments/codellama_codellama_7b_instruct_hf_zero_shot/preds.jsonl",
        "CodeLLaMA-Instruct-13B": "./experiments/CodeLLaMA_Prompting/preds.jsonl",
        "Magicoder-S-DS-6.7B": "./experiments/llm_outputs/Magicoder-S-DS-6.7B.jsonl",
        "Stable-Code-Instruct-3B": "./experiments/llm_outputs/Stable-Code-Instruct-3b.jsonl",
        "DeepSeekCoder-Instruct-6.7B": "./experiments/llm_outputs/DeepSeekCoder-6.7B-Instruct.jsonl",
        # "ChatGPT"
    }.items():
        if preds_path.endswith(".json"): preds = json.load(open(preds_path))
        elif preds_path.endswith(".jsonl"): preds = read_jsonl(preds_path)
        if model == "BM-25 kNN": preds = [rec[0] for rec in preds]
        elif model in ["LSTM", "CodeReviewer", 
                       "CodeLLaMA-Instruct-7B", 
                       "CodeLLaMA-Instruct-13B"]:
            preds = [rec['pred'] for rec in preds]
        else: preds = [process_magicoder_output(rec['pred_review']) for rec in preds]
        BERTSCORE = np.mean(bert_score.compute(predictions=preds, references=trues, model_type="distilbert-base-uncased")['f1'])
        BLEU = bleu_fromstr(predictions=preds, golds=trues, rmstop=False)
        BLEU_WITHOUT_STOP = bleu_fromstr(predictions=preds, golds=trues, rmstop=True)
        EDIT_DISTANCE = np.mean([lev(p, r)/max(len(p), len(r)) for p,r in zip(preds, trues)])
        ROUGE_L = np.mean([rouge_score_.score(p,r)['rougeL'].fmeasure for p,r in zip(preds, trues)])
        print(f"\x1b[34;1m{model}\x1b[0m: BLEU: {BLEU:.2f} BLEU without stop: {BLEU_WITHOUT_STOP:.2f} Edit Distance: {EDIT_DISTANCE:.3f} BERTScore: {BERTSCORE:.3f} ROUGE-L: {ROUGE_L:.3f}")
        # bleu_score.compute(predictions=preds, references=trues) 
