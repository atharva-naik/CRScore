import evaluate
import numpy as np
import pandas as pd
from typing import *
import statsmodels.api as sm
from scipy.stats import spearmanr
from collections import defaultdict
from rouge_score import rouge_scorer
from scipy.stats import pointbiserialr
from statsmodels.formula.api import ols
from Levenshtein import distance as lev
from src.metrics.DELETE_cr_score import CRScorer

# def create_histogram(values: Union[List[float], np.ndarray], nbins=5):
#     maxv = np.max(values)
#     minv = np.min(values)
#     step = (maxv-minv)/nbins
#     ranges = np.array([minv+step*i for i in range(nbins+1)])
#     counts = np.zeros(nbins)
#     for v in values: 
#         for i in range(nbins):
#             if ranges[i] <= v <= ranges[i-1]:
#                 counts[i] += 1
#                 break    

#     return ranges, counts

def compute_buckets_of_x_by_y(x, y):
    buckets = defaultdict(lambda: [])
    for x_i, y_i in zip(x, y):
        buckets[y_i].append(x_i)
    ranges = [(np.min(vals), np.max(vals)) for _, vals in sorted(buckets.items(), key=lambda x: x[0], reverse=False)]
    # mins = [r[0] for r in ranges]
    # maxs = [r[1] for r in ranges]  
    means = [np.mean(vals) for _, vals in sorted(buckets.items(), key=lambda x: x[0], reverse=False)]

    return ranges, means

# main
if __name__ == "__main__":
    df = pd.read_csv('cr_manual_rel_annot_likert_scale.csv')
    k = 100
    predictions = list(df['pred'])[:k]
    references = list(df['msg'])[:k]
    codes = list(df['patch'])[:k]
    # human eval scores for reference.
    likert_scores = np.array(list(df['rel_score_pred'])[:k])
    # code changes and review. relevance score.
    cr_score = CRScorer(checkpoint_path="./ckpts/crr_rcr_ccr_0.005/best_model.pth")
    # bleu score.
    bleu_score = evaluate.load("bleu")
    # BERT score.
    bert_score = evaluate.load("bertscore")
    # ROUGE-L score.
    rouge_score_ = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # compute CR score.
    inst_cr_scores = np.array(cr_score.compute(predictions=predictions, codes=codes)["inst_scores"])
    # compute BLEU score.
    inst_bleu_scores = [bleu_score.compute(predictions=[p], references=[r]) for p,r in zip(predictions, references)]
    inst_bleu_scores = np.array([b["bleu"] for b in inst_bleu_scores])
    # compute BERT score.
    inst_bert_scores = np.array(bert_score.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")['f1'])
    # compute ROUGE-L score.
    inst_rouge_scores = np.array([rouge_score_.score(p,r)['rougeL'].fmeasure for p,r in zip(predictions, references)])
    # print(inst_rouge_scores)
    inst_editd_scores = np.array([lev(p, r) for p,r in zip(predictions, references)])

    pb_corr, pb_p_value = pointbiserialr((likert_scores>2).astype(int), inst_cr_scores)
    print(f"Point-Biserial Correlation for CRScore: {pb_corr:.3f}, p-value: {pb_p_value:0.3e}")
    pb_corr, pb_p_value = pointbiserialr((likert_scores>2).astype(int), inst_bleu_scores)
    print(f"Point-Biserial Correlation for BLEU: {pb_corr:.3f}, p-value: {pb_p_value:0.3e}")
    pb_corr, pb_p_value = pointbiserialr((likert_scores>2).astype(int), inst_rouge_scores)
    print(f"Point-Biserial Correlation for ROUGE-L: {pb_corr:.3f}, p-value: {pb_p_value:0.3e}")
    pb_corr, pb_p_value = pointbiserialr((likert_scores>2).astype(int), inst_bert_scores)
    print(f"Point-Biserial Correlation for BERTScore: {pb_corr:.3f}, p-value: {pb_p_value:0.3e}")
    pb_corr, pb_p_value = pointbiserialr((likert_scores>2).astype(int), inst_editd_scores)
    print(f"Point-Biserial Correlation for Edit Distance: {pb_corr:.3f}, p-value: {pb_p_value:0.3e}")
    corr, p_value = spearmanr(inst_cr_scores, likert_scores)    
    print(f"Spearman Correlation for CRScore: {corr:.3f}, p-value: {p_value:0.3e}")
    corr, p_value = spearmanr(inst_bleu_scores, likert_scores)
    print(f"Spearman Correlation for BLEU: {corr:.3f}, p-value: {p_value:0.3e}")
    corr, p_value = spearmanr(inst_rouge_scores, likert_scores)
    print(f"Spearman Correlation for ROUGE-L: {corr:.3f}, p-value: {p_value:0.3e}")
    corr, p_value = spearmanr(inst_bert_scores, likert_scores)
    print(f"Spearman Correlation for BERTScore: {corr:.3f}, p-value: {p_value:0.3e}")
    corr, p_value = spearmanr(inst_editd_scores, likert_scores)
    print(f"Spearman Correlation for Edit Distance: {corr:.3f}, p-value: {p_value:0.3e}")

    likert_dist = np.unique(likert_scores, return_counts=True)
    print(likert_dist)
    # bleu_dist = create_histogram(inst_bleu_scores)
    print("BLEU score:")
    print(compute_buckets_of_x_by_y(inst_bleu_scores, likert_scores))
    print("CR score:")
    # cr_dist = create_histogram(inst_cr_scores)
    print(compute_buckets_of_x_by_y(inst_cr_scores, likert_scores))

    bleu_df = pd.DataFrame({
        "continuous": inst_bleu_scores,
        "categorical": likert_scores,
    })
    
    model = ols('continuous ~ C(categorical)', data=bleu_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    r_squared = anova_table['sum_sq']['C(categorical)'] / (anova_table['sum_sq']['C(categorical)'] + anova_table['sum_sq']['Residual'])
    print(f"R-squared value ANOVA BLEU: {r_squared:.3f}")

    cr_df = pd.DataFrame({
        "continuous": inst_cr_scores,
        "categorical": likert_scores,
    })
    
    model = ols('continuous ~ C(categorical)', data=cr_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    r_squared = anova_table['sum_sq']['C(categorical)'] / (anova_table['sum_sq']['C(categorical)'] + anova_table['sum_sq']['Residual'])
    print(f"R-squared value ANOVA CR: {r_squared:.3f}")

    rouge_df = pd.DataFrame({
        "continuous": inst_rouge_scores,
        "categorical": likert_scores,
    })
    
    model = ols('continuous ~ C(categorical)', data=rouge_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    r_squared = anova_table['sum_sq']['C(categorical)'] / (anova_table['sum_sq']['C(categorical)'] + anova_table['sum_sq']['Residual'])
    print(f"R-squared value ANOVA CR: {r_squared:.3f}")

    bert_df = pd.DataFrame({
        "continuous": inst_bert_scores,
        "categorical": likert_scores,
    })
    
    model = ols('continuous ~ C(categorical)', data=bert_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    r_squared = anova_table['sum_sq']['C(categorical)'] / (anova_table['sum_sq']['C(categorical)'] + anova_table['sum_sq']['Residual'])
    print(f"R-squared value ANOVA CR: {r_squared:.3f}")

    editd_df = pd.DataFrame({
        "continuous": inst_editd_scores,
        "categorical": likert_scores,
    })
    
    model = ols('continuous ~ C(categorical)', data=editd_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    r_squared = anova_table['sum_sq']['C(categorical)'] / (anova_table['sum_sq']['C(categorical)'] + anova_table['sum_sq']['Residual'])
    print(f"R-squared value ANOVA CR: {r_squared:.3f}")