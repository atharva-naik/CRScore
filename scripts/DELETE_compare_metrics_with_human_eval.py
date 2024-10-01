# compare all the metrics with human binary annotations:
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import pointbiserialr

def flatten_list(t: list):
    flat_list = [item for sublist in t for item in sublist]
    return flat_list
    

# main
if __name__ == "__main__":
    correctness_scores = {rec['index']: rec for rec in json.load(open("human_study_correctness_scores.json"))}
    info_scores = {rec['index']: rec for rec in json.load(open("human_study_info_scores.json"))}
    relevance_scores = {rec['index']: rec for rec in json.load(open("human_study_relevance_scores.json"))}
    bleu_scores = {rec['index']: rec for rec in json.load(open("human_study_bleu_scores.json"))}
    rouge_l_scores = {rec['index']: rec for rec in json.load(open("human_study_rouge_l_scores.json"))}
    editdistance_scores = {rec['index']: rec for rec in json.load(open("human_study_editdistance_scores.json"))}
    bert_scores = {rec['index']: rec for rec in json.load(open("human_study_bert_scores.json"))}

    human_eval = pd.read_excel("pilot_annotations_atharva.xlsx")
    model_names = json.load(open("model_names_key.json"))
    k = len(model_names)
    assert k == len(human_eval)
    
    metric_names = ['relevance', 'informativeness', 'correctness']
    automatic_metric_values = {
        "correctness": correctness_scores,
        "informativeness": info_scores,
        "relevance": relevance_scores,
        "bleu": bleu_scores,
        "rouge-l": rouge_l_scores,
        "bertscore": bert_scores,
        "editdistance": editdistance_scores,
    }
    metric_eval_dict = {metric_name: defaultdict(lambda: []) for metric_name in metric_names}
    
    for i in range(k):
        model_name = model_names[i]
        index = human_eval["index"][i]
        if model_name == "ground_truth": continue
        for mname in metric_names:
            mvalue = human_eval[mname][i]
            if np.isnan(mvalue): continue
            assert mvalue == 1 or mvalue == 0, f"non binary metric value: {mvalue}"
            automatic_metric_value = automatic_metric_values[mname][index][model_name]
            if mname == "correctness": # because greater the correctness distance, the worse the model is
                automatic_metric_value = -automatic_metric_value
            metric_eval_dict[mname][model_name].append({
                "automated_score": automatic_metric_value,
                "bleu": automatic_metric_values['bleu'][index][model_name],
                "rouge-l": automatic_metric_values['rouge-l'][index][model_name],
                "editdistance": automatic_metric_values['editdistance'][index][model_name],
                "bertscore": automatic_metric_values['bertscore'][index][model_name],
                "human_eval_score": mvalue,
            })
    for metric, model_wise_scores in metric_eval_dict.items():
        bleu = flatten_list([[i['bleu'] for i in v] for v in model_wise_scores.values()]) 
        rouge_l = flatten_list([[i['rouge-l'] for i in v] for v in model_wise_scores.values()]) 
        bertscore = flatten_list([[i['bertscore'] for i in v] for v in model_wise_scores.values()])
        editdistance = flatten_list([[-i['editdistance'] for i in v] for v in model_wise_scores.values()]) 
        automated_scores = flatten_list([[i['automated_score'] for i in v] for v in model_wise_scores.values()]) # continuous variable
        human_eval_scores = flatten_list([[i['human_eval_score'] for i in v] for v in model_wise_scores.values()]) # boolean variable
        # print(automated_scores)
        model_ranking = dict(sorted([(k, np.mean([i['automated_score'] for i in v])) for k,v in model_wise_scores.items()], reverse=False, key=lambda x: x[1]))
        print(model_ranking)
        if metric == "relevance":
            with open("rel_scores_DEBUG_DELETE.txt", "w") as f:
                for rel_score, rel_annot in zip(automated_scores, human_eval_scores):
                    f.write(f"{rel_score} {rel_annot}"+"\n")    
        R, p_val = pointbiserialr(human_eval_scores, automated_scores)
        print(f"\x1b[34;1m#{metric} vs ours\x1b[0m: PB R: {R:.3f}, p: {p_val:.3f}")
        R, p_val = pointbiserialr(human_eval_scores, bertscore)
        print(f"\x1b[34;1m#{metric} vs BertScore \x1b[0m: PB R: {R:.3f}, p: {p_val:.3f}")
        R, p_val = pointbiserialr(human_eval_scores, bleu)
        print(f"\x1b[34;1m#{metric} vs BLEU \x1b[0m: PB R: {R:.3f}, p: {p_val:.3f}")
        R, p_val = pointbiserialr(human_eval_scores, rouge_l)
        print(f"\x1b[34;1m#{metric} vs ROUGE-L \x1b[0m: PB R: {R:.3f}, p: {p_val:.3f}")
        R, p_val = pointbiserialr(human_eval_scores, editdistance)
        print(f"\x1b[34;1m#{metric} vs Edit Distance \x1b[0m: PB R: {R:.3f}, p: {p_val:.3f}")
    #         model_he_scores[model_name][mname][0] += int(mvalue)
    #         model_he_scores[model_name][mname][1] += 1
    # for model_name in model_he_scores:
    #     for metric_name in model_he_scores[model_name]:
    #         model_he_scores[model_name][metric_name][0] = (model_he_scores[model_name][metric_name][0]/model_he_scores[model_name][metric_name][1])
    # for metric_name in metric_names:
    #     metric_leaderboard = dict(sorted({model_name: model_he_scores[model_name][metric_name][0] for model_name in model_he_scores}.items(), key=lambda x: x[1], reverse=True))
    #     num_samples = {model_name: model_he_scores[model_name][metric_name][1] for model_name in model_he_scores}
    #     print(f"\x1b[34;1m{metric_name}\x1b[0m")
    #     for k,v in metric_leaderboard.items():
    #         print(f"{k}: {100*v:.2f}")