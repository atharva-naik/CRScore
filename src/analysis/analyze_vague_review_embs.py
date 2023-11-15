import json
import numpy as np
import pandas as pd

k = 100
# main
if __name__ == "__main__":
    likert_scores = np.array(pd.read_csv("./cr_manual_rel_annot_likert_scale.csv")['rel_score'][:k])
    trained_model_rel_scores = json.load(open("./ckpts/crr_rcr_ccr_0.005/test_preds.json"))
    untrained_model_rel_scores = json.load(open("./ckpts/crr_rcr_ccr_zero_shot/test_preds.json"))
    t_scores = []
    u_scores = []
    score_gains = []
    for i in range(k):
        if likert_scores[i] > 0:
            t_scores.append(trained_model_rel_scores[i])
            u_scores.append(untrained_model_rel_scores[i])
            score_gains.append(trained_model_rel_scores[i]-untrained_model_rel_scores[i])
    print(len(t_scores))
    print(f"avg. untrained model scores: {np.mean(u_scores):.3f}")
    print(f"avg. trained model scores: {np.mean(t_scores):.3f}")
    print(f"avg. score gains: {np.mean(score_gains):.3f}")