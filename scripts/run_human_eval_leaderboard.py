# use human evaluation data to create a leaderboard of models.
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# main
if __name__ == '__main__':
    human_eval = pd.read_excel("pilot_annotations_atharva.xlsx")
    model_names = json.load(open("model_names_key.json"))
    k = len(model_names)
    assert k == len(human_eval)
    metric_names = ['relevance', 'informativeness', 'correctness']
    model_scores = defaultdict(lambda: {mname: [0,0] for mname in metric_names})
    for i in range(k):
        model_name = model_names[i]
        for mname in metric_names:
            mvalue = human_eval[mname][i]
            if np.isnan(mvalue): continue
            assert mvalue == 1 or mvalue == 0, f"non binary metric value: {mvalue}"
            model_scores[model_name][mname][0] += int(mvalue)
            model_scores[model_name][mname][1] += 1
    for model_name in model_scores:
        for metric_name in model_scores[model_name]:
            model_scores[model_name][metric_name][0] = (model_scores[model_name][metric_name][0]/model_scores[model_name][metric_name][1])
    for metric_name in metric_names:
        metric_leaderboard = dict(sorted({model_name: model_scores[model_name][metric_name][0] for model_name in model_scores}.items(), key=lambda x: x[1], reverse=True))
        num_samples = {model_name: model_scores[model_name][metric_name][1] for model_name in model_scores}
        print(f"\x1b[34;1m{metric_name}\x1b[0m")
        for k,v in metric_leaderboard.items():
            print(f"{k}: {100*v:.2f} using {num_samples[k]} judgements")
         