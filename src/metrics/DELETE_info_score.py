# import os
# import json
# import torch
# import numpy as np
# from typing import *
# from sentence_transformers import util, SentenceTransformer

# class InfoScorer:
#     def __init__(self, model_path: str="all-roberta-large-v1", batch_size: int=32):
#         self.sbert = SentenceTransformer(model_path)
#         self.generic_reference_reviews = [
#             "why do we need this?", 
#             "why is this needed?",
#             "why this change?",
#             "why was this removed?", 
#             "why do we need this change?",
#             "why did you change this?",
#             "why is this change needed?",
#             "why do we need to change this?",
#             "why did you remove this?",
#             "why is this removed?",
#             'why do we need this method?',
#             'why is this being removed?',
#             'why do we need this interface?',
#             'why do we need this?',
#             'why did this change?',
#             'why do we need this check?',
#             'why do we need this import?',
#             'why do we need this class?'
#         ]
#         self.batch_size = batch_size
#         self.generic_reference_review_embs = self.encode(self.generic_reference_reviews, batch_size=self.batch_size)

#     def compute(self, predictions: List[str], batch_size: int=64):
#         review_embs = self.encode(predictions, batch_size=batch_size)
#         inst_scores = (1-util.cos_sim(review_embs, self.generic_reference_review_embs).max(axis=-1).values).cpu().tolist()
#         # inst_scores = util.cos_sim(review_embs, self.generic_reference_review_embs).mean(axis=-1).cpu().tolist()
#         return {"inst_scores": inst_scores, "score": np.mean(inst_scores)}

#     def encode(self, reviews: List[str], batch_size: int=32):
#         embs = self.sbert.encode(reviews, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
#         if torch.cuda.is_available():
#             return embs.cuda()
#         return embs

# def run_info_scorer():
#     info_score = InfoScorer()
#     import pandas as pd
#     df = pd.read_csv('human_study_data.csv')
#     corr_scores = []
#     codes = list(df['patch'])
#     model_wise_scores = {}
#     for ref_subset, model_name in [
#         ('msg', 'ground_truth'),
#         ('knn_pred', 'knn'),
#         ('lstm_pred', 'lstm'),
#         ('magicoder_pred', 'magicoder'),
#         ('codereviewer_pred', 'codereviewer')
#     ]:
#         predictions = list(df[ref_subset])
#         score = info_score.compute(predictions=predictions)
#         model_wise_scores[model_name] = score['inst_scores']
#         print(model_name, score['score'])
#     indices = list(df['index'])
#     for i in range(len(indices)):
#         corr_scores.append({
#             "index": indices[i],
#             "ground_truth": model_wise_scores['ground_truth'][i],
#             "knn": model_wise_scores['knn'][i],
#             "lstm": model_wise_scores['lstm'][i],
#             "magicoder": model_wise_scores['magicoder'][i],
#             "codereviewer": model_wise_scores['codereviewer'][i]
#         })
#     with open("./human_study_info_scores.json", "w") as f:
#         json.dump(corr_scores, f, indent=4) 

# # main
# if __name__ == "__main__":
#     run_info_scorer()