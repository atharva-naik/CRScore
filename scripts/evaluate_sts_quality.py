import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import pointbiserialr, spearmanr, kendalltau
from sentence_transformers import SentenceTransformer, util


def sbert_filt_encode(sbert, stop, claims, show_progress_bar: bool=False):
    token_vecs = sbert.encode(claims, output_value="token_embeddings", show_progress_bar=show_progress_bar)
    filt_pooled_vecs = []
    for i,token_vecs in enumerate(token_vecs):
        claim = claims[i]
        tokens = sbert.tokenizer.batch_decode(sbert.tokenize([claim])['input_ids'][0])
        # print([token for token in tokens if token not in stop]) # TODO: DEBUG
        filt_token_vecs = torch.stack([vec for tok,vec in zip(tokens, token_vecs) if tok not in stop])
        pooled_filt_sent_vec = torch.sum(filt_token_vecs, 0) / len(filt_token_vecs)
        pooled_normed_filt_sent_vec = F.normalize(pooled_filt_sent_vec.unsqueeze(0), p=2, dim=1).squeeze()
        filt_pooled_vecs.append(pooled_normed_filt_sent_vec.tolist())

    return filt_pooled_vecs

def load_claims_addressed_data():
    claims_addressed = []
    marcus_annot_end_points = {"py": 501-2, "java": 501-2, "js": 505-2}
    index_to_lang = {}
    index_to_claims = defaultdict(lambda: [])
    langs = list(marcus_annot_end_points.keys())
    for lang in langs:
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


            if str(rec['claim']) != "nan":
                index_to_claims[index].append(rec['claim'])
            
            if str(rec["Rel (F)"]) != "nan" and system != "msg": # skip CodeReviewer ground truth/references among the evaluated systems, because we don't count it for the correlations as reference based metrics would default to 1 on them and disadvantage their correlation values.
                try: claims_addressed_annot = [int(v) for v in rec['claims addressed'].split(",")]
                except (ValueError, AttributeError): claims_addressed_annot = []
                claims_addressed.append({"index": index, "system": system, "diff": diff, "review": rec["review"], 'claims_addressed': claims_addressed_annot, 'lang': lang})
            
    return claims_addressed, index_to_claims

# main
if __name__ == "__main__":
    stop = set(stopwords.words("english")+[".",","])
    model_path="mixedbread-ai/mxbai-embed-large-v1"
    sbert = SentenceTransformer(model_path)
    claims_addressed, index_to_claims = load_claims_addressed_data()
    claim_and_review_pos_pairs = []
    claim_and_review_neg_pairs = []
    # mislabeled_cases = 0
    for rec in tqdm(claims_addressed):
        ind = rec['index']
        claims_covered = rec['claims_addressed']
        review = rec['review']
        if len(claims_covered) > 0:
            claims = index_to_claims[ind]
            for claim_index, claim in enumerate(claims):
                if claim_index+1 in claims_covered:
                    claim_and_review_pos_pairs.append((claim, review))
                else: claim_and_review_neg_pairs.append((claim, review))
            # for claim_index in claims_covered:
            #     try: 
            #         claim_and_review_pos_pairs.append((rec['review'], claims[claim_index-1]))
            #         print(ind, rec['lang'])
            #         print(claims)
            #         print(claims_covered)
            #         exit()
            #     except IndexError:
            #         print(ind, rec['lang'])
            #         print(claims)
            #         print(claims_covered)
            #         mislabeled_cases += 1
    # print(len(claim_and_review_pos_pairs))
    # print(len(claim_and_review_neg_pairs))

    human_label = []
    sts_score = []

    claims, reviews = [], []
    for claim, review in claim_and_review_pos_pairs:
        claims.append(claim)
        reviews.append(review)
    claim_encs = sbert_filt_encode(sbert=sbert, stop=stop, show_progress_bar=True, claims=claims)
    review_encs = sbert_filt_encode(sbert=sbert, stop=stop, show_progress_bar=True, claims=reviews)
    score = util.cos_sim(claim_encs, review_encs)
    sts_score.extend(score.diag().cpu().tolist())
    human_label.extend([1 for _ in range(len(claim_and_review_pos_pairs))])

    # sts_score_binarized = [int(s > thresh) for s in sts_score]

    # print(len(claim_and_review_pos_pairs))
    # print(len(claim_and_review_neg_pairs))
    # print(np.mean([int(p == t) for p,t in zip(sts_score_binarized, human_label)]))

    # human_label = []
    # sts_score = []

    claims, reviews = [], []
    for claim, review in claim_and_review_neg_pairs:
        claims.append(claim)
        reviews.append(review)
    claim_encs = sbert_filt_encode(sbert=sbert, stop=stop, show_progress_bar=True, claims=claims)
    review_encs = sbert_filt_encode(sbert=sbert, stop=stop, show_progress_bar=True, claims=reviews)
    score = util.cos_sim(claim_encs, review_encs)
    sts_score.extend(score.diag().cpu().tolist())
    human_label.extend([0 for _ in range(len(claim_and_review_neg_pairs))])

    print(len(claim_and_review_pos_pairs))
    print(len(claim_and_review_neg_pairs))

    thresholds: list[float] = [0.6, 0.65, 0.6576, 0.7, 0.7314, 0.75, 0.8]
    for thresh in thresholds:
        sts_score_binarized = [int(s > thresh) for s in sts_score]
        # print(sts_score_binarized)
        P = precision_score(human_label, sts_score_binarized)
        R = recall_score(human_label, sts_score_binarized)
        F1 = f1_score(human_label, sts_score_binarized)
        
        print(f"P: {P:.4f} R: {R:.4f} F: {F1:.4f}")
    # print(np.mean([int(p == t) for p,t in zip(sts_score_binarized, human_label)]))