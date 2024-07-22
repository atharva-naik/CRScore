# compute similarity score threshold for STS computation.
import os
import nltk
import torch
import random
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch.nn.functional as F
from nltk.corpus import stopwords
from src.datautils import read_jsonl
from sentence_transformers import SentenceTransformer, util

nltk.download("english")
stop = set(stopwords.words("english")+[".",","])
print(stop)

def sbert_filt_encode(sbert, claims, show_progress_bar: bool=False, batch_size: int=128):
    token_vecs = sbert.encode(
        claims, output_value="token_embeddings", 
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
    )
    filt_pooled_vecs = []
    for i,token_vecs in tqdm(
        enumerate(token_vecs), 
        total=len(token_vecs)
    ):
        claim = claims[0]            # vec_j = sbert_filt_encode(sbert, , show_progress_bar=)
        tokens = sbert.tokenizer.batch_decode(sbert.tokenize([claim])['input_ids'][0])
        filt_token_vecs = torch.stack([vec for tok,vec in zip(tokens, token_vecs) if tok not in stop])
        pooled_filt_sent_vec = torch.sum(filt_token_vecs, 0) / len(filt_token_vecs)
        pooled_normed_filt_sent_vec = F.normalize(pooled_filt_sent_vec.unsqueeze(0), p=2, dim=1).squeeze()
        filt_pooled_vecs.append(pooled_normed_filt_sent_vec.tolist())

    return filt_pooled_vecs

def find_quartiles(data):
    # Sort the data in ascending order
    sorted_data = np.sort(data)
    # Find the median (Q2)
    median = np.median(sorted_data)
    # Find the indices for Q1 and Q3
    n = len(sorted_data)
    index_q1 = int(np.floor((n + 1) / 4)) - 1  # Q1 is at the 25th percentile
    index_q3 = int(np.floor(3 * (n + 1) / 4)) - 1  # Q3 is at the 75th percentile
    # Calculate Q1 and Q3
    q1 = sorted_data[index_q1]
    q3 = sorted_data[index_q3]
    
    return median, q1, q3

def plot_value_dist(values):
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Histogram of Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig("plots/STS_score_values_dist.png")

    # # Q-Q plot to check normality
    # plt.figure(figsize=(10, 6))
    # stats.probplot(values, dist="norm", plot=plt)
    # plt.title('Q-Q Plot')
    # plt.savefig("plots/STS_score_values_QQ.png")

# main
if __name__ == "__main__":
    test_data = read_jsonl("data/Comment_Generation/msg-test.jsonl")#[:100]
    N = len(test_data)
    sbert = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    all_vecs = sbert_filt_encode(sbert, [rec['msg'] for rec in test_data], show_progress_bar=True, batch_size=1024)
    # sim_scores = []
    sim_score_values = []
    JUMP = 10000
    for i in tqdm(range(0, N, JUMP)):
        vec_is = torch.as_tensor(all_vecs[i:min(i+JUMP,N)]) # I (<=500) x emb_dim 
        # for j in range(i+1, min(i+501, N)):
        vec_js = torch.as_tensor(all_vecs[i+1:min(i+JUMP,N)]) # J (<=500) x emb_dim
        scores = util.cos_sim(vec_is, vec_js).view(-1).tolist() # flatten IxJ array to a list of size IJ
        sim_score_values.extend(scores)
        # sim_scores.append((i,j,score))
    # sim_score_values = np.array([s for _,_,s in sim_scores])
    print(len(sim_score_values))

    median, q1, q3 = find_quartiles(sim_score_values)
    mean_value = np.mean(sim_score_values)
    max_value = np.max(sim_score_values)
    min_value = np.min(sim_score_values)
    std_dev = np.std(sim_score_values)
    
    u = mean_value
    s = std_dev
    z_scores = [(x/100, (x/100-u)/s) for x in range(60, 100, 5)]
    print(f"centrality: Q1: {q1:.3f} Q2: {median:.3f} Q3: {q3:.3f} mean: {mean_value:.3f}")
    print(f"spread: std_dev: {std_dev:.3f}, max: {max_value:.3f}, min: {min_value:.3f}")

    random_values = random.sample(sim_score_values, k=100000)
    plot_value_dist(sim_score_values)