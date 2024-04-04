import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Generating sample data
num_models = 5
num_samples = 1000
# data = np.random.rand(num_samples, num_models)

rel_scores = json.load(open("all_model_rel_scores.json"))
P, R, F = {}, {}, {}
for model, metrics in rel_scores.items(): 
    P[model] = metrics['P']
    R[model] = metrics['R']
    F[model] = metrics['F']

def plot_histogram(data, path, title):
    # Create bins for the histogram
    plt.clf()
    bins = np.linspace(0, 1, 11)
    
    # Plot histograms in separate subplots
    fig, axes = plt.subplots(nrows=1, ncols=num_models, figsize=(15, 5), sharey=True)
    fig.suptitle(title)

    # bar colors.
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    # Iterate over each array and plot its histogram
    for i, model in enumerate(data):
        ax = axes[i]
        ax.hist(
            data[model], bins=bins, color=colors[i],
            alpha=0.7, label=model.title(), width=0.08,
        )
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plt.savefig(path)

# def plot_histogram(data, path, title):
#     plt.clf()
#     # Creating histogram
#     bins = np.arange(0, 1.1, 0.1)
#     bar_width = 0.02
#     # plt.figure(figsize=(10, 6))

#     for i, model in enumerate(data):
#         hist, _ = np.histogram(data[model], bins=bins)
#         plt.bar(
#             bins[:-1] + (i * bar_width), 
#             hist, width=bar_width, 
#             alpha=0.5, label=model.title()
#         )

#     plt.xlabel('Score')
#     plt.ylabel('Frequency')
#     plt.title(title)
#     # plt.xticks(bins)
#     plt.xticks(
#         bins[:-1] + bar_width * (num_models - 1) / 2, 
#         [f'{x:.1f}-{x+.1:.1f}' for x in bins[:-1]], 
#         rotation=90
#     )
#     plt.legend()
#     plt.tight_layout()

#     # # Adding values on top of bars
#     # for i in range(len(bins) - 1):
#     #     for j in range(num_models):
#     #         count = np.sum((data[:, j] >= bins[i]) & (data[:, j] < bins[i+1]))
#     #         plt.text(bins[i] + 0.05 + j * bar_width, count + 10, str(count), ha='center')
#     plt.savefig(path)

plot_histogram(P, "plots/rel_score_P_dist.png", 'Histogram of Conciseness Scores')
plot_histogram(R, "plots/rel_score_R_dist.png", 'Histogram of Comprehensiveness Scores')
plot_histogram(F, "plots/rel_score_F_dist.png", 'Histogram of Relevance Scores')