# analyze agreement between annotations for pilot study.
from itertools import permutations, product
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score

score_based_rankings = [">".join([str(rank) for rank in ranking]) for ranking in 
[[2, 4, 5, 1, 3],
 [2, 5, 1, 4, 3],
 [5, 1, 2, 3, 4],
 [1, 4, 3, 2, 5],
 [4, 2, 5, 3, 1],
 [2, 4, 1, 5, 3],
 [1, 4, 5, 3, 2],
 [5, 2, 1, 3, 4],
 [5, 2, 3, 1, 4],
 [2, 4, 3, 1, 5]]]


def generate_equivalent_rankings(ranking):
    # Parse the ranking into a list of elements
    elements = ranking.split('>')

    # Generate permutations of the elements
    permutations_list = [permutations(par_ords.split('=')) for par_ords in elements]
    # print(list(permutations_list))

    # Reassemble permutations into ranking format
    equivalent_rankings = []
    for sample in list(product(*permutations_list)):
        ranking = []
        for element in sample:
            ranking += list(element)
        ranking = [int(i) for i in ranking]
        # convert rankings to scores.
        # scores = [0 for _ in range(len(ranking))]
        # for i,element in enumerate(ranking):
        #     scores[element-1] = (len(ranking)-i)
        # equivalent_rankings.append(scores)
        equivalent_rankings.append(ranking)

    return equivalent_rankings

def convert_ranking_to_scores(ranking):
    # Split the ranking string into elements
    elements = ranking.split('>')
    # Initialize a dictionary to store scores
    scores = {}

    # Assign scores based on ranks
    for i, element in enumerate(elements):
        if '=' in element:
            tied_elements = element.split('=')
            for tied_element in tied_elements:
                if tied_element not in scores:
                    scores[tied_element.strip()] = 5 - i
        else:
            if element not in scores:
                scores[element.strip()] = 5 - i

    # Convert scores to a list
    score_list = [scores[str(i+1)] for i in range(5)]

    return score_list

# main
if __name__ == "__main__":
    paths = ["human_study/pilot/Pilot Study Annotations - Akshay Goindani.csv", "human_study/pilot/Pilot Study Annotations - Atharva Kulkarni.csv", "human_study/pilot/Pilot Study Annotations - Yash Mathur.csv"]
    annotators = []
    for path in paths:
        annotators.append(pd.read_csv(path))#.to_dict("records"))
    for i in range(len(paths)):
        for j in range(i):
            x_concise = annotators[i]["Conciseness"]
            y_concise = annotators[j]["Conciseness"]
            x_comp = annotators[i]["Comprehensiveness"]
            y_comp = annotators[j]["Comprehensiveness"]
            x_concise_collapsed = annotators[i]["Conciseness"].apply(lambda x: {1:1, 2:1, 3:2, 4:2, 5:3}[x])
            y_concise_collapsed = annotators[j]["Conciseness"].apply(lambda x: {1:1, 2:1, 3:2, 4:2, 5:3}[x])
            x_comp_collapsed = annotators[i]["Comprehensiveness"].apply(lambda x: {1:1, 2:1, 3:2, 4:2, 5:3}[x])
            y_comp_collapsed = annotators[j]["Comprehensiveness"].apply(lambda x: {1:1, 2:1, 3:2, 4:2, 5:3}[x])
            IAA_concise = cohen_kappa_score(x_concise, y_concise)
            IAA_concise_collapsed = cohen_kappa_score(x_concise_collapsed, y_concise_collapsed)
            IAA_comp = cohen_kappa_score(x_comp, y_comp)
            IAA_comp_collapsed = cohen_kappa_score(x_comp_collapsed, y_comp_collapsed)
            print(f"annotator-{j+1} & annotator-{i+1}: Conciseness: {IAA_concise:.3f} {IAA_concise_collapsed:.3f}  Comprehensiveness: {IAA_comp:.3f} {IAA_comp_collapsed:.3f}")
    for i in range(len(paths)):
        for j in range(i):
            x_ranks = annotators[i]["Overall Quality Ranking"].dropna()
            y_ranks = annotators[j]["Overall Quality Ranking"].dropna()
            max_spearman_rs = []
            # spearmanr_scores = []
            kendal_tau_scores = []
            for x_rl, y_rl in zip(x_ranks, y_ranks):
                # print("x_rl:", x_rl)
                x_scores = convert_ranking_to_scores(x_rl)
                y_scores = convert_ranking_to_scores(y_rl)
                # spearmanr_scores.append(spearmanr(x_scores, y_scores).statistic)
                kendal_tau_scores.append(kendalltau(x_scores, y_scores))

                x_rl = generate_equivalent_rankings(x_rl)
                y_rl = generate_equivalent_rankings(y_rl)

                max_spearman_r = 0
                for possible_x_rl in x_rl:
                    for possible_y_rl in y_rl:
                        # print(x_rl, y_rl)
                        max_spearman_r = max(max_spearman_r, spearmanr(possible_x_rl, possible_y_rl).statistic)
                # print(max_spearman_r)
                max_spearman_rs.append(max_spearman_r)
            mean_max_spearman_r = np.mean(max_spearman_rs)
            mean_kendall_tau = np.mean(kendal_tau_scores)
            # mean_spearmanr_with_ties = np.mean(spearmanr_scores)
            print(f"annotator-{j+1} & annotator-{i+1}: {mean_max_spearman_r:.3f}")
            print(f"annotator-{j+1} & annotator-{i+1}: Kendall Tau-b (with ties): {mean_kendall_tau:.3f}")

    for i in range(len(paths)):
        x_ranks = annotators[i]["Overall Quality Ranking"].dropna()
        y_ranks = score_based_rankings
        max_spearman_rs = []
        # spearmanr_scores = []
        kendal_tau_scores = []
        for x_rl, y_rl in zip(x_ranks, y_ranks):
            # print("x_rl:", x_rl)
            x_scores = convert_ranking_to_scores(x_rl)
            y_scores = convert_ranking_to_scores(y_rl)
            # spearmanr_scores.append(spearmanr(x_scores, y_scores).statistic)
            kendal_tau_scores.append(kendalltau(x_scores, y_scores))

            x_rl = generate_equivalent_rankings(x_rl)
            y_rl = generate_equivalent_rankings(y_rl)

            max_spearman_r = 0
            for possible_x_rl in x_rl:
                for possible_y_rl in y_rl:
                    # print(x_rl, y_rl)
                    max_spearman_r = max(max_spearman_r, spearmanr(possible_x_rl, possible_y_rl).statistic)
            # print(max_spearman_r)
            max_spearman_rs.append(max_spearman_r)
        mean_max_spearman_r = np.mean(max_spearman_rs)
        mean_kendall_tau = np.mean(kendal_tau_scores)
        # mean_spearmanr_with_ties = np.mean(spearmanr_scores)
        print(f"annotator-{i+1} score: {mean_max_spearman_r:.3f}")
        print(f"annotator-{i+1} & score: Kendall Tau-b (with ties): {mean_kendall_tau:.3f}")