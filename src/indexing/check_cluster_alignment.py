# script to load clusters dumped based on code changes and reviews and then aligned based on index overlaps.
import json
import numpy as np

# main
if __name__ == "__main__":
    code_changes = json.load(open("./data/Comment_Generation/dev_test_code_change_clusters.json"))
    reviews = json.load(open("./data/Comment_Generation/dev_test_review_clusters.json"))
    dice_coeff_matrix = np.zeros((len(code_changes), len(reviews)))
    for i in range(len(code_changes)):
        for j in range(len(reviews)):
            X = set(code_changes[i])
            Y = set(reviews[j])
            X_n_Y = len(X.intersection(Y))
            dice_coeff_matrix[i][j] = (2*X_n_Y)/(len(X)+len(Y))
    many_to_one_optimal_alignment = []
    # for each cluster of code changes, find the most similar cluster of review changes (with replacement).
    alignment_score = 0
    for i in range(len(code_changes)):
        ind = np.argmax(dice_coeff_matrix[i])
        # print(dice_coeff_matrix[i][ind])
        many_to_one_optimal_alignment.append((ind, dice_coeff_matrix[i][ind]))
        alignment_score += dice_coeff_matrix[i][ind]
    alignment_score /= len(code_changes)
    print(f"score = {alignment_score:.3f}")