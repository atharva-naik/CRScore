# compute the correctness score by detecting contradictions in generated reviews.
import os
import re
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from src.datautils import generate_before_after_code_from_patch, read_jsonl

def get_embedded_code(review_text: str):
    pattern = r'`([^`]+)`'
    matches = re.findall(pattern, review_text)

    return matches if matches is not None else []

def get_review_template(review_text: str):
    pattern = r'`([^`]+)`'
    # Replace embedded code with an empty string
    output_text = re.sub(pattern, '', review_text)

    return output_text

# file = open("TEMP.txt", "w")
TEMPLATES = defaultdict(lambda: 0)
def heuristic_based_correctness_check(code_change, review):
    before_code, after_code = generate_before_after_code_from_patch(code_change)
    emb_codes = get_embedded_code(review)
    review_template = get_review_template(review)
    TEMPLATES[review_template.lower()] += 1
    if (("->" in review_template.lower() or "can be replaced with" in review_template.lower()
        or "is equivalent to" in review_template.lower() or "instead of" in review_template.lower()) 
        and len(emb_codes) == 2 and emb_codes[0] == emb_codes[1]):
        return 0
    if (("shouldn't this be" in review_template.lower() or
         "why not just" in review_template.lower() or
         "why not" in review_template.lower() or
         "you can use" in review_template.lower())
        and len(emb_codes) == 1 and emb_codes[0] in after_code):
        return 0
    if (("should this be" in review_template.lower() or 
         "this should be" in review_template.lower())
        and len(emb_codes) == 1 and emb_codes[0] not in after_code):
        return 0
    if (("why did you remove" in review_template.lower())
        and len(emb_codes) == 1 and not(emb_codes[0] in before_code 
        and emb_codes[0] not in after_code)):
        return 0
    return 1

# main
if __name__ == "__main__":
    test_data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    correctness_score = []
    with open("ckpts/gen_study_inf/checkpoints-1800-5.72/preds.txt", "r") as f:
        i = 0
        for line in tqdm(f):
            code_change = test_data[i]['patch']
            review = line.strip()
            score = heuristic_based_correctness_check(code_change, review)
            # if score == 0: print(review)
            i += 1
            correctness_score.append(score)
    print(np.mean(correctness_score))
    TEMPLATES = dict(sorted(TEMPLATES.items(), reverse=True, key=lambda x: x[1]))
    with open("TEMPLATES.json", "w") as f:
        json.dump(TEMPLATES, f, indent=4)