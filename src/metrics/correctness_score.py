# compute the correctness score by detecting contradictions in generated reviews.
import os
import re
import json
import torch
import random
import numpy as np
from typing import *
from tqdm import tqdm
import torch.nn as nn
from fuzzywuzzy import fuzz
from collections import defaultdict
from transformers import AutoTokenizer
from src.models.correctness_codebert import CorrectnessCodeBERT, CRDataLoader, CRDataset
from src.datautils import generate_before_after_code_from_patch, read_jsonl, write_jsonl

class CorrectnessScorer:
    def __init__(self, checkpoint_path: str, model_path: str, code_enc_dim: int=768, review_enc_dim: int=768, use_simple_cc_network: bool=False, device: str="cuda:0"):
        self.model = CorrectnessCodeBERT(
            model_path=model_path,
            code_enc_dim=code_enc_dim,
            review_enc_dim=review_enc_dim,
            use_simple_cc_network=use_simple_cc_network
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_path = model_path
        self.device = device
        ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt_dict['model_state_dict'])
        self.model.to(device)
        self.pdist = nn.PairwiseDistance(p=2)

    def compute(self, codes: List[str], predictions: List[str], batch_size: int=64):
        data = []
        for i in range(len(codes)):
            before, after = generate_before_after_code_from_patch(codes[i])
            data.append({"anchor_before": before, "anchor_after": after, "pos": predictions[i], "neg": predictions[i]})
        dataset = CRDataset(data)
        dataloader = CRDataLoader(dataset, model_path=self.model_path, batch_size=batch_size, shuffle=False, tokenizer=self.tokenizer)
        self.model.eval()
        pbar = tqdm(enumerate(dataloader), 
                    total=len(dataloader))
        inst_scores = []
        with torch.no_grad():
            for step, batch in pbar:
                anchor_before = {k: v.to(self.device) for k,v in batch["anchor_before"].items()}
                anchor_after = {k: v.to(self.device) for k,v in batch["anchor_after"].items()}
                pos = {k: v.to(self.device) for k,v in batch["pos"].items()}
                neg = {k: v.to(self.device) for k,v in batch["neg"].items()}
                # Forward pass and Calculate contrastive loss
                anchor_emb, pos_emb, _, loss = self.model(
                    anchor_before=anchor_before, 
                    anchor_after=anchor_after, 
                    pos=pos, neg=neg,
                )
                batch_scores = self.pdist(anchor_emb, pos_emb).tolist()
                inst_scores.extend(batch_scores)
        
        return {"inst_scores": inst_scores, "score": np.mean(inst_scores)}        

def augment_data(path: str):
    data = read_jsonl(path)
    new_data = []
    for rec in tqdm(data):
        before_code, after_code = generate_before_after_code_from_patch(rec['patch'])
        lines_removed = []
        lines_added = []
        for line in rec['patch'].split("\n"):
            # skip empty lines
            line_content = line.strip('+').strip('-').strip()
            if line_content == "": continue
            if line.startswith("+"):
                lines_added.append(line[1:])
            elif line.startswith("-"):
                lines_removed.append(line[1:])
        # align some lines
        aligned_line_pairs = []
        for i in range(len(lines_added)):
            alignment_scores = []
            for j in range(len(lines_removed)):
                before = lines_removed[j]
                after = lines_added[i]
                alignment_scores.append(fuzz.partial_token_set_ratio(before, after))
            if len(alignment_scores) > 0:
                ind = np.argmax(alignment_scores)
                if alignment_scores[ind] >= 90:
                    before = lines_removed[ind]
                    after = lines_added[i]
                    aligned_line_pairs.append((before, after))
        for line in lines_removed:
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Shouldn't this be `{line}`?", 
                "neg": f"Should this be `{line}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Shouldn't it be `{line}`?", 
                "neg": f"Should it be `{line}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"This shouldn't be `{line}`?", 
                "neg": f"This should be `{line}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Why not use `{line}`?", 
                "neg": f"Why use `{line}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Why not `{line}`?", 
                "neg": f"Why `{line}`?"
            })
        for line in lines_added:
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Should this be `{line}`?", 
                "neg": f"Shouldn't this be `{line}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Should it be `{line}`?", 
                "neg": f"Shouldn't it be `{line}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"This should be `{line}`?", 
                "neg": f"This shouldn't be `{line}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Why use `{line}`?", 
                "neg": f"Why not use `{line}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Why `{line}`?", 
                "neg": f"Why not `{line}`?"
            })
        for before, after in aligned_line_pairs:
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Why did you remove `{before}`?", 
                "neg": f"Why did you remove `{after}`?"                
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"How about `{before}`?", 
                "neg": f"How about `{after}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Should `{before}` be `{after}`?", 
                "neg": f"Shouldn't `{before}` be `{after}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Should this `{before}` be instead of `{after}`?", 
                "neg": f"Shouldn't this `{before}` be instead of `{after}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Should `{before}` be `{after}`?", 
                "neg": f"Should `{after}` be `{before}`?"
            })
            new_data.append({
                "anchor_before": before_code, 
                "anchor_after": after_code, 
                "pos": f"Should this `{before}` be instead of `{after}`?", 
                "neg": f"Should this `{after}` be instead of `{before}`?"
            })
            # print(before)
            # print(after)
            # exit()

    return new_data

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

def dump_correctness_model_training_data(folder: str):
    for split in ["train", "valid"]:
        path = os.path.join(folder, f"msg-{split}.jsonl")
        aug_data = augment_data(path)
        aug_data_path = os.path.join(folder, f"correctness-aug-msg-{split}.jsonl")
        if split == "valid":
            aug_data = random.sample(aug_data, k=40000)
        print(f"writing {len(aug_data)} {split} instances to {aug_data_path}")
        write_jsonl(data=aug_data, path=aug_data_path)

def run_correctness_scorer():
    corr_score = CorrectnessScorer(model_path="microsoft/unixcoder-base", checkpoint_path="./ckpts/uxcoder_complex_cc_net/best_model.pth")
    import pandas as pd
    # df = pd.read_csv('cr_manual_rel_annot_likert_scale.csv')
    df = pd.read_csv('human_study_data.csv')
    # k = 100
    # predictions = list(df['pred'])[:k]
    corr_scores = []
    codes = list(df['patch'])#[:k]
    model_wise_scores = {}
    for ref_subset, model_name in [
        ('msg', 'ground_truth'),
        ('knn_pred', 'knn'),
        ('lstm_pred', 'lstm'),
        ('magicoder_pred', 'magicoder'),
        ('codereviewer_pred', 'codereviewer')
    ]:
        predictions = list(df[ref_subset])
        score = corr_score.compute(predictions=predictions, codes=codes)
        model_wise_scores[model_name] = score['inst_scores']
    indices = list(df['index'])
    for i in range(len(indices)):
        corr_scores.append({
            "index": indices[i],
            "ground_truth_rel": model_wise_scores['ground_truth'][i],
            "knn_rel": model_wise_scores['knn'][i],
            "lstm_rel": model_wise_scores['lstm'][i],
            "magicoder_rel": model_wise_scores['magicoder'][i],
            "codereviewer_rel": model_wise_scores['codereviewer'][i]
        })
    with open("./human_study_correctness_scores.json", "w") as f:
        json.dump(corr_scores, f, indent=4) 

# main
if __name__ == "__main__":
    # NOTE: uncomment and run this line for data creation.
    # dump_correctness_model_training_data("./data/Comment_Generation")

    run_correctness_scorer()


    # test_data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    # correctness_score = []
    # with open("ckpts/gen_study_inf/checkpoints-1800-5.72/preds.txt", "r") as f:
    #     i = 0
    #     for line in tqdm(f):
    #         code_change = test_data[i]['patch']
    #         review = line.strip()
    #         score = heuristic_based_correctness_check(code_change, review)
    #         # if score == 0: print(review)
    #         i += 1
    #         correctness_score.append(score)
    # print(np.mean(correctness_score))
    # TEMPLATES = dict(sorted(TEMPLATES.items(), reverse=True, key=lambda x: x[1]))
    # with open("TEMPLATES.json", "w") as f:
    #     json.dump(TEMPLATES, f, indent=4)