# Load model directly
import os
import json
from typing import *
from tqdm import tqdm
from src.datautils import read_jsonl, ReviewExample
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def do_inference(data: List[ReviewExample], model_path: str="microsoft/codereviewer", device: str="cuda:0", 
                 save_path: str="experiments/MS_CR_ZeroShot", **inference_params):
    """function to do zero inference with model.generate for Microsoft's Code Reviewer model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    gen_reviews_and_labels = []
    for example in tqdm(data):
        inp = tokenizer(example.input, return_tensors="pt").input_ids
        op = model.generate(inp.to(device), **inference_params)
        gen_review = tokenizer.decode(op.squeeze().cpu())
        gen_reviews_and_labels.append({
            "idx": example.idx,
            "diff": example.diff, 
            "input": example.input,
            "gold_review": example.msg,
            "gen_review": gen_review,
        })
    # save model outputs and gold labels/inputs.
    with open(save_path, "w") as f:
        json.dump(gen_reviews_and_labels, f, indent=4)

# main
if __name__ == "__main__":
    data_path = "./data/Comment_Generation/msg-test.jsonl"
    data = [ReviewExample.from_dict(item) for item in read_jsonl(data_path)]
    # do zero shot inference with microsoft's Code Reviewer model.
    do_inference(data)