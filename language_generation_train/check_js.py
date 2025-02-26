import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import torch
# export HF_HOME=/data/datasets/hf_cache/your_username
os.environ["HF_HOME"] = "/data/datasets/hf_cache/mkapadni"
#  export HF_HUB_CACHE=/data/datasets/hf_cache/your_username/models
os.environ["HF_HUB_CACHE"] = "/data/datasets/hf_cache/mkapadni/models"


df = pd.read_json("/home/mkapadni/work/crscore_plus_plus/Comment_Generation/msg-train.jsonl", lines=True)
# print shape
print(df.shape)


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/stack-edu-classifier-javascript",cache_dir="/data/datasets/hf_cache/mkapadni/models")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/stack-edu-classifier-javascript",cache_dir="/data/datasets/hf_cache/mkapadni/models")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def score_patch(patch_text):
    inputs = tokenizer(
        patch_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=1024  # Matches model's context window
    ).to(device)
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy()
    score = logits.item()
    return {
        "raw_score": score,
        "int_score": int(round(max(0, min(score, 5)))),
    }


raw_scores = []
int_scores = []
for i in tqdm(df['patch'], total=df.shape[0], desc="Scoring patches"):
    scores = score_patch(i)
    raw_scores.append(scores["raw_score"])
    int_scores.append(scores["int_score"])

df['raw_js_score'] = raw_scores
df['int_js_score'] = int_scores

df = df[df['raw_js_score'] >= 1.5]
print(df.shape)

df.to_json("/data/datasets/hf_cache/mkapadni/crscore_plus_plus/dataset/msg-train_js_classified.jsonl", lines=True, orient='records')

