import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

train_path = "/home/mkapadni/work/crscore_plus_plus/Comment_Generation/msg-train_all_classified.json"
train_df = pd.read_json(train_path, lines=True)


# tokenize first 1000 samples and then give me avg token length

token_lens = []
for i in tqdm(range(1000), total=1000):
    token_lens.append(len(tokenizer(train_df["patch"][i])["input_ids"]))


print(sum(token_lens)/len(token_lens))