import json
from datasets import load_from_disk
from src.datautils import read_jsonl

humaneval_data = json.load(open("human_study/phase1/raw_data.json"))
humaneval_data_projs = set(rec['proj'] for rec in humaneval_data)
GPT_data = load_from_disk("./data/GPT_code_change_summ_labels.hf/")
val_data = read_jsonl("./data/Comment_Generation/msg-valid.jsonl")
val_data_map = {rec['patch']: rec for rec in val_data}
GPT_data_projs = set()
keyerrors = 0
for rec in GPT_data:
    diff = rec['instruction'].split('GIT DIFF:')[1].split("Lines ")[0].strip()
    try: GPT_data_projs.add(val_data_map[diff]['proj'])
    except: keyerrors += 1
print(keyerrors)