import os
import re
import json
from tqdm import tqdm
from text_generation import Client

def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            try:
                js = json.loads(line.strip())
            except:
                print("Error during reading json data.")
                continue
            data.append(js)
    return data

def remove_patch_header(code: str):
    # Define the regex pattern to match expressions like "@@ -53,7 +53,7 @@" or "@@ -12,8 +12,20 @@"
    pattern = r'@@ -\d+,\d+ \+\d+,\d+ @@'

    # Use re.sub() with count=1 to replace only the first matched pattern with an empty string
    result = re.sub(pattern, '', code, count=1)

    return result

def generate_before_after_code_from_patch(patch: str):
    patch = remove_patch_header(patch).strip()
    old_lines = []
    new_lines = []
    for line in patch.split("\n"):
        if line.startswith("+"):
            line = line[1:]
            new_lines.append(line)
        elif line.startswith("-"):
            line = line[1:]
            old_lines.append(line)
        else:
            new_lines.append(line)
            old_lines.append(line)

    return "\n".join(old_lines), "\n".join(new_lines)

if __name__ == "__main__":
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    client = Client("http://tir-1-32:8880", timeout=60)
    preds = []
    if os.path.exists("./experiments/StarCoder_ZeroShot/preds.jsonl"):
        response = input("overwrite current preds (y/n)?")
        if response.lower() == "n": exit()
    open("./experiments/StarCoder_ZeroShot/preds.jsonl", "w")
    for rec in tqdm(data):
        patch = rec["patch"]
        old_code, new_code = generate_before_after_code_from_patch(patch)
        # commit message generation using StarCoder.

        # prompt = f"<commit_before>{old_code}<commit_msg>text<commit_after>{new_code}<eos>"
        prompt = f"<commit_before>{old_code}<commit_msg>"
        completion = client.generate(prompt, max_new_tokens=64, top_p=0.95, temperature=0.0001, do_sample=True).generated_text
        commit_msg = completion.split("<commit_after>")[0].strip()
        with open("./experiments/StarCoder_ZeroShot/preds.jsonl", "a") as f:
            f.write(json.dumps({
                "gold": rec['msg'],
                "pred": commit_msg,
            })+"\n")