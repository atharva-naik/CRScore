# LLM based evaluation classifiers.

import os
import re
import json
import openai
from typing import *
from tqdm import tqdm
from src.datautils import read_jsonl

def get_GPT_response(prompt: str, model_id: str='text-davinci-003') -> dict:
    prompt = prompt.strip()
    response = openai.Completion.create(
        engine=model_id,
        prompt=prompt,
        temperature=0.2,
        max_tokens=256,
        top_p=0.95,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response.to_dict()

# def filter_completions_from_response(response: dict) -> List[Tuple[str, bool]]:
#     completions = []
#     for choice in response["choices"]:
#         completion = choice["text"].split("question:")[-1].strip()
#         is_valid = bool("question:" in choice["text"])
#         completions.append((completion, is_valid))

#     return completions

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

CODE_QUALITY_PROMPT = """Read the following code quality guidelines carefully:
1. documentation: Issues with documenting changes or inaccurate, out of date or misleading comments.
2. presentation: Issues with variable naming (confusing or misleading names etc.), formatting issues, etc. 
3. algorithm: Issues with control flow, expressions, statements, performance, loop structure or other kinds of bugs.
4. structure: Issues with modularity, decomposition, duplication, etc. in the code.

Now carefully look at the two version of the {} code shown below:
OLD_CODE:
{}

NEW_CODE:
{}

Which code quality guideline does the NEW_CODE violate.
ANSWER:"""

# main
if __name__ == "__main__":
    model_id: str='text-davinci-003'
    secret_key = json.load(open("./secrets/openai.json"))["key"]
    # openai.organization = "Carnegie Mellon University"
    openai.api_key = secret_key
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    write_path = f"./data/Code_Quality/{model_id}_preds.jsonl"
    overwrite = False
    if os.path.exists(write_path):
        overwrite = bool(input("Overwrite preds? (y/n)").lower() == "y")
    if overwrite: open(write_path, "w")
    i = 0
    for rec in tqdm(data):
        patch = rec["patch"]
        old_code, new_code = generate_before_after_code_from_patch(patch)
        prompt = CODE_QUALITY_PROMPT.format(rec["lang"], old_code, new_code)
        response = get_GPT_response(prompt)
        with open(write_path, "a") as f:
            f.write(json.dumps({"id": i, "prompt": prompt, "completions": response["choices"][0]["text"].strip()})+"\n")
        i += 1