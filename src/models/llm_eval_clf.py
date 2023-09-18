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

CODE_REVIEW_ASSESMENT_PROMPTS = """Classify code change reviews according to the following taxonomy:
1. documentation: Review covers issues with documenting changes or inaccurate, out of date or misleading comments.
2. presentation: Review covers issues with variable naming (confusing or misleading names etc.), formatting issues, etc.
3. algorithm: Review covers issues with control flow, expressions, statements, performance, loop structure or other kinds of bugs.
4. structure: Review covers issues with modularity, decomposition, duplication, etc. in the code.
5. alternative solution: Review poses a question to discuss options and alternative solutions
6. correct understanding: Review tries to ensure that the reviewer has captured the real meaning of changes under review
7. rationale: Review asks for missing information or rationale to justify code changes
8. code context: Review asks for information to clarify context or code relevant to understanding the changes
9. necessity: Review asks whether a change is really necessary or can simplified or removed
10. specialized expertise: Review asks another reviewer to step in and contribute with their specialized expertise.
11. splittable: Review questions whether a code change could be split into multiple ones.

REVIEW: {}
LABEL:"""

CODE_REVIEW_ASSESMENT_WITH_EG_PROMPTS = """Classify code change reviews:

REVIEW: Since the change owner is always admin, this code might be able to move out of
the loop? The following should be enough for this
LABEL: alternative solution

REVIEW: This is now an empty heading. Or do you feel it is important to point out that these are C++ classes?
LABEL: correct understanding

REVIEW: Can you explain why you replaced that with this and where exactly was failing?
LABEL: rationale

REVIEW: In what situations would [this condition] be false, but not undefined?
LABEL: code context

REVIEW: Is this needed?
LABEL: Necessity

REVIEW: Lars, Simon, any ideas? We really need to fix this for [the next release] and the
time draws nigh
LABEL: specialized expertise

REVIEW: This looks like an unrelated change. Should it be in a separate commit?
LABEL: splittable 

REVIEW: I think it’s really neat that he uses enhanced for loops.
LABEL: algorithm

REVIEW: This is not DRY, as you have the same test: one in the if statement, and the other in the while.
LABEL: algorithm

REVIEW: Inverting lists and printing them, we don’t do that. That is when we need to have a separate function for inverting and a separate function for printing.
LABEL: structure

REVIEW: you rarely have too many classes
LABEL: structure

REVIEW: I think that is very nice, very symmetrical. I like symmetric code; if you do similar things they should be similarly formatted.
LABEL: presentation

REVIEW: Old code and ordering for code in a file. Instance variables at the bottom of the module.
LABEL: presentation

REVIEW: The comments should be at a higher level of abstraction. You should describe the intent behind what you are doing, not provide a textual version of what follows.
LABEL: documentation

REVIEW: I would prefer that variables, if they contain multiple words, use camel casing, so I can easily read them.
LABEL: documentation

REVIEW: {}
LABEL:"""

def run_code_quality_testing(model_id: str='text-davinci-003', limit: int=1000):
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
    for rec in tqdm(data[:limit]):
        patch = rec["patch"]
        old_code, new_code = generate_before_after_code_from_patch(patch)
        prompt = CODE_QUALITY_PROMPT.format(rec["lang"], old_code, new_code)
        response = get_GPT_response(prompt)
        with open(write_path, "a") as f:
            f.write(json.dumps({"id": i, "prompt": prompt, "completions": response["choices"][0]["text"].strip()})+"\n")
        i += 1

def run_code_review_classification(model_id: str='text-davinci-003', limit: int=1000, use_eg_demos: bool=True):
    secret_key = json.load(open("./secrets/openai.json"))["key"]
    # openai.organization = "Carnegie Mellon University"
    openai.api_key = secret_key
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    if use_eg_demos:
        write_path = f"./data/Code_Quality/{model_id}_review_taxonomy_preds_and_eg.jsonl"
    else: write_path = f"./data/Code_Quality/{model_id}_review_taxonomy_preds.jsonl"
    overwrite = False
    if os.path.exists(write_path):
        overwrite = bool(input("Overwrite preds? (y/n)").lower() == "y")
        if not(overwrite): exit()
    if overwrite: open(write_path, "w")
    i = 0
    for rec in tqdm(data[:limit]):
        review = rec["msg"]
        if use_eg_demos:
            prompt = CODE_REVIEW_ASSESMENT_WITH_EG_PROMPTS.format(review)
        else: prompt = CODE_REVIEW_ASSESMENT_PROMPTS.format(review)
        response = get_GPT_response(prompt)
        with open(write_path, "a") as f:
            f.write(json.dumps({"id": i, "prompt": prompt, "completions": response["choices"][0]["text"].strip()})+"\n")
        i += 1

# main
if __name__ == "__main__":
    # run_code_quality_testing(model_id="gpt-3.5-turbo")
    # run_code_review_classification(model_id="gpt-3.5-turbo")
    run_code_review_classification(model_id="gpt-3.5-turbo", use_eg_demos=True)
    run_code_review_classification(model_id="text-davinci-003", use_eg_demos=True)