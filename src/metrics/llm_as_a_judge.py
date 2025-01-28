import os
import sys
import json
import torch
import openai
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_human_eval_samples():
    human_annot_datapoints = {}
    marcus_annot_end_points = {"py": 501-2, "java": 501-2, "js": 505-2}
    index_to_lang = {}
    langs = list(marcus_annot_end_points.keys())
    for lang in langs:
        human_annot_datapoints[lang] = []
        marcus_annot = pd.read_csv(f"human_study/phase2/{lang}_marcus_review_qual_final.csv").to_dict("records")
        atharva_annot = pd.read_csv(f"human_study/phase2/{lang}_atharva_review_qual_final.csv").to_dict("records")
        index = None
        for i, rec in enumerate(marcus_annot):
            # if beyond the boundary of Marcus' annotations then switch to Atharva's annotations.
            if i > marcus_annot_end_points[lang]: 
                rec = atharva_annot[i]
            if str(rec['index']) != "nan":
                index = int(rec["index"])
                diff = rec['diff']
                index_to_lang[index] = lang
            system = rec['system']
            if str(rec["Rel (F)"]) != "nan" and system != "msg": # skip CodeReviewer ground truth/references among the evaluated systems, because we don't count it for the correlations as reference based metrics would default to 1 on them and disadvantage their correlation values.
                human_annot_datapoints[lang].append({"index": index, "system": system, "diff": diff, "review": rec["review"]})
            
    return human_annot_datapoints


GPT_MODEL="gpt-4o"
MAGICODER_MODEL = "ise-uiuc/Magicoder-S-DS-6.7B"

CODE_CHANGE_AND_REVIEW_SYSTEM_PROMPT = """You are a highly skilled software engineer who has a lot of experience reviewing code changes. Your task is to rate the relevance of any given code change"""

CODE_CHANGE_AND_REVIEW_JUDGE_PROMPT = """You will be asked to rate the relevance of reviews for given Python, Java or Javascript code changes. A relevant review is one which is both concise and comprehensive. A concise review contains very little text not related to the code change. A comprehensive review contains all the information about a code change that should be covered by a review. A relevant review is comprehensive while being concise.

Now look at the {lang} code change and review below and score the relevance of the review on a scale of 1 to 5

Code Change:
{code_change}

Review:
{review}

Your score: """

LANG_MAP = {
    "py": "Python",
    "js": "Javascript",
    "java": "Java",
}

class LLM_as_a_Judge:
    def __init__(self, model: str, api_key: Union[str, None]=None):
        self.model = model
        if model.startswith("gpt"):
            self.client: Union[OpenAI, AutoModelForCausalLM] = OpenAI(api_key=api_key)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            # self.client: Union[OpenAI, AutoModelForCausalLM] = AutoModelForCausalLM.from_pretrained(
            #     self.model, 
            #     trust_remote_code=True, 
            #     device_map="auto",    
            # )
            self.client: Union[OpenAI, AutoModelForCausalLM] = pipeline(
                model="ise-uiuc/Magicoder-S-DS-6.7B",
                task="text-generation",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

    def __call__(self, code_change: str, review: str, lang: str):
        if self.model.startswith("gpt"):
            inst_to_be_judged = CODE_CHANGE_AND_REVIEW_JUDGE_PROMPT.format(lang=LANG_MAP[lang], code_change=code_change[:5000], review=review)
            messages = [
                {"role": "system", "content": CODE_CHANGE_AND_REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": inst_to_be_judged}
            ]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            response = str(completion.choices[0].message.content).strip()
        else:
            inst_to_be_judged = CODE_CHANGE_AND_REVIEW_JUDGE_PROMPT.format(lang=LANG_MAP[lang], code_change=code_change[:5000], review=review)
            # print(inst_to_be_judged)
            # messages = [
            #     {"role": "user", "content": CODE_CHANGE_AND_REVIEW_SYSTEM_PROMPT+"\n"+inst_to_be_judged}
            # ]
            # tokenized_chat = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=True,
            #     add_generation_prompt=True,
            #     return_tensors="pt"
            # )

            # outputs = self.client.generate(
            #     tokenized_chat.to(self.client.device),
            #     max_new_tokens=50, do_sample=False,
            #     eos_token_id=self.tokenizer.eos_token_id,
            #     pad_token_id=self.tokenizer.eos_token_id,
            # )
            # response = self.tokenizer.decode(outputs[0][len(tokenized_chat[0]):], skip_special_tokens=True)
            prompt = CODE_CHANGE_AND_REVIEW_SYSTEM_PROMPT+"\n"+inst_to_be_judged
            result = self.client(
                prompt, max_new_tokens=5, 
                num_return_sequences=1, temperature=0.0
            )
            response = result[0]["generated_text"].replace(prompt,'').split("\n")[0].strip()
            # print("\x1b[34;1mresponse:\x1b[0m", response)
        if response.startswith("1"): score = 1
        elif response.startswith("2"): score = 2
        elif response.startswith("3"): score = 3
        elif response.startswith("4"): score = 4
        elif response.startswith("5"): score = 5
        else: score = 1

        return score/5, inst_to_be_judged

# main
if __name__ == "__main__":
    human_annot_datapoints = load_human_eval_samples()
    total = 0
    for lang in LANG_MAP:
        total += len(human_annot_datapoints[lang]) 
        print(len(human_annot_datapoints[lang]))
    print(total)
    # print(human_annot_datapoints['py'][0])
    
    LLM_judgement_save_path: str = "./Magicoder-as-a-judge_metric_scores.jsonl"
    # LLM_judgement_save_path: str = "./GPT-4o-as-a-judge_metric_scores.jsonl"
    if os.path.exists(LLM_judgement_save_path):
        overwrite = bool(input("overwrite (y/N)?").lower().strip() in ["yes","y"])
        if not overwrite: exit()

    open(LLM_judgement_save_path, "w")
    GPT4_key = "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"
    judge = LLM_as_a_Judge(model=MAGICODER_MODEL)
    # judge = LLM_as_a_Judge(model=GPT_MODEL, api_key=GPT4_key)
    llm_judgements = []
    for lang in human_annot_datapoints:
        # # NOTE: hardcoded for now.
        # if lang == "py":
        #     human_annot_datapoints[lang] = human_annot_datapoints[lang][187:]
        #     print(human_annot_datapoints[lang][0])
        for rec in tqdm(human_annot_datapoints[lang]):
            score, prompt = judge(review=rec['review'], code_change=rec['diff'], lang=lang)
            judge_rec = rec
            judge_rec['lang'] = lang
            judge_rec['score'] = score
            judge_rec['prompt'] = prompt

            llm_judgements.append(judge_rec)
            with open(LLM_judgement_save_path, "a") as f:
                f.write(json.dumps(judge_rec)+"\n")