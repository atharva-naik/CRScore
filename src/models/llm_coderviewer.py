# script for inferencing with various LLMs.
import os
import json
import pandas as pd
from vllm import LLM
from typing import *
from tqdm import tqdm
import huggingface_hub
from dotenv import load_dotenv
from src.llm_utils import RequestOutputV2
from vllm.sampling_params import SamplingParams
from CodeBERT.CodeReviewer.code.evaluator.smooth_bleu import bleu_fromstr
from src.datautils import read_jsonl, generate_before_after_code_from_patch

load_dotenv()
access_token = os.environ.get("ACCESS_TOKEN")
huggingface_hub.login(token=access_token)

class CodeLLaMAEngine:
    def __init__(self, path: str="codellama/CodeLlama-13b-Instruct-hf"):
        self.model_path = path
        self.llm = LLM(path)

    def __call__(self, prompt: Union[str, List[str]], use_tqdm: bool=False, max_tokens: int=20):
        res_op_list = self.llm.generate(prompt, SamplingParams(max_tokens=max_tokens), use_tqdm=use_tqdm)
        return res_op_list # [RequestOutputV2.from_v1_object(res_op).to_json() for res_op in res_op_list]

    def extract_output(self, outputs):
        return outputs[0]["outputs"][0]["text"].strip()

PROMPT_FOR_CODE_REVIEW = {
    "codellama": """Look at the following code change:

Old Code:
{}

New Code:
{}

Now generate a brief review of the following code change:
Review:""" 
}

# main
if __name__ == "__main__":
    model_type = "codellama"
    prompt_type = "zero_shot"
    model_name = "codellama/CodeLlama-13b-Instruct-hf"
    if model_type == "codellama":
        model = CodeLLaMAEngine(model_name)
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    golds = []
    pred_nls = []
    save_dir = f"./experiments/{model_name.lower().replace('-','_').replace('/','_')}_{prompt_type}"
    os.makedirs(save_dir, exist_ok=True)
    preds_path = os.path.join(save_dir, "preds.jsonl")
    metrics_path = os.path.join(save_dir, "bleu_scores.json")
    if os.path.exists(preds_path):
        overwrite = input("Overwrite preds (y/n)?").lower()
        if overwrite == "n": exit()
    # overwrite predictions:
    open(preds_path, "w")
    for rec in tqdm(data):
        old_code, new_code = generate_before_after_code_from_patch(rec["patch"])
        # truncate old and new code to avoid prompt being too long.
        old_code = old_code[:1000]
        new_code = new_code[:1000]
        # print(old_code, new_code)
        extracted_output = model(prompt=PROMPT_FOR_CODE_REVIEW[model_type].format(old_code, new_code), max_tokens=100)[0].outputs[0].text.strip() # codellama.extract_output(codellama(prompt=PROMPT_FOR_CODE_REVIEW["codellama"].format(old_code, new_code), max_tokens=100))
        golds.append(rec["msg"])
        pred_nls.append(extracted_output)    
        with open(preds_path, "a") as f:
            f.write(json.dumps({"pred": extracted_output, "gold": rec["msg"]})+"\n")
    bleu_with_stop = bleu_fromstr(pred_nls, golds, rmstop=False)
    bleu_without_stop = bleu_fromstr(pred_nls, golds, rmstop=True)
    print("BLEU with STOP:", bleu_with_stop)
    print("BLEU without STOP:", bleu_without_stop)
    with open(metrics_path, "w") as f:
        json.dump({
            "BLEU with stop": bleu_with_stop,
            "BLEU": bleu_without_stop,
        }, f, indent=4)