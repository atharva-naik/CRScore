# LLM-prompting based approach to review generation.
import os
import re
import json
import torch
import warnings
from typing import *
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from transformers import logging
from transformers import pipeline
from src.datautils import read_jsonl, remove_patch_header
from CodeBERT.CodeReviewer.code.evaluator.smooth_bleu import bleu_fromstr

logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()

class OpenAIEngine:
    def __init__(self, path: str="gpt-4-0613"):
        self.model_path = path
        # secret_key = os.environ["OPENAI_ACCESS_TOKEN"]
        # openai.organization = "org-rBWf7SAdxrBRh3V0O34DlEFT"
        # openai.organization = "Carnegie Mellon University"
        # openai.api_key = secret_key
        self.client = OpenAI()

    def get_GPT_response(self, prompt: str, **args) -> dict:
        prompt = prompt.strip()
        response = openai.Completion.create(
            engine=self.model_path,
            prompt=prompt, **args,
        )

        return response.to_dict()

    def __call__(self, prompt: str, temperature=0.2, max_tokens=256,
                 top_p=0.95, frequency_penalty=0.0, presence_penalty=0.0):
        # responses = self.get_GPT_response(
        #     prompt=prompt, temperature=temperature,
        #     frequency_penalty=frequency_penalty,
        #     max_tokens=max_tokens, top_p=top_p,
        #     presence_penalty=presence_penalty,
        # )
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return response.choices[0].message.content.strip()

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

CC1 = """Review the code changes shown below by providing appropriate feedback. Don't just summarize the code change, instead point out how they can be corrected if there are any mistakes or how they can be improved:

Code Change:
@@ -587,6 +587,8 @@ static void *evp_md_from_dispatch(const OSSL_DISPATCH *fns,
     if ((md = EVP_MD_meth_new(NID_undef, NID_undef)) == NULL)
         return NULL;
 
+    md->name = OPENSSL_strdup(name);
+
     for (; fns->function_id != 0; fns++) {
         switch (fns->function_id) {
         case OSSL_FUNC_DIGEST_NEWCTX:

Your Review:
"""
CR1 = """Should this be NULL checked? Not having the name isn't critical I guess."""
CC2 = """Code Change:
@@ -105,7 +105,7 @@ int SSL_SRP_CTX_init(struct ssl_st *s)
     s->srp_ctx.b = NULL;
     s->srp_ctx.v = NULL;
     s->srp_ctx.login = NULL;
-    s->srp_ctx.info = ctx->srp_ctx.info;
+    s->srp_ctx.info = NULL;
     s->srp_ctx.strength = ctx->srp_ctx.strength;
 
     if (((ctx->srp_ctx.N != NULL) &&

Your Review:
"""
CR2 = """Probably a memset() of srp_ctx is more appropriate here, rather than all these NULL assignments"""
CC3 = """Code Change:
@@ -1087,7 +1087,8 @@ export class AmpA4A extends AMP.BaseElement {
     dev().assert(!!this.element.ownerDocument, 'missing owner document?!');
     this.protectedEmitLifecycleEvent_('renderFriendlyStart');
     // Create and setup friendly iframe.
-    const iframe = /** @type {!HTMLIFrameElement} */(
+    dev().assert(!this.iframe);
+    this.iframe = /** @type {!HTMLIFrameElement} */(
         createElementWithAttributes(
             /** @type {!Document} */(this.element.ownerDocument), 'iframe', {
               // NOTE: It is possible for either width or height to be 'auto',

Your Review:
"""
CR3 = """You can remove these now given we have an explicit check earlier in layoutCallback"""
CHAT_TEMPLATE = [
    {"role": "user", "content": CC1},
    {"role": "system", "content": CR1},
    {"role": "user", "content": CC2},
    {"role": "system", "content": CR2},
    {"role": "user", "content": CC3},
    {"role": "system", "content": CR3},
]

MAGICODER_CODEREVIEW_PROMPT_PREFIX = """Review the code changes shown below by providing appropriate feedback. Don't just summarize the code change, instead point out how they can be corrected if there are any mistakes or how they can be improved:

@@ Code Change
@@ -587,6 +587,8 @@ static void *evp_md_from_dispatch(const OSSL_DISPATCH *fns,
     if ((md = EVP_MD_meth_new(NID_undef, NID_undef)) == NULL)
         return NULL;
 
+    md->name = OPENSSL_strdup(name);
+
     for (; fns->function_id != 0; fns++) {
         switch (fns->function_id) {
         case OSSL_FUNC_DIGEST_NEWCTX:

@@ Review
Should this be NULL checked? Not having the name isn't critical I guess.

@@ Code Change
@@ -105,7 +105,7 @@ int SSL_SRP_CTX_init(struct ssl_st *s)
     s->srp_ctx.b = NULL;
     s->srp_ctx.v = NULL;
     s->srp_ctx.login = NULL;
-    s->srp_ctx.info = ctx->srp_ctx.info;
+    s->srp_ctx.info = NULL;
     s->srp_ctx.strength = ctx->srp_ctx.strength;
 
     if (((ctx->srp_ctx.N != NULL) &&

@@ Review 
Probably a memset() of srp_ctx is more appropriate here, rather than all these NULL assignments

@@ Code Change
@@ -1087,7 +1087,8 @@ export class AmpA4A extends AMP.BaseElement {
     dev().assert(!!this.element.ownerDocument, 'missing owner document?!');
     this.protectedEmitLifecycleEvent_('renderFriendlyStart');
     // Create and setup friendly iframe.
-    const iframe = /** @type {!HTMLIFrameElement} */(
+    dev().assert(!this.iframe);
+    this.iframe = /** @type {!HTMLIFrameElement} */(
         createElementWithAttributes(
             /** @type {!Document} */(this.element.ownerDocument), 'iframe', {
               // NOTE: It is possible for either width or height to be 'auto',

@@ Review
You can remove these now given we have an explicit check earlier in layoutCallback
"""
MAGICODER_CODEREVIEW_PROMPT = """@@ Code Change
{code_change}

@@ Review
"""

def test_magicoder():
    instruction = "Write Python code to print 'Hello World'"
    prompt = MAGICODER_PROMPT.format(instruction=instruction)
    generator = pipeline(
        model="ise-uiuc/Magicoder-S-DS-6.7B",
        task="text-generation",
        torch_dtype=torch.bfloat16,
        device="cuda:0",
    )
    result = generator(prompt, max_length=1024, num_return_sequences=1, temperature=0.0)
    
    return result

def test_magicoder_codereview(code_change):
    prompt = MAGICODER_CODEREVIEW_PROMPT_PREFIX + MAGICODER_CODEREVIEW_PROMPT.format(code_change=code_change)
    generator = pipeline(
        model="ise-uiuc/Magicoder-S-DS-6.7B",
        task="text-generation",
        torch_dtype=torch.bfloat16,
        device="cuda:0",
    )
    result = generator(prompt, max_length=1024, num_return_sequences=1, temperature=0.0)
    
    return result

def main(save_dir: str, data_path: str="./data/Comment_Generation/msg-test.jsonl", 
         model_path: str="ise-uiuc/Magicoder-S-DS-6.7B", device: str="cuda:0",
         model_name: str="Magicoder-S-DS-6.7B", task: str="text-generation",
         skip: Union[int, None]=None):
    os.makedirs(save_dir, exist_ok=True)
    write_path = os.path.join(save_dir, model_name+".jsonl")
    data = read_jsonl(data_path)
    if model_name.startswith("Magicoder"):
        generator = pipeline(
            model=model_path, task=task,
            torch_dtype=torch.bfloat16,
            device=device,
        )        
        tokenizer = generator.tokenizer
        model = generator.model
    elif model_name.startswith("GPT-3.5"):
        generator = OpenAIEngine(path=model_path)
    else:
        generator = pipeline(
            model=model_path, task=task,
            device=device,
        )
        tokenizer = generator.tokenizer
        model = generator.model
    if os.path.exists(write_path):
        overwrite_file = input("overwrite (y/n)?").lower().strip()
        if overwrite_file not in ["y", "yes"]: exit()
    open(write_path, "w")
    id = 0
    preds = []
    golds = []
    patch_lengths = []
    for rec in tqdm(data, desc=f"{model_name} inference"):
        if skip is not None and id < skip: 
            id += 1
            patch_lengths.append(len(rec['patch']))
            continue
        # print("max patch legth till now:", max(patch_lengths))
        prompt = MAGICODER_CODEREVIEW_PROMPT_PREFIX + MAGICODER_CODEREVIEW_PROMPT.format(code_change=rec['patch'][:5000])
        # print("size of patch causing error:", len(rec['patch']))
        if "Mistral" in model_name:
            temperature = 1e-12
        elif "Stable" in model_name:
            temperature = 1e-12
        # elif "CodeGemma" in model_name:
        #     temperature = 0.2
        else:
            temperature = 0
        # temperature = 0 if "Mistral" not in model_name else 1e-12
        if model_name.startswith("Magicoder"):
            result = generator(prompt, max_length=1024, num_return_sequences=1, temperature=temperature)
        elif model_name.startswith("CodeGemma"):
            CC4 = "Code Change:\n"+rec['patch'][:5000]+"\n\nYour Review:\n"
            chat = CHAT_TEMPLATE + [{'role': 'user', 'content': CC4}]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            result = model.generate(input_ids=inputs.to(device), max_new_tokens=200, temperature=temperature)
        elif model_name.startswith("GPT-3.5"):
            prompt = MAGICODER_CODEREVIEW_PROMPT_PREFIX + MAGICODER_CODEREVIEW_PROMPT.format(code_change=rec['patch'][:2000])
            result = generator(prompt)
        else: 
            result = generator(prompt, max_new_tokens=200, num_return_sequences=1, temperature=temperature)
        
        if model_name.startswith("CodeGemma"):
            op = tokenizer.decode(result[0]).replace(prompt,'') # do something here.
        elif model_name.startswith("GPT-3.5"):
            op = result
        else:
            op = result[0]['generated_text'].replace(prompt,'')

        with open(write_path, "a") as f:
            f.write(json.dumps({"id": id, "code_change": rec['patch'], "gold_review": rec['msg'], 'prompt': prompt, 'pred_review': op})+"\n")
            golds.append(rec['msg'])
            preds.append(rec['patch'])
        id += 1
    bleu_with_stop = bleu_fromstr(predictions=preds, golds=golds, rmstop=False)
    print(f"bleu_with_stop = {bleu_with_stop}")
    bleu_without_stop = bleu_fromstr(predictions=preds, golds=golds, rmstop=True)
    print(f"bleu_without_stop = {bleu_without_stop}")

# main
if __name__ == "__main__":
    # main(save_dir="./data/Comment_Generation/llm_outputs", skip=6923)
    # main(
    #     save_dir="./experiments/llm_outputs",
    #     model_path="deepseek-ai/deepseek-coder-6.7b-instruct",
    #     model_name="DeepSeekCoder-6.7B-Instruct"
    # )
    # main(
    #     save_dir="./experiments/llm_outputs",
    #     model_path="ajibawa-2023/Code-Mistral-7B",
    #     model_name="Code-Mistral-7B",
    # )
    # main(
    #     save_dir="./experiments/llm_outputs",
    #     model_path="google/codegemma-7b-it",
    #     model_name="CodeGemma-7B-Instruct",
    # )
    # main(
    #     save_dir="./experiments/llm_outputs",
    #     model_path="stabilityai/stable-code-instruct-3b",
    #     model_name="Stable-Code-Instruct-3b",
    # )

    main(
        save_dir="./experiments/llm_outputs",
        model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        model_name="Llama-3-8B-Instruct",
    )
    # main(
    #     save_dir="./experiments/llm_outputs",
    #     model_path="gpt-3.5-turbo-0125",
    #     model_name="GPT-3.5-Turbo",
    # )