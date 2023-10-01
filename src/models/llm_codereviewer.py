# script for inferencing with various LLMs.
import os
import json
import openai
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

class OpenAIEngine:
    def __init__(self, path: str="gpt-3.5-turbo"):
        openai
        self.model_path = path
        secret_key = os.environ.get("OPENAI_ACCESS_TOKEN") # json.load(open("./secrets/openai.json"))["key"]
        openai.organization = "org-rBWf7SAdxrBRh3V0O34DlEFT"
        # openai.organization = "Carnegie Mellon University"
        openai.api_key = secret_key

    def __call__(self, prompt: str):
        return get_GPT_response(prompt)

class CodeLLaMAEngine:
    def __init__(self, path: str="codellama/CodeLlama-13b-Instruct-hf"):
        access_token = os.environ.get("ACCESS_TOKEN")
        huggingface_hub.login(token=access_token)
        self.model_path = path
        self.llm = LLM(path)

    def __call__(self, prompt: Union[str, List[str]], use_tqdm: bool=False, max_tokens: int=20):
        res_op_list = self.llm.generate(prompt, SamplingParams(max_tokens=max_tokens), use_tqdm=use_tqdm)
        return res_op_list # [RequestOutputV2.from_v1_object(res_op).to_json() for res_op in res_op_list]

    def extract_output(self, outputs):
        return outputs[0]["outputs"][0]["text"].strip()

PROMPT_FOR_CODE_REVIEW = {
    "zero_shot": {
    "codellama": """Look at the following code change:

Old Code:
{}

New Code:
{}

Now generate a brief review of the following code change:
Review:""",
    "gpt-3.5": """Look at the following code change:

Old Code:
{}

New Code:
{}

Now generate a brief review of the following code change. Please don't just summarize the changes, but give your feedback on them. Keep the review brief:
Review:""",
},
    "few_shot": {
        "gpt-3.5": """Look at the given code changes and generate the code reviews:

Code Change: @@ -587,6 +587,8 @@ static void *evp_md_from_dispatch(const OSSL_DISPATCH *fns,
     if ((md = EVP_MD_meth_new(NID_undef, NID_undef)) == NULL)
         return NULL;
 
+    md->name = OPENSSL_strdup(name);
+
     for (; fns->function_id != 0; fns++) {
         switch (fns->function_id) {
         case OSSL_FUNC_DIGEST_NEWCTX:
Review: Should this be NULL checked? Not having the name isn't critical I guess.

Code Change: @@ -105,7 +105,7 @@ int SSL_SRP_CTX_init(struct ssl_st *s)
     s->srp_ctx.b = NULL;
     s->srp_ctx.v = NULL;
     s->srp_ctx.login = NULL;
-    s->srp_ctx.info = ctx->srp_ctx.info;
+    s->srp_ctx.info = NULL;
     s->srp_ctx.strength = ctx->srp_ctx.strength;
 
     if (((ctx->srp_ctx.N != NULL) &&
Review: Probably a memset() of srp_ctx is more appropriate here, rather than all these NULL assignments

Code Change: @@ -1087,7 +1087,8 @@ export class AmpA4A extends AMP.BaseElement {
     dev().assert(!!this.element.ownerDocument, 'missing owner document?!');
     this.protectedEmitLifecycleEvent_('renderFriendlyStart');
     // Create and setup friendly iframe.
-    const iframe = /** @type {!HTMLIFrameElement} */(
+    dev().assert(!this.iframe);
+    this.iframe = /** @type {!HTMLIFrameElement} */(
         createElementWithAttributes(
             /** @type {!Document} */(this.element.ownerDocument), 'iframe', {
               // NOTE: It is possible for either width or height to be 'auto',
Review: You can remove these now given we have an explicit check earlier in layoutCallback"""
    },
    "code_quality_labels_prompting": {
        "gpt-3.5": """Look at the given code changes and code quality aspects and generate the code reviews:

Code Change: @@ -587,6 +587,8 @@ static void *evp_md_from_dispatch(const OSSL_DISPATCH *fns,
     if ((md = EVP_MD_meth_new(NID_undef, NID_undef)) == NULL)
         return NULL;
 
+    md->name = OPENSSL_strdup(name);
+
     for (; fns->function_id != 0; fns++) {
         switch (fns->function_id) {
         case OSSL_FUNC_DIGEST_NEWCTX:
Code Quality Aspect: expressions
Review: Should this be NULL checked? Not having the name isn't critical I guess.

Code Change: @@ -25,13 +25,16 @@ module View
       def should_render_revenue?
         revenue = @tile.revenue_to_render

+        # special case: city with multi-revenue - no choice but to draw separate revenue
+        return true if revenue.any? { |r| !r.is_a?(Numeric) }
+
         return false if revenue.empty?

         return false if revenue.first.is_a?(Numeric) && (@tile.cities + @tile.towns).one?

         return false if revenue.uniq.size > 1

-        return false if @tile.cities.sum(&:slots) < 3 && @tile.stops.size == 2
+        return false if @tile.cities.sum(&:slots) < 3 && (@tile.cities + @tile.towns).size == 2

         true
       end
Code Quality Aspect: modularization
Review: we call cities + towns . size a lot, maybe make a helper method on tiles

Code Change: @@ -160,6 +160,11 @@ func (r *routeBuilder) profileHandler() (request.Handler, error) {
        return middleware.Wrap(h, backendMiddleware(r.cfg, r.authenticator, r.ratelimitStore, profile.MonitoringMap)...)
 }

+func (r *routeBuilder) firehoseLogHandler() (request.Handler, error) {
+       h := firehose.Handler(r.batchProcessor, r.authenticator)
+       return middleware.Wrap(h, firehoseMiddleware(r.cfg, intake.MonitoringMap)...)
+}
+
 func (r *routeBuilder) backendIntakeHandler() (request.Handler, error) {
        requestMetadataFunc := emptyRequestMetadata
        if r.cfg.AugmentEnabled {
Code Quality Aspect: names
Review: nit: `firehoseLogHandler` vs. `firehoseMiddleware` looks like a naming inconsistency? (`log` is not used anywhere else)."""
    }
}


# main
if __name__ == "__main__":
    model_type =  "gpt-3.5" # "codellama"
    prompt_type = "code_quality_labels_prompting" # "few_shot"  # "zero_shot"
    model_name = "gpt-3.5-turbo" # "codellama/CodeLlama-7b-Instruct-hf"
    if model_type == "codellama":
        model = CodeLLaMAEngine(model_name)
        cutoff_point = None
    elif model_type == "gpt-3.5":
        model = OpenAIEngine(model_name)
        cutoff_point = 1000
    if prompt_type == "code_quality_labels_prompting":
        data = pd.read_csv("./data/Comment_Generation/code_reviews_with_code_quality_labels.csv").to_dict("records")
        cutoff_point = 55
    else: data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
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
    if cutoff_point: data = data[:cutoff_point]
    # print(data[-1]["code quality"])
    # exit()
    for rec in tqdm(data):
        old_code, new_code = generate_before_after_code_from_patch(rec["patch"])
        # truncate old and new code to avoid prompt being too long.
        old_code = old_code[:1000]
        new_code = new_code[:1000]
        # print(old_code, new_code)
        if prompt_type == "few_shot":
            prompt = PROMPT_FOR_CODE_REVIEW[prompt_type][model_type]+"""
            Code Change: {}
Review:""".format(rec["patch"])
        elif prompt_type == "code_quality_labels_prompting":
            prompt = PROMPT_FOR_CODE_REVIEW[prompt_type][model_type]+"""
Code Change: {}
Code Quality Aspect: {}
Review:""".format(rec["patch"], rec["code quality"])
        else: prompt = PROMPT_FOR_CODE_REVIEW[prompt_type][model_type].format(old_code, new_code)
        if model_type == "codellama":
            extracted_output = model(prompt=prompt, max_tokens=100)[0].outputs[0].text.strip() # codellama.extract_output(codellama(prompt=PROMPT_FOR_CODE_REVIEW["codellama"].format(old_code, new_code), max_tokens=100))
        elif model_type == "gpt-3.5":
            extracted_output = model(prompt=prompt)["choices"][0]["text"].strip()
        golds.append(rec["msg"])
        pred_nls.append(extracted_output)    
        with open(preds_path, "a") as f:
            f.write(json.dumps({"pred": extracted_output, "gold": rec["msg"], "prompt": prompt})+"\n")
    bleu_with_stop = bleu_fromstr(pred_nls, golds, rmstop=False)
    bleu_without_stop = bleu_fromstr(pred_nls, golds, rmstop=True)
    print("BLEU with STOP:", bleu_with_stop)
    print("BLEU without STOP:", bleu_without_stop)
    with open(metrics_path, "w") as f:
        json.dump({
            "BLEU with stop": bleu_with_stop,
            "BLEU": bleu_without_stop,
        }, f, indent=4)