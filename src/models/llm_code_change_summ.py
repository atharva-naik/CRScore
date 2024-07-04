# LLM-prompting based approach to review generation.
import os
import re
import json
import torch
import openai
import warnings
from typing import *
from tqdm import tqdm
from openai import OpenAI
from datasets import Dataset
from dotenv import load_dotenv
from transformers import logging, pipeline, AutoTokenizer, AutoModelForCausalLM
from src.datautils import read_jsonl, remove_patch_header, generate_added_and_removed_lines_from_patch
# from CodeBERT.CodeReviewer.code.evaluator.smooth_bleu import bleu_fromstr

logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=UserWarning)

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

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

# MAGICODER_CODEREVIEW_PROMPT_PREFIX = """Review the code changes shown below by providing appropriate feedback. Don't just summarize the code change, instead point out how they can be corrected if there are any mistakes or how they can be improved:

# @@ Code Change
# @@ -587,6 +587,8 @@ static void *evp_md_from_dispatch(const OSSL_DISPATCH *fns,
#      if ((md = EVP_MD_meth_new(NID_undef, NID_undef)) == NULL)
#          return NULL;
 
# +    md->name = OPENSSL_strdup(name);
# +
#      for (; fns->function_id != 0; fns++) {
#          switch (fns->function_id) {
#          case OSSL_FUNC_DIGEST_NEWCTX:

# @@ Review
# Should this be NULL checked? Not having the name isn't critical I guess.

# @@ Code Change
# @@ -105,7 +105,7 @@ int SSL_SRP_CTX_init(struct ssl_st *s)
#      s->srp_ctx.b = NULL;
#      s->srp_ctx.v = NULL;
#      s->srp_ctx.login = NULL;
# -    s->srp_ctx.info = ctx->srp_ctx.info;
# +    s->srp_ctx.info = NULL;
#      s->srp_ctx.strength = ctx->srp_ctx.strength;
 
#      if (((ctx->srp_ctx.N != NULL) &&

# @@ Review 
# Probably a memset() of srp_ctx is more appropriate here, rather than all these NULL assignments

# @@ Code Change
# @@ -1087,7 +1087,8 @@ export class AmpA4A extends AMP.BaseElement {
#      dev().assert(!!this.element.ownerDocument, 'missing owner document?!');
#      this.protectedEmitLifecycleEvent_('renderFriendlyStart');
#      // Create and setup friendly iframe.
# -    const iframe = /** @type {!HTMLIFrameElement} */(
# +    dev().assert(!this.iframe);
# +    this.iframe = /** @type {!HTMLIFrameElement} */(
#          createElementWithAttributes(
#              /** @type {!Document} */(this.element.ownerDocument), 'iframe', {
#                // NOTE: It is possible for either width or height to be 'auto',

# @@ Review
# You can remove these now given we have an explicit check earlier in layoutCallback
# """

MAGICODER_CODESUMM_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

Now look at the GIT DIFF below and summarize the changes:
@@ GIT DIFF
{code_change}

@@ Summary of changes
"""

MAGICODER_CODESUMM_CODECHANGE_PROMPT = """You are an exceptionally intelligent and experienced senior software engineer and code reviewer that consistently provides helpful and reliable reviews on code changes. Now look at instance of code changes with the code before and code after being shown and enumerate and summarize the changes that happened in neat points. The points should be independent and non overlaping and should comprehensively cover all changes. The points shouldn't be repetitive so try to minimize the number of points while being comprehensive.

@@ Code before:
{code_before}

@@ Code after:
{code_after}

@@ List of code changes:"""

CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT_PREFIX = open("src/models/few_shot_impl_code_smell_detection_prompt.txt").read()

CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT = """

GIT DIFF:
{code_change}

Lines removed:
{lines_removed}

Lines added:
{lines_added}

Summary of changes:
"""

CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT_NOLA = """

GIT DIFF:
{code_change}

Lines removed:
{lines_removed}

Summary of changes:
"""

CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT_NOLR = """

GIT DIFF:
{code_change}

Lines added:
{lines_added}

Summary of changes:
"""

GPT4_CODESUMM_LINECHANGE_IMPL_PROMPT = """Summarize the code changes in the GIT DIFF below and their possible implications. Make two sections called "summary of changes" and "implications" and write numbered points under each section. Don't make the points hierarchical.

GIT DIFF:
{code_change}

Lines removed:
{lines_removed}

Lines added:
{lines_added}

Summary of changes:
"""

GPT4_CODESUMM_LINECHANGE_IMPL_PROMPT_NOLA = """Summarize the code changes in the GIT DIFF below and their possible implications. Make two sections called "summary of changes" and "implications" and write numbered points under each section. Don't make the points hierarchical.

GIT DIFF:
{code_change}

Lines removed:
{lines_removed}

Summary of changes:
"""

GPT4_CODESUMM_LINECHANGE_IMPL_PROMPT_NOLR = """Summarize the code changes in the GIT DIFF below and their possible implications. Make two sections called "summary of changes" and "implications" and write numbered points under each section. Don't make the points hierarchical.

GIT DIFF:
{code_change}

Lines added:
{lines_added}

Summary of changes:
"""

MAGICODER_CODESUMM_LINECHANGE_PROMPT = """You are an exceptionally intelligent and experienced senior software engineer who is very skilled at reviewing code and providing reliable and helpful feedback. Now look at code change represented as a GIT DIFF as well as the added and removed lines and summarize all the important changes.

@@ GIT DIFF
{code_change}

@@ Lines removed
{lines_removed}

@@ Lines added
{lines_added}

@@ Code changes:"""

MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_PREFIX = """You are an exceptionally intelligent and experienced senior software engineer who is very skilled at reviewing code and providing reliable and helpful feedback. Now look at code change represented as a GIT DIFF as well as the added and removed lines and summarize all the important changes.\n\n@@ GIT DIFF\n@@ -182,7 +182,9 @@ abstract class AbstractSolrBackendFactory implements FactoryInterface\n      */\n     protected function createBackend(Connector $connector)\n     {\n+        $config = $this->config->get($this->mainConfig);\n         $backend = new $this->backendClass($connector);\n+        $backend->setPageSize($config->Index->record_batch_size);\n         $backend->setQueryBuilder($this->createQueryBuilder());\n         $backend->setSimilarBuilder($this->createSimilarBuilder());\n         if ($this->logger) {\n\n@@ Lines added\n$config = $this->config->get($this->mainConfig);\n$backend->setPageSize($config->Index->record_batch_size);\n\n@@ Code changes:"""+"""\nThe code added a new line to retrieve a configuration from the main configuration file. It then sets the page size of the backend to the value of the \"record_batch_size\" key in the configuration."""+"""\n\n@@ GIT DIFF
@@ -53,7 +53,7 @@ public class ProtocGapicPluginGeneratorTest {
                 model.getFiles().stream().map(ProtoFile::getProto).collect(Collectors.toList()))
             // Only the file to generate a client for (don't generate dependencies)
             .addFileToGenerate("multiple_services.proto")
-            .setParameter("language=java")
+            .setParameter("language=java,transport=grpc")
             .build();
 
     CodeGeneratorResponse response = ProtocGeneratorMain.generate(codeGeneratorRequest);

@@ Lines removed
.setParameter("language=java")

@@ Lines added
.setParameter("language=java,transport=grpc")

@@ Code changes:
The code change represented by the GIT DIFF replaces an existing line of code. The replaced line of code now sets the parameter "language=java,transport=grpc" instead of the original line of code that set the parameter as "language=java". This indicates that the code is now generating a gRPC client instead of a regular Java client."""

MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_NOLA = """

@@ GIT DIFF
{code_change}

@@ Lines removed
{lines_removed}

@@ Code changes:
"""

MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_NOLR = """

@@ GIT DIFF
{code_change}

@@ Lines added
{lines_added}

@@ Code changes:
"""

MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT = """

@@ GIT DIFF
{code_change}

@@ Lines removed
{lines_removed}

@@ Lines added
{lines_added}

@@ Code changes:
"""

MAGICODER_CODESUMM_LINECHANGE_PROMPT_NOLA = """You are an exceptionally intelligent and experienced senior software engineer who is very skilled at reviewing code and providing reliable and helpful feedback. Now look at code change represented as a GIT DIFF as well as the removed lines and summarize all the important changes.

@@ GIT DIFF
{code_change}

@@ Lines removed
{lines_removed}

@@ Code changes:"""

MAGICODER_CODESUMM_LINECHANGE_PROMPT_NOLR = """You are an exceptionally intelligent and experienced senior software engineer who is very skilled at reviewing code and providing reliable and helpful feedback. Now look at code change represented as a GIT DIFF as well as the added lines and summarize all the important changes.

@@ GIT DIFF
{code_change}

@@ Lines added
{lines_added}

@@ Code changes:"""

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

def test_magicoder_codesumm(code_change):
    prompt = MAGICODER_CODESUMM_PROMPT.format(code_change=code_change)
    generator = pipeline(
        model="ise-uiuc/Magicoder-S-DS-6.7B",
        task="text-generation",
        torch_dtype=torch.bfloat16,
        device="cuda:0",
    )
    result = generator(prompt, max_length=1024, num_return_sequences=1, temperature=0.0)
    
    return result

def gen_GPT_labels(
        save_dir: str, data_path: str="./data/Comment_Generation/msg-valid.jsonl", 
        model_name: str="gpt-4-0613", sample_size: int=1000, use_exemplars: bool=True,
        gen_implications: bool=False,
    ):
    import random
    os.makedirs(save_dir, exist_ok=True)
    gen_implications = True # overrides use exemplars.
    use_exemplars = True
    write_path = os.path.join(save_dir, model_name+".jsonl")
    # create a pool of samples that are not too long.
    ctr, data = 0, read_jsonl(data_path)
    candidates = []
    for rec in data:
        if 10 <= len(rec['patch']) <= 1000:
            ctr += 1
            candidates.append(rec)
    print(f"{ctr} samples between 10 and 2000 chars")
    random.seed(42)
    sampled_data = random.sample(candidates, k=sample_size)
    print(len(sampled_data))

    # check if user wants to overwrite the data.
    if os.path.exists(write_path):
        overwrite_file = input("overwrite (y/n)?").lower().strip()
        if overwrite_file not in ["y", "yes"]: exit()
    open(write_path, "w")
    id = 0
    preds = []
    responses = []
    instructions = []
    LLM = OpenAIEngine(path=model_name)
    for rec in tqdm(sampled_data, desc=f"{model_name} inference"):
        # if skip is not None and id < skip: 
        #     id += 1
        #     continue
        # code change summarize
        lines_added, lines_removed = generate_added_and_removed_lines_from_patch(rec['patch'])
        NLA = len(lines_added)
        NLR = len(lines_removed)
        lines_added = "\n".join(lines_added)[:4000]
        lines_removed = "\n".join(lines_removed)[:4000]

        if NLA != 0 and NLR != 0:
            if gen_implications:
                prompt = GPT4_CODESUMM_LINECHANGE_IMPL_PROMPT.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                    lines_added=lines_added,
                )  
            elif use_exemplars:
                prompt = MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_PREFIX + MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                    lines_added=lines_added,
                ) 
            else: 
                prompt = MAGICODER_CODESUMM_LINECHANGE_PROMPT.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                    lines_added=lines_added,
                ) 
        elif NLR == 0:
            if gen_implications:
                prompt = GPT4_CODESUMM_LINECHANGE_IMPL_PROMPT_NOLR.format(
                    code_change=rec['patch'][:4000],
                    lines_added=lines_added,
                ) 
            elif use_exemplars:
                prompt = MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_PREFIX + MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_NOLR.format(
                    code_change=rec['patch'][:4000],
                    lines_added=lines_added,
                ) 
            else:
                prompt = MAGICODER_CODESUMM_LINECHANGE_PROMPT_NOLR.format(
                    code_change=rec['patch'][:4000],
                    lines_added=lines_added,
                )
        elif NLA == 0:
            if gen_implications:
                prompt = GPT4_CODESUMM_LINECHANGE_IMPL_PROMPT_NOLA.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                )
            elif use_exemplars:
                prompt = MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_PREFIX + MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_NOLA.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                ) 
            else: 
                prompt = MAGICODER_CODESUMM_LINECHANGE_PROMPT_NOLA.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                )
        else: continue   
        op = LLM(prompt)
        instructions.append(prompt)
        responses.append(op)
        with open(write_path, "a") as f:
            f.write(json.dumps({"id": id, "code_change": rec['patch'], 'prompt': prompt, 'change_summary': op})+"\n")
            preds.append(rec['patch'])
        id += 1

    dataset = Dataset.from_dict({"instruction": instructions, "response": responses})
    dataset.save_to_disk("./data/GPT_code_change_summ_labels.hf") 

def main_no_pipeline(
        save_dir: str, data_path: str="./data/Comment_Generation/msg-test.jsonl", 
        model_path: str="ise-uiuc/Magicoder-S-DS-6.7B", device: str="cuda:0",
        model_name: str="Magicoder-S-DS-6.7B", task: str="text-generation",
        skip: Union[int, None]=None, checkpoint_path:Union[str, None]=None,
        gen_implications: bool=False, use_exemplars: bool=False
    ):
    os.makedirs(save_dir, exist_ok=True)
    # use_exemplars = True
    write_path = os.path.join(save_dir, model_name+".jsonl")
    data = read_jsonl(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path if checkpoint_path is not None else model_path, device_map="auto")
    # model.to(device)
    if os.path.exists(write_path):
        overwrite_file = input("overwrite (y/n)?").lower().strip()
        if overwrite_file not in ["y", "yes"]: exit()
    open(write_path, "w")
    id = 0
    preds = []
    # golds = []
    patch_lengths = []
    for rec in tqdm(data, desc=f"{model_name} inference"):
        if skip is not None and id < skip: 
            id += 1
            patch_lengths.append(len(rec['patch']))
            continue
        # print("max patch legth till now:", max(patch_lengths))
        # prompt = MAGICODER_CODESUMM_PROMPT.format(code_change=rec['patch'][:5000])

        # code change summarize
        lines_added, lines_removed = generate_added_and_removed_lines_from_patch(rec['patch'])
        NLA = len(lines_added)
        NLR = len(lines_removed)
        lines_added = "\n".join(lines_added)[:4000]
        lines_removed = "\n".join(lines_removed)[:4000]

        if NLA != 0 and NLR != 0:
            if gen_implications:
                if use_exemplars:
                    prompt = CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT_PREFIX+CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT.format(
                        code_change=rec['patch'][:4000],
                        lines_removed=lines_removed,
                        lines_added=lines_added,                        
                    )
                else:
                    prompt = GPT4_CODESUMM_LINECHANGE_IMPL_PROMPT.format(
                        code_change=rec['patch'][:4000],
                        lines_removed=lines_removed,
                        lines_added=lines_added,
                    )  
            elif use_exemplars:
                prompt = MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_PREFIX + MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                    lines_added=lines_added,
                ) 
            else: 
                prompt = MAGICODER_CODESUMM_LINECHANGE_PROMPT.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                    lines_added=lines_added,
                ) 
        elif NLR == 0:
            if gen_implications:
                if use_exemplars:
                    prompt = CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT_PREFIX+CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT_NOLR.format(
                        code_change=rec['patch'][:4000],
                        lines_added=lines_added,  
                    )
                else:
                    prompt = GPT4_CODESUMM_LINECHANGE_IMPL_PROMPT_NOLR.format(
                        code_change=rec['patch'][:4000],
                        lines_added=lines_added,
                    ) 
            elif use_exemplars:
                prompt = MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_PREFIX + MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_NOLR.format(
                    code_change=rec['patch'][:4000],
                    lines_added=lines_added,
                ) 
            else:
                prompt = MAGICODER_CODESUMM_LINECHANGE_PROMPT_NOLR.format(
                    code_change=rec['patch'][:4000],
                    lines_added=lines_added,
                )
        elif NLA == 0:
            if gen_implications:
                if use_exemplars:
                    prompt = CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT_PREFIX+CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT_NOLA.format(
                        code_change=rec['patch'][:4000],
                        lines_removed=lines_removed,                  
                    )
                else:
                    prompt = GPT4_CODESUMM_LINECHANGE_IMPL_PROMPT_NOLA.format(
                        code_change=rec['patch'][:4000],
                        lines_removed=lines_removed,
                    ) 
            elif use_exemplars:
                prompt = MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_PREFIX + MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_NOLA.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                ) 
            else: 
                prompt = MAGICODER_CODESUMM_LINECHANGE_PROMPT_NOLA.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                )   
        # print("added:", NLA)
        # print("removed:", NLR)
        # print("prompt:", len(prompt))   
        # code_before, code_after = generate_before_after_code_from_patch(rec['patch'])
        # prompt = MAGICODER_CODESUMM_CODECHANGE_PROMPT.format(
        #     code_before=code_before, code_after=code_after
        # )

        # print("size of patch causing error:", len(rec['patch']))
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150, temperature=0.2)
        op = tokenizer.decode(outputs[0]).replace(prompt,'').replace('<｜end▁of▁sentence｜>','')
        # avoid saving very long versions of prompts.
        prompt_to_save = prompt.replace(CODESTRAL_CODESUMM_LINECHANGE_EXEMPLARS_IMPL_PROMPT_PREFIX, "")

        with open(write_path, "a") as f:
            f.write(json.dumps({"id": id, "code_change": rec['patch'], 'instruction': prompt_to_save, 'response': op}, ensure_ascii=False)+"\n")
            # golds.append(rec['msg'])
            preds.append(rec['patch'])
        id += 1

def main(save_dir: str, data_path: str="./data/Comment_Generation/msg-test.jsonl", 
         model_path: str="ise-uiuc/Magicoder-S-DS-6.7B", device: str="cuda:0",
         model_name: str="Magicoder-S-DS-6.7B", task: str="text-generation",
         skip: Union[int, None]=None):
    os.makedirs(save_dir, exist_ok=True)
    use_exemplars = True
    write_path = os.path.join(save_dir, model_name+".jsonl")
    data = read_jsonl(data_path)
    generator = pipeline(
        model=model_path, task=task,
        torch_dtype=torch.bfloat16,
        device=device,
    )
    if os.path.exists(write_path):
        overwrite_file = input("overwrite (y/n)?").lower().strip()
        if overwrite_file not in ["y", "yes"]: exit()
    open(write_path, "w")
    id = 0
    preds = []
    # golds = []
    patch_lengths = []
    for rec in tqdm(data, desc=f"{model_name} inference"):
        if skip is not None and id < skip: 
            id += 1
            patch_lengths.append(len(rec['patch']))
            continue
        # print("max patch legth till now:", max(patch_lengths))
        # prompt = MAGICODER_CODESUMM_PROMPT.format(code_change=rec['patch'][:5000])

        # code change summarize
        lines_added, lines_removed = generate_added_and_removed_lines_from_patch(rec['patch'])
        NLA = len(lines_added)
        NLR = len(lines_removed)
        lines_added = "\n".join(lines_added)[:4000]
        lines_removed = "\n".join(lines_removed)[:4000]

        if NLA != 0 and NLR != 0:
            if use_exemplars:
                prompt = MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_PREFIX + MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                    lines_added=lines_added,
                ) 
            else: 
                prompt = MAGICODER_CODESUMM_LINECHANGE_PROMPT.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                    lines_added=lines_added,
                ) 
        elif NLR == 0:
            if use_exemplars:
                prompt = MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_PREFIX + MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_NOLR.format(
                    code_change=rec['patch'][:4000],
                    lines_added=lines_added,
                ) 
            else:
                prompt = MAGICODER_CODESUMM_LINECHANGE_PROMPT_NOLR.format(
                    code_change=rec['patch'][:4000],
                    lines_added=lines_added,
                )
        elif NLA == 0:
            if use_exemplars:
                prompt = MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_PREFIX + MAGICODER_CODESUMM_LINECHANGE_EXEMPLARS_PROMPT_NOLA.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                ) 
            else: 
                prompt = MAGICODER_CODESUMM_LINECHANGE_PROMPT_NOLA.format(
                    code_change=rec['patch'][:4000],
                    lines_removed=lines_removed,
                )   
        # print("added:", NLA)
        # print("removed:", NLR)
        # print("prompt:", len(prompt))   
        # code_before, code_after = generate_before_after_code_from_patch(rec['patch'])
        # prompt = MAGICODER_CODESUMM_CODECHANGE_PROMPT.format(
        #     code_before=code_before, code_after=code_after
        # )

        # print("size of patch causing error:", len(rec['patch']))
        result = generator(prompt, max_length=1024, num_return_sequences=1, temperature=0.2)
        op = result[0]['generated_text'].replace(prompt,'')
        with open(write_path, "a") as f:
            f.write(json.dumps({"id": id, "code_change": rec['patch'], 'instruction': prompt, 'response': op})+"\n")
            # golds.append(rec['msg'])
            preds.append(rec['patch'])
        id += 1
    # bleu_with_stop = bleu_fromstr(predictions=preds, golds=golds, rmstop=False)
    # print(f"bleu_with_stop = {bleu_with_stop}")
    # bleu_without_stop = bleu_fromstr(predictions=preds, golds=golds, rmstop=True)
    # print(f"bleu_without_stop = {bleu_without_stop}")

# main
if __name__ == "__main__":
    # main(save_dir="./experiments/code_change_summ_v3")

    # main(
    #     save_dir="./experiments/code_change_summ_v3", 
    #     model_path="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    #     model_name="deepseek-coder-7b-instruct-v1.5",
    # )

    # gen_GPT_labels("./experiments/GPT_code_change_summ_labels")
    # gen_GPT_labels("./experiments/GPT_code_change_summ_labels_with_impl", gen_implications=True, sample_size=500)

    # fine-tuned Magicoder generate claims with implications:
    # main_no_pipeline(
    #     save_dir="./experiments/code_change_summ_finetune_impl", 
    #     checkpoint_path="/data/tir/projects/tir3/users/arnaik/magicoder_code_change_summ_impl",
    #     gen_implications=True,
    # )

    # Codestral generate claims with implications:
    main_no_pipeline(
        save_dir="./experiments/code_change_summ_finetune_impl", 
        model_path="mistralai/Codestral-22B-v0.1", model_name="codestral-22b",
        gen_implications=True, use_exemplars=True,
    )

# generator = pipeline(
#     model="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
#     task="text-generation",
#     torch_dtype=torch.bfloat16,
#     device="cuda:0",
# )