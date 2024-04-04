# LLM-prompting based approach to review generation.
import os
import re
import json
import torch
import warnings
from typing import *
from tqdm import tqdm
from transformers import logging
from transformers import pipeline
from src.datautils import read_jsonl, remove_patch_header, generate_added_and_removed_lines_from_patch
# from CodeBERT.CodeReviewer.code.evaluator.smooth_bleu import bleu_fromstr

logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=UserWarning)

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

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
            f.write(json.dumps({"id": id, "code_change": rec['patch'], 'prompt': prompt, 'change_summary': op})+"\n")
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
    main(
        save_dir="./experiments/code_change_summ_v3", 
        model_path="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        model_name="deepseek-coder-7b-instruct-v1.5",
    )

# generator = pipeline(
#     model="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
#     task="text-generation",
#     torch_dtype=torch.bfloat16,
#     device="cuda:0",
# )