import os
import json
import torch
from typing import List, Dict, Union, Any, Optional
from tqdm import tqdm
from collections import defaultdict

# Choose one of these based on your needs
from openai import OpenAI  # For GPT models
# For Magicoder or other transformer models
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Constants from original code
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
    def __init__(self, model: str, api_key: Optional[str]=None):
        self.model = model
        if model.startswith("gpt"):
            self.client = OpenAI(api_key=api_key)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.client = pipeline(
                model=model,
                task="text-generation",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

    def __call__(self, code_change: str, review: str, lang: str):
        if self.model.startswith("gpt"):
            inst_to_be_judged = CODE_CHANGE_AND_REVIEW_JUDGE_PROMPT.format(
                lang=LANG_MAP[lang], 
                code_change=code_change[:5000], 
                review=review
            )
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
            inst_to_be_judged = CODE_CHANGE_AND_REVIEW_JUDGE_PROMPT.format(
                lang=LANG_MAP[lang], 
                code_change=code_change[:5000], 
                review=review
            )
            prompt = CODE_CHANGE_AND_REVIEW_SYSTEM_PROMPT + "\n" + inst_to_be_judged
            result = self.client(
                prompt, max_new_tokens=5, 
                num_return_sequences=1, temperature=0.0
            )
            response = result[0]["generated_text"].replace(prompt,'').split("\n")[0].strip()
            
        # Parse the score from the response
        if response.startswith("1"): score = 1
        elif response.startswith("2"): score = 2
        elif response.startswith("3"): score = 3
        elif response.startswith("4"): score = 4
        elif response.startswith("5"): score = 5
        else: score = 1

        return score/5, inst_to_be_judged


def score_code_reviews(
    code_diffs: List[str], 
    code_reviews: List[str], 
    language_tags: List[str], 
    indices: Optional[List[int]] = None,
    system_names: Optional[List[str]] = None,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate the relevance of code reviews for given code differences.
    
    Args:
        code_diffs (List[str]): List of code changes/diffs
        code_reviews (List[str]): List of reviews to evaluate
        language_tags (List[str]): List of language tags ('py', 'js', 'java')
        indices (Optional[List[int]]): Optional indices for each sample
        system_names (Optional[List[str]]): Optional name of the system that generated each review
        model (str): Model to use as judge ('gpt-4o' or a HuggingFace model name)
        api_key (Optional[str]): API key for OpenAI (required for GPT models)
        output_file (Optional[str]): Path to save results, if specified
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing evaluation results
    """
    if len(code_diffs) != len(code_reviews) or len(code_diffs) != len(language_tags):
        raise ValueError("code_diffs, code_reviews, and language_tags must have the same length")
    
    # Initialize the judge
    judge = LLM_as_a_Judge(model=model, api_key=api_key)
    
    # Create indices and system names if not provided
    if indices is None:
        indices = list(range(len(code_diffs)))
    if system_names is None:
        system_names = ["model"] * len(code_diffs)
    
    # Prepare results
    results = []
    
    # Evaluate each review
    for i, (code_diff, review, lang, idx, system) in enumerate(
        tqdm(zip(code_diffs, code_reviews, language_tags, indices, system_names), 
             total=len(code_diffs))
    ):
        # Get score from judge
        score, prompt = judge(code_change=code_diff, review=review, lang=lang)
        
        # Create result record
        result = {
            "index": idx,
            "system": system,
            "lang": lang,
            "diff": code_diff,
            "review": review,
            "score": score,
            "prompt": prompt
        }
        
        results.append(result)
        
        # Write to file if specified
        if output_file:
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
    
    return results


# Example usage
if __name__ == "__main__":
    # Example data
    diffs = [
        "def add(a, b):\n    return a + b\n\n# Changed to\ndef add(a, b):\n    return a + b + 0  # Added unnecessary zero",
        "function greet(name) {\n    console.log('Hello ' + name);\n}\n\n// Changed to\nfunction greet(name) {\n    console.log(`Hello ${name}`);\n}"
    ]
    
    reviews = [
        "The function was modified to add zero to the result, which is unnecessary and might confuse readers.",
        "Changed string concatenation to template literals, which is a more modern approach."
    ]
    
    languages = ["py", "js"]
    
    # Make sure to set your API key if using GPT models
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Score the reviews
    results = score_code_reviews(
        code_diffs=diffs,
        code_reviews=reviews,
        language_tags=languages,
        model="gpt-4o",  # or "ise-uiuc/Magicoder-S-DS-6.7B"
        api_key=api_key,
        output_file="review_scores.jsonl"
    )
    
    print(f"Evaluated {len(results)} reviews")
    for i, result in enumerate(results):
        print(f"Review {i+1} score: {result['score']:.2f}")