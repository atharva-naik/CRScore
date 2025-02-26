import pandas as pd
import numpy as np

import ast
from tqdm import tqdm

import esprima
from pygments import lex
from pygments.lexers import JavaLexer
from pygments.token import Error

tqdm.pandas()

def check_syntax_java(code: str) -> bool:
    try:
        tokens = list(lex(code, JavaLexer()))
        return not any(token[0] is Error for token in tokens)
    except Exception:
        return False

def check_syntax_javascript(code: str) -> bool:
    try:
        esprima.parseScript(code)
        return True
    except esprima.Error:
        return False

def check_syntax(code: str, language: str) -> bool:
    if language.lower() == "java":
        return check_syntax_java(code)
    elif language.lower() == "javascript":
        return check_syntax_javascript(code)
    else:
        raise ValueError("Unsupported language. Use 'java' or 'javascript'.")

    

df = pd.read_json("/data/datasets/hf_cache/mkapadni/crscore_plus_plus/dataset/msg-train_js_classified.jsonl", lines=True)

print("Original shape:", df.shape)
print("Columns:", df.columns)

# Filter out patches with invalid Python syntax

df['is_valid_syntax'] = df['oldf'].progress_apply(check_syntax_javascript)
df = df[df['is_valid_syntax']].reset_index(drop=True)
print("Shape after filtering out invalid syntax:", df.shape)


df.to_json("/home/mkapadni/work/crscore_plus_plus/Comment_Generation/msg-train_js_classified_valid_syntax.json", lines=True, orient='records')

