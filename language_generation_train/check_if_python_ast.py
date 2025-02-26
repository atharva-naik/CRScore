import pandas as pd
import numpy as np

import ast
from tqdm import tqdm

tqdm.pandas()

def is_syntax_valid(code: str) -> bool:
    """
    Checks whether the given code string is syntactically correct Python.
    
    :param code: A string containing Python code
    :return: True if the code is valid, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    
df = pd.read_json("/data/datasets/hf_cache/mkapadni/crscore_plus_plus/dataset/msg-train_python_classified.jsonl", lines=True).reset_index(drop=True)
print("Original shape:", df.shape)
print("Columns:", df.columns)

# Filter out patches with invalid Python syntax

df['is_valid_syntax'] = df['oldf'].progress_apply(is_syntax_valid)
df = df[df['is_valid_syntax']].reset_index(drop=True)
print("Shape after filtering out invalid syntax:", df.shape)


df.to_json("/home/mkapadni/work/crscore_plus_plus/Comment_Generation/msg-train_python_classified_valid_syntax.json", lines=True, orient='records')

