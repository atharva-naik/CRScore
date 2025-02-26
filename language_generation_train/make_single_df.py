import pandas as pd
import numpy as np
import os
from tqdm import tqdm


files_list = {"c":"/data/user_data/mkapadni/crscore_plus_plus/dataset/msg-train_c_classified.jsonl",
              "go":"/data/user_data/mkapadni/crscore_plus_plus/dataset/msg-train_go_classified.jsonl",
              "php":"/data/user_data/mkapadni/crscore_plus_plus/dataset/msg-train_php_classified.jsonl",
              "cpp":"/data/user_data/mkapadni/crscore_plus_plus/dataset/msg-train_cpp_classified.jsonl",
              "ruby":"/data/user_data/mkapadni/crscore_plus_plus/dataset/msg-train_ruby_classified.jsonl",
              "java":"/home/mkapadni/work/crscore_plus_plus/Comment_Generation/msg-train_java_classified_valid_syntax.json",
              "js":"/home/mkapadni/work/crscore_plus_plus/Comment_Generation/msg-train_js_classified_valid_syntax.json",
              "python":"/home/mkapadni/work/crscore_plus_plus/Comment_Generation/msg-train_python_classified_valid_syntax.json"}

# make single big df with these columns - 'patch', 'y', 'oldf', 'idx', 'id', 'msg', 'proj', 'lang', 'raw_score', 'int_score'



df = pd.DataFrame(columns=['patch', 'y', 'oldf', 'idx', 'id', 'msg', 'proj', 'lang', 'raw_score', 'int_score'])

# This will hold all DataFrames temporarily
df_list = []

for k, v in files_list.items():
    # open datafram
    temp_df = pd.read_json(v, lines=True)
    
    # rewrite the "raw_k_score" to "raw_score" and "int_k_score" to "int_score"
    temp_df = temp_df.rename(columns={"raw_" + k + "_score": "raw_score", "int_" + k + "_score": "int_score"})
    
    # add the language column
    temp_df['lang'] = k
    
    # Add the DataFrame to the list
    df_list.append(temp_df)
    print(f"Added {k} to the big df")

# After the loop, concatenate all DataFrames in the list
df = pd.concat(df_list, ignore_index=True)


print(df.shape)
print(df.columns)
df.to_json("/data/user_data/mkapadni/crscore_plus_plus/dataset/msg-train_all_classified.jsonl", lines=True, orient='records')
df.to_json("/home/mkapadni/work/crscore_plus_plus/Comment_Generation/msg-train_all_classified.json", orient='records')