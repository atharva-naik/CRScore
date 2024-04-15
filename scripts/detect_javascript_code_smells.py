import os
from tqdm import tqdm

output_dir = "./experiments/javascript_code_smells"
os.makedirs(output_dir, exist_ok=True)
# os.system()
JAVASCRIPT_PROJS_DIR = "/home/arnaik/code-review-test-projects/javascript"
for proj_name in tqdm(os.listdir(JAVASCRIPT_PROJS_DIR)):
    smell_output_file = os.path.join(output_dir, f"{proj_name}.txt")
    proj_file = os.path.join(JAVASCRIPT_PROJS_DIR, proj_name, f"{proj_name}.js")
    os.system(f"jshint {proj_file}")