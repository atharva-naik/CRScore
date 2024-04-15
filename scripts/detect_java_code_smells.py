import os
from tqdm import tqdm

output_dir = "./experiments/java_code_smells"
os.makedirs(output_dir, exist_ok=True)
# os.system()
JAVA_PROJS_DIR = "/home/arnaik/code-review-test-projects/java"
for proj_name in tqdm(os.listdir(JAVA_PROJS_DIR)):
    smell_output_file = os.path.join(output_dir, f"{proj_name}.txt")
    proj_dir = os.path.join(JAVA_PROJS_DIR, proj_name)
    os.system(f"/home/arnaik/pmd-bin-7.0.0/bin/pmd check -d {proj_dir} -R rulesets/java/quickstart.xml -f text -r {smell_output_file}")