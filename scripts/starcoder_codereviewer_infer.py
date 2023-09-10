import re
from text_generation import Client
from src.datautils import read_jsonl

def remove_patch_header(code: str):
    # Define the regex pattern to match expressions like "@@ -53,7 +53,7 @@" or "@@ -12,8 +12,20 @@"
    pattern = r'@@ -\d+,\d+ \+\d+,\d+ @@'

    # Use re.sub() with count=1 to replace only the first matched pattern with an empty string
    result = re.sub(pattern, '', code, count=1)

    return result

def generate_before_after_code_from_patch(patch: str):
    patch = remove_patch_header(patch).strip()
    old_lines = []
    new_lines = []
    for line in patch.split("\n"):
        if line.startswith("+"):
            line = line[1:]
            new_lines.append(line)
        elif line.startswith("-"):
            line = line[1:]
            old_lines.append(line)
        else:
            new_lines.append(line)
            old_lines.append(line)
    return "\n".join(old_lines), "\n".join(new_lines)

if __name__ == "__main__":
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    client = Client("http://tir-1-32:8880", timeout=60)
    print(client.generate("def read_file(filename):\n", max_new_tokens=64, top_p=0.95, temperature=0.2, do_sample=True).generated_text)