import os
import json
from typing import *
from tqdm import tqdm
from src.datautils import read_jsonl

def remove_space_clean(line):
    """
        Remove start and end empty chars.
    """
    rep = " \t\r"
    totallen = len(line)
    i = 0
    while i < totallen and line[i] in rep:
        i += 1
    j = totallen - 1
    while j >= 0 and line[j] in rep:
        j -= 1
    line = line[i : j + 1]
    return line

def generate_newf(oldf, diff) -> Tuple[str, Tuple[int, int]]:
    import re

    oldflines = oldf.split("\n")
    difflines = diff.split("\n")
    first_line = difflines[0]
    difflines = difflines[1:]
    difflines = [line for line in difflines if line != r"\ No newline at end of file"]
    regex = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
    matchres = re.match(regex, first_line)
    avail = None
    
    if matchres:
        startline, rangelen, startpos, endpos = matchres.groups()
        avail = True
    else:
        avail = False
        return "", (-1,-1)
    
    startline, rangelen = int(startline) - 1, int(rangelen)
    endline = startline + rangelen
    prevlines = oldflines[:startline]
    afterlines = oldflines[endline:]
    lines = []
    
    for line in difflines:
        if line.startswith("-"): pass
        elif line.startswith("+"):
            lines.append(line[1:])
        else: 
            if line[0] == " ": line = line[1:]
            lines.append(line)

    prevlines = [line for line in prevlines]
    afterlines = [line for line in afterlines]
    lines = [line for line in lines]
    merged_lines = prevlines+lines+afterlines
    patch_lines = (len(prevlines)+1, len(prevlines)+len(lines))

    return "\n".join(merged_lines), patch_lines

# main
if __name__ == "__main__":
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    all_patch_lines = {}
    for i,rec in tqdm(enumerate(data)):
        content, patch_lines = generate_newf(rec['oldf'], rec['patch'])
        all_patch_lines[f"test{i}"] = patch_lines
        if rec['lang'] == "py":
            folder = os.path.join("/home/arnaik/code-review-test-projects/python", f"test{i}")
            file = os.path.join(folder, f"test{i}.py")
            os.makedirs(folder, exist_ok=True)
            with open(file, "w") as f:
                f.write(content+"\n")
        elif rec["lang"] == "java":
            folder = os.path.join("/home/arnaik/code-review-test-projects/java", f"test{i}")
            file = os.path.join(folder, f"test{i}.java")
            os.makedirs(folder, exist_ok=True)
            with open(file, "w") as f:
                f.write(content+"\n")
        elif rec["lang"] == "js":
            folder = os.path.join("/home/arnaik/code-review-test-projects/javascript", f"test{i}")
            file = os.path.join(folder, f"test{i}.js")
            os.makedirs(folder, exist_ok=True)
            with open(file, "w") as f:
                f.write(content+"\n")
        with open("/home/arnaik/code-review-test-projects/patch_lines.json", "w") as f:
            json.dump(all_patch_lines, f, indent=4)