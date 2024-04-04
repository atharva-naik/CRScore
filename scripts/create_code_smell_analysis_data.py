import os
import json
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

def generate_newf(oldf, diff) -> str:
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
        return ""
    
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

    return "\n".join(prevlines+lines+afterlines)

# main
if __name__ == "__main__":
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    for i,rec in enumerate(data):
        content = generate_newf(rec['oldf'], rec['patch'])
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
            