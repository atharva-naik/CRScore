import os
import sys
import shlex
import fnmatch
from shutil import *
from tqdm import tqdm
from src import detectorv2
from subprocess import run, PIPE

def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        print(arg)

def file_extractor(proj_path):
    if len(sys.argv) != 2:
        print("target directory not specified")

    matches = []
    path = os.path.join("code-dump", os.path.basename(proj_path))
    if not os.path.exists(path):
        os.makedirs(path)
    dst = os.path.join("code-dump", os.path.basename(proj_path))

    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    for root, dirnames, filenames in os.walk(proj_path):
        for filename in fnmatch.filter(filenames, '*.py'):
            matches.append(os.path.join(root, filename))

    for idx,src in enumerate(matches):
        copy(src, dst)
        # print(dst+"/"+os.path.basename(src))
        os.rename(dst+"/"+os.path.basename(src), dst+"/"+str(idx)+os.path.basename(src))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("target directory not specified")
    # file_extractor(sys.argv[1])
    pbar = tqdm(os.listdir(sys.argv[1]))
    err_count, tot = 0, 0
    for project in pbar:
        project_path = os.path.join(sys.argv[1], project)
        tot += 1
        try: 
            detectorv2.detect_main(
                directory=project_path,
                #os.path.join("./code-dump/", os.path.basename(sys.argv[1])), 
                logdir="./detected_smells",
            )
        except SyntaxError as e:
            err_count += 1
            print(f"\x1b[31;mSyntax ERROR in {project}:\x1b[0m{e}")
        # except Exception as e:
        #     err_count += 1
        #     print(f"\x1b[31;mERROR in {project}:\x1b[0m{e}")
            # exit()
        pbar.set_description(f"{err_count}/{tot} ERRORs")
    print('*****     Output Generated     *****')

#if len(sys.argv) != 2:
#    print ("target directory not specified")
#cmd = "python3 tools/py_file_extractor ./sourcecode/"+sys.argv[1]
#result = run(shlex.split(cmd), stdout=PIPE)

#detect_main("./code-dump/"+sys.argv[1])



