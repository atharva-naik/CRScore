import os
import json
import time
import requests
import pandas as pd
from typing import *
from tqdm import tqdm
from github import Github
from urllib.parse import quote_plus

def get_files_and_raw_urls(github_repo_url, github_token=None):
    # Extract username and repository name from the GitHub URL
    username, repo_name = github_repo_url.split('/')[-2:]

    # Create a Github instance
    g = Github(github_token)

    # Get the repository object
    repo = g.get_repo(f"{username}/{repo_name}")

    # Get the list of contents (files and directories) in the root of the repository
    contents = repo.get_contents("")

    # Get the list of files and generate raw GitHub file URLs
    file_list = [content.name for content in contents if content.type == "file"]
    raw_urls = [
        # f"{github_repo_url}/raw/{quote_plus(content.sha)}/{quote_plus(content.path)}"
        f"https://raw.githubusercontent.com/{username}/{repo_name}/main/{quote_plus(content.path)}"
        for content in contents
        if content.type == "file"
    ]

    return file_list, raw_urls

def get_all_files_and_raw_urls(github_repo_url, github_token=None, filter_by_ext: List[str]=[".py", ".go", ".js", ".java", ".cs", ".rb", ".php", ".c", ".cpp", ".cs"]):
    # Extract username and repository name from the GitHub URL
    username, repo_name = github_repo_url.split('/')[-2:]
    # Create a Github instance
    g = Github(github_token)
    # Get the repository object
    repo = g.get_repo(f"{username}/{repo_name}")
    data = []
    # Function to recursively get files and raw URLs
    def get_files_recursively(contents):
        data = []
        for content in contents:
            if content.type == "file":
                _, ext = os.path.splitext(content.path)
                rec = {}
                rec["lang"] = ext.replace(".","")
                rec["reponame"] = repo_name
                rec["username"] = username
                rec["path"] = content.path
                if ext not in filter_by_ext: continue
                raw_main_url = f"https://raw.githubusercontent.com/{username}/{repo_name}/main/{quote_plus(content.path)}" 
                rec["raw_main_url"] = raw_main_url
                # f"{github_repo_url}/raw/{quote_plus(content.sha)}/{quote_plus(content.path)}"
                raw_master_url = f"https://raw.githubusercontent.com/{username}/{repo_name}/master/{quote_plus(content.path)}"
                rec["raw_master_url"] = raw_master_url
                # download file content/text.
                # use main as the default branch and fallback to master if main not present.
                resp = requests.get(raw_main_url)
                if resp.status_code != 200:
                    resp = requests.get(raw_master_url)
                rec["content"] = resp.text   
                data.append(rec)
                with open("./data/Comment_Generation/all_repo_data.jsonl", "a") as f:
                    f.write(json.dumps(rec)+"\n")
                time.sleep(1)
            elif content.type == "dir":
                # Recursively get files from subdirectories
                subcontents = repo.get_contents(content.path)
                subdata = get_files_recursively(subcontents)
                data.extend(subdata)

        return data

    # Get the list of contents (files and directories) in the root of the repository
    root_contents = repo.get_contents("")

    # Get all files and raw URLs recursively
    data = get_files_recursively(root_contents)

    return data

creds = json.load(open("./src/scraping/gh_access_token.json"))
# Optional: If the repository is private, provide a GitHub personal access token
github_token = creds["access_token"]

if __name__ == "__main__":
    open("./data/Comment_Generation/all_repo_data.jsonl", "w")
    all_valtest_gitrepos_final = pd.read_csv("./data/Comment_Generation/all_valtest_gitrepos_final.csv").to_dict("records")[30:]
    all_data = []

    for rec in tqdm(all_valtest_gitrepos_final):
        github_repo_url = rec["url (actual)"]
        repo_data = get_all_files_and_raw_urls(github_repo_url, github_token=github_token)
        all_data.extend(repo_data)
    
    with open("./data/Comment_Generation/all_repo_data.json", "w") as f:
        json.dump(all_data, f, indent=4)