# collect Code Review data from Pull Requests.

import os
import json
from typing import *
from tqdm import tqdm
# Authentication is defined via github.Auth
from github import Auth
from github import Github
from collections import defaultdict
from backoff import on_exception, expo
from ratelimit import limits, RateLimitException
from github.GithubException import RateLimitExceededException

@on_exception(expo, RateLimitException, max_tries=32)
@limits(calls=2.5, period=1) # at most 2 calls/second
def rate_conscious_api_call(obj, call, *args, **kwargs):
    try: return getattr(obj, call)(*args)
    except RateLimitExceededException:
        print("rate limit exceeded")
        raise RateLimitException()

# scraper class.
class GithubPullRequestCommentsScraper:
    def __init__(self, github_object):
        self.data = []
        self.g = github_object

    def build_old_file_index(self, pull_request):
        file_histories = defaultdict(lambda:[])
        for commit in pull_request.get_commits():
            for file in commit.files:
                file_histories[file.filename].append(file)
        old_file_index = {}
        for filename, file_history in file_histories.items():
            old_file_index[filename] = {}
            for i in range(1, len(file_history)):
                curr_file_sha = file_history[i].sha
                old_file = file_history[i-1]
                old_file_index[filename][curr_file_sha] = old_file 

        return old_file_index

    def run(self, repo_names: List[str], file_name: str, 
            reset_file: bool=False, filt_no_comments: bool=True):
        if reset_file: open(file_name, "w")
        for repo_name in repo_names:
            repo = rate_conscious_api_call(self.g, "get_repo", repo_name) # self.g.get_repo(repo_name)
            # main_pulls = repo.get_pulls(state='closed', base='main')
            # master_pulls = repo.get_pulls(state='closed', base='master')
            all_closed_pulls = rate_conscious_api_call(repo, "get_pulls", state="closed") # repo.get_pulls(state='closed')
            for pr in tqdm(all_closed_pulls, desc=repo_name):
                # print(old_commit_index)
                # old_file_index = self.build_old_file_index(pr)
                rec = {}
                rec["pr_id"] = pr.id
                rec["pr_title"] = pr.title
                rec["pr_poster"] = pr.user.login
                rec["pr_body"] = pr.body
                rec["repo_url"] = repo.url
                rec["repo_name"] = repo_name
                commit_map = {commit.sha: commit for commit in rate_conscious_api_call(pr, "get_commits")}
                # old_commit_index = {commits[0].sha: commits[0]}
                # for i in range(1, len(commits)):
                #     old_commit_index[commits[i].sha] = commits[i-1]
                rec["comments"] = [self.extract_comment_json(comment, repo, commit_map) for comment in rate_conscious_api_call(pr, "get_review_comments")]
                if filt_no_comments and len(rec["comments"]) == 0: continue
                self.data.append(rec)
                with open(file_name, "a") as f:
                    f.write(json.dumps(rec)+"\n")
            # for pr in master_pulls:
            #     rec = {}
            #     rec["repo_name"] = repo_name
            #     rec["user"] = pr.user.login
            #     rec["pr_title"] = pr.title
            #     rec["comments"] = [self.extract_comment_json(comment) for comment in pr.get_comments()]
            #     self.data.append(rec)

    def extract_comment_json(self, comment, repo, commit_map):
        # prev_commit = old_commit_index[comment.original_commit_id]
        # print(comment.diff_hunk)
        file_raw_url = ""
        file_blob_url = ""
        try: 
            commit = commit_map[comment.original_commit_id]
            try: 
                file = [file for file in commit.files if file.filename == comment.path or file.previous_filename == comment.path][0]
                file_raw_url = file.raw_url
                file_blob_url = file.blob_url
            except IndexError: pass
        except KeyError: pass

        return {
            "user": comment.user.login if comment.user is not None else None,
            "diff": comment.diff_hunk,
            "path": comment.path,
            "body": comment.body,
            "id": comment.id,
            "url": comment.url,
            "original_commit_id": comment.original_commit_id,
            "created_at": comment.created_at.strftime("%d/%m%/Y, %H:%M:%S"),
            "updated_at": comment.updated_at.strftime("%d/%m%/Y, %H:%M:%S"),
            "html_url": comment.html_url,
            "pr_url": comment.pull_request_url,
            "file_raw_url": file_raw_url,
            "file_blob_url": file_blob_url,
            # "pr_review_id": comment.pull_request_review_id,
            "commit_id": comment.commit_id, 
            # "node_id": comment.node_id,
            # "old_file": repo.get_contents(comment.path, ref=prev_commit.sha).decoded_content.decode()
        }

# main
if __name__ == "__main__":
    # using an access token
    creds = json.load(open("datautils/scraping/gh_access_token.json"))
    auth = Auth.Token(creds["access_token"])

    # Public Web Github
    g = Github(auth=auth)

    scraper = GithubPullRequestCommentsScraper(g)
    scraper.run(repo_names=["pytorch/pytorch"], reset_file=True, 
                file_name="./data/Comment_Generation/pytorch_pr_closed.jsonl")