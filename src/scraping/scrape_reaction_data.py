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

    def run(self, repo_names: List[str], file_name: str, 
            reset_file: bool=False, filt_no_comments: bool=True):
        if reset_file: open(file_name, "w")
        for repo_name in repo_names:
            repo = rate_conscious_api_call(self.g, "get_repo", repo_name) # self.g.get_repo(repo_name)
            all_closed_pulls = rate_conscious_api_call(repo, "get_pulls", state="closed") # repo.get_pulls(state='closed')
            for pr in tqdm(all_closed_pulls, desc=repo_name):
                rec = {}
                rec["pr_id"] = pr.id
                rec["pr_title"] = pr.title
                rec["pr_poster"] = pr.user.login
                rec["pr_body"] = pr.body
                rec["repo_url"] = repo.url
                rec["repo_name"] = repo_name
                rec["comments"] = [self.extract_comment_json(comment) for comment in rate_conscious_api_call(pr, "get_review_comments")]
                if filt_no_comments and len(rec["comments"]) == 0: continue
                self.data.append(rec)
                with open(file_name, "a") as f:
                    f.write(json.dumps(rec)+"\n")

    def extract_comment_json(self, comment):
        return {
            "user": comment.user.login if comment.user is not None else None,
            "diff": comment.diff_hunk,
            "path": comment.path,
            "body": comment.body,
            "id": comment.id,
            "url": comment.url,
            "reactions": comment.reactions,
            "original_commit_id": comment.original_commit_id,
            "created_at": comment.created_at.strftime("%d/%m%/Y, %H:%M:%S"),
            "updated_at": comment.updated_at.strftime("%d/%m%/Y, %H:%M:%S"),
            "html_url": comment.html_url,
            "pr_url": comment.pull_request_url,
            "commit_id": comment.commit_id, 
        }

# main
if __name__ == "__main__":
    # using an access token
    creds = json.load(open("./src/scraping/gh_access_token.json"))
    auth = Auth.Token(creds["access_token"])

    # Public Web Github
    g = Github(auth=auth)

    scraper = GithubPullRequestCommentsScraper(g)
    scraper.run(repo_names=["pytorch/pytorch", "PyGithub/PyGithub"], reset_file=True, 
                file_name="./data/Comment_Generation/")