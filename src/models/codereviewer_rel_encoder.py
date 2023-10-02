# encoder model to prohect code diff and reviews to a space where co-sine similarity corresponds to relevance.
# script to use contrastive learning to finetune a model of review relevance.

import os
import sys
import torch
import logging
import argparse
import random
import json
from tqdm import tqdm
import time
from itertools import cycle
from sentence_transformers import util
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup

sys.path.append("./CodeBERT")
sys.path.append("./CodeBERT/CodeReviewer")
sys.path.append("./CodeBERT/CodeReviewer/code")

from CodeBERT.CodeReviewer.code.models import build_or_load_rel_model
from CodeBERT.CodeReviewer.code.configs import add_args, set_seed, set_dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
from CodeBERT.CodeReviewer.code.utils import CommentGenDataset, SimpleRelDataset, read_jsonl
# from evaluator.smooth_bleu import bleu_fromstr

def get_loaders(data_files, args, tokenizer, pool=None, eval=False):
    def fn(features):
        return features
    try: global_rank = args.global_rank
    except AttributeError: global_rank = 0
    for data_file in data_files:
        print("\x1b[33;1mdata_file:\x1b[0m", data_file)
        dataset = SimpleRelDataset(tokenizer, pool, args, data_file)
        data_len = len(dataset)
        if global_rank == 0:
            print(f"Data length: {data_len}.")
        if eval:
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset, sampler=sampler, collate_fn=fn,
            batch_size=args.train_batch_size if not eval else args.eval_batch_size)
        yield dataset, sampler, dataloader

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    # args.cpu_count = multiprocessing.cpu_count()
    config, model, tokenizer = build_or_load_rel_model(args)
    random.seed(args.seed)
    # pool = multiprocessing.Pool(args.cpu_count)
    test_file = "./data/Comment_Generation/msg-test.jsonl"
    _, _, test_dataloader = list(get_loaders(
        data_files=[test_file], args=args, 
        tokenizer=tokenizer, eval=True,
    ))[0]
    model.eval()
    raw_test_data = read_jsonl(test_file)
    # id_to_data = {rec["id"] for rec in}
    relevance_scores = []
    for step, examples in tqdm(enumerate(test_dataloader, 1)):
        # print(dir(examples[0]))
        diff_ids = torch.tensor(
            [ex.source_ids for ex in examples], dtype=torch.long
        ).to(args.local_rank)
        review_ids = torch.tensor(
            [ex.target_ids for ex in examples], dtype=torch.long
        ).to(args.local_rank)
        ids = [ex.example_id for ex in examples]
        review_mask = review_ids.ne(tokenizer.pad_id)
        diff_mask = diff_ids.ne(tokenizer.pad_id)
        with torch.no_grad():
            review_encoder_outputs = model.encoder(
                input_ids=review_ids, 
                attention_mask=review_mask,
                output_attentions=False,
                return_dict=False,
            )
            review_hidden_states = review_encoder_outputs[0]
            review_first_hidden = review_hidden_states[:, 0, :]
            diff_encoder_outputs = model.encoder( \
                input_ids=diff_ids,
                attention_mask=diff_mask,
                output_attentions=False,
                return_dict=False
            )
            diff_hidden_states = diff_encoder_outputs[0]
            diff_first_hidden = diff_hidden_states[:, 0, :]
            rel_scores = util.cos_sim(review_first_hidden, diff_first_hidden).diag().cpu().tolist()
            relevance_scores += [
                {
                    "id": id, "score": score,
                    "msg": raw_test_data[id]["msg"],
                    "patch": raw_test_data[id]["patch"]  
                } for id, score in zip(ids, rel_scores)
            ]
            # break
    with open("./experiments/codereviewer_rel_scores.json", "w") as f:
        json.dump(relevance_scores, f, indent=4)