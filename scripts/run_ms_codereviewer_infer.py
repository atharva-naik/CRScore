import time
import torch
import os, json
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
from itertools import cycle
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from src.models import build_or_load_gen_model
from CodeBERT.CodeReviewer.code.configs import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from CodeBERT.CodeReviewer.code.utils import CommentGenDataset, SimpleGenDataset
from CodeBERT.CodeReviewer.code.evaluator.smooth_bleu import bleu_fromstr


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_loader(data_file, args, tokenizer, pool):
    def fn(features):
        return features
    logger.info(f"Start data file {data_file}.")
    if args.raw_input:
        dataset = SimpleGenDataset(tokenizer, pool, args, data_file)
    else:
        dataset = CommentGenDataset(tokenizer, pool, args, data_file)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.cpu_count, collate_fn=fn)
    logger.info(f"Finish data files {data_file}.")
    return dataset, sampler, dataloader


def eval_epoch_bleu(args, eval_dataloader, model, tokenizer, device_id=None, preds_folder=None):
    logger.info(f"  ***** Running bleu evaluation on {args.eval_file} *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    pred_ids, ex_ids = [], []
    for step, examples in tqdm(enumerate(eval_dataloader, 1)):
        if device_id:
            source_ids = torch.tensor(
                [ex.source_ids for ex in examples], dtype=torch.long
            ).to(device_id)   
        else:
            source_ids = torch.tensor(
                [ex.source_ids for ex in examples], dtype=torch.long
            ).to(args.local_rank)
        ids = [ex.example_id for ex in examples]
        source_mask = source_ids.ne(tokenizer.pad_id)
        preds = model.generate(source_ids,
                            attention_mask=source_mask,
                            use_cache=True,
                            num_beams=args.beam_size,
                            early_stopping=True,
                            max_length=args.max_target_length)
        top_preds = list(preds.cpu().numpy())
        pred_ids.extend(top_preds)
    pred_nls = [tokenizer.decode(id[2:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    valid_file = args.eval_file
    golds = []
    with open(valid_file, "r") as f:
        for line in f:
            golds.append(json.loads(line)["msg"])
    golds = golds[:len(pred_nls)]
    if preds_folder: 
        os.makedirs(preds_folder, exist_ok=True)
        preds_path = os.path.join(preds_folder, "preds.jsonl")
    else: preds_path = os.path.join(args.model_name_or_path, "preds.jsonl")
    open(preds_path, "w") # clear existing file.
    for pred, gold in zip(pred_nls, golds):
        with open(preds_path, "a") as f:
            f.write(json.dumps({"pred": pred, "gold": gold})+"\n")
    # with open(os.path.join(args.model_name_or_path, "preds.txt"), "w", encoding="utf-8") as f:
    #     for pred in pred_nls:
    #         f.write(pred.strip() + "\n")
    # with open(os.path.join(args.model_name_or_path, "golds.txt"), "w", encoding="utf-8") as f:
    #     for gold in golds:
    #         f.write(gold.strip() + "\n")
    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    logger.warning(f"WithStop BLEU: {bleu}")
    bleu = bleu_fromstr(pred_nls, golds, rmstop=True)
    return bleu

def main_single_launch(args):
    device_id = "cuda:0" # dist.get_rank() % args.gpu_per_node
    logger.warning(f"GPU ID: {device_id}")
    torch.cuda.set_device(device_id)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    # model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model.to(device_id)
    pool = multiprocessing.Pool(args.cpu_count)
    data_file = args.eval_file
    set_seed(args)
    _, _, dataloader = get_loader(data_file, args, tokenizer, pool)        # WARNING: this is a iterator, to save memory
    model.eval()
    bleu = eval_epoch_bleu(args, dataloader, model, tokenizer, 
                           device_id=device_id, preds_folder="./experiments/MS_CR_ZeroShot")
    logger.warning(f"BLEU: {bleu}")

def main(args):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % args.gpu_per_node
    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank, \
                   torch.distributed.get_world_size(), \
                   args.eval_batch_size)
    torch.cuda.set_device(local_rank)

    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    pool = multiprocessing.Pool(args.cpu_count)
    data_file = args.eval_file
    set_seed(args)
    _, _, dataloader = get_loader(data_file, args, tokenizer, pool)        # WARNING: this is a iterator, to save memory
    model.eval()
    bleu = eval_epoch_bleu(args, dataloader, model, tokenizer)
    logger.warning(f"BLEU: {bleu}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    # main(args)
    main_single_launch(args)
    logger.info("Test finished.")
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
    
    # command for launching:
    # python -m scripts.run_ms_codereviewer_infer --model_name_or_path microsoft/CodeReviewer --eval_file data/Comment_Generation/msg-test.jsonl --max_source_length 512 --max_target_length 128 --eval_batch_size 12 --beam_size 10  --seed 2233 --raw_input --break_cnt 20