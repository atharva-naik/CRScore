# script to use contrastive learning to finetune a model of review relevance.

import os
import torch
import logging
import argparse
import random
import json
from tqdm import tqdm
import multiprocessing
import time
from itertools import cycle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_rel_model
from configs import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import CommentGenDataset, SimpleRelDataset
# from evaluator.smooth_bleu import bleu_fromstr


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



def get_loaders(data_files, args, tokenizer, pool, eval=False):
    def fn(features):
        return features
    global_rank = args.global_rank
    for data_file in data_files:
        print("\x1b[33;1mdata_file:\x1b[0m", data_file)
        if args.raw_input:
            dataset = SimpleRelDataset(tokenizer, pool, args, data_file)
        else:
            dataset = CommentGenDataset(tokenizer, pool, args, data_file)
        data_len = len(dataset)
        if global_rank == 0:
            logger.info(f"Data length: {data_len}.")
        if eval:
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size if not eval else args.eval_batch_size, \
                                num_workers=args.cpu_count, collate_fn=fn)
        yield dataset, sampler, dataloader


def eval_loss_epoch(args, eval_dataloader, model, tokenizer):
    logger.info(f"  ***** Running evaluation on {args.eval_file} *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    # pred_ids, ex_ids = [], []
    ev_loss = 0
    nb_ev_examples = 0
    nb_ev_steps = 0
    for step, examples in tqdm(enumerate(eval_dataloader, 1)):
        diff_ids = torch.tensor(
            [ex.source_ids for ex in examples], dtype=torch.long
        ).to(args.local_rank)
        review_ids = torch.tensor(
            [ex.target_ids for ex in examples], dtype=torch.long
        ).to(args.local_rank)
        source_mask = review_ids.ne(tokenizer.pad_id)
        target_mask = diff_ids.ne(tokenizer.pad_id)

        loss = model(
            review_ids=review_ids,
            diff_ids=diff_ids,
            review_attention_mask=source_mask,
            diff_attention_mask=target_mask,
            encoder_loss=False
        )

        if args.gpu_per_node > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        ev_loss += loss.item()

        nb_ev_examples += review_ids.size(0)
        nb_ev_steps += 1

        # source_ids = torch.tensor(
        #     [ex.source_ids for ex in examples], dtype=torch.long
        # ).to(args.local_rank)
        # ids = [ex.example_id for ex in examples]
        # source_mask = source_ids.ne(tokenizer.pad_id)
        # preds = model.generate(source_ids,
        #                     attention_mask=source_mask,
        #                     use_cache=True,
        #                     num_beams=args.beam_size,
        #                     early_stopping=True,
        #                     max_length=args.max_target_length)
        # top_preds = list(preds.cpu().numpy())
        # pred_ids.extend(top_preds)
    # [1:] to remove beginning '<msg>'
    # pred_nls = [tokenizer.decode(id[1:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    # valid_file = args.dev_filename
    # golds = []
    # with open(valid_file, "r") as f:
        # for line in f:
            # golds.append(json.loads(line)["msg"])
    # golds = golds[:len(pred_nls)]
    # bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    eval_loss = round(ev_loss * args.gradient_accumulation_steps / nb_ev_steps, 4)
    
    return eval_loss

def save_model(model, optimizer, scheduler, output_dir, config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    config.save_pretrained(output_dir)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
    torch.save(
        optimizer.state_dict(),
        output_optimizer_file,
        _use_new_zipfile_serialization=False,
    )
    output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
    torch.save(
        scheduler.state_dict(),
        output_scheduler_file,
        _use_new_zipfile_serialization=False,  
    )


def main(args):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % args.gpu_per_node
    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank, \
                   torch.distributed.get_world_size(), \
                   args.train_batch_size)
    torch.cuda.set_device(local_rank)

    set_seed(args)
    config, model, tokenizer = build_or_load_rel_model(args)
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    pool = multiprocessing.Pool(args.cpu_count)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    args.warmup_steps = int(args.train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    if os.path.exists("{}/checkpoints-last/optimizer.pt".format(args.output_dir)):
        optimizer.load_state_dict(
            torch.load(
                "{}/checkpoints-last/optimizer.pt".format(args.output_dir),
                map_location="cpu",
            )
        )
        scheduler.load_state_dict(
            torch.load(
                "{}/checkpoints-last/scheduler.pt".format(args.output_dir),
                map_location="cpu",
            )
        )

    global_step = 0
    save_steps = args.save_steps
    train_file = args.train_filename
    valid_file = args.dev_filename
    print("\x1b[33;1mDEBUG:\x1b[0m", train_file, os.path.isdir(train_file))
    if os.path.isdir(train_file):
        train_files = [file for file in os.listdir(train_file) if file.startswith("train") and file.endswith(".jsonl")]
    else:
        train_files = [train_file]
    random.seed(args.seed)
    random.shuffle(train_files)
    # train_files = [os.path.join(train_file, file) for file in train_files]
    valid_files = [valid_file]
    # bleu = eval_bleu_epoch(args, valid_dataloader, model, tokenizer)
    # logger.warning("Initial bleu: {}".format(bleu))
    for epoch in range(1, args.train_epochs + 1):
        # set seed for reproducible data split
        save_seed = args.seed
        args.seed += epoch
        set_seed(args)
        args.seed = save_seed
        model.train()
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        for _, _, train_dataloader in get_loaders(train_files, args, tokenizer, pool):        # WARNING: this is an iterator, to save memory
            for step, examples in tqdm(enumerate(train_dataloader, 1)):
                if step == 1:
                    ex = examples[0]
                    logger.info(f"batch size: {len(examples)}")
                    logger.info(f"example source: {tokenizer.convert_ids_to_tokens(ex.source_ids)}")
                    # logger.info(f"example label: {tokenizer.convert_ids_to_tokens(ex.source_labels)}")
                    logger.info(f"example target: {tokenizer.convert_ids_to_tokens(ex.target_ids)}")
                diff_ids = torch.tensor(
                    [ex.source_ids for ex in examples], dtype=torch.long
                ).to(local_rank)
                review_ids = torch.tensor(
                    [ex.target_ids for ex in examples], dtype=torch.long
                ).to(local_rank)
                source_mask = review_ids.ne(tokenizer.pad_id)
                target_mask = diff_ids.ne(tokenizer.pad_id)

                loss = model(
                    review_ids=review_ids,
                    diff_ids=diff_ids,
                    review_attention_mask=source_mask,
                    diff_attention_mask=target_mask,
                    encoder_loss=False
                )

                if args.gpu_per_node > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += review_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    if args.global_rank == 0 and global_step % args.log_steps == 0:
                        train_loss = round(
                            tr_loss * args.gradient_accumulation_steps / nb_tr_steps,
                            4,
                        )
                        logger.info(
                            "step {}/{}: Train loss {}".format(
                                global_step,
                                args.train_steps,
                                round(train_loss, 3),
                            )
                        )
                if True:#global_step == args.train_steps and args.global_rank == 0:
                    # end training
                    _, _, valid_dataloader = next(get_loaders(valid_files, args, tokenizer, pool, eval=True))
                    eval_loss = eval_loss_epoch(args, valid_dataloader, model, tokenizer)
                    output_dir = os.path.join(args.output_dir, "checkpoints-last" + "-" + str(eval_loss))
                    save_model(model, optimizer, scheduler, output_dir, config)
                    logger.info(f"Reach max steps {args.train_steps}.")
                    time.sleep(5)
                    return
                if args.global_rank == 0 and \
                        global_step % save_steps == 0 and \
                        nb_tr_steps % args.gradient_accumulation_steps == 0:
                    _, _, valid_dataloader = next(get_loaders(valid_files, args, tokenizer, pool, eval=True))
                    eval_loss = eval_loss_epoch(args, valid_dataloader, model, tokenizer)
                    output_dir = os.path.join(args.output_dir, "checkpoints-" + str(global_step) + "-" + str(eval_loss))
                    save_model(model, optimizer, scheduler, output_dir, config)
                    logger.info(
                        "Save the {}-step model and optimizer into {}".format(
                            global_step, output_dir
                        )
                    )
                    time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    main(args)
    logger.info("Training finished.")
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
