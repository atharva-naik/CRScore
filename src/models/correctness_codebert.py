import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from CodeBERT.UniXcoder.unixcoder import UniXcoder
from src.datautils import read_jsonl

class CorrectnessCodeBERT(nn.Module):
    def __init__(self, model_path, code_enc_dim: int=768, 
                 review_enc_dim: int=768, use_simple_cc_network: bool=False):
        super().__init__()
        self.model_path = model_path
        if "unixcoder" in model_path:
            self.code_encoder = UniXcoder(model_path)
            self.review_encoder = UniXcoder(model_path)            
        else:
            self.code_encoder = AutoModel.from_pretrained(model_path)
            self.review_encoder = AutoModel.from_pretrained(model_path)
        self.code_enc_dim = code_enc_dim
        self.review_enc_dim = review_enc_dim
        if use_simple_cc_network:
            self.code_comparison_network = nn.Sequential(
                nn.Linear(4*code_enc_dim, 5*code_enc_dim),
                nn.GELU(),
                nn.Linear(5*code_enc_dim, code_enc_dim)
            )
        else:
            self.code_comparison_network = nn.Linear(4*code_enc_dim, code_enc_dim)
        self.loss_fn = nn.TripletMarginLoss(margin=1, p=2)

    def forward(self, anchor_before, anchor_after, pos, neg):
        if "unixcoder" in self.model_path:
            before_enc = self.code_encoder(anchor_before['input_ids'])[1]
            after_enc = self.code_encoder(anchor_after['input_ids'])[1]
        else:
            before_enc = self.code_encoder(**anchor_before).pooler_output
            after_enc = self.code_encoder(**anchor_after).pooler_output
        anchor = self.code_comparison_network(torch.cat([
            after_enc-before_enc,
            after_enc*before_enc, 
            before_enc, after_enc
        ], axis=-1))
        if "unixcoder" in self.model_path:
            pos = self.review_encoder(pos['input_ids'])[1]
            neg = self.review_encoder(neg['input_ids'])[1]
        else:
            pos = self.review_encoder(**pos).pooler_output
            neg = self.review_encoder(**neg).pooler_output
        loss = self.loss_fn(anchor, pos, neg)

        return anchor, pos, neg, loss

def test_function():
    code_before = "print('hello')"
    code_after = "print('bye')"
    pos_review = "positive review" 
    neg_review = "negative review"
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    correctness_codebert = CorrectnessCodeBERT("microsoft/codebert-base")
    correctness_codebert.cuda()
    # tokenize inputs.
    before_code = tok(code_before, return_tensors="pt")
    for k in before_code: before_code[k] = before_code[k].cuda()
    after_code = tok(code_after, return_tensors="pt")
    for k in after_code: after_code[k] = after_code[k].cuda()
    pos_review = tok(pos_review, return_tensors="pt")
    for k in pos_review: pos_review[k] = pos_review[k].cuda()
    neg_review = tok(neg_review, return_tensors="pt")
    for k in neg_review: neg_review[k] = neg_review[k].cuda()
    _, _, _, loss = correctness_codebert(
        anchor_before=before_code, 
        anchor_after=after_code, 
        pos=pos_review, 
        neg=neg_review,
    )
    print(loss)

class CRDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

class CRDataLoader(DataLoader):
    def __init__(self, *args, tokenizer=None, model_path, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=self.collate_fn)
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.max_length = 200

    def collate_fn(self, batch):
        op = {}
        for k in batch[0]:
            values = []
            for rec in batch: values.append(rec[k])
            if "unixcoder" in self.model_path:
                op[k] = self.tokenizer(
                    values, max_length=self.max_length, 
                    padding="longest", return_tensors="pt",
                    truncation=True#, mode="<encoder-only>"
                )
            else:
                op[k] = self.tokenizer(
                    values, max_length=self.max_length, 
                    padding="longest", return_tensors="pt",
                    truncation=True
                )
        
        return op

def get_args():
    parser = argparse.ArgumentParser(description="Contrastive Learning for Correctness CodeBERT")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    # parser.add_argument("--eval", action="store_true", help="no training, only evaluate checkpoint")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=5000, help="Validation steps")
    # parser.add_argument("--code_model_type", default="codereviewer", type=str, help="type of model/model class to be used")
    parser.add_argument("--model_path", default="microsoft/codereviewer", type=str, help="model name or path")
    # parser.add_argument("--review_model_type", default="codereviewer", type=str, help="type of model/model class to be used")
    # parser.add_argument("--review_model_path", default="microsoft/codereviewer", type=str, help="model name or path")
    # parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    # parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--use_simple_cc_network", action="store_true", help="use single layer projection network for code change representation")
    parser.add_argument("--output_dir", type=str, default=None, help="directory where checkpoints will be stored.")
    parser.add_argument(
        "--seed", type=int, default=2233, help="random seed for initialization"
    )
    # parser.add_argument("--use_unnormalized_loss", action="store_true", help="use regular unnormalized CE loss")
    # parser.add_argument("--asym_code_first", action="store_true", help="assymetric InfoNCE objective with code first")
    # parser.add_argument("--asym_review_first", action="store_true", help="assymetric InfoNCE objective with review first")
    # parser.add_argument("--temperature", default=0.0001, type=float, help='Temperature parameter for InfoNCE objective')
    # parser.add_argument(
    #     "--max_source_length",
    #     default=200,
    #     type=int,
    #     help="The maximum total source sequence length after tokenization. Sequences longer "
    #     "than this will be truncated, sequences shorter will be padded.",
    # )
    # parser.add_argument(
    #     "--max_target_length",
    #     default=100,
    #     type=int,
    #     help="The maximum total target sequence length after tokenization. Sequences longer "
    #     "than this will be truncated, sequences shorter will be padded.",
    # )
    parser.add_argument("--device", default="cuda:0", type=str, help="device to be used for training")
    parser.add_argument(
        "--train_filename", type=str,
        default="./data/Comment_Generation/correctness-aug-msg-train.jsonl",
        help="The train filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--dev_filename", type=str,
        default="./data/Comment_Generation/correctness-aug-msg-valid.jsonl",
        help="The dev filename. Should contain the .jsonl files for this task.",
    )
    # parser.add_argument(
    #     "--test_filename",
    #     default=None,
    #     type=str,
    #     help="The test filename. Should contain the .jsonl files for this task.",
    # )
    # Add other relevant arguments
    # Apply simple checks/constraints over arguments
    args = parser.parse_args()
    # if not args.eval: # for training mode, make sure output_dir is set.
    #     assert args.output_dir is not None, f"You forgot to set output_dir in training mode"
    return args

def validate(model, dataloader, args) -> float:
    model.eval()
    total_loss = 0.0
    # if return_preds:
        # all_review_vecs = []
        # all_code_vecs = []
    device = args.device
    pbar = tqdm(enumerate(dataloader), 
                total=len(dataloader), 
                desc="validating")
    with torch.no_grad():
        for step, batch in pbar:
            # Get inputs
            anchor_before = {k: v.to(device) for k,v in batch["anchor_before"].items()}
            anchor_after = {k: v.to(device) for k,v in batch["anchor_after"].items()}
            pos = {k: v.to(device) for k,v in batch["pos"].items()}
            neg = {k: v.to(device) for k,v in batch["neg"].items()}

            # Forward pass and Calculate contrastive loss
            _, _, _, loss = model(
                anchor_before=anchor_before, 
                anchor_after=anchor_after, 
                pos=pos, neg=neg,
            )
            # if return_preds:
            #     all_review_vecs.extend(review_enc.cpu().numpy())
            #     all_code_vecs.extend(code_enc.cpu().numpy())
            total_loss += loss.item()
            pbar.set_description(f"bl: {loss:.3f} l: {(total_loss/(step+1)):.3f}")
            # break # TODO: DEBUG
    # if return_preds:
    #     all_code_vecs = np.stack(all_code_vecs)
    #     all_review_vecs = np.stack(all_review_vecs)
    #     if model.asym_review_first:
    #         scores = util.cos_sim(all_code_vecs, all_review_vecs).numpy()
    #     else: scores = util.cos_sim(all_code_vecs, all_review_vecs).numpy()
    #     # print("scores shape:", scores.shape)
    #     return total_loss / len(dataloader), scores, np.argsort(scores)[:,::-1]
    return total_loss / len(dataloader)

def train(args):
    model = CorrectnessCodeBERT(args.model_path, use_simple_cc_network=args.use_simple_cc_network)
    if "unixcoder" in args.model_path:
        tokenizer = model.code_encoder.tokenizer#.tokenize
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # move model to device.
    model.to(args.device)

    model.train()
    ckpt_dict = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dict, exist_ok=True)

    traindata = read_jsonl(args.train_filename)
    trainset = CRDataset(traindata)
    trainloader = CRDataLoader(trainset, model_path=args.model_path, batch_size=args.batch_size, shuffle=True, tokenizer=tokenizer)
    valdata = read_jsonl(args.dev_filename)
    valset = CRDataset(valdata)
    valloader = CRDataLoader(valset, model_path=args.model_path, batch_size=args.batch_size, shuffle=False, tokenizer=tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr)


    device = args.device
    # train-eval loop.
    best_loss = 10000
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(enumerate(trainloader), 
                    total=len(trainloader), 
                    desc="training")
        for step, batch in pbar:
            # Get inputs
            anchor_before = {k: v.to(device) for k,v in batch["anchor_before"].items()}
            anchor_after = {k: v.to(device) for k,v in batch["anchor_after"].items()}
            pos = {k: v.to(device) for k,v in batch["pos"].items()}
            neg = {k: v.to(device) for k,v in batch["neg"].items()}
            
            # Forward pass and Calculate contrastive loss
            _, _, _, loss = model(
                anchor_before=anchor_before, 
                anchor_after=anchor_after, 
                pos=pos, neg=neg,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"bl: {loss:.3f} l: {(total_loss/(step+1)):.3f}")

            if (step + 1) % args.n_steps == 0:
                val_loss = validate(model, valloader, args)
                print(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {val_loss:.4f}")

                # Save the model if it has the best contrastive loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    ckpt_dict = {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "val_loss": val_loss,
                    }
                    print(f"\x1b[32;1msaving best model with loss={best_loss}\x1b[0m")
                    ckpt_save_path = os.path.join(args.output_dir, "best_model.pth")
                    torch.save(ckpt_dict, ckpt_save_path)
                ckpt_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "val_loss": val_loss,
                }
                ckpt_save_path = os.path.join(args.output_dir, "last_model.pth")
                torch.save(ckpt_dict, ckpt_save_path)

# main
if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    train(args)