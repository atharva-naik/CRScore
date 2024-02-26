import os
import copy
import json
import torch
import random
import pathlib
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
import torch.optim as optim
from src.datautils import read_jsonl
from sentence_transformers import util
from torch.utils.data import DataLoader, Dataset
from CodeBERT.CodeReviewer.code.utils import MyTokenizer
from transformers import RobertaModel, RobertaTokenizerFast, AutoModel, T5Tokenizer, RobertaTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ReviewClfFeatures(object):
    def __init__(self, example_id, source_ids, source_mask, target_cls, type):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.target_cls = target_cls
        assert type in ("label", "line", "genmsg", "daemsg", "revrel", "revclf")
        self.type = type

class ReviewClfDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, MyTokenizer): tok_type = "mytok"
        elif isinstance(tokenizer, T5Tokenizer): tok_type = ""
        elif isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)): tok_type = "rb"
        else: tok_type = "unk"
        
        savep = file_path.replace(".jsonl", f"_{tok_type}.simprevclf")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            self.feats = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            data = read_jsonl(file_path)
            # data = [dic for dic in data if len(dic["patch"].split("\n")) <= 20]
            for i in range(len(data)):
                data[i]["idx"] = i
                data[i]["cls"] = 1
            all_data = []
            N = len(data)
            neg_ids = random.sample(range(N), k=N)
            for i,j in zip(range(N), neg_ids):
                if i == j: j = i-1 if i > 0 else 1
                rec = copy.deepcopy(data[i])
                rec["cls"] = 0
                rec["msg"] = data[j]["msg"]
                all_data.append(rec)
            all_data += data
            data = all_data
            assert len(data) == 2*N
            logger.info(f"Tokenize examples: {file_path}")
            # self.feats = pool.map(self.convert_examples_to_features, \
            #     [(dic, tokenizer, args) for dic in data])
            self.feats = [self.convert_examples_to_features((dic, args)) for dic in tqdm(data)]
            torch.save(self.feats, savep)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]

    def reset_len(self, data_len):
        assert len(self.feats) >= data_len
        self.feats = self.feats[:data_len]

    def set_start_end_ids(self, examples):
        for example in examples:
            labels = example.labels
            start_id = 0
            end_id = len(labels) - 1
            for i, label in enumerate(labels):
                if label != -100:               # find the first label
                    start_id = i
                    break
            for i in range(len(labels) - 1, -1, -1):
                label = labels[i]
                if label != -100:
                    end_id = i
                    break
            example.start_id = start_id
            example.end_id = end_id

    def encode_remove(self, tokenizer, text, args):
        text = tokenizer.encode(text, max_length=args.max_source_length, truncation=True)
        if type(tokenizer) == T5Tokenizer: return text[:-1]
        elif type(tokenizer) in [RobertaTokenizer, RobertaTokenizerFast]: return text[1:-1]
        elif type(tokenizer) == MyTokenizer: return text
        else: raise NotImplementedError

    def pad_assert(self, source_ids, target_cls, args):
        source_ids = source_ids[:args.max_source_length - 2]
        source_ids = [self.tokenizer.bos_id] + source_ids + [self.tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [self.tokenizer.pad_id] * pad_len
        assert target_cls in [0,1]
        assert len(source_ids) == args.max_source_length, "Not equal length."
        return source_ids, target_cls

    def convert_examples_to_features(self, item):
        dic, args = item
        diff, msg, target_cls = dic["patch"], dic["msg"], dic["cls"]
        difflines = diff.split("\n")[1:]        # remove start @@
        difflines = [line for line in difflines if len(line.strip()) > 0]
        map_dic = {"-": 0, "+": 1, " ": 2}
        def f(s):
            if s in map_dic: return map_dic[s]
            else: return 2
        labels = [f(line[0]) for line in difflines]
        difflines = [line[1:].strip() for line in difflines]
        inputstr = ""
        for label, line in zip(labels, difflines):
            if label == 1: inputstr += "<add>" + line
            elif label == 0: inputstr += "<del>" + line
            else: inputstr += "<keep>" + line
        source_ids = self.encode_remove(self.tokenizer, msg + " <msg> "+ inputstr, args)
        source_ids, target_cls = self.pad_assert(source_ids, target_cls, args)

        source_ids = torch.as_tensor(source_ids)
        target_cls = torch.as_tensor(target_cls)
        source_mask = source_ids.ne(self.tokenizer.pad_id)

        return ReviewClfFeatures(
            example_id=dic["idx"], 
            source_ids=source_ids, 
            target_cls=target_cls,
            source_mask=source_mask,
            type="revclf"
        )

def get_loaders(data_file, args, tokenizer, eval=False):
    def fn(features): 
        # print(features[0].source_ids)
        return {
            "input_ids": torch.stack([feat.source_ids for feat in features]),
            "attention_mask": torch.stack([feat.source_mask for feat in features]),
            "target_cls": torch.stack([feat.target_cls for feat in features]),
        }
    print("\x1b[33;1mdata_file:\x1b[0m", data_file)
    dataset = ReviewClfDataset(
        tokenizer=tokenizer,
        args=args, file_path=data_file
    )
    data_len = len(dataset)
    logger.info(f"Data length: {data_len}.")
    # if eval: sampler = SequentialSampler(dataset)
    # else: sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, shuffle=not(eval), collate_fn=fn,
        batch_size=args.batch_size,
    )
    return dataset, None, dataloader

# Define the contrastive loss
class CECosSimLoss(nn.Module):
    def __init__(self, temp: float=0.01):
        super(CECosSimLoss, self).__init__()
        self.temp = temp

    def forward(self, output1, output2, label):
        # Calculate cosine similarity
        cosine_similarity = nn.functional.cosine_similarity(output1, output2)/self.temp
        # Use cross-entropy loss
        loss = nn.functional.cross_entropy(cosine_similarity, label)
        return loss

class ReviewRelevanceClf(nn.Module):
    def __init__(self, model_type: str="codereviewer", model_path: str="microsoft/codereviewer", emb_size: int=768, clf_dropout_rate: float=0.2):
        super().__init__()
        self.model_type = model_type
        self.model_path = model_path
        if model_type == "codereviewer":
            model = AutoModel.from_pretrained(model_path)
            self.encoder = model.encoder
        self.emb_size = emb_size
        self.clf_dropout_rate = clf_dropout_rate
        self.clf = nn.Linear(emb_size, 2)
        self.dropout = nn.Dropout(p=clf_dropout_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attn_mask, target_cls=None):#, label):
        enc = self.encoder(input_ids, attn_mask).last_hidden_state.sum(axis=1)
        logits = self.clf(self.dropout(enc))
        if target_cls is None:
            return enc, logits, None
        loss = self.loss_fn(logits, target_cls)

        return enc, logits, loss

# Define the training loop
def train(model, dataloader, val_dataloader, epoch, optimizer, device, best_loss, best_acc, args):
    model.train()
    total_loss = 0.0

    pbar = tqdm(enumerate(dataloader), 
                total=len(dataloader), 
                desc="training")
    train_matches = 0
    tot = 0
    for step, batch in pbar:
        # Get inputs
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_cls = batch["target_cls"].to(device)
        # Forward pass and Calculate contrastive loss
        _, logits, loss = model(input_ids, attention_mask, target_cls)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_matches = (target_cls.cpu() == logits.max(axis=-1).indices.cpu()).sum().item()
        train_matches += batch_matches
        tot += len(input_ids)
        bacc = batch_matches / len(input_ids)
        tacc = train_matches / tot
        pbar.set_description(f"bl: {loss:.3f} l: {(total_loss/(step+1)):.3f}, ba: {100*bacc:.2f} a: {100*tacc:.2f}")

        if (step + 1) % args.n_steps == 0:
            val_loss, val_acc = validate(model, val_dataloader, device)
            print(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {val_loss:.4f}")

            # Save the model if it has the best contrastive loss
            if val_acc > best_acc:
                best_loss = val_loss
                best_acc = val_acc
                ckpt_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss, 
                    "val_loss": val_loss,
                    "val_acc": val_acc, 
                }
                print(f"\x1b[32;1msaving best model with acc={best_acc:.2f}\x1b[0m")
                ckpt_save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(ckpt_dict, ckpt_save_path)
            ckpt_dict = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "val_loss": val_loss,
                "val_acc": val_acc, 
            }
            ckpt_save_path = os.path.join(args.output_dir, "last_model.pth")
            torch.save(ckpt_dict, ckpt_save_path)

    return total_loss / len(dataloader), best_loss, best_acc

# Define the validation loop
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    val_matches = 0
    vacc = 0
    tot = 0
    pbar = tqdm(enumerate(dataloader), 
                total=len(dataloader), 
                desc="validating")
    with torch.no_grad():
        for step, batch in pbar:
            # Get inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_cls = batch["target_cls"].to(device)
            # Forward pass and Calculate contrastive loss
            enc, logits, loss = model(input_ids, attention_mask, target_cls)
            total_loss += loss.item()
            batch_matches = (target_cls.cpu() == logits.max(axis=-1).indices.cpu()).sum().item()
            val_matches += batch_matches
            tot += len(input_ids)
            bacc = batch_matches / len(input_ids)
            vacc = val_matches / tot
            pbar.set_description(f"bl: {loss:.3f} l: {(total_loss/(step+1)):.3f}, ba: {100*bacc:.2f} a: {100*vacc:.2f}")
            # pbar.set_description(f"bl: {loss:.3f} l: {(total_loss/(step+1)):.3f}")
    return total_loss / len(dataloader), vacc

# Define the prediction function
def predict(model, input_ids, attention_mask, device):
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        output = model(input_ids, attention_mask)
    return output

def build_tokenizer(tokenizer_class, model_name_or_path):
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)

    try: # need to do this for codereviewer
        tokenizer.special_dict = {
            f"<e{i}>" : tokenizer.get_vocab()[f"<e{i}>"] for i in range(99, -1, -1)
        }
    except KeyError: pass

    tokenizer.mask_id = tokenizer.get_vocab()["<mask>"]
    tokenizer.bos_id = tokenizer.get_vocab()["<s>"]
    tokenizer.pad_id = tokenizer.get_vocab()["<pad>"]
    tokenizer.eos_id = tokenizer.get_vocab()["</s>"]
    tokenizer.msg_id = tokenizer.get_vocab()["<msg>"]
    tokenizer.keep_id = tokenizer.get_vocab()["<keep>"]
    tokenizer.add_id = tokenizer.get_vocab()["<add>"]
    tokenizer.del_id = tokenizer.get_vocab()["<del>"]
    tokenizer.start_id = tokenizer.get_vocab()["<start>"]
    tokenizer.end_id = tokenizer.get_vocab()["<end>"]

    return tokenizer

def get_args():
    parser = argparse.ArgumentParser(description="Contrastive Learning for BERT")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--eval", action="store_true", help="no training, only evaluate checkpoint")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=200, help="Validation steps")
    parser.add_argument("--model_type", default="codereviewer", type=str, help="type of model/model class to be used")
    parser.add_argument("--model_path", default="microsoft/codereviewer", type=str, help="model name or path")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default=None, help="directory where checkpoints will be stored.")
    parser.add_argument(
        "--seed", type=int, default=2233, help="random seed for initialization"
    )
    # parser.add_argument("--use_unnormalized_loss", action="store_true", help="use regular unnormalized CE loss")
    # parser.add_argument("--temperature", default=0.0001, type=float, help='Temperature parameter for InfoNCE objective')
    parser.add_argument(
        "--max_source_length",
        default=200,
        type=int,
        help="The maximum total source sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=100,
        type=int,
        help="The maximum total target sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--train_filename",
        default=None,
        type=str,
        help="The train filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--dev_filename",
        default=None,
        type=str,
        help="The dev filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. Should contain the .jsonl files for this task.",
    )
    # Add other relevant arguments
    
    # Apply simple checks/constraints over arguments
    args = parser.parse_args()
    if not args.eval: # for training mode, make sure output_dir is set.
        assert args.output_dir is not None, f"You forgot to set output_dir in training mode"

    return args

def recall_at_k(indices, labels, k):
    assert len(indices) == len(labels)
    tot, score = len(indices), 0
    for i in range(len(indices)):
        preds_set = set(indices[i][:k])
        labels_set = set(labels[i])
        score += int(len(labels_set.intersection(preds_set)) > 0)

    return score/tot

def eval_checkpoint(args):
    # Initialize model, optimizer, criterion, and other necessary components
    print(f"\x1b[34;1mloading checkpoint: {args.checkpoint_path}\x1b[0m")
    model = ReviewRelevanceClf(
        model_type=args.model_type,
        model_path=args.model_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the checkpoint.
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

    test_file = args.test_filename
    valid_file = args.dev_filename
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = build_tokenizer(RobertaTokenizerFast, args.model_path)

    # Load your dataset and create DataLoader instances
    _, _, test_dataloader = get_loaders(
        data_file=test_file, args=args,
        tokenizer=tokenizer, eval=True,
    )
    _, _, val_dataloader = get_loaders(
        data_file=valid_file, args=args,
        tokenizer=tokenizer, eval=True,
    )

    val_loss, val_scores, val_indices = validate(model, val_dataloader, 
                                                 device=device, return_preds=True)
    val_labels = [[i] for i in range(len(val_scores))]
    val_recall_at_5 = recall_at_k(val_indices, val_labels, k=5)
    print("val_loss:", val_loss, "val_recall@5:", val_recall_at_5)

    
    test_loss, test_scores, test_indices = validate(model, test_dataloader, 
                                                    device=device, return_preds=True)
    test_labels = [[i] for i in range(len(test_scores))]
    test_recall_at_5 = recall_at_k(test_indices, test_labels, k=5)
    print("test_loss:", test_loss, "test_recall@5:", test_recall_at_5)
    if args.checkpoint_path is not None:
        test_ids_path = os.path.join(pathlib.Path(args.checkpoint_path).parent, "test_ids.json") 
        test_preds_path = os.path.join(pathlib.Path(args.checkpoint_path).parent, "test_preds.json")
    else:
        test_ids_path = os.path.join(args.output_dir, "test_ids.json")
        test_preds_path = os.path.join(args.output_dir, "test_preds.json")
    with open(test_ids_path, "w") as f:
        json.dump(test_indices.tolist(), f, indent=4)
    with open(test_preds_path, "w") as f:
        json.dump(np.diag(test_scores).tolist(), f, indent=4)
    
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize model, optimizer, criterion, and other necessary components
    model = ReviewRelevanceClf(
        model_type=args.model_type,
        model_path=args.model_path,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optionally resume training
    if args.resume:
        # Load checkpoint and update model, optimizer, etc.
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"Resuming training from epoch {epoch}, best loss: {best_loss}")

    model.to(device)
        
    train_file = args.train_filename
    valid_file = args.dev_filename
    train_files = [train_file]

    random.shuffle(train_files)
    tokenizer = build_tokenizer(RobertaTokenizerFast, args.model_path)

    # Load your dataset and create DataLoader instances
    _, _, train_dataloader = get_loaders(
        data_file=train_file, args=args,
        tokenizer=tokenizer, eval=False,
    )
    _, _, val_dataloader = get_loaders(
        data_file=valid_file, args=args,
        tokenizer=tokenizer, eval=True,
    )

    best_acc = 0
    best_loss = float("inf")

    # Training loop
    for epoch in range(args.epochs):
        train_loss, best_loss, best_acc = train(
            model=model, dataloader=train_dataloader, 
            val_dataloader=val_dataloader, epoch=epoch,
            optimizer=optimizer, device=device, 
            best_loss=best_loss, args=args,
            best_acc=best_acc,
        )
        print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {train_loss:.4f}")

if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.eval: eval_checkpoint(args)
    else: main(args)