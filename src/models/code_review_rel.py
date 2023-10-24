import os
import copy
import torch
import random
import logging
import argparse
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
import torch.optim as optim
from src.datautils import read_jsonl
from torch.utils.data import DataLoader, Dataset
from CodeBERT.CodeReviewer.code.utils import MyTokenizer
from src.losses.info_nce import InfoNCE
from transformers import RobertaModel, RobertaTokenizerFast, AutoModel, T5Tokenizer, RobertaTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ReviewFeatures(object):
    def __init__(self, example_id, source_ids, target_ids, 
                 source_mask, target_mask, type):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.target_ids = target_ids
        assert type in ("label", "line", "genmsg", "daemsg", "revrel")
        self.type = type

class ReviewRelDataset(Dataset):
    def __init__(self, code_tokenizer, 
                 review_tokenizer, args, 
                 file_path):
        self.code_tokenizer = code_tokenizer
        self.review_tokenizer = review_tokenizer
        if isinstance(code_tokenizer, MyTokenizer): code_tok_type = "mytok"
        elif isinstance(code_tokenizer, T5Tokenizer): code_tok_type = ""
        elif isinstance(code_tokenizer, RobertaTokenizer): code_tok_type = "rb"
        else: code_tok_type = "unk"

        if isinstance(review_tokenizer, MyTokenizer): review_tok_type = "mytok"
        elif isinstance(review_tokenizer, T5Tokenizer): review_tok_type = ""
        elif isinstance(review_tokenizer, RobertaTokenizer): review_tok_type = "rb"
        else: review_tok_type = "unk"
        
        savep = file_path.replace(".jsonl", "_ctt_"+code_tok_type+"_rtt_"+review_tok_type+".simprevrel")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            self.feats = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            data = read_jsonl(file_path)
            # data = [dic for dic in data if len(dic["patch"].split("\n")) <= 20]
            for i in range(len(data)):
                data[i]["idx"] = i
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

    def pad_assert(self, source_ids, target_ids, args):
        source_ids = source_ids[:args.max_source_length - 2]
        source_ids = [self.code_tokenizer.bos_id] + source_ids + [self.code_tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [self.code_tokenizer.pad_id] * pad_len
        target_ids = target_ids[:args.max_target_length - 1]
        target_ids = target_ids + [self.review_tokenizer.eos_id]
        pad_len = args.max_target_length - len(target_ids)
        target_ids += [self.review_tokenizer.pad_id] * pad_len
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(target_ids) == args.max_target_length, "Not equal length."
        return source_ids, target_ids

    def convert_examples_to_features(self, item):
        dic, args = item
        diff, msg = dic["patch"], dic["msg"]
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
        source_ids = self.encode_remove(self.code_tokenizer, inputstr, args)
        target_ids = []
        target_ids.append(self.review_tokenizer.msg_id)
        msg = self.encode_remove(self.review_tokenizer, dic["msg"], args)
        target_ids.extend(msg)
        source_ids, target_ids = self.pad_assert(source_ids, target_ids, args)

        source_ids = torch.as_tensor(source_ids)
        target_ids = torch.as_tensor(target_ids)
        source_mask = source_ids.ne(self.code_tokenizer.pad_id)
        target_mask = target_ids.ne(self.review_tokenizer.pad_id)

        return ReviewFeatures(
            example_id=dic["idx"], 
            source_ids=source_ids, 
            target_ids=target_ids,
            source_mask=source_mask,
            target_mask=target_mask,
            type="revrel"
        )

def get_loaders(data_file, args, code_tokenizer, review_tokenizer, eval=False):
    def fn(features): 
        # print(features[0].source_ids)
        return {
            "code_input_ids": torch.stack([feat.source_ids for feat in features]),
            "code_attention_mask": torch.stack([feat.source_mask for feat in features]),
            "review_input_ids": torch.stack([feat.target_ids for feat in features]),
            "review_attention_mask": torch.stack([feat.target_mask for feat in features]),
        }
    print("\x1b[33;1mdata_file:\x1b[0m", data_file)
    dataset = ReviewRelDataset(
        code_tokenizer=code_tokenizer,
        review_tokenizer=review_tokenizer,
        args=args, file_path=data_file
    )
    data_len = len(dataset)
    logger.info(f"Data length: {data_len}.")
    # if eval: sampler = SequentialSampler(dataset)
    # else: sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, shuffle=not(eval), collate_fn=fn,
        batch_size=args.batch_size if not eval else args.batch_size,
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

class ReviewRelevanceModel(nn.Module):
    def __init__(self, code_encoder_type: str="codebert", code_encoder_path: str="microsoft/codebert-base", 
                 review_encoder_type: str="codereviewer", review_encoder_path: str="microsoft/codereviewer"):
        super().__init__()
        self.code_encoder_type = code_encoder_type
        self.code_encoder_path = code_encoder_path
        self.review_encoder_type = review_encoder_type
        self.review_encoder_path = review_encoder_path
        if review_encoder_type == "codereviewer":
            review_model = AutoModel.from_pretrained(review_encoder_path)
            self.review_encoder = copy.deepcopy(review_model.encoder)
            del review_model
        if code_encoder_type == "codereviewer":
            code_model = AutoModel.from_pretrained(review_encoder_path)
            self.code_encoder = copy.deepcopy(code_model.encoder)
            del code_model
        else: self.code_encoder = AutoModel.from_pretrained(code_encoder_path)
        # self.loss_fn = CECosSimLoss()
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = InfoNCE()

    def encode_review(self, review_input_ids, review_attn_mask):
        return self.review_encoder(review_input_ids, review_attn_mask)

    def encode_code(self, code_input_ids, code_attn_mask):
        return self.code_encoder(code_input_ids, code_attn_mask)

    def forward(self, code_input_ids, code_attn_mask, review_input_ids, review_attn_mask, label):
        review_enc = self.encode_review(review_input_ids, review_attn_mask).last_hidden_state[:,0,:]
        if self.code_encoder_type == "codebert": code_enc = self.encode_code(code_input_ids, code_attn_mask).pooler_output
        else: code_enc = self.encode_code(code_input_ids, code_attn_mask).last_hidden_state[:,0,:]
        # loss = self.loss_fn(review_enc, code_enc, label)
        # loss = self.loss_fn(code_enc @ review_enc.T, label)

        # symmetric version of InfoNCE
        loss = self.loss_fn(review_enc, code_enc)+self.loss_fn(code_enc, review_enc)

        return review_enc, code_enc, loss

# Define the training loop
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(enumerate(dataloader), 
                total=len(dataloader), 
                desc="training")
    for step, batch in pbar:
        # Get inputs
        input_ids1 = batch["code_input_ids"]
        attention_mask1 = batch["code_attention_mask"]
        input_ids2 = batch["review_input_ids"]
        attention_mask2 = batch["review_attention_mask"]
        label = torch.as_tensor(range(len(input_ids1))).to(device)
        input_ids1, attention_mask1, input_ids2, attention_mask2, label = (
            input_ids1.to(device),
            attention_mask1.to(device),
            input_ids2.to(device),
            attention_mask2.to(device),
            label.to(device),
        )

        # Forward pass and Calculate contrastive loss
        _, _, loss = model(input_ids1, attention_mask1, input_ids2, attention_mask2, label)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description(f"bl: {loss:.3f} l: {(total_loss/(step+1)):.3f}")

    return total_loss / len(dataloader)

# Define the validation loop
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(enumerate(dataloader), 
                total=len(dataloader), 
                desc="validating")
    with torch.no_grad():
        for step, batch in pbar:
            # Get inputs
            input_ids1 = batch["code_input_ids"].to
            attention_mask1 = batch["code_attention_mask"]
            input_ids2 = batch["review_input_ids"]
            attention_mask2 = batch["review_attention_mask"]
            label = torch.as_tensor(range(len(input_ids1))).to(device)

            # Forward pass and Calculate contrastive loss
            _, _, loss = model(input_ids1, attention_mask1, input_ids2, attention_mask2, label)

            total_loss += loss.item()
            pbar.set_description(f"bl: {loss:.3f} l: {(total_loss/(step+1)):.3f}")

    return total_loss / len(dataloader)

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
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=100, help="Validation steps")
    parser.add_argument("--code_model_type", default="codebert", type=str, help="type of model/model class to be used")
    parser.add_argument("--code_model_path", default="microsoft/codebert-base", type=str, help="model name or path")
    parser.add_argument("--review_model_type", default="codereviewer", type=str, help="type of model/model class to be used")
    parser.add_argument("--review_model_path", default="microsoft/codereviewer", type=str, help="model name or path")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth", help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, required=True, help="directory where checkpoints will be stored.")
    parser.add_argument(
        "--seed", type=int, default=2233, help="random seed for initialization"
    )
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
    # Add other relevant arguments

    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model, optimizer, criterion, and other necessary components
    model = ReviewRelevanceModel(
        code_encoder_type=args.code_model_type,
        code_encoder_path=args.code_model_path,
        review_encoder_type=args.review_model_type,
        review_encoder_path=args.review_model_path,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optionally resume training
    if args.resume:
        # Load checkpoint and update model, optimizer, etc.
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"Resuming training from epoch {epoch}, best loss: {best_loss}")
        
    train_file = args.train_filename
    valid_file = args.dev_filename
    train_files = [train_file]

    random.seed(args.seed)
    random.shuffle(train_files)

    code_tokenizer = build_tokenizer(RobertaTokenizerFast, args.code_model_path)
    review_tokenizer = build_tokenizer(RobertaTokenizerFast, args.review_model_path)

    # Load your dataset and create DataLoader instances
    _, _, train_dataloader = get_loaders(
        data_file=train_file, args=args,
        code_tokenizer=code_tokenizer,
        review_tokenizer=review_tokenizer,
    )
    _, _, val_dataloader = get_loaders(
        data_file=valid_file, args=args,
        code_tokenizer=code_tokenizer,
        review_tokenizer=review_tokenizer,
    )

    best_loss = float("inf")

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {train_loss:.4f}")

        if (epoch + 1) % args.n_steps == 0:
            val_loss = validate(model, val_dataloader, device)
            print(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {val_loss:.4f}")

            # Save the model if it has the best contrastive loss
            if val_loss < best_loss:
                best_loss = val_loss
                ckpt_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                }
                ckpt_save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(ckpt_dict, ckpt_save_path)

if __name__ == "__main__":
    main()