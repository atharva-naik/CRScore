# use an LSTM model to generate reviews (review comments) from code patches in a seq2seq fashion.
import io
import os
import re
import torch
import random
import argparse
import unicodedata
import numpy as np
from tqdm import tqdm
# from io import open
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
# from __future__ import unicode_literals, print_function, division
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from CodeBERT.CodeReviewer.code.utils import SimpleGenDataset, CommentGenDataset

class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int=1, dropout_p: float=0.1):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded)
        return output, hidden

class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, 
                 max_length: int=200, num_layers: int=1, 
                 device: str="cuda:0", SOS_token: int=1,
                 EOS_token: int=2):
        super().__init__()
        self.device = device
        self.EOS_token = EOS_token
        self.SOS_token = SOS_token
        self.max_length = max_length
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 
                            num_layers=num_layers, batch_first=True)    
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class LSTMAttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, device: str="cuda:0", 
                 dropout_p=0.1, max_length: int=200, SOS_token: int=1, 
                 EOS_token: int=2):
        super(LSTMAttnDecoder, self).__init__()
        self.device = device
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.max_length = max_length
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, 
            dtype=torch.long, 
            device=self.device
        ).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_lstm = torch.cat((embedded, context), dim=2)

        output, hidden = self.lstm(input_lstm, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

class LSTMCodeReviewer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        self.encoder = LSTMEncoder(vocab_size, hidden_size)
        self.decoder = LSTMAttnDecoder(hidden_size, vocab_size)
    
    def forward(self, input_tensor, target_tensor=None):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = self.decoder(encoder_outputs, encoder_hidden, target_tensor)

        return decoder_outputs, decoder_hidden, decoder_attn

def evaluate(model, val_dataloader, criterion, tokenizer, args):
    model.eval()
    batch_losses = []
    pbar = tqdm(enumerate(val_dataloader),
                total=len(val_dataloader))
    index2word = {v: k for k,v in tokenizer.vocab.items()}

    for step, batch in pbar:
        with torch.no_grad():
            input_tensor, target_tensor = batch
            input_tensor = input_tensor.to(args.device)
            decoder_outputs, decoder_hidden, decoder_attn = model(input_tensor)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            batch_losses.append(loss.item())

            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == args.EOS_token:
                    decoded_words.append('</s>')
                    break
                decoded_words.append(index2word[idx.item()])
        pbar.set_description(f"bl: {(batch_losses[-1]):.3f} l: {np.mean(batch_losses):.3f}")
    # return decoded_words, decoder_attn

def get_args():
    parser = argparse.ArgumentParser("Script for fine-tuning Badhanu attention LSTM on review comment generation")
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # parser.add_argument(
    #     "--load_model_path",
    #     default=None,
    #     type=str,
    #     required=False
    # )
    # parser.add_argument(
    #     "--model_name_or_path",
    #     default=None,
    #     type=str,
    #     help="Path to trained model: Should contain the .bin files",
    # )
    # parser.add_argument(
    #     "--train_path",
    #     default=None,
    #     type=str,
    #     help="The pretrain files path. Should contain the .jsonl files for this task.",
    # )
    # parser.add_argument(
    #     "--eval_chunkname",
    #     default=None,
    #     type=str,
    #     help="The eval file name.",
    # )
    parser.add_argument(
        "--train_filename", default=None, type=str,
        help="The train filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--valid_filename", default=None, type=str,
        help="The validation filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. Should contain the .jsonl files for this task.",
    )
    # parser.add_argument(
    #     "--gold_filename",
    #     default=None,
    #     type=str,
    #     help="The gold filename. Should contain the .jsonl files for this task.",
    # )
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
        "--do_train", action="store_true", help="Whether to run eval on the train set."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    # parser.add_argument(
    #     "--raw_input", action="store_true", help="Whether to use simple input format (set for baselines)."
    # )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--mask_rate", default=0.15, type=float, help="The masked rate of input lines.",
    )
    parser.add_argument(
        "--beam_size", default=6, type=int, help="beam size for beam search"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--save_steps", default=-1, type=int,
    )
    parser.add_argument(
        "--log_steps", default=-1, type=int,
    )
    parser.add_argument("--eval_steps", default=-1, type=int, help="")
    parser.add_argument("--eval_file", default="", type=str)
    parser.add_argument("--out_file", default="", type=str)
    parser.add_argument("--break_cnt", default=-1, type=int)
    parser.add_argument("--train_steps", default=-1, type=int, help="")
    parser.add_argument(
        "--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--gpu_per_node",
        type=int,
        default=4,
        help="gpus per node",
    )
    parser.add_argument(
        "--node_index",
        type=int,
        default=0,
        help="For distributed training: node_index",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--seed", type=int, default=2233, help="random seed for initialization"
    )  # previous one 42

    args = parser.parse_args()
    return args

def train(args):
    def fn(features): return features
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
    vocab_size = len(tokenizer.vocab)
    model = LSTMCodeReviewer(vocab_size=vocab_size, hidden_size=768)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()
    best_loss = 10000
    # create dataloaders:
    train_dataset = SimpleGenDataset(tokenizer, pool=None, args=args, data_file=args.train_filename)
    val_dataset = SimpleGenDataset(tokenizer, pool=None, args=args.valid_filename)
    train_dataloader = 

    for epoch in range(1, args.n_epochs + 1):
        batch_losses = []
        pbar = tqdm(enumerate(train_dataloader),
                    total=len(train_dataloader))
        for step, batch in pbar:
            input_tensor, target_tensor = batch
            input_tensor = input_tensor.to(args.device)
            target_tensor = target_tensor.to(args.device)
            optimizer.zero_grad()
            decoder_outputs = model(input_tensor, target_tensor)
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            pbar.set_description(f"{epoch}/{args.n_epochs} bl: {batch_losses[-1]:.3f} l: {np.mean(batch_losses):.3f}")
            if step % args.log_steps == 0:
                evaluate(model, val_dataloader, criterion, tokenizer, args)
        # if epoch % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0

# main
if __name__ == "__main__":
    args = get_args()
    train(args)