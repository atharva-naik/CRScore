import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from src.losses.info_nce import normalize
from transformers import RobertaTokenizerFast
from src.models.code_review_rel import ReviewRelevanceModel, ReviewFeatures, build_tokenizer, get_loaders

def get_args():
    parser = argparse.ArgumentParser(description="Do clustering with contrastively trained relevance model")
    parser.add_argument("--code_model_type", default="codereviewer", type=str, help="type of model/model class to be used")
    parser.add_argument("--code_model_path", default="microsoft/codereviewer", type=str, help="model name or path")
    parser.add_argument("--review_model_type", default="codereviewer", type=str, help="type of model/model class to be used")
    parser.add_argument("--device", default="cuda:0", type=str, help="device on which model should be loaded")
    parser.add_argument("--review_model_path", default="microsoft/codereviewer", type=str, help="model name or path")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth", help="Path to the checkpoint file")
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
    # Add other relevant arguments

    return parser.parse_args()

def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v

# main 
if __name__ == "__main__":
    args = get_args()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model = ReviewRelevanceModel(
        code_encoder_type=args.code_model_type,
        code_encoder_path=args.code_model_path,
        review_encoder_type=args.review_model_type,
        review_encoder_path=args.review_model_path,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)

    code_tokenizer = build_tokenizer(RobertaTokenizerFast, args.code_model_path)
    review_tokenizer = build_tokenizer(RobertaTokenizerFast, args.review_model_path)

    valid_file = "data/Comment_Generation/msg-valid.jsonl"
    _, _, val_dataloader = get_loaders(
        data_file=valid_file, args=args,
        code_tokenizer=code_tokenizer,
        review_tokenizer=review_tokenizer,
    )
    test_file = "data/Comment_Generation/msg-test.jsonl"
    _, _, test_dataloader = get_loaders(
        data_file=test_file, args=args,
        code_tokenizer=code_tokenizer,
        review_tokenizer=review_tokenizer,
    )
    model.eval()

    cc_vectors, cr_vectors = [], []
    for batch in tqdm(itr_merge(val_dataloader, test_dataloader)):
        # Get inputs
        code_input_ids = batch["code_input_ids"].to(args.device)
        code_attention_mask = batch["code_attention_mask"].to(args.device)
        review_input_ids = batch["review_input_ids"].to(args.device)
        review_attention_mask = batch["review_attention_mask"].to(args.device)

        with torch.no_grad():
            review_emb = model.encode_review(review_input_ids, review_attention_mask).last_hidden_state[:,0,:]
            code_emb = model.encode_code(code_input_ids, code_attention_mask).last_hidden_state[:,0,:]
            review_emb, code_emb = normalize(review_emb, code_emb)
            cc_vectors.extend(code_emb.cpu().numpy())
            cr_vectors.extend(review_emb.cpu().numpy())

    C = np.stack(cc_vectors)
    R = np.stack(cr_vectors)
    print("C.shape:", C.shape)
    print("R.shape:", R.shape)
    # # reduce embedding dimension with PCA.
    # pca = PCA(n_components=100)
    # C_ = pca.fit_transform(C)
    C_ = C
    R_ = R

    np.save('./data/Comment_Generation/code_change_sem_vecs.npy', C_)
    np.save('./data/Comment_Generation/code_review_sem_vecs.npy', R_)

    k = 100

    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init='auto').fit(C_)
    clusters = [[] for _ in range(k)]
    for i, label in enumerate(kmeans.labels_.tolist()):
        clusters[label].append(i)
    with open("./data/Comment_Generation/dev_test_code_change_clusters.json", "w") as f:
        json.dump(clusters, f, indent=4)

    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init='auto').fit(R_)
    clusters = [[] for _ in range(k)]
    for i, label in enumerate(kmeans.labels_.tolist()):
        clusters[label].append(i)
    with open("./data/Comment_Generation/dev_test_review_clusters.json", "w") as f:
        json.dump(clusters, f, indent=4)