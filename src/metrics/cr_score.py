import json
import torch
import random
import numpy as np
from typing import *
from tqdm import tqdm
from argparse import Namespace
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, T5Tokenizer, RobertaTokenizer, logging
from src.models.code_review_rel import ReviewRelDataset, ReviewRelevanceModel, build_tokenizer
from src.models.code_review_rel_clf import ReviewRelevanceClf, ReviewClfDataset, build_tokenizer as build_tokenizer_v2
from sentence_transformers import util

logging.set_verbosity_error() # to silence fine-tuning related warnings.

# Review Relevance Dataset.
class ReviewRelDatasetFromData(ReviewRelDataset):
    def __init__(self, code_tokenizer, review_tokenizer,
                 code_changes: List[str], code_reviews: List[str], **kwargs):
        self.code_tokenizer = code_tokenizer
        args = Namespace()
        self.review_tokenizer = review_tokenizer
        data = [
            {
                "idx": None, 
                "patch": patch, 
                "msg": msg
            } for patch, msg in zip(code_changes, code_reviews)
        ]
        args.max_source_length = kwargs.get("max_source_length", 200)
        args.max_target_length = kwargs.get("max_target_length", 100)
        for i in range(len(data)): data[i]["idx"] = i
        self.feats = [self.convert_examples_to_features((dic, args)) for dic in tqdm(data)]

# Review Relevance Classification Dataset.
class ReviewClsDatasetFromData(ReviewClfDataset):
    def __init__(self, tokenizer, code_changes: List[str], code_reviews: List[str], **kwargs):
        self.tokenizer = tokenizer
        args = Namespace()
        data = [
            {
                "idx": None, 
                "patch": patch, 
                "msg": msg,
                "cls": 1 # just a placeholder value (not really used)
            } for patch, msg in zip(code_changes, code_reviews)
        ]
        args.max_source_length = kwargs.get("max_source_length", 300)
        # args.max_target_length = kwargs.get("max_target_length", 100)
        for i in range(len(data)): data[i]["idx"] = i
        self.feats = [self.convert_examples_to_features((dic, args)) for dic in tqdm(data)]

def get_loader(code_changes, code_reviews, code_tokenizer, review_tokenizer, eval=False, batch_size: int=32):
    def fn(features): 
        # print(features[0].source_ids)
        return {
            "code_input_ids": torch.stack([feat.source_ids for feat in features]),
            "code_attention_mask": torch.stack([feat.source_mask for feat in features]),
            "review_input_ids": torch.stack([feat.target_ids for feat in features]),
            "review_attention_mask": torch.stack([feat.target_mask for feat in features]),
        }
    # print("\x1b[33;1mdata_file:\x1b[0m", data_file)
    dataset = ReviewRelDatasetFromData(
        code_tokenizer=code_tokenizer,
        review_tokenizer=review_tokenizer,
        code_changes=code_changes,
        code_reviews=code_reviews,
    )
    # data_len = len(dataset)
    # logger.info(f"Data length: {data_len}.")
    # if eval: sampler = SequentialSampler(dataset)
    # else: sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, shuffle=not(eval), 
        collate_fn=fn, batch_size=batch_size,
    )
    
    return dataset, None, dataloader

def get_loader_v2(code_changes, code_reviews, tokenizer, eval=False, batch_size: int=32):
    def fn(features): 
        # print(features[0].source_ids)
        return {
            "input_ids": torch.stack([feat.source_ids for feat in features]),
            "attention_mask": torch.stack([feat.source_mask for feat in features]),
            # "review_input_ids": torch.stack([feat.target_ids for feat in features]),
            # "review_attention_mask": torch.stack([feat.target_mask for feat in features]),
        }
    # print("\x1b[33;1mdata_file:\x1b[0m", data_file)
    dataset = ReviewClsDatasetFromData(
        tokenizer=tokenizer,
        code_changes=code_changes,
        code_reviews=code_reviews,
    )
    # data_len = len(dataset)
    # logger.info(f"Data length: {data_len}.")
    # if eval: sampler = SequentialSampler(dataset)
    # else: sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, shuffle=not(eval), 
        collate_fn=fn, batch_size=batch_size,
    )
    
    return dataset, None, dataloader

# CR scoring model.
class CRScorer2:
    def __init__(self, checkpoint_path: str, seed: int=2233,
                 model_type: str="codereviewer", model_path: str="microsoft/codereviewer"):
        # Initialize model, optimizer, criterion, and other necessary components
        print(f"\x1b[34;1mloading checkpoint: {checkpoint_path}\x1b[0m for CRScore")
        self.model = ReviewRelevanceClf(
            model_type=model_type,
            model_path=model_path
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load the checkpoint.
        assert checkpoint_path is not None
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.tokenizer = build_tokenizer_v2(RobertaTokenizerFast, model_path)

    def compute(self, codes: List[str]=[], predictions: List[str]=[], 
                references: List[str]=[], batch_size: int=32, 
                use_tqdm: bool=False):
        _, _, dataloader = get_loader_v2(
            code_changes=codes, code_reviews=predictions, 
            tokenizer=self.tokenizer, eval=True, batch_size=batch_size,
        )
        pbar = tqdm(enumerate(dataloader), 
                    total=len(dataloader), 
                    desc="computing CR score",
                    disable=not(use_tqdm))
        inst_scores = []
        with torch.no_grad():
            for step, batch in pbar:
                # Get inputs
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                # label = torch.as_tensor(range(len(input_ids1))).to(device)

                # Forward pass and Calculate contrastive loss
                _, logits, _ = self.model(input_ids, attention_mask)
                inst_scores.extend(torch.nn.functional.softmax(logits)[:,1].cpu().tolist())

        return {"inst_scores": inst_scores, "score": np.mean(inst_scores)}

# CR scoring model.
class CRScorer:
    def __init__(self, checkpoint_path: str, temperature: float=0.005, seed: int=2233,
                 code_model_type: str="codereviewer", code_model_path: str="microsoft/codereviewer",
                 review_model_type: str="codereviewer", review_model_path: str="microsoft/codereviewer",
                 asym_code_first: bool=False, asym_review_first: bool=False):
        # Initialize model, optimizer, criterion, and other necessary components
        print(f"\x1b[34;1mloading checkpoint: {checkpoint_path}\x1b[0m for CRScore")
        self.model = ReviewRelevanceModel(
            code_encoder_type=code_model_type,
            code_encoder_path=code_model_path,
            review_encoder_type=review_model_type,
            review_encoder_path=review_model_path,
            temperature=temperature,
            asym_code_first=asym_code_first,
            asym_review_first=asym_review_first,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load the checkpoint.
        assert checkpoint_path is not None
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.code_tokenizer = build_tokenizer(RobertaTokenizerFast, code_model_path)
        self.review_tokenizer = build_tokenizer(RobertaTokenizerFast, review_model_path)

    def compute(self, codes: List[str]=[], predictions: List[str]=[], 
                references: List[str]=[], batch_size: int=32, 
                use_tqdm: bool=False):
        _, _, dataloader = get_loader(
            code_changes=codes, code_reviews=predictions, 
            code_tokenizer=self.code_tokenizer,
            review_tokenizer=self.review_tokenizer,
            eval=True, batch_size=batch_size,
        )
        pbar = tqdm(enumerate(dataloader), 
                    total=len(dataloader), 
                    desc="computing CR score",
                    disable=not(use_tqdm))
        inst_scores = []
        with torch.no_grad():
            for step, batch in pbar:
                # Get inputs
                input_ids1 = batch["code_input_ids"].to(self.device)
                attention_mask1 = batch["code_attention_mask"].to(self.device)
                input_ids2 = batch["review_input_ids"].to(self.device)
                attention_mask2 = batch["review_attention_mask"].to(self.device)
                # label = torch.as_tensor(range(len(input_ids1))).to(device)

                # Forward pass and Calculate contrastive loss
                review_enc, code_enc, loss = self.model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                inst_scores.extend(util.cos_sim(code_enc, review_enc).diag().tolist())

        return {"inst_scores": inst_scores, "score": np.mean(inst_scores)}
    
# main
if __name__ == "__main__":
    # cr_score = CRScorer(checkpoint_path="./ckpts/crr_rcr_ccr_0.005/best_model.pth")
    cr_score = CRScorer2(checkpoint_path="ckpts/rr_clf/best_model.pth")
    import pandas as pd
    # df = pd.read_csv('cr_manual_rel_annot_likert_scale.csv')
    df = pd.read_csv('human_study_data.csv')
    # k = 100
    # predictions = list(df['pred'])[:k]
    rel_scores = []
    codes = list(df['patch'])#[:k]
    model_wise_scores = {}
    for ref_subset, model_name in [
        ('msg', 'ground_truth'),
        ('knn_pred', 'knn'),
        ('lstm_pred', 'lstm'),
        ('magicoder_pred', 'magicoder'),
        ('codereviewer_pred', 'codereviewer')
    ]:
        predictions = list(df[ref_subset])
        score = cr_score.compute(predictions=predictions, codes=codes)
        model_wise_scores[model_name] = score['inst_scores']
        print(model_name, score['score'])
    indices = list(df['index'])
    for i in range(len(indices)):
        rel_scores.append({
            "index": indices[i],
            "ground_truth": model_wise_scores['ground_truth'][i],
            "knn": model_wise_scores['knn'][i],
            "lstm": model_wise_scores['lstm'][i],
            "magicoder": model_wise_scores['magicoder'][i],
            "codereviewer": model_wise_scores['codereviewer'][i]
        })
    with open("./human_study_relevance_scores.json", "w") as f:
        json.dump(rel_scores, f, indent=4) 
    # print(score)