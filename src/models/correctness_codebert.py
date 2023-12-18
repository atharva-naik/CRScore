import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class CorrectnessCodeBERT(nn.Module):
    def __init__(self, model_path, 
                 code_enc_dim: int=768, 
                 review_enc_dim: int=768):
        super().__init__()
        self.code_encoder = AutoModel.from_pretrained(model_path)
        self.review_encoder = AutoModel.from_pretrained(model_path)
        self.code_enc_dim = code_enc_dim
        self.review_enc_dim = review_enc_dim
        self.code_comparison_network = nn.Linear(4*code_enc_dim, code_enc_dim)
        self.loss_fn = nn.TripletMarginLoss(margin=1, p=2)

    def forward(self, anchor_before, anchor_after, pos, neg):
        before_enc = self.code_encoder(**anchor_before).pooler_output
        after_enc = self.code_encoder(**anchor_after).pooler_output
        anchor = self.code_comparison_network(torch.cat([
            after_enc-before_enc,
            after_enc*before_enc, 
            before_enc, after_enc
        ], axis=-1))
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

# main
if __name__ == "__main__":
    test_function()