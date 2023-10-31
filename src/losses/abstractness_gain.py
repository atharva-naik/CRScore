import torch
from typing import *
from tqdm import tqdm
from torch import Tensor
import sentence_transformers
from sentence_transformers import util, SentenceTransformer

CONCRETE_WORDS = ["donut", "antlers", "aquarium", "nursemaid", "pyrethrum", "swallowwort", "strongbox", "sixth-former", "restharrow", "recorder", "sawmill", "vulval", "tenrecidae", "hairpiece", "sturnus", "gadiformes", "cobbler", "bullet", "dioxin", "usa"]
ABSTRACT_WORDS = ["sense", "indulgent", "bedevil", "improbable", "purvey", "pigheadedness", "ranging", "quietus", "regularisation", "creditably", "arcella", "nonproductive", "couth", "repulsion", "palsgrave", "goof-proof", "meshuga", "dillydally", "reliance", "lumbus"]

class AbstractnessScorer:
    def __init__(self, model_name: str):
        self.sbert = SentenceTransformer(model_name)
        self.model_name = model_name 
        self.abstract_paradigm_encodings = self.encode_paradigms(ABSTRACT_WORDS)
        self.concrete_paradigm_encodings = self.encode_paradigms(CONCRETE_WORDS)

    def score_word_by_word(self, inp_list: List[str]):
        inp_embs = self.sbert.encode(inp_list, convert_to_tensor=True)
        abs_score = util.cos_sim(self.abstract_paradigm_encodings, inp_embs).max(axis=0).values.mean().cpu().item()
        conc_score = util.cos_sim(self.concrete_paradigm_encodings, inp_embs).max(axis=0).values.mean().cpu().item()
        total_score = (abs_score - conc_score)/2

        return round(abs_score, 3), round(conc_score, 3), round(total_score, 3)

    def score(self, inp: str):
        abs_score, conc_score, total_score = self.score_word_by_word(inp.split())
        p_abs_score, p_conc_score, p_total_score = self.score_word_by_word([inp])
        return {"word_by_word": {"abs_score": abs_score, "conc_score": conc_score, "total_score": total_score}, 
                "phrase": {"abs_score": p_abs_score, "conc_score": p_conc_score, "total_score": p_total_score}}

    def encode_paradigms(self, word_list) -> Tensor:
        return self.sbert.encode(word_list, convert_to_tensor=True)

# main
if __name__ == "__main__":
    from src.datautils import read_jsonl
    abs_scorer = AbstractnessScorer("all-mpnet-base-v2")
    # op = abs_scorer.score("His music tastes are very fickle")
    # print(op["word_by_word"]["total_score"], op["phrase"]["total_score"])
    # op = abs_scorer.score("The nile river is 3 metres long")
    # print(op["word_by_word"]["total_score"], op["phrase"]["total_score"])
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")
    for item in tqdm(data):
        item[""]