# script for clustering code changes based on the kind of modifications.
import os
import json
import tree_sitter 
import numpy as np
from typing import *
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import util
from tree_sitter import Parser, Language

ext_to_languages = {
    "c": Language('/home/arnaik/CodeReviewEval/src/indexing/my-languages.so', 'c'),
    "cpp": Language('/home/arnaik/CodeReviewEval/src/indexing/my-languages.so', 'cpp'),
    "cs": Language('/home/arnaik/CodeReviewEval/src/indexing/my-languages.so', 'c_sharp'),
    "py": Language('/home/arnaik/CodeReviewEval/src/indexing/my-languages.so', 'python'),
    "rb": Language('/home/arnaik/CodeReviewEval/src/indexing/my-languages.so', 'ruby'),
    "js": Language('/home/arnaik/CodeReviewEval/src/indexing/my-languages.so', 'javascript'),
    "php": Language('/home/arnaik/CodeReviewEval/src/indexing/my-languages.so', 'php'),
    "java": Language('/home/arnaik/CodeReviewEval/src/indexing/my-languages.so', 'java'),
    "go": Language('/home/arnaik/CodeReviewEval/src/indexing/my-languages.so', 'go'),
}

def extract_tokens(code, node):
    for child in node.children:
        yield from extract_tokens(code, child)
    yield (node.type, code[node.start_byte:node.end_byte])

FILT_LIST = ["ERROR"]#['"', "'", ":", "::", ";", ".", ",", "ERROR", "=", "}", "{", ")", "(", "[", "]", "\n", "", "!", "!=", "!==", "!~", "\\", "$", "&&"]
def lex_code(code: str, parser, lang: str, 
             return_code: bool=True, 
             filt_list: List[str]=FILT_LIST) -> Union[List[Tuple[str, str]], List[str]]:
    global ext_to_languages
    language = ext_to_languages[lang]
    parser.set_language(language)
    tree = parser.parse(bytes(code, 'utf-8'))
    
    node_types_and_code = [(typ, code) for typ, code in list(extract_tokens(code, tree.root_node)) if typ not in filt_list]
    if not return_code: return [typ for typ, _ in node_types_and_code]

    return node_types_and_code

def test_lexer():
    parser = Parser()
    code = """
    public class HelloWorld {
        public static void main(String[] args) {
            System.out.println("Hello, World!");
        }
    }
    """
    print(lex_code(code, parser, "java"))

def get_dist(l: List[str], sort_keys: bool=False):
    d = {}
    keys, values = np.unique(l, return_counts=True)
    for i in range(len(keys)): d[keys[i]] = values[i]
    if sort_keys: d = dict(sorted(d.items()))

    return d

def build_code_change_vector(code_change: str, lang: str, vocab: dict, 
                             use_line_change_stats: bool=False):
    lines_added = 0
    lines_removed = 0
    lines_unchanged = 0
    # inv_vocab = {v: k for k,v in vocab.items()}
    before_code, after_code = generate_before_after_code_from_patch(code_change)
    b_node_type_dist = lex_code(before_code, parser, lang)
    a_node_type_dist = lex_code(after_code, parser, lang)
    before_node_type_dist = get_dist([typ for typ,_ in b_node_type_dist])
    after_node_type_dist = get_dist([typ for typ,_ in a_node_type_dist])

    for line in item["patch"].split("\n"):
        if line.startswith("+"): lines_added += 1
        elif line.startswith("-"): lines_removed += 1
        else: lines_unchanged += 1

    
    vec = np.zeros(len(vocab)+3)
    if use_line_change_stats:
        vec[0] = lines_added
        vec[1] = lines_removed
        vec[2] = lines_unchanged
    offset = 3 if use_line_change_stats else 0

    for key, value in before_node_type_dist.items(): vec[vocab[key]+offset] -= value
    for key, value in after_node_type_dist.items(): vec[vocab[key]+offset] += value

    return vec

def build_vocab(data, parser):
    vocab = set()
    for item in tqdm(data):
        lang = item["lang"].replace(".","")
        before_code, after_code = generate_before_after_code_from_patch(item["patch"])
        for typ in lex_code(before_code, parser, lang, return_code=False): vocab.add(typ)
        for typ in lex_code(after_code, parser, lang, return_code=False): vocab.add(typ)
    vocab = {k: i for i,k in enumerate(sorted(vocab))}

    return vocab

def load_vocab(data, parser, path: str="./data/Comment_Generation/node_type_vocab.json"):
    if os.path.exists(path): return json.load(open(path))
    vocab = build_vocab(data, parser)
    with open(path, "w") as f: json.dump(vocab, f, indent=4)

    return vocab

# main 
if __name__ == "__main__":
    from src.datautils import read_jsonl, generate_before_after_code_from_patch
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")+read_jsonl("./data/Comment_Generation/msg-valid.jsonl")
    parser = Parser()
    vocab = load_vocab(data, parser)
    print("vector dimension:", len(vocab)+3)
    cc_vectors = []
    for item in tqdm(data):
        lang = item["lang"].replace(".","")
        cc_vector = build_code_change_vector(item['patch'], lang, vocab)
        cc_vectors.append(cc_vector)
        # print(cc_vector, sum(cc_vector), sum(cc_vector[3:]))
    X = np.stack(cc_vectors)
    pca = PCA(n_components=100)
    Y = pca.fit_transform(X)
    k = 10
    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init='auto').fit(Y)
    clusters = [[] for _ in range(k)]
    for i, label in enumerate(kmeans.labels_.tolist()):
        clusters[label].append(i)
    with open("./data/Comment_Generation/dev_test_code_change_clusters.json", "w") as f:
        json.dump(clusters, f, indent=4)
    # print(pca.explained_variance_ratio_.sum())

    # # some sanity checking.
    # inds = util.cos_sim(Y, Y).argsort()
    # print("\x1b[34;1mref code:\x1b[0m", data[0]["patch"])
    # print("\x1b[34;1mref review:\x1b[0m", data[0]["msg"])
    # for i in inds[0][:5]:
    #     print("\x1b[1mref code:\x1b[0m", data[i]["patch"])
    #     print("\x1b[1mref review:\x1b[0m", data[i]["msg"])