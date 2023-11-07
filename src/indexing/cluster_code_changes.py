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
from sentence_transformers import SentenceTransformer, util
from src.datautils import read_jsonl, generate_before_after_code_from_patch

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

def collapse_key(key: str) -> str:
    if key == "" or not key[0].isalpha(): return "punct"

    elif key.endswith("operator"): return "operator"
    elif key.endswith("declarator"): return "declarator"
    elif key.endswith("declaration"): return "declaration"
    elif key.endswith("expression"): return "expression"
    elif key.endswith("statement"): return "statement"
    elif key.endswith("comment"): return "comment"
    elif key.endswith("literal"): return "literal"
    elif key.endswith("type"): return "type"
    elif key.endswith("directive"): return "directive"
    elif key.endswith("pattern"): return "pattern"
    elif key.endswith("element"): return "element"
    elif key.endswith("clause"): return "clause"
    elif key.endswith("list"): return "list"
    elif key.endswith("argument"): return "argument"
    elif key.endswith("parameter"): return "parameter"
    elif key.endswith("function"): return "function"
    elif key.endswith("class"): return "class"
    elif key.endswith("definition"): return "definition"
    elif key.endswith("modifier"): return "modifier"
    elif key.endswith("assignment"): return "assignment"
    elif key.endswith("identifier"): return "identifier"
    elif key.endswith("comprehension"): return "comprehension"
    elif key.endswith("specifier"): return "specifier"
    elif key.endswith("method"): return "method"
    elif key.endswith("spec") or key.endswith("specification"): return "spec"
    elif key.endswith("attribute"): return "attribute"
    elif key.endswith("block"): return "block"
    elif key.endswith("parameters"): return "parameter"
    elif key.endswith("text"): return "text"
    elif key.endswith("expr"): return "expression"
    elif key.endswith("case"): return "case"
    elif key.endswith("variable"): return 'variable'
    elif key.endswith("_name"): return "name"
    elif key.endswith("string"): return "string"
    elif key.endswith("assert"): return "assert"
    elif key.endswith("interfaces"): return "interface"
    elif key.endswith("interpolation"): return "interplolation"
    elif key.endswith("types"): return "type"

    elif key == "elseif": return "elif"
    elif key == "definition": return "def"
    elif key == 'requires': return "require"
    elif key == "const": return "constant"
    elif key == "arguments": return "argument"
    elif key == "defined?": return "defined"
    elif key == "modifiers": return "modifier"
    elif key == "nil": return "null"
    elif key == "asm": return "assembly"
    elif key == "exports": return "export"
    elif key == "del": return "delete"
    elif key == "int": return "integer"

    elif "import" in key: return "import"
    elif "array" in key: return "array"

    elif key.startswith("ref"): return "ref"
    elif key.startswith("raw_string"): return "string"
    elif key.startswith("print"): return "print"
    elif key.startswith("hash"): return "hash"
    elif key.startswith("interface"): return "interface"
    elif key.startswith("method"): return "method"
    elif key.startswith("literal"): return "literal"
    elif key.startswith("type"): return "type"
    elif key.startswith("throw"): return "throw"
    elif key.startswith("template"): return "template"
    elif key.startswith("switch"): return "switch"
    elif key.startswith("struct"): return "struct"
    elif key.startswith("module"): return "module"
    elif key.startswith("lambda"): return "lambda"
    elif key.startswith("constructor"): return "constructor"
    elif key.startswith("block"): return "block"
    elif key.startswith("preproc"): return "preproc"
    elif key.startswith("char_"): return "char"
    elif key.startswith("character"): return "char"
    elif key.startswith("class"): return "class"
    elif key.startswith("delete"): return "delete"
    elif key.startswith("range"): return "range"
    elif key.lower().startswith("null"): return "null"
    elif key.startswith("regex"): return "regex"
    elif key.startswith("enum_"): return "enum"
    elif key.startswith("with_"): return "with"
    elif key.startswith("namespace_"): return "namespace"
    elif key.startswith("annotat"): return "annotation"
    elif key.startswith("case_"): return "case"
    elif key.startswith("heredoc_"): return "heredoc"
    # elif key.startswith("assignment_"): return "assignment"
    elif key.startswith("__") or key.endswith("__"): return "internal"
    elif key.startswith("string_"): return "string"
    elif key.startswith("alias"): return "alias"
    elif key.startswith("as_"): return "as"
    elif key.startswith("bare"): return "bare"
    elif key.startswith("field"): return "field"
    elif key.startswith("extends"): return "extends"
    elif key.startswith("element"): return "element"

    else: return key

def build_code_change_vector(code_change: str, lang: str, vocab: dict, 
                             parser, use_line_change_stats: bool=False,
                             use_reduced_vocab: bool=True):
    lines_added = 0
    lines_removed = 0
    lines_unchanged = 0
    # inv_vocab = {v: k for k,v in vocab.items()}
    before_code, after_code = generate_before_after_code_from_patch(code_change)
    b_node_type_dist = lex_code(before_code, parser, lang)
    a_node_type_dist = lex_code(after_code, parser, lang)
    before_node_type_dist = get_dist([collapse_key(typ) if use_reduced_vocab else typ for typ,_ in b_node_type_dist])
    after_node_type_dist = get_dist([collapse_key(typ) if use_reduced_vocab else typ for typ,_ in a_node_type_dist])

    for line in code_change.split("\n"):
        if line.startswith("+"): lines_added += 1
        elif line.startswith("-"): lines_removed += 1
        else: lines_unchanged += 1
    
    offset = 3 if use_line_change_stats else 0
    vec = np.zeros(len(vocab)+offset)
    if use_line_change_stats:
        vec[0] = lines_added
        vec[1] = lines_removed
        vec[2] = lines_unchanged

    for key, value in before_node_type_dist.items(): vec[vocab[key]+offset] -= value
    for key, value in after_node_type_dist.items(): vec[vocab[key]+offset] += value

    return vec

def build_review_vector(review: str, sbert, sbert_vocab_index):
    sbert_review_index = sbert.encode(review, convert_to_tensor=True, show_progress_bar=False)
    return util.cos_sim(sbert_review_index, sbert_vocab_index).squeeze().cpu().numpy()

def build_vocab(data, parser):
    vocab = set()
    for item in tqdm(data):
        lang = item["lang"].replace(".","")
        before_code, after_code = generate_before_after_code_from_patch(item["patch"])
        for typ in lex_code(before_code, parser, lang, return_code=False): vocab.add(typ)
        for typ in lex_code(after_code, parser, lang, return_code=False): vocab.add(typ)
    vocab = {k: i for i,k in enumerate(sorted(vocab))}

    return vocab

def reduce_vocab(vocab):
    reduced_vocab = set()
    for key, value in vocab.items():
        reduced_vocab.add(collapse_key(key))

    return {k: i for i,k in enumerate(sorted(reduced_vocab))}

def load_vocab(data, parser, apply_vocab_reduction: bool=True,
               path: str="./data/Comment_Generation/node_type_vocab.json"):
    if os.path.exists(path): vocab = json.load(open(path))
    else:
        vocab = build_vocab(data, parser)
        with open(path, "w") as f: json.dump(vocab, f, indent=4)
    if apply_vocab_reduction:
        vocab = reduce_vocab(vocab)
        with open(path.replace("vocab", "reduced_vocab"), "w") as f: json.dump(vocab, f, indent=4)

    return vocab

# main 
if __name__ == "__main__":
    data = read_jsonl("./data/Comment_Generation/msg-test.jsonl")+read_jsonl("./data/Comment_Generation/msg-valid.jsonl")
    parser = Parser()
    vocab = load_vocab(data, parser)
    
    cc_vectors = []
    for item in tqdm(data):
        lang = item["lang"].replace(".","")
        cc_vector = build_code_change_vector(code_change=item['patch'], lang=lang, 
                                             vocab=vocab, parser=parser)
        cc_vectors.append(cc_vector)
        # print(cc_vector, sum(cc_vector), sum(cc_vector[3:]))
    print("vector dimension:", len(cc_vector))
    
    sbert = SentenceTransformer("all-mpnet-base-v2")
    sbert.cuda()
    sbert_vocab_index = sbert.encode(list(vocab.keys()), convert_to_tensor=True)
    cr_vectors = []
    for item in tqdm(data):
        cr_vector = build_review_vector(review=item['msg'], sbert=sbert, 
                                        sbert_vocab_index=sbert_vocab_index)
        cr_vectors.append(cr_vector)
    
    C = np.stack(cc_vectors)
    R = np.stack(cr_vectors)
    # # reduce embedding dimension with PCA.
    # pca = PCA(n_components=100)
    # C_ = pca.fit_transform(C)
    C_ = C
    R_ = R

    # do normalization.
    length = np.sqrt((C_**2).sum(axis=1))[:,None]+1e-12
    # print(length.shape)
    # print(C_.shape)
    C_ = C_ / length
    length = np.sqrt((R_**2).sum(axis=1))[:,None]+1e-12
    R_ = R_ / length

    np.save('./data/Comment_Generation/code_change_vecs.npy', C_)
    np.save('./data/Comment_Generation/code_reviews.npy', R_)

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
    # print(pca.explained_variance_ratio_.sum())

    # # some sanity checking.
    # inds = util.cos_sim(Y, Y).argsort()
    # print("\x1b[34;1mref code:\x1b[0m", data[0]["patch"])
    # print("\x1b[34;1mref review:\x1b[0m", data[0]["msg"])
    # for i in inds[0][:5]:
    #     print("\x1b[1mref code:\x1b[0m", data[i]["patch"])
    #     print("\x1b[1mref review:\x1b[0m", data[i]["msg"])