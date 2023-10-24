import json
from tqdm import tqdm
from src.datautils import read_jsonl
from tree_sitter import Language, Parser


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
def extract_functions_and_classes(node, code):
    items = []

    if node.type == 'function_definition':
        function_name = "FUNC_NAME_NOT_FOUND"
        for child in node.children:
            if child.type == 'identifier':
                function_name = child.text.decode('utf-8')
                break
        function_source = code[node.start_byte:node.end_byte].decode('utf-8')
        items.append(('function', function_name, function_source))

    elif node.type == 'class_declaration':
        class_name = "CLASS_NAME_NOT_FOUND"
        for child in node.children:
            if child.type == 'identifier':
                class_name = child.text.decode('utf-8')
                break
        class_source = code[node.start_byte:node.end_byte].decode('utf-8')
        items.append(('class', class_name, class_source))

    for child in node.children:
        try: items.extend(extract_functions_and_classes(child, code))
        # avoid over-recusion.
        except RecursionError: return items

    return items


# # Extract functions
# def extract_functions_rec(node, code):
#     functions = []

#     if node.type == 'function_definition':
#         function_name = node.children[1].children[0].string
#         function_source = code[node.start_byte : node.end_byte].decode('utf-8')
#         functions.append((function_name, function_source))

#     for child in node.children:
#         functions.extend(extract_functions_rec(child, code))

#     return functions

# main
if __name__ == '__main__':
    # Initialize parser
    parser = Parser()
    # Read source code data.
    data = read_jsonl("./data/Comment_Generation/all_repo_data.jsonl")
    data += read_jsonl("./data/Comment_Generation/all_repo_data1.jsonl")
    indexed_data = []
    write_path = "./data/Comment_Generation/all_repo_indexed.jsonl"
    open(write_path, "w")
    for rec in tqdm(data):
        language = ext_to_languages[rec["lang"]]
        parser.set_language(language)
        tree = parser.parse(bytes(rec['content'], 'utf8'))
        functions = extract_functions_and_classes(tree.root_node, bytes(rec['content'], 'utf8'))
        del rec["content"]
        rec['items'] = functions
        with open(write_path, "a") as f:
            f.write(json.dumps(rec)+"\n")