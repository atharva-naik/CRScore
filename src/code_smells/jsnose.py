# Javascript code smell detection
import os
import json
import esprima
import esprima.nodes
from typing import *

# def highest_nesting_chain_rec(lst, current_chain=[]):
#     if not isinstance(lst, list):
#         return current_chain
#     if all(not isinstance(item, list) for item in lst):
#         return current_chain
#     chains = [highest_nesting_chain_rec(item, current_chain + [item]) for item in lst if isinstance(item, list)]
#     max_chain = max(chains, key=len)
#     return max_chain

# def highest_nesting_chain(lst):
#     max_chain = highest_nesting_chain_rec(lst)
#     print(max_chain)
#     return [item[0] for item in max_chain]

def highest_nesting_level(lst):
    if not isinstance(lst, list):
        return 0
    if all(not isinstance(item, list) for item in lst):
        return 1
    return 1 + max(highest_nesting_level(item) for item in lst if isinstance(item, list))

def extract_nested_functions(node_or_list):
    result = []
    if isinstance(node_or_list, esprima.nodes.Node):
        body = node_or_list.body
        body = get_descendant(node_or_list)
        return extract_nested_functions(body)

    elif isinstance(node_or_list, list):
        for node in node_or_list:
            if node.type == "FunctionDeclaration":
                body = get_descendant(node)
                sub_result = extract_nested_functions(body)
                # print(sub_result+[node.id.name])
                result.append(sub_result+[node.id.name])

    return result
# def extract_nested_functions(node, current_path=[], function_names=[]):
#     if isinstance(node, esprima.nodes.Node):
#         if node.type == 'FunctionDeclaration':
#             function_names.append(current_path + [node.id.name])
#         for key, value in node.items():
#             print(key)
#             if isinstance(value, list):
#                 for i, item in enumerate(value):
#                     if item.type != "FunctionDeclaration": continue
#                     extract_nested_functions(item, current_path + [key, str(i)], function_names)
#             else:
#                 extract_nested_functions(value, current_path, function_names)

#     return function_names

def get_descendant(node):
    if isinstance(node, esprima.nodes.VariableDeclaration):
        return node.declarations
    elif isinstance(node, esprima.nodes.VariableDeclarator):
        return node.init
    elif isinstance(node, esprima.nodes.ExpressionStatement):
        return [node.expression]
    elif isinstance(node, esprima.nodes.AssignmentExpression):
        return [node.left, node.right]
    elif isinstance(node, esprima.nodes.ExpressionStatement):
        return node.expression
    else: return node.body

class CodeSmellDetector:
    def __init__(self):
        self.js_ast = None
        self.all_smells = []
        self.LONGEST_SCOPING_CHAIN_THRESH = 3
        self.EXCESSIVE_GLOBAL_VAR_THRESH = 10
        self.MIN_OBJECT_PROPERTIES = 3 # small/lazy object.
        self.MAX_OBJECT_PROPERTIES = 20 # large object
        self.MAX_OBJECT_LOC = 750 # large object

    def parse(self, js_code: str):
        self.all_smells = []
        self.js_ast = esprima.parseScript(js_code)

    def count_global_variables(self, js_ast) -> int:
        global_vars = set()

        def traverse(node):
            if hasattr(node, 'type'):
                if node.type == 'VariableDeclaration':
                    for declaration in node.declarations:
                        if declaration.init is not None and declaration.init.type != 'FunctionExpression':
                            global_vars.add(declaration.id.name)
                elif node.type == 'FunctionDeclaration':
                    global_vars.add(node.id.name)
                elif node.type == 'FunctionExpression':
                    pass  # Skip function expressions as they don't create global variables

            for prop in dir(node):
                if prop.startswith('body'):
                    body = getattr(node, prop)
                    if isinstance(body, list):
                        for item in body:
                            traverse(item)
                    else:
                        traverse(body)
        traverse(js_ast)

        return len(global_vars)

    def check_closures_in_loop(self, node_or_list, inside_for_loop: bool=False):
        if isinstance(node_or_list, esprima.nodes.Node):
            node = node_or_list
            # print(node.type)
            if isinstance(node, esprima.nodes.ForStatement):
                inside_for_loop = True
            body = get_descendant(node)
            self.check_closures_in_loop(body, inside_for_loop)

        elif isinstance(node_or_list, list):
            for node in node_or_list:
                # print("type:", node.type)
                body = get_descendant(node)
                if node.type == "ForStatement":
                    inside_for_loop = True
                    # print("INSIDE FOR LOOP")
                    self.check_closures_in_loop(body, inside_for_loop)
                    inside_for_loop = False
                elif node.type == "FunctionDeclaration":
                    if inside_for_loop:
                        self.all_smells.append(("Closure Smell", f"Function {node.id.name} declared inside a for loop (closure in loop)"))
                elif node.type == "FunctionExpression":
                    if inside_for_loop:
                        self.all_smells.append(("Closure Smell", f"Function expression inside a for loop (closure in loop)"))
                self.check_closures_in_loop(body, inside_for_loop)

    def closure_smell_detection(self):
        ## CLOSURE SMELLS:
        # detect long scope chaining.
        scope_chains = extract_nested_functions(self.js_ast)
        # if nested functions found:
        try:
            scope_chains = scope_chains[0]
        except IndexError: pass # no nested functions found

        longest_scope_chain_length = highest_nesting_level(scope_chains)
        # longest_scope_chain = highest_nesting_chain(scope_chains)
        if longest_scope_chain_length >= self.LONGEST_SCOPING_CHAIN_THRESH:
            self.all_smells.append(("Closure Smell", f"Found nested function declaration (closure) of nesting = {longest_scope_chain_length} (≥ {self.LONGEST_SCOPING_CHAIN_THRESH} - max threshold for longest scoping chain)"))
        # closures in loops.
        self.check_closures_in_loop(self.js_ast)
        ## GLOBAL VARIABLE SMELLS:
        global_var_count = self.count_global_variables(self.js_ast)
        if global_var_count >= self.EXCESSIVE_GLOBAL_VAR_THRESH: 
            self.all_smells.append(("Excessive Global Variables", f"Found {global_var_count} global variables (> {self.EXCESSIVE_GLOBAL_VAR_THRESH} - max threshold for global variables). Consider creating a class for all of these global variables."))

    def detect_all(self, js_code):
        self.parse(js_code)
        self.closure_smell_detection()

        return self.all_smells

# main
if __name__ == "__main__":
    JS_CODES = {"longest scope chaining": """function foo(x) {
var tmp = 3;
    function bar(y) {
        ++tmp;
    }
    function sar(z) {
        document.write(x + y + z + tmp);
        function baz(z) {
            document.write(x + y + z + tmp);
            function sez(z) {
                document.write(x + y + z + tmp);
            }
        }
    }
}
foo(2); // writes 19 i.e., 2+10+3+4""",

"closures in loop (1/2)": """var addTheHandler = function (nodes) {
    for (i = 0; i < nodes.length; i++) {
        nodes[i].onclick = function (e) {
            document.write(i);
        };
    }
};
addTheHandler(document.getElementsByTagName("div"));""",

"closures in loop (2/2)": """var addTheHandler = function noida(nodes) {
    for (i = 0; i < nodes.length; i++) {
        nodes[i].onclick = function delta(e) {
            document.write(i);
        };
    }
};
addTheHandler(document.getElementsByTagName("div"));""",

"": """function outside() {
    var a = 10;
    function inside(a) {
        return a;
    }
    return inside;
}
result = outside()(20); \\ result: 20"""
}
    jsnose = CodeSmellDetector()
    for case, code in JS_CODES.items():
        print(f"\x1b[34;1mtesting case: {case}\x1b[0m")
        print(jsnose.detect_all(code))
        print()