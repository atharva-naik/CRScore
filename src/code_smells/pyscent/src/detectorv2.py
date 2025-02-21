import os
import ast
import sys
import json
from pathlib import Path
from tools.viz_generator import add_viz
from .Detector.class_coupling_detector import detect_class_cohesion
from .Detector.cyclomatic_complexity_detector import detect_cyclomatic_complexity
from .Detector.long_lambda_detector import detect_long_lambda
from .Detector.long_list_comp_detector import detect_long_list_comp
from .Detector.pylint_output_detector import detect_pylint_output
from .Detector.shotgun_surgery_detector import detect_shotgun_surgery
from .Detector.useless_exception_detector import detect_useless_exception

def get_stats(directory):
    total_num_method = 0
    total_num_class = 0
    total_num_lambda = 0
    total_num_try_catch = 0
    total_num_list_comp = 0
    total_num_code_blocks = 0
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            with open(directory + "/" + filename, encoding='UTF8') as f:
                data = f.read()
                tree = ast.parse(data)
                for node in ast.walk(tree):
                    if isinstance(node,ast.FunctionDef):
                        total_num_method+=1
                    if isinstance(node,ast.ClassDef):
                        total_num_class+=1
                    if isinstance(node,ast.Lambda):
                        total_num_lambda+=1
                    if isinstance(node,ast.Try):
                        total_num_try_catch += 1
                    if isinstance(node,ast.ListComp):
                        total_num_list_comp+=1
    total_num_code_blocks = total_num_method + total_num_class
    return {"methods":total_num_method,"classes":total_num_class,"lambdas":total_num_lambda,\
            "try":total_num_try_catch,"listcomps":total_num_list_comp,\
            "codeblocks":total_num_code_blocks}

def get_function_arguments(node):
    if isinstance(node, ast.FunctionDef):
        arguments = []
        for arg in node.args.args:
            arguments.append(arg.arg)
        return arguments
    return []

def get_info(d):
    if 'line length' in d[1]:
        metric = d[1]['line length']
    else: metric = d[1]['metric']

    return int(d[1]['lineno']), metric, d[1]['filename']

def get_ast_node_at_line(filename, line_number):
    with open(filename, "r") as file:
        code = file.read()
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.AST):
                if hasattr(node, "lineno") and node.lineno == line_number:
                    return node
    return None

def get_class_attributes(node):
    if isinstance(node, ast.ClassDef):
        attributes = []
        for class_body_node in node.body:
            if isinstance(class_body_node, ast.Assign):
                for target in class_body_node.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        return attributes
    return []

def get_class_methods(node):
    if isinstance(node, ast.ClassDef):
        attributes = []
        for class_body_node in node.body:
            if isinstance(class_body_node, ast.FunctionDef):
                attributes.append(class_body_node.name)
        return attributes
    return []

def get_init_attributes(node):
    if isinstance(node, ast.ClassDef):
        init_attributes = []
        for class_body_node in node.body:
            if isinstance(class_body_node, ast.FunctionDef) and class_body_node.name == "__init__":
                for stmt in class_body_node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute):
                                init_attributes.append(target.attr)
        return init_attributes
    return []

def detect_main(directory, logdir):
    # Get stats for files in directory
    stats_dict = get_stats(directory)
    # print("directory:", directory)
    dirname = Path(directory).name
    log_json = {
        "project_path": dirname,
        "stats": stats_dict,
        "smells": []
    }
    COHESION_LIMIT = 30
    long_method, long_params, long_branches, many_attrbs, many_methods = detect_pylint_output(directory)
    useless_try = detect_useless_exception(directory)
    num_shotgun, most_external = detect_shotgun_surgery(directory)
    cohesion_output = detect_class_cohesion(directory, COHESION_LIMIT)
    cc_output = detect_cyclomatic_complexity(directory)
    long_lambda_output = detect_long_lambda(directory, 60)
    long_list_comp_output = detect_long_list_comp(directory, 72)
    # print(long_method)
    if long_method[0] > 0:
        lineno, metric, fname = get_info(long_method)
        method_name = ""
        node = get_ast_node_at_line(os.path.join(directory, fname), lineno)
        if node is not None and isinstance(node, ast.FunctionDef): method_name = node.name
        log_json["smells"].append([f"Long method smell detected with {long_method[0]} long methods and longest method `{method_name}` at line {lineno} containing {metric} lines", lineno])
    if long_params[0] > 0:
        lineno, metric, fname = get_info(long_params)
        method_name = ""
        method_arguments = ""
        node = get_ast_node_at_line(os.path.join(directory, fname), lineno)
        if node is not None and isinstance(node, ast.FunctionDef): 
            method_name = node.name
            arguments = [f'`{arg}`' for arg in get_function_arguments(node)]
            method_arguments = " ("+", ".join(arguments)+")"
        log_json["smells"].append([f"Long parameter list smell detected with {long_params[0]} methods with long parameter lists and method `{method_name}` at line {lineno} having {metric} parameters"+method_arguments, lineno])
    if long_branches[0] > 0:
        lineno, metric, fname = get_info(long_branches)
        node = get_ast_node_at_line(os.path.join(directory, fname), lineno)
        method_name = ''
        if node is not None and isinstance(node, ast.FunctionDef): 
            method_name = node.name
        log_json["smells"].append([f"Long branch smell detected with {long_branches[0]} methods with long branching and method with most branches `{method_name}` at line {lineno} having {metric} branches", lineno])
    if many_attrbs[0] > 0:
        lineno, metric, fname = get_info(many_attrbs)
        node = get_ast_node_at_line(os.path.join(directory, fname), lineno)
        class_name = ''
        class_attributes = ""
        if node is not None and isinstance(node, ast.ClassDef):
            class_name = node.name
            class_attributes = " ("+", ".join([f"`{attr}`" for attr in get_init_attributes(node)])+")"
        log_json["smells"].append([f"Many attribute smell detected with {many_attrbs[0]} classes having many attributes and the class with most attributes, `{class_name}` having {metric} attributes"+class_attributes, lineno])
    if many_methods[0] > 0:
        lineno, metric, fname = get_info(many_methods)
        node = get_ast_node_at_line(os.path.join(directory, fname), lineno)
        class_name = ''
        class_methods = ""
        if node is not None and isinstance(node, ast.ClassDef):
            class_name = node.name
            class_methods = " ("+", ".join([f"`{method}`" for method in get_class_methods(node) if not method.startswith("_")])+")"
        log_json["smells"].append([f"Many methods smell detected with {many_methods[0]} classes having many methods and the class with most methods, `{class_name}` having {metric} methods"+class_methods, lineno])
    for fname, line_and_msgs in useless_try[0]:
        for lineno, msg in line_and_msgs:
            node = get_ast_node_at_line(os.path.join(directory, fname), lineno)
            # if node is not None and isinstance(node, ast.ExceptHandler): pass
            # print(f"\x1b[32;1mUseless try except smell with \"{msg.lower()}\" at line {lineno}\x1b[0m")
            log_json["smells"].append([f"Useless try except smell with \"{msg.lower()}\" at line {lineno}", lineno])
    if num_shotgun > 0:
        fname = most_external[0]
        class_name = most_external[1]
        log_json["smells"].append([f"Shotgun smell detected with {num_shotgun} classes with too many external functions and class with most external function calls, `{class_name}` having {most_external[2]} external function calls", -1])
        # print(f"Shotgun smell detected with {num_shotgun} classes with too many external functions and class with most external function calls, `{class_name}` having {most_external[2]} external function calls")
    if cohesion_output > 0:
        log_json["smells"].append([f"Class cohesion smell detected with {cohesion_output} out of {stats_dict['classes']} classes having cohesion < {COHESION_LIMIT}%", -1])
        # print(f"\x1b[31;1mClass cohesion smell detected with {cohesion_output} out of {stats_dict['classes']} classes having cohesion < {COHESION_LIMIT}%\x1b[0m")
    if cc_output > 0:
        log_json["smells"].append([f"Code complexity smell detected with {cc_output} of {stats_dict['codeblocks']} code blocks having cyclomatic complexity of rank C or worse (rank C is moderate to slighly complex with cyclomatic complexity between 11 and 20).",-1])
        # print(f"\x1b[33;1mCode complexity smell detected with {cc_output} of {stats_dict['codeblocks']} code blocks having cyclomatic complexity of rank C or worse (rank C is moderate to slighly complex with cyclomatic complexity between 11 and 20).\x1b[0m")
    if long_lambda_output[0] > 0:
        lineno, metric, fname = get_info(long_lambda_output)
        node = get_ast_node_at_line(os.path.join(directory, fname), lineno)
        lamb_func = ''
        if node is not None: lamb_func = ast.unparse(node)
        log_json["smells"].append([f"Long lambda smell detected with {long_lambda_output[0]} long lambda functions and the longest lambda function `{lamb_func}` at line {lineno} having line length {metric}", lineno])
        # print(f"\x1b[33;1mLong lambda smell detected with {long_lambda_output[0]} long lambda functions and the longest lambda function `{lamb_func}` at line {lineno} having line length {metric}\x1b[0m")
    if long_list_comp_output[0] > 0:
        # print(long_list_comp_output)
        lineno, metric, fname = get_info(long_list_comp_output)
        node = get_ast_node_at_line(os.path.join(directory, fname), lineno)
        list_comp = ''
        if node is not None: list_comp = ast.unparse(node)
        log_json["smells"].append([f"Long list comprehension smell detected with {long_list_comp_output[0]} long list comprehensions and the longest list comprehension `{list_comp}` at line {lineno} having line length {metric}", lineno])
    # save path for final JSON log:
    savepath = os.path.join(logdir, f"{dirname}.json")
    with open(savepath, "w") as f:
        json.dump(log_json, f, indent=4)
    
    return log_json

