# metrics computation.

def taxonomy_conf_mat(preds, labels, save_path: str,
                      pred_key: str="completions", 
                      label_key: str="code quality"):
    class_mapping = {"documentation": 0, "structure": 1, "presentation": 2, "algorithm": 3, "no issues": 4}
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    y_test, y_pred = [], []
    for pred, label in zip(preds, labels):
        pred = pred[pred_key].lower()
        label = label[label_key].lower()
        code_quality_guidelines = ["documentation", "structure", "presentation", "algorithm"]
        if label not in code_quality_guidelines: label = "no issues"
        if "does not violate any" in pred: pred = "no issues"
        if pred.startswith("none."): pred = "no issues"
        for guideline in code_quality_guidelines:
            if guideline in pred: 
                pred = guideline
                break
        pred = class_mapping[pred]
        label = class_mapping[label]
        y_test.append(label)
        y_pred.append(pred)
    # print(set(y_pred))
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)
    hmap = sns.heatmap(conf_mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
    figure = hmap.get_figure()
    figure.savefig(save_path)

def taxonomy_code_quality_accuracy(preds, labels, pred_key: str="completions", label_key: str="code quality"):
    """accuracy wrt to labels."""
    tot = len(preds)
    ctr = 0
    for pred, label in zip(preds, labels):
        pred = pred[pred_key].lower()
        label = label[label_key].lower()
        code_quality_guidelines = ["documentation", "structure", "presentation", "algorithm"]
        if label not in code_quality_guidelines: label = "no issues"
        if "does not violate any" in pred: pred = "no issues"
        if pred.startswith("none."): pred = "no issues"
        for guideline in code_quality_guidelines:
            if guideline in pred: 
                pred = guideline
                break
        if pred == label: ctr += 1

    return 100*ctr/tot

def taxonomy_review_accuracy(preds, labels, pred_key: str="completions", label_key: str="code quality"):
    """accuracy wrt to labels."""
    import fuzzywuzzy
    import numpy as np
    from fuzzywuzzy import fuzz
    tot = len(preds)
    ctr = 0
    for pred, label in zip(preds, labels):
        pred = pred[pred_key].lower()
        label = label[label_key].lower()
        allowed_labels = ["documentation", "structure", "presentation", "algorithm", "N5:Necessity", "N4:Code Context", "N2:Correct Understanding ", "N1:Alternative Solution", "N6:Specialize Expertise", "N3:Rationale"]
        pred = allowed_labels[np.argmax([fuzz.token_set_ratio(pred.lower(), allowed_label.replace(":", " ").lower()) for allowed_label in allowed_labels])]
        # if label not in code_quality_guidelines: label = "no issues"
        # if "does not violate any" in pred: pred = "no issues"
        # if pred.startswith("none."): pred = "no issues"
        # for guideline in code_quality_guidelines:
        #     if guideline in pred: 
        #         pred = guideline
        #         break
        if pred == label: ctr += 1

    return 100*ctr/tot