import os
import json
import random
import itertools
import pandas as pd
from collections import defaultdict

# main
if __name__ == "__main__":
    # load phase1 annotated data and raw data.
    raw_data = json.load(open("human_study/phase1/raw_data.json"))
    raw_data = {rec['index']: rec for rec in raw_data}

    # load Marcus' annotations:
    marcus_claim_acc = {}
    marcus_claim_acc["js"] = pd.read_csv("human_study/phase1/js_claim_acc_annot_marcus.csv").to_dict("records")
    marcus_claim_acc["py"] = pd.read_csv("human_study/phase1/py_claim_acc_annot_marcus.csv").to_dict("records")
    marcus_claim_acc["java"] = pd.read_csv('human_study/phase1/java_claim_acc_annot_marcus.csv').to_dict("records")
    # load Atharva's annotations:
    atharva_claim_acc = {}
    atharva_claim_acc["js"] = pd.read_csv("human_study/phase1/js_claim_acc_annot_atharva.csv").to_dict("records")
    atharva_claim_acc["py"] = pd.read_csv("human_study/phase1/py_claim_acc_annot_atharva.csv").to_dict("records")
    atharva_claim_acc["java"] = pd.read_csv('human_study/phase1/java_claim_acc_annot_atharva.csv').to_dict("records")

    boundary_point = {"py":  248-2, 'java': 250-2, 'js': 247-2}
    claim_acc = {lang: [atharva_claim_acc[lang][i] if i >= boundary_point[lang] else marcus_claim_acc[lang][i] for i in range(len(atharva_claim_acc[lang]))] for lang in ["py", "java", "js"]}
    
    claim_acc_label = 'Correctness\n0 - incorrect\n1 - correct\n-1 - unverifiable\n-2 - incomplete'

    review_models = ["knn_pred", "lstm_pred", "codereviewer_pred", "magicoder_pred", "deepseekcoder_pred", "stable_code_pred", "llama3_pred", "codellama_13b_pred", "gpt3.5_pred", "msg"]
    langs = ["py", "java", "js"]
    os.makedirs('human_study/phase2', exist_ok=True)
    for ii, lang_data in enumerate([claim_acc["py"], claim_acc["java"], claim_acc["js"]]):
        missing_additional_claims = []
        phase2_rows = []
        lang = langs[ii]
        index2data = defaultdict(lambda: [])
        # convert to dict of lists where index is the key.
        index = None
        for row in lang_data:
            if str(row['index']).strip() != "nan":
                index = int(row['index'])
            index2data[index].append(row)
            
        for index, all_rows in index2data.items():
            raw_record = raw_data[index]
            issues = [r for r in raw_record["claims"] if r[0] == "issue"]
            claims, new_row = [], {}
            current_review_index = 0
            ID = all_rows[0]['id']
            index = all_rows[0]['index']
            diff = all_rows[0]["diff"]
            old_file = all_rows[0]['old_file']
            new_file = all_rows[0]['new_file']
            
            for row in all_rows:
                # ammend claims based on accuracy labels.
                if row[claim_acc_label] == 0:
                    if str(row["additional claims"]).strip() not in ["nan",'']:
                        claims.append(("claim", row["additional claims"]))
                elif row[claim_acc_label] == -1:
                    if str(row["additional claims"]).strip() not in ["nan",'']:
                        # capture the missing additional claims added next to the unverifiable claims:
                        missing_additional_claims.append({
                            "id": ID, "index": index, "diff": diff, 
                            "oldf": old_file, "newf": new_file,
                            'additional claims': row['additional claims'], 
                        })
                        claims.append(("claim", row["additional claims"]))
                elif row[claim_acc_label] == 1:
                    claims.append(("claim", row['claim']))
                    if str(row["additional claims"]).strip() not in ["nan",""]:
                        claims.append(("claim", row["additional claims"]))
                # elif row[claim_acc_label] in [-1, 2, -2]: pass
            claims_and_issues_with_types = claims+issues
            review_and_systems = [(raw_record[model], model) for model in review_models]
            random.shuffle(review_and_systems)
            ITER = itertools.zip_longest(
                claims_and_issues_with_types, 
                review_and_systems, fillvalue=("","")
            )
            instance_rows = []
            for claim_and_type, review_and_system in ITER:
                instance_rows.append({
                    "id": ID, "index": index, "diff": diff, "claim_no": "",
                    "claim": claim_and_type[1], "necessary?": "", "type": claim_and_type[0], 
                    "review": review_and_system[0], "system": review_and_system[1],   
                    "claims addressed": "", "Con (P)": "", "Comp (R)": "", "Rel (F)": "",
                    "old_file": old_file, "new_file": new_file,
                })
            instance_rows[0]["claim_no"] = 1
            for i in range(1, len(instance_rows)):
                instance_rows[i]["claim_no"] = i+1
                instance_rows[i]["diff"] = ""
                instance_rows[i]['id'] = "" 
                instance_rows[i]['index'] = "" 
                instance_rows[i]['old_file'] = "" 
                instance_rows[i]['new_file'] = "" 
            phase2_rows += instance_rows
                # # delete irrelevant rows.
                # try: del row["error type"]
                # except: pass
                # del row["additional claims"]
                # del row[claim_acc_label]                
                # # get current review:
                # try:
                #     current_review = raw_record[review_models[current_review_index]]
                # except IndexError:
                #     current_review = ""

                # new_row = {}
                # new_row.update(row)
                # new_row.update({"type": "claim", "review": current_review, "con (P)": "", "comp (R)": "", "rel (F)": ""})
                # phase2_rows.append(new_row)

                # # update current review index.
                # current_review_index += 1
            # if current_review_index < 3:
            #     # get current review:
            #     try:
            #         current_review = raw_record[review_models[current_review_index]]
            #     except IndexError:
            #         current_review = ""

            #     new_row["claim"] = ""
            #     new_row["review"] = current_review
            #     phase2_rows.append(new_row)

            #     # update current review index.
            #     current_review_index += 1
        pd.DataFrame(missing_additional_claims).to_csv(f"human_study/phase2/{lang}_missing_additional_claims.csv", index=False)
        pd.DataFrame(phase2_rows).to_csv(f"human_study/phase2/{lang}_review_qual_annot_fixed.csv", index=False)