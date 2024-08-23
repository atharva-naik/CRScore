# compute the combined claim accuracy across annotatations.
import os
import json
import pandas as pd
from collections import defaultdict

boundary_point = {"py": 248-2, 'java': 250-2, 'js': 247-2}
for lang in ['py', 'java', 'js']:
    atharva = pd.read_csv(f"human_study/phase1/{lang}_claim_acc_annot_atharva.csv").to_dict("records")
    marcus = pd.read_csv(f"human_study/phase1/{lang}_claim_acc_annot_marcus.csv").to_dict("records")
    # print(atharva[0].keys())
    claim_acc_key = 'Correctness\n0 - incorrect\n1 - correct\n-1 - unverifiable\n-2 - incomplete'
    # claim_accs = []
    claim_dist = defaultdict(lambda: 0)
    added_claims = 0
    removed_claims = 0
    num_code_changes = set()
    for i in range(len(atharva)):
        if str(atharva[i]["index"]) != "nan":
            num_code_changes.add(int(atharva[i]["index"]))
        if i >= boundary_point[lang]: 
            if str(atharva[i]["additional claims"]) != "nan":
                added_claims += 1
            acc = atharva[i][claim_acc_key]
            if str(acc) == "nan": 
                # print("row", i+2)
                removed_claims += 1
                continue
            # claim_accs.append(acc)
            claim_dist[atharva[i][claim_acc_key]] += 1
        else: 
            if str(marcus[i]["additional claims"]) != "nan":
                added_claims += 1
            acc = marcus[i][claim_acc_key]
            if str(acc) == "nan": 
                # print("row", i+2)
                removed_claims += 1
                continue
            # claim_accs.append(marcus[i][claim_acc_key])
            claim_dist[acc] += 1

    num_code_changes = len(num_code_changes)
    claim_dist = dict(claim_dist)
    total = claim_dist[0]+claim_dist[1]+claim_dist[-1]
    print("Accuracy", round(100*claim_dist[1]/total, 2))
    print("Error Rate", round(100*claim_dist[0]/total, 2))
    print("Unverifiable Rate", round(100*claim_dist[-1]/total, 2))
    print("Missing Rate", round(100*added_claims/total, 2))
    print("Total Claims", total)
    print("Incomplete Claims", claim_dist[-2])
    print("Discarded Claims due to Code Change", removed_claims)
    print("Evaluated Claims", total+claim_dist[-2]+removed_claims)
    print("Code Changes", num_code_changes)
    print()