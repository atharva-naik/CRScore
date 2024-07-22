import os
import json
import numpy as np
import pandas as pd
import krippendorff as kd

def IOU(a, b):
    a = set([r.strip() for r in a.split(",") if r != "-"])
    b = set([r.strip() for r in b.split(",") if r != "-"])
    U = len(a.union(b))
    I = len(a.intersection(b))
    # if union is zero it means both sets are equal and empty.
    if U == 0: return 1

    return I/U

# main
if __name__ == "__main__":
    marcus = pd.read_csv("human_study/phase2/phase2_agreement_annot_Marcus.csv").to_dict("records")
    atharva = pd.read_csv("human_study/phase2/phase2_agreement_annot_Atharva.csv").to_dict("records")
    con = [[],[]]
    comp = [[],[]]
    rel = [[],[]]
    IOUs = []
    ck = "claims addressed" # claims key.
    for marcus_row, atharva_row in zip(marcus, atharva):
        if str(marcus_row["Con (P)"]) == "nan": continue
        con[0].append(marcus_row["Con (P)"])
        con[1].append(atharva_row["Con (P)"])
        comp[0].append(marcus_row["Comp (R)"])
        comp[1].append(atharva_row["Comp (R)"])
        rel[0].append(marcus_row["Rel (F)"])
        rel[1].append(atharva_row["Rel (F)"])
        IOUs.append(IOU(marcus_row[ck], atharva_row[ck]))
    print("number of coded items:", len(con[0]), len(con[1]))
    print("Con (P) IAA:", round(kd.alpha(con, level_of_measurement='ordinal'), 4))
    print("Comp (R) IAA:", round(kd.alpha(comp, level_of_measurement='ordinal'), 4))
    print("Rel (F) IAA:", round(kd.alpha(rel, level_of_measurement='ordinal'), 4))
    print("Mean IOU:", round(np.mean(IOUs), 4))