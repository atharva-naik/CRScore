# compute point-biserial correlation (for correlating )

import json
import numpy as np
from scipy.stats import pointbiserialr
import pandas as pd
import pingouin as pg

k = 100
categorical_variable = np.array(pd.read_csv("./cr_manual_rel_annot_likert_scale.csv")['rel_score'][:k])
continuous_variable = json.load(open("./ckpts/crr_rcr_ccr_0.005/test_preds.json"))[:k]

assert len(categorical_variable) == len(continuous_variable)

# Point-Biserial Correlation
pb_corr, pb_p_value = pointbiserialr((categorical_variable>2).astype(int), continuous_variable)
print(f"Point-Biserial Correlation: {pb_corr:.3f}, p-value: {pb_p_value}")

data = pd.DataFrame({'Categorical': categorical_variable, 'Continuous': continuous_variable})

# Eta-squared using ANOVA
eta_squared = pg.anova(data=data, dv='Continuous', between='Categorical')['np2'][0]
print(f"Eta-squared: {eta_squared:.3f}")