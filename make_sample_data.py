"""
Generates a SYNTHETIC stand-in with the same columns as Kaggle's insurance.csv.
Only for testing the pipeline when you don't have the real file yet.
For your submission, use the REAL dataset: data/insurance.csv from
https://www.kaggle.com/datasets/mirichoi0218/insurance
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 1338
age = rng.integers(18, 65, n)
sex = rng.choice(["male", "female"], n)
bmi = np.clip(rng.normal(30.7, 6.1, n), 16, 53).round(2)
children = rng.choice([0, 1, 2, 3, 4, 5], n, p=[.43, .24, .18, .12, .02, .01])
smoker = rng.choice(["yes", "no"], n, p=[.2, .8])
region = rng.choice(["southwest", "southeast", "northwest", "northeast"], n)
s = (smoker == "yes")
charges = (2000 + 260 * (age - 18) + 80 * bmi + 480 * children
           + s * (14000 + 1100 * np.maximum(bmi - 30, 0) * 1.5 + 5000 * (bmi > 30))
           + (region == "southeast") * 600)
charges = charges * rng.lognormal(0, 0.28, n)
pd.DataFrame(dict(age=age, sex=sex, bmi=bmi, children=children, smoker=smoker,
                  region=region, charges=charges.round(2))).to_csv("data/insurance.csv", index=False)
print("synthetic data/insurance.csv written")
