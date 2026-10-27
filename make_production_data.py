import pandas as pd
import numpy as np

test = pd.read_csv("data/test.csv")
production = test.copy()

np.random.seed(99)

# Shift ages upward by 5 years on average
production["Age"] = production["Age"] + np.random.normal(5, 2, len(production))

# Increase premiums by 15%
production["AnnualPremium"] = production["AnnualPremium"].str.replace("£", "").str.replace(",", "").astype(float)
production["AnnualPremium"] = production["AnnualPremium"] * 1.15
production["AnnualPremium"] = "£" + production["AnnualPremium"].round(2).astype(str)

# Save
production.to_csv("data/production.csv", index=False)
print("Production data saved:", production.shape)