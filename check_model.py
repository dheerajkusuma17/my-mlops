import pandas as pd
import joblib
from clean import clean_data

model = joblib.load("models/model.pkl")
test = clean_data(pd.read_csv("data/test.csv"))

X_test = test.drop(columns=["Result"])
probabilities = model.predict_proba(X_test)[:, 1]

print("Total customers:", len(probabilities))
print("Probability > 0:", (probabilities > 0).sum())
print("Probability > 0.2:", (probabilities > 0.2).sum())
print("\nSample probabilities (first 20):")
print(probabilities[:20].round(4))
