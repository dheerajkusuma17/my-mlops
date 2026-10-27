import pandas as pd

train = pd.read_csv("data/train.csv")

print(train.shape)
print(train.head())

print("\n---Column types---")
train.info()

print("\n--- Missing values per column ---")
print(train.isnull().sum())

print("\n--- How many 0s and 1s in the Result ---")
print(train["Result"].value_counts())