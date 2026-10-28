import pandas as pd
from clean import clean_data, remove_outliers

def test_clean_removes_id_column():
    raw = pd.read_csv("data/train.csv")
    cleaned = clean_data(raw)
    assert "id" not in cleaned.columns

def test_clean_no_missing_values():
    raw = pd.read_csv("data/train.csv")
    cleaned = clean_data(raw)
    assert cleaned.isnull().sum().sum() == 0

def test_cleaned_premium_is_numeric():
    raw = pd.read_csv("data/train.csv")
    cleaned = clean_data(raw)
    assert cleaned["AnnualPremium"].dtype == float

def test_outlier_removal_reduces_rows():
    raw = pd.read_csv("data/train.csv")
    cleaned = clean_data(raw)
    trimmed = remove_outliers(cleaned)
    assert len(trimmed) < len(cleaned)