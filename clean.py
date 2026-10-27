import pandas as pd

def clean_data(data):
    """Fix types, fill blanks. Safe for any data, anytime."""
    data = data.copy()

    data = data.drop(columns=["id", "SalesChannelID", "VehicleAge", "DaysSinceCreated"])

    data["AnnualPremium"] = (
        data["AnnualPremium"]
        .str.replace("£", "")
        .str.replace(",", "")
        .astype(float)
    )

    data["Gender"] = data["Gender"].fillna(data["Gender"].mode()[0])
    data["RegionID"] = data["RegionID"].fillna(data["RegionID"].mode()[0])
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["HasDrivingLicense"] = data["HasDrivingLicense"].fillna(1)
    data["Switch"] = data["Switch"].fillna(-1)
    data["PastAccident"] = data["PastAccident"].fillna("Unknown")

    return data

def remove_outliers(data):
    """Remove extreme premiums. Use only on training data."""
    q1 = data["AnnualPremium"].quantile(0.25)
    q3 = data["AnnualPremium"].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr
    return data[data["AnnualPremium"] <= upper_limit]

if __name__== "__main__":
    train = pd.read_csv("data/train.csv")
    cleaned = clean_data(train)
    trimmed = remove_outliers(cleaned)
