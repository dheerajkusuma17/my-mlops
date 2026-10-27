import joblib
import pandas as pd
from clean import clean_data
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset

model = joblib.load("models/model.pkl")

# Reference = what the model trained on
reference = clean_data(pd.read_csv("data/train.csv"))
reference["prediction"] = model.predict(reference.drop(columns=["Result"]))

# Production = new data arriving now
production = clean_data(pd.read_csv("data/production.csv"))
production["prediction"] = model.predict(reference.drop(columns=["Result"]))

# Define column types for Evidently
definition = DataDefinition(
    numerical_columns=["Age", "AnnualPremium", "HasDrivingLicense", "RegionID", "Switch"],
    categorical_columns=["Gender", "PastAccident"],
)

# Wrap dataframes as Evidently Datasets
ref_dataset = Dataset.from_pandas(reference, data_definition=definition)
prod_dataset = Dataset.from_pandas(production, data_definition=definition)

# Build the report
report = Report(metrics=[
    DataDriftPreset(),
    DataSummaryPreset(),
])

snapshot = report.run(reference_data=ref_dataset, current_data=prod_dataset)
snapshot.save_html("monitoring_report.html")
print("Report saved to monitoring_report.html")