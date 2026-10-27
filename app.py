from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import yaml

with open("config.yml") as f:
    config = yaml.safe_load(f)

app = FastAPI()
model = joblib.load(config["model"]["store_path"])
threshold = config["model"]["threshold"]

class Customer(BaseModel):
    Gender: str
    Age: float
    HasDrivingLicense: int
    RegionID: float
    Switch: float
    PastAccident: str
    AnnualPremium: float

@app.get("/")
def health_check():
    return {"status": "OK", "model": "insurance_model", "threshold": threshold}

@app.post("/predict")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    probability = model.predict_proba(row)[:, 1][0]
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "threshold": threshold,
    }