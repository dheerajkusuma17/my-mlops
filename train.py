import pandas as pd
import joblib
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import recall_score, precision_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


from clean import clean_data, remove_outliers

with open("config.yml") as file:
    config = yaml.safe_load(file)

# Load and clean both datasets
train = remove_outliers(clean_data(pd.read_csv(config["data"]["train_path"])))
test = clean_data(pd.read_csv(config["data"]["test_path"]))

# Data Partition (Seperating Clues X from Results y)
X_train = train.drop(columns=["Result"])
y_train = train["Result"]
X_test = test.drop(columns=["Result"])
y_test = test["Result"]

# Translator: Words to Numbers:
preprocessor = ColumnTransformer(
    transformers = [
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["Gender", "PastAccident"]),
    ],
    remainder = "passthrough",
)

# Translator(Preprocessor + SMOTE) + model:
steps = [("preprocessor", preprocessor)]

if config["model"]["use_smote"]:
    steps.append(("smote", SMOTE(random_state=42)))

steps.append(("model", DecisionTreeClassifier(
    criterion=config["model"]["criterion"],
    max_depth=config["model"]["max_depth"],
    random_state=42,
)))

pipeline = Pipeline(steps)

# Training the model:
mlflow.set_experiment("insurance-model")

with mlflow.start_run():

    # Train
    pipeline.fit(X_train, y_train)
    train_accuracy = pipeline.score(X_train, y_train)

    # Test, using out own cut-off line

    probabilities = pipeline.predict_proba(X_test)[:,1]
    predictions = (probabilities >= config["model"]["threshold"]).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)

    print("Train accuracy:", round(train_accuracy, 4))
    print("Accuracy: ", round(accuracy, 4))
    print("ROC AUC: ", round(roc_auc, 4))
    print("Recall: ", round(recall, 4))
    print("Precision: ", round(precision, 4))

    # Write everything into the MLflow notebook
    mlflow.log_params(config["model"])
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.sklearn.log_model(pipeline, "model")

    joblib.dump(pipeline, config["model"]["store_path"])
    print("Model saved and logged to MLflow")
