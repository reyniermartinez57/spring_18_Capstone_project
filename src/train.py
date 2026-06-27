import os
import yaml
import mlflow
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1

def train_and_log():
    # 1. Load config and datasets
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv")).values.ravel()
    
    # 2. Setup MLflow Experiment
    mlflow.set_experiment("Heart_Disease_Classification_Capstone")
    
    # 3. Define the 5 distinct runs required by the rubric
    # We will use variations of our 3 core configurations from the YAML file
    run_configurations = [
        {"name": "Logistic_Regression_Base", "model_type": "lr", "params": config["models"]["logistic_regression"]},
        {"name": "Logistic_Regression_High_Reg", "model_type": "lr", "params": {**config["models"]["logistic_regression"], "C": 0.01}},
        {"name": "Random_Forest_Base", "model_type": "rf", "params": config["models"]["random_forest"]},
        {"name": "Random_Forest_Deeper", "model_type": "rf", "params": {**config["models"]["random_forest"], "max_depth": 12, "n_estimators": 200}},
        {"name": "Gradient_Boosting_Base", "model_type": "gb", "params": config["models"]["gradient_boosting"]}
    ]
    
    print("Starting MLflow Experiment tracking across 5 distinct runs...")
    
    for run in run_configurations:
        with mlflow.start_run(run_name=run["name"]):
            print(f" Running: {run['name']}...")
            
            # Initialize model based on configuration type
            if run["model_type"] == "lr":
                model = LogisticRegression(**run["params"])
            elif run["model_type"] == "rf":
                model = RandomForestClassifier(**run["params"])
            elif run["model_type"] == "gb":
                model = GradientBoostingClassifier(**run["params"])
                
            # Train model
            model.fit(X_train, y_train)
            
            # Predict & Evaluate
            preds = model.predict(X_val)
            precision, recall, f1 = evaluate_model(y_val, preds)
            
            # Log Parameters to MLflow
            mlflow.log_params(run["params"])
            
            # Log Metrics to MLflow
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log Model Artifacts cleanly
            mlflow.sklearn.log_model(model, artifact_path="model")
            
    print("All 5 runs successfully tracked in MLflow!")

if __name__ == "__main__":
    train_and_log()