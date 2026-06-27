import os
import pytest
import pandas as pd
import mlflow
from sklearn.metrics import f1_score

@pytest.fixture
def production_model_and_data():
    """Loads the programmatically selected best model artifact and the validation dataset."""
    # 1. Read the saved best run ID
    meta_path = "configs/best_model_meta.txt"
    if not os.path.exists(meta_path):
        pytest.fail("Missing configs/best_model_meta.txt. Run src/evaluate.py first!")
        
    with open(meta_path, "r") as f:
        run_id = f.read().strip()
    
    # 2. Load the model directly from MLflow artifacts
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # 3. Load validation data splits
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()
    
    return model, X_val, y_val

def test_prediction_shape_and_type(production_model_and_data):
    """Test 1: Verifies that the model outputs predictions of correct type and matching shape."""
    model, X_val, y_val = production_model_and_data
    
    preds = model.predict(X_val)
    
    # Assert type and dimensions
    assert isinstance(preds, type(y_val)), "Predictions should be a numpy array matching target type"
    assert preds.shape == y_val.shape, f"Shape mismatch: Expected {y_val.shape}, got {preds.shape}"

def test_model_performance_threshold(production_model_and_data):
    """Test 2: Verifies that the production model meets the minimum grading performance threshold."""
    model, X_val, y_val = production_model_and_data
    
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, zero_division=0)
    
    # Assert the model performs well (e.g., F1 score above baseline 0.75)
    threshold = 0.75
    assert f1 >= threshold, f"Model performance dropped! Expected F1 >= {threshold}, got {f1:.4f}"