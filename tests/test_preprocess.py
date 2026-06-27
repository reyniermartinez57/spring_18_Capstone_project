import pytest
import pandas as pd
import numpy as np
import os
import shutil
from src.preprocess import preprocess_data

@pytest.fixture
def sample_data_path():
    """Generates a micro mock version of your data containing enough samples for stratification."""
    data_dir = "tests/mock_data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "test_raw.csv")
    
    # Perfectly aligned indentation levels
    mock_df = pd.DataFrame({
        'age': [63, 67, 37, 41, 56, 62, 53, 57],
        'sex': [1, 1, 1, 0, 1, 0, 1, 1],
        'cp': [1, 4, 3, 2, 2, 4, 4, 4],
        'trestbps': [145, 160, 130, 130, np.nan, 140, 140, 140], 
        'chol': [233, 286, 250, 204, 236, 268, 203, 192],
        'fbs': [1, 0, 0, 0, 0, 0, 1, 0],
        'restecg': [2, 2, 0, 2, 0, 2, 2, 0],
        'thalach': [150, 108, 187, 172, 178, 160, 155, 148],
        'exang': [0, 1, 0, 0, 0, 0, 1, 0],
        'oldpeak': [2.3, 1.5, 3.5, 1.4, 0.8, 3.6, 3.1, 0.4],
        'slope': [3, 2, 3, 1, 1, 3, 3, 2],
        'ca': [0.0, 3.0, 0.0, 0.0, np.nan, 2.0, 0.0, 0.0],      
        'thal': [6.0, 3.0, 3.0, 3.0, 3.0, 3.0, 7.0, 6.0],
        'num': [0, 2, 0, 0, 0, 3, 1, 0]
    })
    
    mock_df.to_csv(file_path, index=False)
    yield file_path
    
    # Cleanup after execution completes
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree("tests/mock_processed", ignore_errors=True)

def test_preprocessing_pipeline(sample_data_path):
    output_dir = "tests/mock_processed"
    
    # Cache original dataset to assert immutability rules later
    original_df = pd.read_csv(sample_data_path)
    
    # Run pipeline
    preprocess_data(input_path=sample_data_path, output_dir=output_dir)
    
    # Load outputs
    X_train = pd.read_csv(os.path.join(output_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(output_dir, "X_val.csv"))
    
    # Test 1: Handle Missing Values correctly (No NaNs remain)
    assert X_train.isnull().sum().sum() == 0, "Imputation failed: NaNs found in X_train"
    assert X_val.isnull().sum().sum() == 0, "Imputation failed: NaNs found in X_val"
    
    # Test 2: Categorical variables are encoded (Check for generated dummy keys)
    assert any("cp_" in col for col in X_train.columns), "Encoding failed: One-hot feature tags missing"
    
    # Test 3: Numerical scaling features normalized properly (Mean near 0)
    assert abs(X_train['age'].mean()) < 1.0, "Scaling failed: Numerical values not standardized properly"
    
    # Test 4: Verify original dataframe immutability
    post_run_df = pd.read_csv(sample_data_path)
    pd.testing.assert_frame_equal(original_df, post_run_df, obj="Original Dataframe modified during preprocessing pipeline execution!")