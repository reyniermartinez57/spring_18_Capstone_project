import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(input_path="data/raw_data.csv", output_dir="data/processed/"):
    # 1. Create processed directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load raw data
    print(f"Reading raw data from {input_path}...")
    raw_df = pd.read_csv(input_path)
    df = raw_df.copy() # Ensure we do not mutate the original dataframe reference
    
    # 3. Handle column renaming safely
    if 'num' in df.columns:
        df.rename(columns={'num': 'target'}, inplace=True)
    elif 'target' not in df.columns:
        raise KeyError(f"Could not find a target column ('num' or 'target'). Available columns: {list(df.columns)}")
        
    # 4. Binary target conversion
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # 5. Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Define features explicitly based on your heart disease dataset structure
    # NOTE: Adjust these lists to match your exact column names!
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    # 6. Split data safely
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. Impute missing values on training distributions to prevent leakage
    for col in numerical_features:
        median_value = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_value)
        X_val[col] = X_val[col].fillna(median_value)
        
    for col in categorical_features:
        mode_value = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_value)
        X_val[col] = X_val[col].fillna(mode_value)
    
    # 8. Build ColumnTransformer for explicit scaling and encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Fit on training data and transform both sets
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    
    # Reconstruct dynamic column names for the output dataframes
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(categorical_features))
    final_feature_names = numerical_features + encoded_cat_cols
    
    # 9. Save processed datasets back to disk
    pd.DataFrame(X_train_transformed, columns=final_feature_names).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_val_transformed, columns=final_feature_names).to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    
    print("Preprocessing complete! Saved split datasets to", output_dir)

if __name__ == "__main__":
    preprocess_data()