import mlflow
import pandas as pd

def find_and_register_best_model():
    print("Querying MLflow tracking server for experiment runs...")
    
    # 1. Fetch the experiment ID
    experiment = mlflow.get_experiment_by_name("Heart_Disease_Classification_Capstone")
    if not experiment:
        raise ValueError("Experiment not found! Make sure you ran train.py first.")
    
    # 2. Programmatically search all runs in this experiment using mlflow.search_runs()
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs_df.empty:
        raise ValueError("No runs found in this experiment.")
        
    print(f"Found {len(runs_df)} logged runs. Sorting by top F1-Score...")
    
    # 3. Sort runs programmatically to find the best configuration
    # MLflow names logged metrics as metrics.metric_name in the resulting DataFrame
    best_run = runs_df.sort_values(by="metrics.f1_score", ascending=False).iloc[0]
    
    print("\n" + "="*50)
    print("🏆 BEST RUN IDENTIFIED PROGRAMMATICALLY 🏆")
    print(f"Run Name:  {best_run.get('tags.mlflow.runName', 'Unnamed Run')}")
    print(f"Run ID:    {best_run['run_id']}")
    print(f"F1-Score:  {best_run['metrics.f1_score']:.4f}")
    print(f"Precision: {best_run['metrics.precision']:.4f}")
    print(f"Recall:    {best_run['metrics.recall']:.4f}")
    print("="*50 + "\n")
    
    # 4. Save the best run ID to a local text file so your app.py interface knows what to load
    with open("configs/best_model_meta.txt", "w") as f:
        f.write(best_run['run_id'])
    print("Best model run metadata safely exported to configs/best_model_meta.txt!")

if __name__ == "__main__":
    find_and_register_best_model()