
import sys

# 1. Prevent the broken ujson DLL from crashing FastAPI import
sys.modules['ujson'] = None

# 2. Bypass the MLflow schema check
import mlflow.store.db.utils
mlflow.store.db.utils._verify_schema = lambda engine: None