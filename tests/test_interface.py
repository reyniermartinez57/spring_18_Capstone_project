import sys
from unittest.mock import patch

sys.modules['ujson'] = None
import mlflow.store.db.utils
mlflow.store.db.utils._verify_schema = lambda engine: None

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

@patch('src.app.query_llm')
@patch('src.app.model')
def test_input_parsing_success(mock_model, mock_query_llm):
    """Test 1: Check natural language parsing and successful inference output."""
    mock_query_llm.side_effect = [
        '{"features": ' + str([0.0] * 28) + '}', # Parser response
        "The model indicates a low risk profile." # Explainer response
    ]
    mock_model.predict.return_value = [0]
    
    response = client.post("/chat", json={"query": "I am a healthy 30 year old."})
    assert response.status_code == 200
    assert response.json()["mode"] == "prediction"
    assert response.json()["prediction"] == 0

@patch('src.app.query_llm')
@patch('src.app.model')
def test_input_parsing_edge_case(mock_model, mock_query_llm):
    """Test 2: Check edge case handling for out-of-scope requests."""
    mock_query_llm.return_value = '{"error": "clarification_needed", "message": "Please input health details."}'
    
    response = client.post("/chat", json={"query": "How do I make chocolate chip cookies?"})
    assert response.status_code == 200
    assert response.json()["mode"] == "clarification"
    assert "Please input health details." in response.json()["response"]