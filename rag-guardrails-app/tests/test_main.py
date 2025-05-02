from fastapi.testclient import TestClient
from app.main import app
import pytest
from unittest.mock import patch, AsyncMock

client = TestClient(app)

def test_query_rag():
    response = client.post(
        "/query",
        json={"question": "What is the Eiffel Tower?"},
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "metrics" in response.json()

@pytest.mark.asyncio
async def test_query_rag_with_evaluation():
    """Test the query endpoint with evaluation enabled, simulating the exact API call scenario"""
    # Prepare test data
    test_question = "What is France?"
    test_request = {
        "question": test_question,
        "evaluate": True
    }
    
    # Make the request
    response = client.post(
        "/query",
        json=test_request
    )
    
    # Verify response structure
    assert response.status_code == 200
    response_data = response.json()
    assert "question" in response_data
    assert "answer" in response_data
    assert "contexts" in response_data
    assert "metrics" in response_data
    
    # Verify metrics structure
    metrics = response_data["metrics"]
    expected_metrics = [
        "faithfulness",
        "answerrelevancy",
        "contextprecision",
        "contextrecall",
        "contextrelevance"
    ]
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be a number"
        assert 0 <= metrics[metric] <= 1, f"Metric {metric} should be between 0 and 1"
