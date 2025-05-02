from fastapi.testclient import TestClient
from app.main import app
import pytest
from unittest.mock import patch, AsyncMock
from langchain_community.embeddings import HuggingFaceEmbeddings

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

@pytest.mark.asyncio
def test_query_rag_with_evaluation_embedder(monkeypatch):
    """Integration test: /query endpoint with evaluation, checks answerrelevancy metric and embedder."""
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    called = {}
    # Patch AnswerRelevancy to check embedder
    from ragas.metrics import AnswerRelevancy
    orig_init = AnswerRelevancy.__init__
    def custom_init(self, *args, **kwargs):
        if 'embeddings' in kwargs:
            assert kwargs['embeddings'] is not None, "App did not set embedder in AnswerRelevancy"
            called['ok'] = True
        orig_init(self, *args, **kwargs)
    monkeypatch.setattr(AnswerRelevancy, "__init__", custom_init)
    client = TestClient(app)
    test_question = "What is France?"
    test_request = {
        "question": test_question,
        "evaluate": True
    }
    response = client.post(
        "/query",
        json=test_request
    )
    assert response.status_code == 200
    response_data = response.json()
    assert "metrics" in response_data
    metrics = response_data["metrics"]
    assert "answerrelevancy" in metrics
    assert isinstance(metrics["answerrelevancy"], (int, float))
    assert 0 <= metrics["answerrelevancy"] <= 1
    assert called.get('ok'), "App did not set the embedder in AnswerRelevancy in the real API flow"
