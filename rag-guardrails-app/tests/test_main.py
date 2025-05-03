from fastapi.testclient import TestClient
from app.main import app
import pytest
from unittest.mock import patch, AsyncMock
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_rag_index_and_docs(monkeypatch):
    """Automatically mock the RAGService's indexer.search and documents for all tests."""
    # Patch the search method to return dummy distances and indices
    dummy_distances = np.array([[0.1, 0.2]])
    dummy_indices = np.array([[0, 1]])
    monkeypatch.setattr(
        "app.main.rag_service.indexer.search",
        lambda query, k=2: (dummy_distances, dummy_indices)
    )
    # Patch the documents to return dummy docs (use absolute import path)
    monkeypatch.setattr(
        "app.main.rag_service.documents",
        [
            {"text": "Dummy context 1"},
            {"text": "Dummy context 2"}
        ],
        raising=False
    )
    yield

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
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_relevance"
    ]
    for metric in expected_metrics:
        # Accept both snake_case and camelCase
        found = any(m.replace("_", "").lower() == metric.replace("_", "").lower() for m in metrics.keys())
        assert found, f"Missing metric: {metric}"
        value = metrics.get(metric)
        # Accept None as valid if metric could not be computed
        if value is not None:
            assert isinstance(value, (int, float)), f"Metric {metric} should be a number or None"
            assert 0 <= value <= 1, f"Metric {metric} should be between 0 and 1 or None"

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

def test_query_with_selected_metrics():
    payload = {
        "question": "What is the Eiffel Tower?",
        "metrics": {
            "faithfulness": True,
            "answer_relevancy": False,
            "context_precision": True,
            "context_recall": False,
            "context_relevance": False
        }
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    # Only faithfulness and context_precision should be present (others should be None)
    assert data["metrics"]["faithfulness"] is not None
    assert data["metrics"]["context_precision"] is not None
    assert data["metrics"]["answer_relevancy"] is None
    assert data["metrics"]["context_recall"] is None
    assert data["metrics"]["context_relevance"] is None

def test_query_with_all_metrics_default():
    payload = {"question": "What is the Eiffel Tower?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    # All metrics should be present by default (allow None if not computable)
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "context_relevance"]:
        assert key in data["metrics"]
        value = data["metrics"][key]
        if value is not None:
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1
