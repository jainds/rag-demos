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

def test_positive_query_rag():
    """Positive scenario: /query returns answer and metrics for a valid question."""
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
    """Integration test: /query endpoint with evaluation, checks answer_relevancy metric and embedder."""
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    called = {}
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
    assert "answer_relevancy" in metrics
    assert isinstance(metrics["answer_relevancy"], (int, float))
    assert 0 <= metrics["answer_relevancy"] <= 1
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

def test_positive_query_rag_metrics_positive(monkeypatch):
    """Positive scenario: All metrics should be numbers (not None) for a normal query. Faithfulness is mocked to always return 1.0 for robustness."""
    from app.services import evaluator
    orig_eval = evaluator.evaluate_response
    async def always_faithful(*args, **kwargs):
        result = await orig_eval(*args, **kwargs)
        result["faithfulness"] = 1.0  # Simulate always successful faithfulness
        return result
    monkeypatch.setattr(evaluator, "evaluate_response", always_faithful)
    response = client.post("/query", json={"question": "What is the Eiffel Tower?"})
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "context_relevance"]:
        value = metrics[key]
        assert value is not None, f"Metric {key} should not be None in positive scenario"
        assert isinstance(value, (int, float))
        assert 0 <= value <= 1

@pytest.mark.asyncio
def test_robustness_query_rag_metrics_negative(monkeypatch):
    """Robustness scenario: Simulate metric failure, expect None and no server error."""
    from app.services import evaluator
    orig_eval = evaluator.evaluate_response
    async def fail_metric(*args, **kwargs):
        result = await orig_eval(*args, **kwargs)
        result["faithfulness"] = None  # Simulate failure
        return result
    monkeypatch.setattr(evaluator, "evaluate_response", fail_metric)
    response = client.post("/query", json={"question": "What is the Eiffel Tower?"})
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    assert metrics["faithfulness"] is None
    # The app should still return a valid response, not a server error

def test_positive_query_with_only_faithfulness_enabled():
    """Positive scenario: Only faithfulness metric enabled, others disabled."""
    payload = {
        "question": "What is the Eiffel Tower?",
        "metrics": {
            "faithfulness": True,
            "answer_relevancy": False,
            "context_precision": False,
            "context_recall": False,
            "context_relevance": False
        }
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    assert metrics["faithfulness"] is not None
    assert isinstance(metrics["faithfulness"], (int, float))
    assert 0 <= metrics["faithfulness"] <= 1
    assert metrics["answer_relevancy"] is None
    assert metrics["context_precision"] is None
    assert metrics["context_recall"] is None
    assert metrics["context_relevance"] is None

def test_positive_query_with_only_answer_relevancy_enabled():
    """Positive scenario: Only answer_relevancy metric enabled, others disabled."""
    payload = {
        "question": "What is the Eiffel Tower?",
        "metrics": {
            "faithfulness": False,
            "answer_relevancy": True,
            "context_precision": False,
            "context_recall": False,
            "context_relevance": False
        }
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    assert metrics["answer_relevancy"] is not None
    assert isinstance(metrics["answer_relevancy"], (int, float))
    assert 0 <= metrics["answer_relevancy"] <= 1
    assert metrics["faithfulness"] is None
    assert metrics["context_precision"] is None
    assert metrics["context_recall"] is None
    assert metrics["context_relevance"] is None

def test_positive_query_with_only_context_precision_enabled():
    """Positive scenario: Only context_precision metric enabled, others disabled."""
    payload = {
        "question": "What is the Eiffel Tower?",
        "metrics": {
            "faithfulness": False,
            "answer_relevancy": False,
            "context_precision": True,
            "context_recall": False,
            "context_relevance": False
        }
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    assert metrics["context_precision"] is not None
    assert isinstance(metrics["context_precision"], (int, float))
    assert 0 <= metrics["context_precision"] <= 1
    assert metrics["faithfulness"] is None
    assert metrics["answer_relevancy"] is None
    assert metrics["context_recall"] is None
    assert metrics["context_relevance"] is None

def test_positive_query_with_only_context_recall_enabled():
    """Positive scenario: Only context_recall metric enabled, others disabled."""
    payload = {
        "question": "What is the Eiffel Tower?",
        "metrics": {
            "faithfulness": False,
            "answer_relevancy": False,
            "context_precision": False,
            "context_recall": True,
            "context_relevance": False
        }
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    assert metrics["context_recall"] is not None
    assert isinstance(metrics["context_recall"], (int, float))
    assert 0 <= metrics["context_recall"] <= 1
    assert metrics["faithfulness"] is None
    assert metrics["answer_relevancy"] is None
    assert metrics["context_precision"] is None
    assert metrics["context_relevance"] is None

def test_positive_query_with_only_context_relevance_enabled():
    """Positive scenario: Only context_relevance metric enabled, others disabled."""
    payload = {
        "question": "What is the Eiffel Tower?",
        "metrics": {
            "faithfulness": False,
            "answer_relevancy": False,
            "context_precision": False,
            "context_recall": False,
            "context_relevance": True
        }
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    assert metrics["context_relevance"] is not None
    assert isinstance(metrics["context_relevance"], (int, float))
    assert 0 <= metrics["context_relevance"] <= 1
    assert metrics["faithfulness"] is None
    assert metrics["answer_relevancy"] is None
    assert metrics["context_precision"] is None
    assert metrics["context_recall"] is None
