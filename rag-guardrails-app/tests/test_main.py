from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_query_rag():
    response = client.post(
        "/query",
        json={"question": "What is the Eiffel Tower?"},
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "metrics" in response.json()
