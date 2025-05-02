import pytest
from app.services.rag import rag_pipeline

@pytest.mark.asyncio
async def test_rag_pipeline():
    response = await rag_pipeline("What is the Eiffel Tower?")
    assert isinstance(response, dict)
    assert "content" in response
    assert len(response["content"]) > 0
