import pytest
from app.services.rag import rag_pipeline, get_faiss_index_and_documents
import os
import json

@pytest.mark.asyncio
async def test_positive_rag_pipeline_normal():
    """Positive scenario: rag_pipeline returns dict or str for a valid query."""
    result = await rag_pipeline("What is the Eiffel Tower?")
    assert isinstance(result, dict) or isinstance(result, str)

@pytest.mark.asyncio
async def test_rag_pipeline_empty_query():
    result = await rag_pipeline("")
    assert isinstance(result, dict) or isinstance(result, str)

@pytest.mark.asyncio
async def test_robustness_rag_pipeline_missing_files(monkeypatch):
    """Robustness scenario: Missing file raises FileNotFoundError."""
    from app.services import rag as rag_mod
    # Patch the file check to simulate missing file
    monkeypatch.setattr(rag_mod, "get_faiss_index_and_documents", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("Missing file")))
    with pytest.raises(FileNotFoundError):
        await rag_pipeline("Test")

@pytest.mark.asyncio
async def test_rag_pipeline_long_query():
    long_query = "What is the Eiffel Tower? " * 1000
    result = await rag_pipeline(long_query)
    assert isinstance(result, dict) or isinstance(result, str)

@pytest.mark.asyncio
async def test_robustness_rag_pipeline_non_string_query():
    """Robustness scenario: Non-string query raises Exception."""
    with pytest.raises(Exception):
        await rag_pipeline(12345)  # Should raise due to non-string input

@pytest.mark.asyncio
async def test_rag_pipeline_no_results(monkeypatch):
    # Patch index.search to return -1 indices (no results)
    from app.services import rag as rag_mod
    orig_get = rag_mod.get_faiss_index_and_documents
    def fake_get(*a, **k):
        index, docs = orig_get(*a, **k)
        class FakeIndex:
            def search(self, x, k):
                import numpy as np
                return np.array([[0.0, 0.0]]), np.array([[-1, -1]])
        return FakeIndex(), docs
    monkeypatch.setattr(rag_mod, "get_faiss_index_and_documents", fake_get)
    result = await rag_pipeline("No results query")
    assert isinstance(result, dict) or isinstance(result, str)

@pytest.mark.asyncio
async def test_robustness_rag_pipeline_documents_not_list(tmp_path, monkeypatch):
    """Robustness scenario: Malformed documents.json (not a list) raises ValueError."""
    # Create a malformed documents.json
    bad_docs_path = tmp_path / "bad_docs.json"
    with open(bad_docs_path, "w") as f:
        f.write("{\"not\": \"a list\"}")
    from app.services import rag as rag_mod
    with pytest.raises(ValueError):
        await rag_pipeline("Test", documents_path=str(bad_docs_path))

@pytest.mark.asyncio
async def test_robustness_rag_pipeline_documents_empty(tmp_path):
    """Robustness scenario: Empty documents and index raises IndexError."""
    # Create an empty list for documents
    docs_path = tmp_path / "empty_docs.json"
    with open(docs_path, "w") as f:
        json.dump([], f)
    # Create a matching empty index
    import faiss, numpy as np
    dim = 384
    index = faiss.IndexFlatL2(dim)
    faiss.write_index(index, str(tmp_path / "empty_index.faiss"))
    with pytest.raises(IndexError):
        await rag_pipeline("Test", index_path=str(tmp_path / "empty_index.faiss"), documents_path=str(docs_path))

@pytest.mark.asyncio
async def test_robustness_rag_pipeline_index_corrupt(tmp_path):
    """Robustness scenario: Corrupt index file raises Exception."""
    # Create a corrupt index file
    index_path = tmp_path / "corrupt.faiss"
    with open(index_path, "wb") as f:
        f.write(b"not a real index")
    docs_path = tmp_path / "docs.json"
    with open(docs_path, "w") as f:
        json.dump([{"text": "doc", "source": "s"}], f)
    with pytest.raises(Exception):
        await rag_pipeline("Test", index_path=str(index_path), documents_path=str(docs_path))

@pytest.mark.asyncio
async def test_robustness_rag_pipeline_env_missing(monkeypatch):
    """Robustness scenario: Missing OPENROUTER_MODEL env variable still returns result."""
    # Remove OPENROUTER_MODEL from env
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    result = await rag_pipeline("Test")
    assert isinstance(result, dict) or isinstance(result, str)

@pytest.mark.asyncio
async def test_robustness_rag_pipeline_llm_exception(monkeypatch):
    """Robustness scenario: LLM exception is raised and handled."""
    from app.services import rag as rag_mod
    class DummyLLM:
        async def generate_async(self, *a, **k): raise Exception("LLM error")
    monkeypatch.setattr(rag_mod, "LLMRails", lambda *a, **k: DummyLLM())
    with pytest.raises(Exception):
        await rag_pipeline("Test")

@pytest.mark.asyncio
async def test_rag_pipeline_large_documents(tmp_path):
    # Create a large documents.json
    docs = [{"text": "x" * 10000, "source": f"doc{i}.txt"} for i in range(10)]
    docs_path = tmp_path / "large_docs.json"
    with open(docs_path, "w") as f:
        json.dump(docs, f)
    # Use normal index
    from app.services.rag import get_faiss_index_and_documents
    index_path = "data/vector_index.faiss"
    result = await rag_pipeline("Test", index_path=index_path, documents_path=str(docs_path))
    assert isinstance(result, dict) or isinstance(result, str)

@pytest.mark.asyncio
async def test_rag_pipeline_multiple_parallel():
    # Run multiple queries in parallel to check for race conditions
    queries = [f"Query {i}" for i in range(5)]
    import asyncio
    results = await asyncio.gather(*(rag_pipeline(q) for q in queries))
    assert all(isinstance(r, (dict, str)) for r in results) 