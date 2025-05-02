import os
import tempfile
import json
import shutil
import numpy as np
import pytest
from pathlib import Path
from app.services.document_loader import DocumentLoader
from app.services.document_indexer import DocumentIndexer
from app.services.rag_service import RAGService
from app.services import evaluator as evaluator_mod

@pytest.fixture(scope="module")
def temp_data_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

@pytest.fixture
def sample_text_file(temp_data_dir):
    file_path = Path(temp_data_dir) / "sample.txt"
    file_path.write_text("This is a test document.\nIt has two sentences.")
    return str(file_path)

@pytest.mark.asyncio
def test_document_loading_and_splitting(sample_text_file):
    loader = DocumentLoader(chunk_size=10, chunk_overlap=0)
    docs = loader.load_documents(sample_text_file, file_type="text")
    assert isinstance(docs, list)
    assert all("text" in d for d in docs)
    assert len(docs) > 1  # Should be split into chunks

@pytest.mark.asyncio
def test_index_creation_and_saving(sample_text_file, temp_data_dir):
    loader = DocumentLoader(chunk_size=10, chunk_overlap=0)
    docs = loader.load_documents(sample_text_file, file_type="text")
    indexer = DocumentIndexer(model_name="all-MiniLM-L6-v2")
    indexer.create_index(docs)
    index_path = Path(temp_data_dir) / "test_index.faiss"
    indexer.save_index(index_path)
    assert index_path.exists()

@pytest.mark.asyncio
def test_index_loading_and_search(sample_text_file, temp_data_dir):
    loader = DocumentLoader(chunk_size=10, chunk_overlap=0)
    docs = loader.load_documents(sample_text_file, file_type="text")
    indexer = DocumentIndexer(model_name="all-MiniLM-L6-v2")
    indexer.create_index(docs)
    index_path = Path(temp_data_dir) / "test_index.faiss"
    indexer.save_index(index_path)
    # New indexer instance
    indexer2 = DocumentIndexer(model_name="all-MiniLM-L6-v2")
    indexer2.load_index(index_path)
    distances, indices = indexer2.search("test", k=1)
    assert isinstance(distances, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert indices.shape[1] == 1

@pytest.mark.asyncio
def test_rag_service_query(sample_text_file, temp_data_dir):
    # Setup RAGService with test data
    rag = RAGService()
    rag.load_and_index_documents(sample_text_file, file_type="text", save_dir=temp_data_dir)
    # Patch LLM to avoid real API call
    rag.llm_provider._async_call = lambda *a, **k: "Test answer."
    async def fake_generate_async(*a, **k):
        return {"content": "Test answer."}
    rag.rails.generate_async = fake_generate_async
    result = pytest.run(rag.query("What is this?")) if hasattr(pytest, 'run') else None
    # If using pytest-asyncio, run with event loop
    if result is None:
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(rag.query("What is this?"))
    assert "answer" in result
    assert "contexts" in result

@pytest.mark.asyncio
def test_evaluator_pipeline():
    # Use the real evaluator with a simple example
    result = pytest.run(evaluator_mod.evaluate_response(
        question="What is AI?",
        answer="AI is artificial intelligence.",
        contexts=["AI is a field of computer science."],
        ground_truths=["AI is artificial intelligence."]
    )) if hasattr(pytest, 'run') else None
    if result is None:
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(evaluator_mod.evaluate_response(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI is a field of computer science."],
            ground_truths=["AI is artificial intelligence."]
        ))
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values()) 