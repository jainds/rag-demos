import pytest
from pathlib import Path
import tempfile
import json
import os
from unittest.mock import Mock, patch, AsyncMock
from app.services.rag_service import RAGService

@pytest.fixture
def mock_env_vars():
    """Setup mock environment variables"""
    with patch.dict(os.environ, {
        "OPENROUTER_MODEL": "test-model",
        "OPENROUTER_API_KEY": "test-key",
        "LANGFUSE_PUBLIC_KEY": "test-public-key",
        "LANGFUSE_SECRET_KEY": "test-secret-key",
        "LANGFUSE_HOST": "http://test.langfuse.com"
    }):
        yield

@pytest.fixture
def sample_config_file(tmp_path):
    """Create a sample config file"""
    config_content = """
    models:
      - type: main
        engine: custom_llm
        model: test-model
    """
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)
    return config_file

@pytest.fixture
def sample_documents():
    return [
        {"text": "This is a test document about AI.", "source": "test1.txt"},
        {"text": "Machine learning is a subset of AI.", "source": "test2.txt"}
    ]

@pytest.fixture
def rag_service(mock_env_vars, sample_config_file):
    return RAGService(config_path=str(sample_config_file))

@pytest.mark.asyncio
async def test_positive_init_rag_service(mock_env_vars, sample_config_file):
    """Positive scenario: RAG service initializes with loader, indexer, and empty documents."""
    service = RAGService(config_path=str(sample_config_file))
    assert service.loader is not None
    assert service.indexer is not None
    assert service.documents == []

@pytest.mark.asyncio
async def test_positive_load_and_index_documents(rag_service, tmp_path):
    """Positive scenario: Documents are loaded, indexed, and files are saved."""
    # Create test document
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document for RAG testing.")
    
    # Test loading and indexing
    rag_service.load_and_index_documents(
        source_path=str(test_file),
        file_type="text",
        save_dir=str(tmp_path)
    )
    
    assert len(rag_service.documents) > 0
    assert (tmp_path / "documents.json").exists()
    assert (tmp_path / "vector_index.faiss").exists()

@pytest.mark.asyncio
async def test_positive_load_existing_index(rag_service, tmp_path, sample_documents):
    """Positive scenario: Existing index and documents can be loaded into a new service."""
    # Save sample documents and create index
    docs_path = tmp_path / "documents.json"
    with open(docs_path, "w") as f:
        json.dump(sample_documents, f)
    
    rag_service.documents = sample_documents
    rag_service.indexer.create_index(sample_documents)
    index_path = tmp_path / "vector_index.faiss"
    rag_service.indexer.save_index(index_path)
    
    # Create new service and load existing index
    new_service = RAGService()
    new_service.load_existing_index(
        documents_path=str(docs_path),
        index_path=str(index_path)
    )
    
    assert len(new_service.documents) == len(sample_documents)
    assert new_service.indexer.index is not None

@pytest.mark.asyncio
async def test_positive_query(rag_service, sample_documents):
    """Positive scenario: Query returns expected answer and contexts, and LLM is called."""
    # Setup test data
    rag_service.documents = sample_documents
    rag_service.indexer.create_index(sample_documents)
    
    # Mock the LLM response
    mock_response = {"content": "This is a test response"}
    rag_service.rails.generate_async = AsyncMock(return_value=mock_response)
    
    # Test query
    response = await rag_service.query("What is AI?")
    assert response["answer"] == mock_response["content"]
    assert "contexts" in response
    
    # Verify LLM was called
    assert rag_service.rails.generate_async.called

@pytest.mark.asyncio
async def test_robustness_query_error_handling(rag_service):
    """Robustness scenario: Query raises exception if LLM fails, and error is handled."""
    # Mock an error in LLM
    rag_service.rails.generate_async = AsyncMock(side_effect=Exception("Test error"))
    
    with pytest.raises(Exception) as exc_info:
        await rag_service.query("Test query")
    assert "Error generating response" in str(exc_info.value) 