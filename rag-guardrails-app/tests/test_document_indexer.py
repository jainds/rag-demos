import pytest
import numpy as np
from pathlib import Path
import tempfile
from app.services.document_indexer import DocumentIndexer

@pytest.fixture
def document_indexer():
    return DocumentIndexer(model_name='all-MiniLM-L6-v2')

@pytest.fixture
def sample_documents():
    return [
        {"text": "This is a test document about AI.", "source": "test1.txt"},
        {"text": "Machine learning is a subset of AI.", "source": "test2.txt"},
        {"text": "Natural language processing is fascinating.", "source": "test3.txt"}
    ]

def test_positive_init_document_indexer():
    """Positive scenario: DocumentIndexer initializes with model and dimension."""
    indexer = DocumentIndexer(model_name='all-MiniLM-L6-v2')
    assert indexer.model is not None
    assert indexer.dimension > 0
    assert indexer.index is None

def test_positive_create_index(document_indexer, sample_documents):
    """Positive scenario: Index is created and has correct number of documents."""
    document_indexer.create_index(sample_documents)
    assert document_indexer.index is not None
    assert document_indexer.index.ntotal == len(sample_documents)

def test_positive_save_and_load_index(document_indexer, sample_documents, tmp_path):
    """Positive scenario: Index can be saved and loaded, preserving document count."""
    # Create and save index
    document_indexer.create_index(sample_documents)
    index_path = tmp_path / "test_index.faiss"
    document_indexer.save_index(index_path)
    
    # Create new indexer and load saved index
    new_indexer = DocumentIndexer(model_name='all-MiniLM-L6-v2')
    new_indexer.load_index(index_path)
    
    assert new_indexer.index is not None
    assert new_indexer.index.ntotal == len(sample_documents)

def test_positive_search(document_indexer, sample_documents):
    """Positive scenario: Search returns correct types and indices in range."""
    document_indexer.create_index(sample_documents)
    
    # Test search
    distances, indices = document_indexer.search("What is artificial intelligence?", k=2)
    
    assert isinstance(distances, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert len(distances[0]) == 2
    assert len(indices[0]) == 2
    assert all(0 <= idx < len(sample_documents) for idx in indices[0])

def test_robustness_search_without_index(document_indexer):
    """Robustness scenario: Searching without index raises ValueError."""
    with pytest.raises(ValueError):
        document_indexer.search("test query")

def test_robustness_save_without_index(document_indexer, tmp_path):
    """Robustness scenario: Saving without index raises ValueError."""
    with pytest.raises(ValueError):
        document_indexer.save_index(tmp_path / "nonexistent.faiss") 