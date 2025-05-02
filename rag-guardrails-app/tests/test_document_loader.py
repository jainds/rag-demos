import pytest
from pathlib import Path
import tempfile
import json
from app.services.document_loader import DocumentLoader

@pytest.fixture
def document_loader():
    return DocumentLoader(chunk_size=100, chunk_overlap=20)

@pytest.fixture
def sample_text_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\nIt has multiple lines.\nThis is for testing.")
        return Path(f.name)

@pytest.fixture
def sample_markdown_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test Document\n\nThis is a markdown test.\n## Section\nMore content.")
        return Path(f.name)

def test_init_document_loader():
    """Test DocumentLoader initialization"""
    loader = DocumentLoader(chunk_size=200, chunk_overlap=50)
    assert loader.chunk_size == 200
    assert loader.chunk_overlap == 50
    assert loader.text_splitter is not None

def test_load_text_document(document_loader, sample_text_file):
    """Test loading a text document"""
    docs = document_loader.load_documents(sample_text_file, file_type="text")
    assert len(docs) > 0
    assert all(isinstance(doc, dict) for doc in docs)
    assert all("text" in doc and "source" in doc for doc in docs)

def test_load_markdown_document(document_loader, sample_markdown_file):
    """Test loading a markdown document"""
    docs = document_loader.load_documents(sample_markdown_file, file_type="markdown")
    assert len(docs) > 0
    assert all(isinstance(doc, dict) for doc in docs)
    assert all("text" in doc and "source" in doc for doc in docs)

def test_save_and_load_json(document_loader, sample_text_file, tmp_path):
    """Test saving and loading documents to/from JSON"""
    # Load initial documents
    docs = document_loader.load_documents(sample_text_file, file_type="text")
    
    # Save to JSON
    json_path = tmp_path / "test_docs.json"
    document_loader.save_documents(docs, json_path)
    
    # Load from JSON
    loaded_docs = document_loader.load_from_json(json_path)
    
    assert len(loaded_docs) == len(docs)
    assert all(isinstance(doc, dict) for doc in loaded_docs)
    assert all("text" in doc and "source" in doc for doc in loaded_docs)

def test_invalid_file_type(document_loader, sample_text_file):
    """Test handling of invalid file type"""
    with pytest.raises(ValueError):
        document_loader.load_documents(sample_text_file, file_type="invalid")

def test_invalid_path(document_loader):
    """Test handling of invalid file path"""
    with pytest.raises(ValueError):
        document_loader.load_documents("nonexistent_file.txt") 