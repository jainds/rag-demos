import pytest
from typing import Generator
from fastapi.testclient import TestClient
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import asyncio
import nest_asyncio

# Configure asyncio for testing
@pytest.fixture(scope="session", autouse=True)
def event_loop():
    """Create and configure event loop for testing"""
    # Use default event loop policy
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    # Create and set new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Apply nest_asyncio
    nest_asyncio.apply(loop)
    
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def client() -> Generator:
    """Create test client after event loop is configured"""
    from app.main import app  # Import here to ensure event loop is configured first
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Automatically mock environment variables for all tests"""
    with patch.dict(os.environ, {
        "OPENROUTER_MODEL": "test-model",
        "OPENROUTER_API_KEY": "test-key",
        "LANGFUSE_PUBLIC_KEY": "test-public-key",
        "LANGFUSE_SECRET_KEY": "test-secret-key",
        "LANGFUSE_HOST": "http://test.langfuse.com"
    }):
        yield

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def cleanup_files():
    """Fixture to clean up temporary files after tests"""
    temp_files = []
    
    def _cleanup():
        for file_path in temp_files:
            try:
                Path(file_path).unlink()
            except Exception:
                pass
    
    yield temp_files
    _cleanup()
