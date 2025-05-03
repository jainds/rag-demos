import pytest
from typing import Generator
from fastapi.testclient import TestClient
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import asyncio
import nest_asyncio
import requests
from dotenv import load_dotenv
import time

# Always load .env.test if it exists, else .env
DOTENV_PATH = Path(__file__).parent.parent / ".env.test"
if not DOTENV_PATH.exists():
    DOTENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

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

@pytest.fixture
def mock_env_vars():
    """Setup mock environment variables for tests that explicitly request it."""
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

@pytest.mark.integration
def test_openrouter_api_key_validity():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_MODEL")
    if not api_key or not model:
        print(f"[Preflight] Skipping: OPENROUTER_API_KEY or OPENROUTER_MODEL not set.\nAPI_KEY={api_key}, MODEL={model}")
        pytest.skip("OPENROUTER_API_KEY or OPENROUTER_MODEL not set.")
    masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
    print(f"[Preflight] Using API_KEY={masked_key}, MODEL={model}")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Your Application Name"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.1,
        "max_tokens": 3
    }
    for attempt in range(2):
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        print(f"[Preflight] Status: {resp.status_code}, Response: {resp.text}")
        if resp.status_code == 429:
            if attempt == 0:
                print("[Preflight] Rate limit hit. Waiting 60 seconds before retry...")
                time.sleep(60)
                continue
            else:
                pytest.skip(f"OpenRouter API rate limit exceeded after retry: {resp.text}")
        elif resp.status_code == 401:
            pytest.fail(f"OpenRouter API key is invalid or unauthorized: {resp.text}")
        elif resp.status_code != 200:
            pytest.skip(f"OpenRouter API returned status {resp.status_code}: {resp.text}")
        else:
            break
