import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
# Import the target module to ensure it exists for patching
from app import main as app_main

@pytest.fixture(autouse=True)
def mock_rag_index_and_docs(monkeypatch):
    """Automatically mock the RAGService's indexer.search and documents for all tests."""
    # Patch the search method to return dummy distances and indices
    dummy_distances = np.array([[0.1, 0.2]])
    dummy_indices = np.array([[0, 1]])
    # Use try-except for robustness, though direct patching should be fine now
    try:
        # Ensure rag_service exists before patching its methods/attributes
        if hasattr(app_main, 'rag_service'):
            monkeypatch.setattr(
                app_main.rag_service.indexer, # Target object
                "search",                     # Attribute name (string)
                lambda query, k=2: (dummy_distances, dummy_indices) # New value
            )
            # Patch the documents attribute directly on the rag_service instance
            monkeypatch.setattr(
                app_main.rag_service, # Target object
                "documents",          # Attribute name (string)
                [                     # New value
                    {"text": "Dummy context 1"},
                    {"text": "Dummy context 2"}
                ]
            )
        else:
            print("Warning: app.main.rag_service not found during fixture setup.")
    except Exception as e:
        print(f"Warning: Error during mock_rag_index_and_docs fixture setup: {e}")
        # If patching fails here, tests might fail later, but setup won't crash
        pass 