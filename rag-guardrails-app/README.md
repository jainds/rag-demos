# RAG Guardrails App

This is a RAG application with Nemo Guardrails, RAGAS evaluation, and Langfuse integration.

## Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   # For production
   pip install -r requirements.txt
   
   # For development (includes testing tools)
   pip install -r requirements-dev.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file with:
   ```
   LANGFUSE_PUBLIC_KEY=your_public_key
   LANGFUSE_SECRET_KEY=your_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com
   OPENROUTER_API_KEY=your_api_key
   OPENROUTER_MODEL=your_model_name
   ```

## Running the App

1. **Build and Run with Docker:**
   ```bash
   docker-compose up --build
   ```

2. **Test the API:**
   ```bash
   # Query endpoint
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the Eiffel Tower?"}'
   
   # Index documents
   curl -X POST http://localhost:8000/index \
     -H "Content-Type: application/json" \
     -d '{
       "source_path": "path/to/documents",
       "file_type": "text",
       "save_dir": "data"
     }'
   ```

## Development

### Running Tests

1. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run all tests:**
   ```bash
   pytest
   ```

3. **Run tests with coverage:**
   ```bash
   pytest --cov=app tests/
   ```

4. **Run specific test files:**
   ```bash
   # Test document loader
   pytest tests/test_document_loader.py
   
   # Test document indexer
   pytest tests/test_document_indexer.py
   
   # Test RAG service
   pytest tests/test_rag_service.py
   ```

### Code Quality

1. **Format code:**
   ```bash
   black app/ tests/
   isort app/ tests/
   ```

2. **Run linter:**
   ```bash
   flake8 app/ tests/
   ```

3. **Type checking:**
   ```bash
   mypy app/ tests/
   ```

## Architecture

The application consists of several components:

1. **Document Loader Service:**
   - Handles document loading from various sources (text, PDF, markdown)
   - Implements text chunking
   - Manages document saving and loading

2. **Document Indexer Service:**
   - Manages vector embeddings
   - Creates and maintains FAISS indices
   - Handles document similarity search

3. **RAG Service:**
   - Coordinates the entire RAG pipeline
   - Integrates document loading, indexing, and LLM
   - Handles query processing with guardrails

4. **API Layer:**
   - Provides REST endpoints for querying and indexing
   - Handles request validation and error handling
   - Integrates with evaluation service

## Monitoring

View traces in Langfuse:
- Go to [Langfuse Cloud](https://cloud.langfuse.com)
- View traces for:
  - Document retrieval
  - Answer generation
  - Guardrail enforcement
  - Evaluation metrics