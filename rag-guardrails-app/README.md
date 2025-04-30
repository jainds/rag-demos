# RAG Guardrails App

This is a RAG application with Nemo Guardrails, RAGAS evaluation, and Langfuse integration.

## Running the App

1.  **Build and Run with Docker:**

    ```bash
    docker-compose up --build
    ```

2.  **Test the API:**

    ```bash
    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"question": "What is the Eiffel Tower?"}'
    ```

3.  **View Traces in Langfuse:**

    *   Go to [Langfuse Cloud](https://cloud.langfuse.com).
    *   View traces for:
        *   Document retrieval
        *   Answer generation
        *   Guardrail enforcement
        *   Evaluation metrics

## Running Tests

1.  **Install pytest:**

    ```bash
    pip install pytest
    ```

2.  **Run tests:**

    ```bash
    pytest tests
    ```

## Architecture
The application now consists of one microservice:

*   **app**: The main application that handles the RAG pipeline and calls the evaluator service.