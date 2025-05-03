import asyncio
import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.services.rag_service import RAGService
from app.services.evaluator import evaluate_response
from langfuse import Langfuse
from langfuse.decorators import observe
import os
from dotenv import load_dotenv
from app.models.data_models import QueryRequest
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, ContextRelevance
from langchain_huggingface import HuggingFaceEmbeddings

# Ensure we're using the default event loop policy
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Apply nest_asyncio if not already applied
try:
    loop = asyncio.get_event_loop()
    if not hasattr(loop, '_nest_patched'):
        nest_asyncio.apply()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply()

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="RAG Guardrails API",
    description="API for RAG with evaluation and guardrails",
    version="1.0.0"
)

# Initialize Langfuse
langfuse = Langfuse(
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    host=os.environ.get("LANGFUSE_HOST")
)

# Initialize RAG service
rag_service = RAGService()

# Load existing index if available
try:
    rag_service.load_existing_index(
        documents_path="data/documents.json",
        index_path="data/vector_index.faiss"
    )
except Exception as e:
    print(f"Warning: Could not load existing index: {e}")

class IndexRequest(BaseModel):
    source_path: str
    file_type: str = "text"
    save_dir: Optional[str] = "data"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/query")
@observe(as_type="generation")
async def query_rag(request: QueryRequest):
    """
    Process a query through the RAG pipeline with optional evaluation.
    """
    try:
        # Get response from RAG pipeline
        result = await rag_service.query(request.question)

        # Determine metrics to use
        metrics = None
        metric_map = {
            "faithfulness": Faithfulness,
            "answer_relevancy": AnswerRelevancy,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
            "context_relevance": ContextRelevance
        }
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        metrics_dict = request.metrics.dict() if hasattr(request.metrics, 'dict') else dict(request.metrics)
        if metrics_dict is not None:
            selected = []
            for name, enabled in metrics_dict.items():
                if not enabled:
                    continue
                key = name.lower()
                if key == "answer_relevancy":
                    selected.append(AnswerRelevancy(llm=rag_service.llm_provider, embeddings=embedder))
                elif key in metric_map:
                    metric_cls = metric_map[key]
                    if key in ["context_precision", "context_recall"]:
                        from app.services.evaluator import patch_metric_prompt
                        selected.append(patch_metric_prompt(metric_cls(llm=rag_service.llm_provider)))
                    else:
                        selected.append(metric_cls(llm=rag_service.llm_provider))
            metrics = selected if selected else None

        # Always evaluate the response
        evaluation = await evaluate_response(
            question=request.question,
            answer=result["answer"],
            contexts=result["contexts"],
            metrics=metrics
        )

        response = {
            "question": request.question,
            "answer": result["answer"],
            "contexts": result["contexts"],
            "metrics": evaluation
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_documents(request: IndexRequest):
    """
    Load and index new documents.
    """
    try:
        rag_service.load_and_index_documents(
            source_path=request.source_path,
            file_type=request.file_type,
            save_dir=request.save_dir
        )
        return {"message": "Documents indexed successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))