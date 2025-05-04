import asyncio
import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.services.rag_service import RAGService, RAGServiceNoGuardrails
from app.services.evaluator import evaluate_response
from langfuse import Langfuse
from langfuse.decorators import observe
import os
from dotenv import load_dotenv
from app.models.data_models import QueryRequest
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, ContextRelevance
from langchain_huggingface import HuggingFaceEmbeddings
from app.models.ChatOpenRouter import ChatOpenRouter
import logging

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
# Initialize RAG service without guardrails
rag_service_no_guardrails = RAGServiceNoGuardrails()
# Initialize ChatOpenRouter for plain endpoint
chat_openrouter = ChatOpenRouter(
    model=os.environ.get("OPENROUTER_MODEL", "opengvlab/internvl3-2b:free"),
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0.1,
    max_tokens=512,
    top_p=0.95
)

# Load existing index for both new services
try:
    rag_service_no_guardrails.load_existing_index(
        documents_path="data/documents.json",
        index_path="data/vector_index.faiss"
    )
except Exception as e:
    print(f"Warning: Could not load existing index for no-guardrails: {e}")

class IndexRequest(BaseModel):
    source_path: str
    file_type: str = "text"
    save_dir: Optional[str] = "data"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/query")
@observe(as_type="generation")
async def query_rag(request: QueryRequest, evaluator=evaluate_response):
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
        metrics_dict = request.metrics.model_dump() if hasattr(request.metrics, 'model_dump') else dict(request.metrics)
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
        evaluation = await evaluator(
            question=request.question,
            answer=result["answer"],
            contexts=result["contexts"],
            metrics=metrics
        )

        # Normalize and ensure all metrics are present in the response
        expected_metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_relevance"
        ]
        # Map possible returned keys to expected keys
        key_map = {
            "faithfulness": "faithfulness",
            "answerrelevancy": "answer_relevancy",
            "answer_relevancy": "answer_relevancy",
            "contextprecision": "context_precision",
            "context_precision": "context_precision",
            "contextrecall": "context_recall",
            "context_recall": "context_recall",
            "contextrelevance": "context_relevance",
            "context_relevance": "context_relevance"
        }
        normalized_metrics = {k: None for k in expected_metrics}
        for k, v in evaluation.items():
            norm_key = key_map.get(k.lower(), k.lower())
            if norm_key in normalized_metrics:
                normalized_metrics[norm_key] = v
        response = {
            "question": request.question,
            "answer": result["answer"],
            "contexts": result["contexts"],
            "metrics": normalized_metrics
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

@app.post("/query_no_guardrails")
async def query_no_guardrails(request: QueryRequest):
    """
    Query endpoint without Nemo Guardrails. Uses retrieval and ChatOpenRouter, can use Ragas if needed.
    """
    logger = logging.getLogger("query_no_guardrails")
    try:
        logger.info(f"Received query: {request.question}")
        result = await rag_service_no_guardrails.query_no_guardrails(request.question)
        logger.info(f"Returning answer: {result['answer']}")
        return {
            "question": request.question,
            "answer": result["answer"],
            "contexts": result["contexts"]
        }
    except Exception as e:
        logger.error(f"Error in /query_no_guardrails: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_openrouter")
async def query_openrouter(request: QueryRequest):
    """
    Plain ChatOpenRouter RAG endpoint: retrieval + LLM, no Nemo Guardrails, no Ragas.
    """
    logger = logging.getLogger("query_openrouter")
    try:
        # Use the same retrieval as rag_service_no_guardrails
        _, indices = rag_service_no_guardrails.indexer.search(request.question, k=2)
        retrieved_docs = [rag_service_no_guardrails.documents[i] for i in indices[0]]
        contexts = [doc['text'] for doc in retrieved_docs]
        context_text = "\n".join(f"• {ctx}" for ctx in contexts) if contexts else ""
        prompt = f"Context:\n{context_text}\n\nQuestion: {request.question}\nAnswer:"
        logger.info(f"[query_openrouter] Prompt: {prompt}")
        answer = await chat_openrouter.agenerate_text(prompt)
        logger.info(f"[query_openrouter] Answer: {answer}")
        return {
            "question": request.question,
            "answer": answer,
            "contexts": contexts
        }
    except Exception as e:
        logger.error(f"Error in /query_openrouter: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))