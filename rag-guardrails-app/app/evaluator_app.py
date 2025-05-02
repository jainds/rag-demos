from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from app.services.evaluator import evaluate_response, batch_evaluate_responses

# Initialize FastAPI
app = FastAPI()

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: Optional[List[str]] = None
    ground_truths: Optional[List[str]] = None
    metrics: Optional[List[str]] = None

class BatchEvaluationRequest(BaseModel):
    questions: List[str]
    answers: List[str]
    contexts: Optional[List[List[str]]] = None
    ground_truths: Optional[List[List[str]]] = None
    metrics: Optional[List[str]] = None

@app.post("/evaluate")
async def evaluate_single_response(request: EvaluationRequest):
    """
    Evaluate a single RAG response
    """
    return evaluate_response(
        question=request.question,
        answer=request.answer,
        contexts=request.contexts,
        ground_truths=request.ground_truths,
        metrics=request.metrics
    )

@app.post("/evaluate/batch")
async def evaluate_multiple_responses(request: BatchEvaluationRequest):
    """
    Evaluate multiple RAG responses in batch
    """
    return batch_evaluate_responses(
        questions=request.questions,
        answers=request.answers,
        contexts=request.contexts,
        ground_truths=request.ground_truths,
        metrics=request.metrics
    )
