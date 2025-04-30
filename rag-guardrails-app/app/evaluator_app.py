from fastapi import FastAPI
from pydantic import BaseModel
from ragas import evaluate
from ragas.test_dataset import TestDataset

# Initialize FastAPI
app = FastAPI()

class EvaluationRequest(BaseModel):
    question: str
    answer: str

@app.post("/evaluate")
async def evaluate_response(request: EvaluationRequest):
    # Load test dataset
    test_data = TestDataset.from_dicts([{"question": request.question, "answer": request.answer, "ground_truths": ["ground truth"]}]) # placeholder
    
    # Evaluate
    results = evaluate(
        test_data.to_pandas(),
        metrics=["faithfulness", "answer_relevancy", "context_relevancy", "context_precision", "context_recall"]
    )
    return results.to_dict()
