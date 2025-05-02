import pytest
from app.services.evaluator import evaluate_response, batch_evaluate_responses
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance
)
from app.models.ChatOpenRouter import ChatOpenRouter
import os

# Initialize test model
test_model = ChatOpenRouter(
    model="anthropic/claude-3-opus-20240229",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0.1,
    max_tokens=50
)

@pytest.fixture
def sample_qa_pair():
    return {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
        "contexts": ["Machine learning is a field of artificial intelligence that focuses on developing systems that can learn from data."],
        "ground_truths": ["Machine learning is an AI technology that allows systems to learn from data and improve over time."]
    }

@pytest.fixture
def sample_qa_batch():
    return {
        "questions": [
            "What is machine learning?",
            "What is deep learning?"
        ],
        "answers": [
            "Machine learning is a subset of AI.",
            "Deep learning is a type of machine learning using neural networks."
        ],
        "contexts": [
            ["Machine learning is a field of AI."],
            ["Deep learning uses neural networks to learn from data."]
        ],
        "ground_truths": [
            ["Machine learning is an AI technology."],
            ["Deep learning is a neural network-based ML approach."]
        ]
    }

@pytest.mark.asyncio
async def test_evaluate_response(sample_qa_pair):
    """Test single response evaluation"""
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"]
    )
    
    assert isinstance(result, dict)
    assert all(isinstance(score, float) for score in result.values())
    assert all(0 <= score <= 1 for score in result.values())

@pytest.mark.asyncio
async def test_evaluate_response_without_context():
    """Test evaluation without context"""
    result = await evaluate_response(
        question="What is AI?",
        answer="AI is artificial intelligence."
    )
    
    assert isinstance(result, dict)
    assert all(isinstance(score, float) for score in result.values())

@pytest.mark.asyncio
async def test_evaluate_response_with_custom_metrics(sample_qa_pair):
    """Test evaluation with custom metrics"""
    metrics = [
        AnswerRelevancy(llm=test_model),
        Faithfulness(llm=test_model)
    ]
    
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        metrics=metrics
    )
    
    assert isinstance(result, dict)
    assert len(result) == len(metrics)

@pytest.mark.asyncio
async def test_batch_evaluate_responses(sample_qa_batch):
    """Test batch response evaluation"""
    result = await batch_evaluate_responses(
        questions=sample_qa_batch["questions"],
        answers=sample_qa_batch["answers"],
        contexts=sample_qa_batch["contexts"],
        ground_truths=sample_qa_batch["ground_truths"]
    )
    
    assert isinstance(result, dict)
    assert all(isinstance(scores, list) for scores in result.values())
    assert all(len(scores) == len(sample_qa_batch["questions"]) 
              for scores in result.values())

@pytest.mark.asyncio
async def test_batch_evaluate_responses_validation():
    """Test batch evaluation input validation"""
    with pytest.raises(ValueError):
        await batch_evaluate_responses(
            questions=["Q1", "Q2"],
            answers=["A1"]  # Mismatched length
        )

@pytest.mark.asyncio
async def test_batch_evaluate_empty_inputs():
    """Test batch evaluation with empty inputs"""
    with pytest.raises(ValueError):
        await batch_evaluate_responses(
            questions=[],
            answers=[]
        )

@pytest.mark.asyncio
async def test_evaluate_response_with_custom_metrics_validation():
    """Test evaluation with invalid metrics"""
    with pytest.raises(TypeError):
        await evaluate_response(
            question="Test?",
            answer="Test.",
            metrics=["invalid_metric"]  # Should be metric objects
        ) 