import asyncio
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance
)
from typing import List, Optional, Dict, Any
import pandas as pd
from app.models.ChatOpenRouter import ChatOpenRouter
import os
from dotenv import load_dotenv
from datasets import Dataset

load_dotenv()

# Initialize OpenRouter for evaluation
evaluation_model = ChatOpenRouter(
    model="anthropic/claude-3-opus-20240229",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0.1,
    max_tokens=50
)

async def evaluate_response(
    question: str,
    answer: str,
    contexts: Optional[List[str]] = None,
    ground_truths: Optional[List[str]] = None,
    metrics: Optional[List[Any]] = None
) -> Dict[str, float]:
    """
    Evaluate a RAG response using Ragas metrics.
    
    Args:
        question: The input question
        answer: The generated answer
        contexts: List of context passages used to generate the answer
        ground_truths: List of ground truth answers (optional)
        metrics: List of metric objects to evaluate (optional)
        
    Returns:
        Dictionary containing evaluation scores
    """
    # Validate metrics if provided
    if metrics is not None:
        if not all(hasattr(m, 'init') for m in metrics):
            raise TypeError("Invalid metrics provided. Each metric must be a valid Ragas metric object.")
    
    # Default metrics if none specified
    if metrics is None:
        metrics = [
            Faithfulness(llm=evaluation_model),
            AnswerRelevancy(llm=evaluation_model),
            ContextPrecision(llm=evaluation_model),
            ContextRecall(llm=evaluation_model),
            ContextRelevance(llm=evaluation_model)
        ]
    
    # Prepare test data with correct column names and format
    test_data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts if contexts else []],
        "ground_truths": [[gt] for gt in (ground_truths if ground_truths else [""])]
    }
    
    # Convert to pandas DataFrame first
    df = pd.DataFrame(test_data)
    
    # Convert DataFrame to Dataset
    test_dataset = Dataset.from_pandas(df)
    
    # Run evaluation
    try:
        # Create a dictionary to store results
        results = {}
        
        # Evaluate each metric individually
        for metric in metrics:
            try:
                score = await metric.single_turn_ascore(test_dataset)
                if isinstance(score, pd.Series):
                    results[metric.__class__.__name__.lower()] = float(score.iloc[0])
                else:
                    results[metric.__class__.__name__.lower()] = float(score)
            except Exception as e:
                print(f"Error evaluating metric {metric.__class__.__name__}: {str(e)}")
                results[metric.__class__.__name__.lower()] = 0.0
                
        return results
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return {}

async def batch_evaluate_responses(
    questions: List[str],
    answers: List[str],
    contexts: Optional[List[List[str]]] = None,
    ground_truths: Optional[List[List[str]]] = None,
    metrics: Optional[List[Any]] = None
) -> Dict[str, List[float]]:
    """
    Batch evaluate multiple RAG responses.
    
    Args:
        questions: List of input questions
        answers: List of generated answers
        contexts: List of context passages for each answer
        ground_truths: List of ground truth answers for each question
        metrics: List of metric objects to evaluate
        
    Returns:
        Dictionary containing evaluation scores for each response
    """
    if not questions or not answers:
        raise ValueError("Questions and answers lists cannot be empty")
        
    if len(questions) != len(answers):
        raise ValueError("Number of questions must match number of answers")
        
    if contexts and len(contexts) != len(questions):
        raise ValueError("Number of contexts must match number of questions")
        
    if ground_truths and len(ground_truths) != len(questions):
        raise ValueError("Number of ground truths must match number of questions")
    
    # Validate metrics if provided
    if metrics is not None:
        if not all(hasattr(m, 'init') for m in metrics):
            raise TypeError("Invalid metrics provided. Each metric must be a valid Ragas metric object.")
    
    # Default metrics if none specified
    if metrics is None:
        metrics = [
            Faithfulness(llm=evaluation_model),
            AnswerRelevancy(llm=evaluation_model),
            ContextPrecision(llm=evaluation_model),
            ContextRecall(llm=evaluation_model),
            ContextRelevance(llm=evaluation_model)
        ]
    
    # Prepare test data with correct column names and format
    test_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts if contexts else [[] for _ in questions],
        "ground_truths": [[gt] if gt else [""] for gt in (ground_truths if ground_truths else [[""] for _ in questions])]
    }
    
    # Convert to pandas DataFrame first
    df = pd.DataFrame(test_data)
    
    # Convert DataFrame to Dataset
    test_dataset = Dataset.from_pandas(df)
    
    # Run evaluation
    try:
        # Create a dictionary to store results
        results = {}
        
        # Evaluate each metric individually
        for metric in metrics:
            try:
                scores = await metric.single_turn_ascore(test_dataset)
                if isinstance(scores, pd.Series):
                    results[metric.__class__.__name__.lower()] = [float(score) for score in scores.tolist()]
                else:
                    results[metric.__class__.__name__.lower()] = [float(scores)] * len(questions)
            except Exception as e:
                print(f"Error evaluating metric {metric.__class__.__name__}: {str(e)}")
                results[metric.__class__.__name__.lower()] = [0.0] * len(questions)
                
        return results
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return {}