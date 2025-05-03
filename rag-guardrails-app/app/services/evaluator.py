import asyncio
import aiohttp
from ragas import evaluate, SingleTurnSample
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
import logging
import math
import pydantic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.exceptions import OutputParserException

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables

# Ensure OpenRouter API key is available
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

# Initialize OpenRouter for evaluation
evaluation_model = ChatOpenRouter(
    model="opengvlab/internvl3-2b:free",
    api_key=openrouter_api_key,
    temperature=0.1,
    max_tokens=1000  # Increased for evaluation purposes
)

def clean_score(score_value: float) -> float:
    """Clean score value to ensure JSON compatibility."""
    if math.isnan(score_value) or math.isinf(score_value):
        return 0.0
    return float(score_value)

def patch_metric_prompt(metric):
    # Patch prompt for metrics that have a context_precision_prompt or context_recall_prompt
    if hasattr(metric, "context_precision_prompt"):
        orig = metric.context_precision_prompt.instruction
        if "Return your answer as a JSON object" not in orig:
            metric.context_precision_prompt.instruction = (
                orig + "\n\nReturn your answer as a JSON object with the required fields."
            )
    if hasattr(metric, "context_recall_prompt"):
        orig = metric.context_recall_prompt.instruction
        if "Return your answer as a JSON object" not in orig:
            metric.context_recall_prompt.instruction = (
                orig + "\n\nReturn your answer as a JSON object with the required fields."
            )
    return metric

async def evaluate_response(
    question: str,
    answer: str,
    contexts: Optional[List[str]] = None,
    ground_truths: Optional[List[str]] = None,
    metrics: Optional[List[Any]] = None
) -> Dict[str, float]:
    """
    Evaluate a single RAG response.
    """
    logger.info("Starting evaluation with:")
    logger.info(f"Question: {question}")
    logger.info(f"Answer: {answer}")
    logger.info(f"Contexts: {contexts}")
    logger.info(f"Ground truths: {ground_truths}")
    
    # Ensure inputs are strings
    question = str(question)
    answer = str(answer)
    
    # Ensure contexts and ground_truths are lists of strings
    contexts = [str(ctx) for ctx in (contexts or [])]
    ground_truths = [str(gt) for gt in (ground_truths or [])]
    
    # Join contexts into string for metrics that expect it
    contexts_str = "\n".join(contexts)
    
    # Default metrics if none specified
    if metrics is None:
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        metrics = [
            Faithfulness(llm=evaluation_model),
            AnswerRelevancy(llm=evaluation_model, embeddings=embedder),
            patch_metric_prompt(ContextPrecision(llm=evaluation_model)),
            patch_metric_prompt(ContextRecall(llm=evaluation_model)),
            ContextRelevance(llm=evaluation_model)
        ]
        logger.info("Using default metrics:")
        for m in metrics:
            logger.info(f"- {m.__class__.__name__}")
    
    # Run evaluation
    try:
        results = {}
        
        # Evaluate each metric individually
        for metric in metrics:
            try:
                logger.info(f"Evaluating metric: {metric.__class__.__name__}")
                # Log the type of metric and input
                logger.debug(f"Metric class: {metric.__class__}")
                logger.debug(f"Metric input sample: question={question}, answer={answer}, contexts={contexts}, ground_truths={ground_truths}")
                if not hasattr(metric, 'single_turn_ascore'):
                    raise TypeError(f"Metric {metric.__class__.__name__} does not implement required method 'single_turn_ascore'")
                # Prepare row as dict of scalars
                row = {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts_str,
                    "ground_truths": ground_truths[0] if ground_truths else "",
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts_str,
                    "reference": ground_truths[0] if ground_truths else ""
                }
                # Ensure all values are strings
                for k, v in row.items():
                    if not isinstance(v, str):
                        row[k] = str(v)
                ragas_metric_classes = (
                    Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, ContextRelevance
                )
                if isinstance(metric, ragas_metric_classes):
                    sample = SingleTurnSample(
                        question=row["question"],
                        answer=row["answer"],
                        contexts=contexts,
                        ground_truths=ground_truths,
                        user_input=row["user_input"],
                        retrieved_contexts=contexts,
                        response=row["answer"],
                        reference=ground_truths[0] if ground_truths else ""
                    )
                    logger.debug(f"Calling single_turn_ascore for {metric.__class__.__name__} with SingleTurnSample: {sample}")
                    try:
                        score = await metric.single_turn_ascore(sample)
                    except AttributeError as e:
                        # Special handling for 'str' object has no attribute 'get' (context relevancy bug)
                        if "'str' object has no attribute 'get'" in str(e):
                            logger.error(f"ContextRelevance metric output type error: {e}")
                            logger.error(f"Likely cause: LLM output is a string or dict, not a list of dicts. Skipping this metric and assigning None.")
                            score = None
                        else:
                            logger.error(f"Exception in single_turn_ascore for {metric.__class__.__name__}: {e}")
                            logger.error(f"Exception type: {type(e)}")
                            logger.exception("Full traceback:")
                            score = None
                    except Exception as e:
                        logger.error(f"Exception in single_turn_ascore for {metric.__class__.__name__}: {e}")
                        logger.error(f"Exception type: {type(e)}")
                        logger.exception("Full traceback:")
                        score = None
                else:
                    logger.debug(f"Calling single_turn_ascore for {metric.__class__.__name__} with row: {row}")
                    try:
                        score = await metric.single_turn_ascore(row)
                    except Exception as e:
                        logger.error(f"Exception in single_turn_ascore for {metric.__class__.__name__} (row): {e}")
                        logger.error(f"Exception type: {type(e)}")
                        logger.exception("Full traceback:")
                        score = None
                logger.info(f"Raw score type: {type(score)}, value: {score}")
                if score is None:
                    score_value = None
                elif isinstance(score, (float, int)):
                    score_value = clean_score(float(score))
                else:
                    try:
                        score_value = clean_score(float(score[0]))
                    except Exception as e:
                        logger.error(f"Error extracting score value for {metric.__class__.__name__}: {e}")
                        logger.error(f"Score object: {score}")
                        logger.exception("Full traceback:")
                        score_value = None
                results[metric.__class__.__name__.lower()] = score_value
                logger.info(f"Final processed score for {metric.__class__.__name__}: {score_value}")
            except Exception as e:
                logger.error(f"Error evaluating metric {metric.__class__.__name__}: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.exception(f"Full traceback for {metric.__class__.__name__}:")
                results[metric.__class__.__name__.lower()] = None
        return results
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.exception("Full traceback:")
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
    """
    logger.info("Starting batch evaluation with:")
    logger.info(f"Questions length: {len(questions)}")
    logger.info(f"Answers length: {len(answers)}")
    logger.info(f"Contexts length: {len(contexts) if contexts else 0}")
    logger.info(f"Ground truths length: {len(ground_truths) if ground_truths else 0}")
    
    if not questions or not answers:
        raise ValueError("Questions and answers lists cannot be empty")
        
    if len(questions) != len(answers):
        raise ValueError("Number of questions must match number of answers")
        
    if contexts and len(contexts) != len(questions):
        raise ValueError("Number of contexts must match number of questions")
        
    if ground_truths and len(ground_truths) != len(questions):
        raise ValueError("Number of ground truths must match number of questions")
    
    # Ensure inputs are strings
    questions = [str(q) for q in questions]
    answers = [str(a) for a in answers]
    
    # Ensure contexts and ground_truths are lists of lists of strings
    contexts = [[str(ctx) for ctx in ctx_list] for ctx_list in (contexts or [[]] * len(questions))]
    ground_truths = [[str(gt) for gt in gt_list] for gt_list in (ground_truths or [[]] * len(questions))]
    
    # Join contexts into strings for metrics that expect it
    contexts_str = ["\n".join(ctx_list) for ctx_list in contexts]
    
    # Default metrics if none specified
    if metrics is None:
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        metrics = [
            Faithfulness(llm=evaluation_model),
            AnswerRelevancy(llm=evaluation_model, embeddings=embedder),
            patch_metric_prompt(ContextPrecision(llm=evaluation_model)),
            patch_metric_prompt(ContextRecall(llm=evaluation_model)),
            ContextRelevance(llm=evaluation_model)
        ]
        logger.info("Using default metrics:")
        for m in metrics:
            logger.info(f"- {m.__class__.__name__}")
    
    # Run evaluation
    try:
        results = {}
        
        # Evaluate each metric individually
        for metric in metrics:
            scores = []
            try:
                logger.info(f"Evaluating metric: {metric.__class__.__name__}")
                # Validate metric type
                if not hasattr(metric, 'single_turn_ascore'):
                    raise TypeError(f"Metric {metric.__class__.__name__} does not implement required method 'single_turn_ascore'")
                for i in range(len(questions)):
                    row = {
                        "question": questions[i],
                        "answer": answers[i],
                        "contexts": contexts_str[i],
                        "ground_truths": ground_truths[i][0] if ground_truths[i] else "",
                        "user_input": questions[i],
                        "response": answers[i],
                        "retrieved_contexts": contexts_str[i],
                        "reference": ground_truths[i][0] if ground_truths[i] else ""
                    }
                    try:
                        score = await metric.single_turn_ascore(row)
                    except Exception as e:
                        logger.info(f"Falling back to ascore for {metric.__class__.__name__}: {str(e)}")
                        df = pd.DataFrame([row])
                        try:
                            score = await metric.ascore(df)
                        except Exception as e2:
                            logger.error(f"Error in ascore fallback for {metric.__class__.__name__}: {str(e2)}")
                            logger.error(f"Error type: {type(e2)}")
                            logger.exception("Full traceback:")
                            score = None
                    if score is None:
                        scores.append(None)
                    elif isinstance(score, pd.Series):
                        scores.append(clean_score(float(score.iloc[0])))
                    elif isinstance(score, (float, int)):
                        scores.append(clean_score(float(score)))
                    else:
                        try:
                            scores.append(clean_score(float(score[0])))
                        except Exception as e:
                            logger.error(f"Error extracting score value for {metric.__class__.__name__}: {e}")
                            logger.error(f"Score object: {score}")
                            logger.exception("Full traceback:")
                            scores.append(None)
                logger.info(f"Final processed scores for {metric.__class__.__name__}: {scores}")
            except (NotImplementedError, pydantic.ValidationError, TypeError, ValueError, AttributeError, OutputParserException) as e:
                logger.error(f"Error evaluating metric {metric.__class__.__name__}: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.exception("Full traceback:")
                scores = [None] * len(questions)
            finally:
                results[metric.__class__.__name__.lower()] = scores
        return results
    except Exception as e:
        logger.error(f"Batch evaluation error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.exception("Full traceback:")
        return {}