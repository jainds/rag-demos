import pytest
import logging
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import collections.abc
import os
from pathlib import Path
from pydantic import ValidationError
from app.services.evaluator import evaluate_response, batch_evaluate_responses, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance
)

@pytest.mark.asyncio
async def test_single_turn_ascore_requires_model(sample_qa_pair):
    # Simulate metric that expects a model, not dict
    class ModelOnlyMetric:
        __name__ = "ModelOnlyMetric"
        async def single_turn_ascore(self, row):
            if not hasattr(row, 'model_dump'):
                raise AttributeError('model_dump')
            return 0.5
    metric = ModelOnlyMetric()
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result[metric.__class__.__name__.lower()] is None # Expect None on error

@pytest.mark.asyncio
async def test_ascore_not_called_on_metrics_without_implementation(sample_qa_pair):
    class NoAscoreMetric:
        __name__ = "NoAscoreMetric"
        async def single_turn_ascore(self, row):
            raise NotImplementedError("No ascore")
    metric = NoAscoreMetric()
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result[metric.__class__.__name__.lower()] is None # Expect None on error

@pytest.mark.asyncio
async def test_evaluator_handles_not_implemented_error(sample_qa_pair):
    class NotImplMetric:
        __name__ = "NotImplMetric"
        async def single_turn_ascore(self, row):
            raise NotImplementedError("not implemented")
    metric = NotImplMetric()
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result[metric.__class__.__name__.lower()] is None # Expect None on error

@pytest.mark.asyncio
async def test_evaluator_handles_pydantic_validation_error(sample_qa_pair):
    class PydanticMetric:
        __name__ = "PydanticMetric"
        async def single_turn_ascore(self, row):
            raise ValidationError([{'loc': ('question',), 'msg': 'Input should be a valid string', 'type': 'string_type'}], type('FakeModel', (), {}))
    metric = PydanticMetric()
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result[metric.__class__.__name__.lower()] is None # Expect None on error

@pytest.mark.asyncio
async def test_evaluator_handles_openrouter_rate_limit(sample_qa_pair):
    """Test evaluator handles rate limit/malformed response from OpenRouter gracefully."""
    # ... (rest of test setup)
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result["faithfulness"] is None # Expect None on error

@pytest.mark.asyncio
def test_context_precision_output_parser_exception(monkeypatch, sample_qa_pair, caplog):
    """Test that ContextPrecision handles OutputParserException and assigns None score when LLM returns non-JSON or errors."""
    # ... (rest of test setup)
    # Check that an error was logged and score is None
    assert any("Exception" in m or "error" in m.lower() for m in caplog.text.splitlines()), "No error was logged for context precision failure"
    assert result["contextprecision"] is None # Expect None on error

@pytest.mark.asyncio
async def test_evaluate_response(sample_qa_pair):
    """Test single response evaluation"""
    # Mock the LLM calls within the metrics for this test
    with patch('app.services.evaluator.evaluation_model', new_callable=AsyncMock) as mock_llm:
        # Configure mock LLM to return different things for different metrics if needed
        # For now, let's assume some metrics succeed and one (ContextPrecision) fails
        async def mock_agenerate(*args, **kwargs):
            # Simulate successful float returns for most
            if "faithfulness" in args[0][0].content.lower(): return MagicMock(generations=[[MagicMock(text='{"verdict": "1"}')]])
            if "answer relevancy" in args[0][0].content.lower(): return MagicMock(generations=[[MagicMock(text='{"reason": "", "verdict": "0.6"}')]])
            if "context precision" in args[0][0].content.lower(): return MagicMock(generations=[[MagicMock(text='This is not JSON')]]) # Simulate failure
            if "context recall" in args[0][0].content.lower(): return MagicMock(generations=[[MagicMock(text='{"reason": "", "verdict": "1"}')]])
            if "context relevance" in args[0][0].content.lower(): return MagicMock(generations=[[MagicMock(text='1.0')]])
            return MagicMock(generations=[[MagicMock(text='0.5')]]) # Default
        mock_llm.agenerate = mock_agenerate

        result = await evaluate_response(
            question=sample_qa_pair["question"],
            answer=sample_qa_pair["answer"],
            contexts=sample_qa_pair["contexts"],
            ground_truths=sample_qa_pair["ground_truths"]
        )

    assert isinstance(result, dict)
    # Allow None values for failed metrics (like contextprecision in this mock)
    assert all(score is None or isinstance(score, (float, int)) for score in result.values())

    # Check specific keys exist (adjust based on expected default metrics)
    expected_keys = ['faithfulness', 'answerrelevancy', 'contextprecision', 'contextrecall', 'contextrelevance']
    print("Result dictionary:", result)
    for key in expected_keys:
        assert key in result, f"Expected key '{key}' not found in results"

    # Optionally, check specific values if the mock is predictable
    assert isinstance(result.get('faithfulness'), float)
    assert result.get('contextprecision') is None # This one failed in the mock
    assert isinstance(result.get('contextrecall'), float)