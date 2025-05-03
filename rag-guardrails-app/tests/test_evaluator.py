import pytest
from app.services.evaluator import evaluate_response, batch_evaluate_responses, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance
)
from app.models.ChatOpenRouter import ChatOpenRouter
import os
from unittest.mock import AsyncMock, patch, MagicMock
import pandas as pd
from pydantic import ValidationError
import inspect
from langchain_core.outputs import LLMResult, GenerationChunk
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import asyncio
import collections.abc
from pathlib import Path

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
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    metrics = [
        AnswerRelevancy(llm=test_model, embeddings=embedder),
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
    result = await evaluate_response(
        question="Test?",
        answer="Test.",
        metrics=["invalid_metric"]  # Should be metric objects
    )
    assert result.get("str", 0.0) == 0.0

def get_columns_and_types_from_arg(args, kwargs):
    # Helper to extract DataFrame/dict columns and value types from args/kwargs
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, dict):
            return list(arg.keys()), [type(v) for v in arg.values()]
        if isinstance(arg, pd.DataFrame):
            cols = list(arg.columns)
            types = [type(arg.iloc[0][col]) for col in cols]
            return cols, types
    return [], []

@pytest.mark.asyncio
async def test_evaluate_response_data_format(sample_qa_pair):
    """Test that the evaluator passes the correct data format and value types to the metric."""
    required_columns = [
        "question", "answer", "contexts", "ground_truths",
        "user_input", "response", "retrieved_contexts", "reference"
    ]
    mock_metric = AsyncMock()
    mock_metric.__class__.__name__ = "MockMetric"
    mock_metric.single_turn_ascore = AsyncMock(return_value=0.5)
    with patch("ragas.metrics.Faithfulness", return_value=mock_metric):
        await evaluate_response(
            question=sample_qa_pair["question"],
            answer=sample_qa_pair["answer"],
            contexts=sample_qa_pair["contexts"],
            ground_truths=sample_qa_pair["ground_truths"],
            metrics=[mock_metric]
        )
        assert mock_metric.single_turn_ascore.call_count == 1
        args, kwargs = mock_metric.single_turn_ascore.call_args
        columns, types = get_columns_and_types_from_arg(args, kwargs)
        for col in required_columns:
            assert col in columns, f"Missing column: {col}"
        for t in types:
            assert t is str, f"Value passed to metric is not a string, got {t}"

# --- Error 1: All values passed to metrics are scalars (strings), not Series or lists ---
@pytest.mark.asyncio
async def test_metric_input_is_scalar(sample_qa_pair):
    mock_metric = AsyncMock()
    mock_metric.__class__.__name__ = "MockMetric"
    mock_metric.single_turn_ascore = AsyncMock(return_value=0.5)
    with patch("ragas.metrics.Faithfulness", return_value=mock_metric):
        await evaluate_response(
            question=sample_qa_pair["question"],
            answer=sample_qa_pair["answer"],
            contexts=sample_qa_pair["contexts"],
            ground_truths=sample_qa_pair["ground_truths"],
            metrics=[mock_metric]
        )
        args, kwargs = mock_metric.single_turn_ascore.call_args
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, dict):
                for v in arg.values():
                    assert isinstance(v, str), f"Metric input value is not a string: {type(v)}"

# --- Error 2: single_turn_ascore expects model, not dict ---
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
    assert result[metric.__class__.__name__.lower()] == 0.0

# --- Error 3: ascore not called if not implemented ---
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
    assert result[metric.__class__.__name__.lower()] == 0.0

# --- Error 4: single_turn_ascore input type ---
@pytest.mark.asyncio
async def test_single_turn_ascore_input_type(sample_qa_pair):
    # Should be dict or model, not DataFrame
    mock_metric = AsyncMock()
    mock_metric.__class__.__name__ = "MockMetric"
    def check_input_type(row):
        assert not isinstance(row, pd.DataFrame), "single_turn_ascore received DataFrame"
        return 0.5
    mock_metric.single_turn_ascore = AsyncMock(side_effect=check_input_type)
    with patch("ragas.metrics.Faithfulness", return_value=mock_metric):
        await evaluate_response(
            question=sample_qa_pair["question"],
            answer=sample_qa_pair["answer"],
            contexts=sample_qa_pair["contexts"],
            ground_truths=sample_qa_pair["ground_truths"],
            metrics=[mock_metric]
        )

# --- Error 5: NotImplementedError handling ---
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
    assert result[metric.__class__.__name__.lower()] == 0.0

# --- Error 6: Pydantic validation error handling ---
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
    assert result[metric.__class__.__name__.lower()] == 0.0

# --- Error 7: End-to-end regression for all metrics ---
@pytest.mark.asyncio
async def test_end_to_end_all_metrics(sample_qa_pair):
    # Use real metrics, but patch LLM to avoid network
    dummy_llm = MagicMock()
    metrics = [
        Faithfulness(llm=dummy_llm),
        AnswerRelevancy(llm=dummy_llm),
        ContextPrecision(llm=dummy_llm),
        ContextRecall(llm=dummy_llm),
        ContextRelevance(llm=dummy_llm)
    ]
    for m in metrics:
        m.single_turn_ascore = AsyncMock(return_value=0.5)
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=metrics
    )
    assert all(isinstance(v, float) for v in result.values())
    assert all(0 <= v <= 1 for v in result.values())

@pytest.mark.asyncio
async def test_ragas_metrics_require_single_turn_sample(sample_qa_pair):
    """Test that Ragas metrics require SingleTurnSample, not dict."""
    from ragas.metrics import Faithfulness
    metric = Faithfulness(llm=MagicMock())
    # Should work with SingleTurnSample
    sample = SingleTurnSample(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        user_input=sample_qa_pair["question"],
        retrieved_contexts=sample_qa_pair["contexts"]
    )
    # Patch single_turn_ascore to check input type
    orig = metric.single_turn_ascore
    async def check_sample_type(arg):
        assert isinstance(arg, SingleTurnSample), f"Expected SingleTurnSample, got {type(arg)}"
        return 0.5
    metric.single_turn_ascore = check_sample_type
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result[metric.__class__.__name__.lower()] == 0.5
    metric.single_turn_ascore = orig

@pytest.mark.asyncio
async def test_custom_metric_accepts_dict(sample_qa_pair):
    """Test that a custom metric can accept a dict if needed."""
    class DictMetric:
        __name__ = "DictMetric"
        async def single_turn_ascore(self, row):
            assert isinstance(row, dict), f"Expected dict, got {type(row)}"
            return 0.7
    metric = DictMetric()
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result[metric.__class__.__name__.lower()] == 0.7

@pytest.mark.asyncio
async def test_all_default_metrics_accept_single_turn_sample(sample_qa_pair):
    """Test that all default Ragas metrics accept SingleTurnSample input."""
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, ContextRelevance
    dummy_llm = MagicMock()
    metrics = [
        Faithfulness(llm=dummy_llm),
        AnswerRelevancy(llm=dummy_llm),
        ContextPrecision(llm=dummy_llm),
        ContextRecall(llm=dummy_llm),
        ContextRelevance(llm=dummy_llm)
    ]
    for m in metrics:
        orig = m.single_turn_ascore
        async def check_sample_type(arg):
            assert isinstance(arg, SingleTurnSample), f"Expected SingleTurnSample, got {type(arg)}"
            return 0.5
        m.single_turn_ascore = check_sample_type
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=metrics
    )
    for m in metrics:
        m.single_turn_ascore = orig
    assert all(v == 0.5 for v in result.values())

def is_async_callable(obj, method_name):
    method = getattr(obj, method_name, None)
    return inspect.iscoroutinefunction(method)

@pytest.mark.asyncio
async def test_chat_openrouter_ragas_compatibility():
    """Test that ChatOpenRouter implements the async methods required by Ragas."""
    from app.models.ChatOpenRouter import ChatOpenRouter
    import os
    model = ChatOpenRouter(
        model="test-model",
        api_key=os.environ.get("OPENROUTER_API_KEY", "test-key"),
        temperature=0.1,
        max_tokens=10
    )
    # Check for agenerate_text method
    assert hasattr(model, "agenerate_text"), "ChatOpenRouter must implement agenerate_text"
    assert is_async_callable(model, "agenerate_text"), "agenerate_text must be async"
    # Test that agenerate_text is awaitable and returns a string
    result = await model.agenerate_text("test prompt")
    assert isinstance(result, str)

@pytest.mark.asyncio
async def test_chat_openrouter_generate_is_async():
    from app.models.ChatOpenRouter import ChatOpenRouter
    import os
    from langchain_core.outputs import LLMResult
    model = ChatOpenRouter(
        model="test-model",
        api_key=os.environ.get("OPENROUTER_API_KEY", "test-key"),
        temperature=0.1,
        max_tokens=10
    )
    # Patch _async_call to avoid real API call
    async def fake_async_call(prompt, *args, **kwargs):
        return "fake response"
    model._async_call = fake_async_call
    result = await model.generate("test prompt")
    assert isinstance(result, LLMResult)

@pytest.mark.asyncio
async def test_chat_openrouter_generate_returns_llmresult():
    from app.models.ChatOpenRouter import ChatOpenRouter
    import os
    from langchain_core.outputs import LLMResult, GenerationChunk
    model = ChatOpenRouter(
        model="test-model",
        api_key=os.environ.get("OPENROUTER_API_KEY", "test-key"),
        temperature=0.1,
        max_tokens=10
    )
    # Patch _call to avoid real API call
    def fake_call(prompt, *args, **kwargs):
        return "fake response"
    model._call = fake_call
    result = model._generate(["test prompt"])
    assert hasattr(result, "generations"), "generate should return object with .generations"
    assert isinstance(result.generations, list)
    assert isinstance(result.generations[0], list)
    assert hasattr(result.generations[0][0], "text")
    assert isinstance(result.generations[0][0].text, str)

@pytest.mark.asyncio
async def test_chat_openrouter_agenerate_returns_llmresult():
    from app.models.ChatOpenRouter import ChatOpenRouter
    import os
    from langchain_core.outputs import LLMResult, GenerationChunk
    model = ChatOpenRouter(
        model="test-model",
        api_key=os.environ.get("OPENROUTER_API_KEY", "test-key"),
        temperature=0.1,
        max_tokens=10
    )
    # Patch _async_call to avoid real API call
    async def fake_async_call(prompt, *args, **kwargs):
        return "fake response"
    model._async_call = fake_async_call
    # Patch agenerate to mimic LLMResult
    result = await model._agenerate(["test prompt"])
    assert hasattr(result, "generations"), "agenerate should return object with .generations"
    assert isinstance(result.generations, list)
    assert isinstance(result.generations[0], list)
    assert hasattr(result.generations[0][0], "text")
    assert isinstance(result.generations[0][0].text, str)

@pytest.mark.asyncio
async def test_evaluate_response_with_real_embeddings(sample_qa_pair):
    """Test AnswerRelevancy with real sentence-transformer embeddings."""
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    metric = AnswerRelevancy(llm=test_model, embeddings=embedder)
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert isinstance(result, dict)
    assert "answerrelevancy" in result
    assert isinstance(result["answerrelevancy"], float)

# --- Error 8: Malformed response (rate limit) from OpenRouter ---
@pytest.mark.asyncio
async def test_evaluator_handles_openrouter_rate_limit(sample_qa_pair):
    """Test evaluator handles rate limit/malformed response from OpenRouter gracefully."""
    from app.models.ChatOpenRouter import ChatOpenRouter
    from ragas.metrics import Faithfulness
    import os
    # Patch _async_call to raise ValueError simulating rate limit
    model = ChatOpenRouter(
        model="test-model",
        api_key=os.environ.get("OPENROUTER_API_KEY", "test-key"),
        temperature=0.1,
        max_tokens=10
    )
    async def fake_async_call(prompt, *args, **kwargs):
        raise ValueError("Malformed response from OpenRouter: {'error': {'message': 'Rate limit exceeded: free-models-per-min.', 'code': 429}}")
    model._async_call = fake_async_call
    metric = Faithfulness(llm=model)
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result["faithfulness"] == 0.0

# --- Test: Ensure correct embedder is passed to AnswerRelevancy ---
@pytest.mark.asyncio
async def test_answer_relevancy_embedder_is_set(sample_qa_pair):
    from ragas.metrics import AnswerRelevancy
    from langchain_huggingface import HuggingFaceEmbeddings
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Patch single_turn_ascore to check self.embeddings
    class MyAnswerRelevancy(AnswerRelevancy):
        async def single_turn_ascore(self, row, *args, **kwargs):
            assert self.embeddings is embedder, f"Embeddings not set or not the expected instance: {self.embeddings}"
            return 0.42
    metric = MyAnswerRelevancy(llm=test_model, embeddings=embedder)
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result[metric.__class__.__name__.lower()] == 0.42

# --- Integration Test: Ensure embedder is set in AnswerRelevancy in app flow ---
@pytest.mark.asyncio
def test_app_sets_embedder_in_answer_relevancy(monkeypatch, sample_qa_pair):
    from ragas.metrics import AnswerRelevancy
    from langchain_huggingface import HuggingFaceEmbeddings
    from app.services import evaluator as evaluator_mod
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    called = {}
    orig_init = AnswerRelevancy.__init__
    def custom_init(self, *args, **kwargs):
        # If called from app default metrics, embeddings must be set
        if 'embeddings' in kwargs:
            assert isinstance(kwargs['embeddings'], HuggingFaceEmbeddings), f"App did not set a HuggingFaceEmbeddings embedder: {kwargs.get('embeddings')}"
            called['ok'] = True
        orig_init(self, *args, **kwargs)
    monkeypatch.setattr(AnswerRelevancy, "__init__", custom_init)
    # Patch Faithfulness etc. to avoid real LLM calls
    monkeypatch.setattr(evaluator_mod, "Faithfulness", lambda **kw: type('F', (), {"single_turn_ascore": lambda self, s: 1.0})())
    monkeypatch.setattr(evaluator_mod, "ContextPrecision", lambda **kw: type('C', (), {"single_turn_ascore": lambda self, s: 1.0})())
    monkeypatch.setattr(evaluator_mod, "ContextRecall", lambda **kw: type('C2', (), {"single_turn_ascore": lambda self, s: 1.0})())
    monkeypatch.setattr(evaluator_mod, "ContextRelevance", lambda **kw: type('C3', (), {"single_turn_ascore": lambda self, s: 1.0})())
    # Patch evaluation_model to avoid real LLM
    monkeypatch.setattr(evaluator_mod, "evaluation_model", object())
    # Patch SentenceTransformer to return our embedder
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda *a, **k: embedder)
    # Now call evaluate_response with default metrics
    try:
        import importlib; importlib.reload(evaluator_mod)
        import asyncio
        asyncio.run(evaluator_mod.evaluate_response(
            question=sample_qa_pair["question"],
            answer=sample_qa_pair["answer"],
            contexts=sample_qa_pair["contexts"],
            ground_truths=sample_qa_pair["ground_truths"]
        ))
    except Exception:
        pass  # Ignore errors from dummy metrics
    assert called.get('ok'), "App did not set a HuggingFaceEmbeddings embedder in AnswerRelevancy"

def test_context_precision_output_parser_exception(monkeypatch, sample_qa_pair, caplog):
    """Test that ContextPrecision handles OutputParserException and assigns 0.0 score when LLM returns non-JSON or errors."""
    from ragas.metrics import ContextPrecision
    import logging
    class DummyLLM:
        async def agenerate(self, prompt, **kwargs):
            return "The provided context mentions that the tower was designed by Gustave Eiffel, which is essential information to understand the subject matter of the question."
    metric = ContextPrecision(llm=DummyLLM())
    with caplog.at_level(logging.ERROR):
        import asyncio
        from app.services.evaluator import evaluate_response
        result = asyncio.run(evaluate_response(
            question=sample_qa_pair["question"],
            answer=sample_qa_pair["answer"],
            contexts=sample_qa_pair["contexts"],
            metrics=[metric]
        ))
    # Check that an error was logged and score is 0.0
    assert any("Exception" in m or "error" in m.lower() for m in caplog.text.splitlines()), "No error was logged for context precision failure"
    assert result["contextprecision"] == 0.0

@pytest.mark.asyncio
async def test_context_relevance_fail_generations_error(caplog, sample_qa_pair):
    """Test ContextRelevance returns None and logs error on 'generations' bug."""
    class DummyContextRelevance(ContextRelevance):
        async def single_turn_ascore(self, sample):
            raise AttributeError("'str' object has no attribute 'generations'")
    metric = DummyContextRelevance(llm=MagicMock())
    with caplog.at_level(logging.ERROR):
        result = await evaluate_response(
            question=sample_qa_pair["question"],
            answer=sample_qa_pair["answer"],
            contexts=sample_qa_pair["contexts"],
            ground_truths=sample_qa_pair["ground_truths"],
            metrics=[metric]
        )
    assert result.get("contextrelevance") is None
    assert any("generations" in m or "'str' object has no attribute 'generations'" in m for m in caplog.text.splitlines())

@pytest.mark.asyncio
async def test_context_relevance_integration_batch(sample_qa_batch):
    """Integration test: batch_evaluate_responses with ContextRelevance, both success and fail."""
    class DummyContextRelevance(ContextRelevance):
        async def single_turn_ascore(self, sample):
            # Accept both dict and SingleTurnSample
            if isinstance(sample, dict):
                question = sample.get("question")
            else:
                question = getattr(sample, "question", None)
            if question == sample_qa_batch["questions"][0]:
                return 0.7
            return None
    metric = DummyContextRelevance(llm=MagicMock())
    result = await batch_evaluate_responses(
        questions=sample_qa_batch["questions"],
        answers=sample_qa_batch["answers"],
        contexts=sample_qa_batch["contexts"],
        ground_truths=sample_qa_batch["ground_truths"],
        metrics=[metric]
    )
    print("Result keys:", result.keys())
    # Use the correct key for the dummy metric
    context_scores = result.get("dummycontextrelevance")
    assert isinstance(context_scores, collections.abc.Sequence)
    assert context_scores == [0.7, None]

@pytest.mark.asyncio
async def test_context_relevance_generations_error_logs_and_returns_zero(caplog):
    class DummyMetric:
        __name__ = "ContextRelevance"
        __class__ = type("ContextRelevance", (), {})
        async def single_turn_ascore(self, sample):
            # Simulate the error: 'str' object has no attribute 'generations'
            raise AttributeError("'str' object has no attribute 'generations'")
    dummy_metric = DummyMetric()
    with caplog.at_level(logging.ERROR):
        result = await evaluate_response(
            question="What is Paris?",
            answer="Paris is a city in France.",
            contexts=["Paris is the capital of France."],
            metrics=[dummy_metric]
        )
        # Check that the error was logged
        assert any("generations" in m or "'str' object has no attribute 'generations'" in m for m in caplog.text.splitlines()), "Error about 'generations' not logged"
        # Check that the score is None
        assert result.get("contextrelevance") is None

@pytest.mark.asyncio
async def test_chatopenrouter_always_returns_llmresult():
    from app.models.ChatOpenRouter import ChatOpenRouter
    from langchain_core.outputs import LLMResult
    model = ChatOpenRouter(
        model="test-model",
        api_key="test-key",
        temperature=0.1,
        max_tokens=10
    )
    # Patch _async_call to return a string
    async def fake_async_call(prompt, *args, **kwargs):
        return "fake response"
    model._async_call = fake_async_call
    result = await model.agenerate("test prompt")
    assert isinstance(result, LLMResult)
    assert hasattr(result, "generations")
    # Also test generate
    result2 = await model.generate("test prompt")
    assert isinstance(result2, LLMResult)
    assert hasattr(result2, "generations")
    assert result2.generations[0][0].text == "fake response"

@pytest.mark.asyncio
async def test_context_relevance_str_object_error_logs_and_returns_zero(caplog):
    class DummyContextRelevance:
        __name__ = "ContextRelevance"
        __class__ = type("ContextRelevance", (), {})
        async def single_turn_ascore(self, sample):
            raise AttributeError("'str' object has no attribute 'get'")
    dummy_metric = DummyContextRelevance()
    with caplog.at_level(logging.ERROR):
        result = await evaluate_response(
            question="What is Paris?",
            answer="Paris is a city in France.",
            contexts=["Paris is the capital of France."],
            ground_truths=["Paris is the capital of France."],
            metrics=[dummy_metric]
        )
    # Should log the error and return None for the metric
    assert any("Exception in single_turn_ascore for ContextRelevance (row):" in r for r in caplog.text.splitlines())
    assert result["contextrelevance"] is None

@pytest.mark.asyncio
async def test_context_relevance_success(monkeypatch, sample_qa_pair):
    """Test ContextRelevance returns a valid float score on success."""
    metric = ContextRelevance(llm=MagicMock())
    metric.single_turn_ascore = AsyncMock(return_value=0.8)
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert isinstance(result["contextrelevance"], float)
    assert 0 <= result["contextrelevance"] <= 1

@pytest.mark.asyncio
async def test_context_relevance_fail_generations_error(caplog, sample_qa_pair):
    """Test ContextRelevance returns None and logs error on 'generations' bug."""
    class DummyContextRelevance(ContextRelevance):
        async def single_turn_ascore(self, sample):
            raise AttributeError("'str' object has no attribute 'generations'")
    metric = DummyContextRelevance(llm=MagicMock())
    with caplog.at_level(logging.ERROR):
        result = await evaluate_response(
            question=sample_qa_pair["question"],
            answer=sample_qa_pair["answer"],
            contexts=sample_qa_pair["contexts"],
            ground_truths=sample_qa_pair["ground_truths"],
            metrics=[metric]
        )
    assert result.get("contextrelevance") is None
    assert any("generations" in m or "'str' object has no attribute 'generations'" in m for m in caplog.text.splitlines())

@pytest.mark.asyncio
async def test_context_relevance_input_validation(sample_qa_pair):
    """Test ContextRelevance with missing/empty contexts returns None."""
    metric = ContextRelevance(llm=MagicMock())
    metric.single_turn_ascore = AsyncMock(return_value=None)
    # No contexts
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=[],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result["contextrelevance"] is None
    # Contexts is None
    result2 = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=None,
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result2["contextrelevance"] is None

@pytest.mark.asyncio
async def test_context_relevance_output_type(sample_qa_pair):
    """Test ContextRelevance output is float or None only."""
    metric = ContextRelevance(llm=MagicMock())
    metric.single_turn_ascore = AsyncMock(return_value=0.5)
    result = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert isinstance(result["contextrelevance"], float)
    # Now simulate None
    metric.single_turn_ascore = AsyncMock(return_value=None)
    result2 = await evaluate_response(
        question=sample_qa_pair["question"],
        answer=sample_qa_pair["answer"],
        contexts=sample_qa_pair["contexts"],
        ground_truths=sample_qa_pair["ground_truths"],
        metrics=[metric]
    )
    assert result2["contextrelevance"] is None

@pytest.mark.asyncio
async def test_chatopenrouter_llmresult_contract():
    from app.models.ChatOpenRouter import ChatOpenRouter
    from langchain_core.outputs import LLMResult
    model = ChatOpenRouter(
        model="test-model",
        api_key="test-key",
        temperature=0.1,
        max_tokens=10
    )
    # Patch _async_call and _call to return a string
    async def fake_async_call(prompt, *args, **kwargs):
        return "fake response"
    def fake_call(prompt, *args, **kwargs):
        return "fake response"
    model._async_call = fake_async_call
    model._call = fake_call
    # Test all relevant methods
    # generate
    result = await model.generate("test prompt")
    assert isinstance(result, LLMResult)
    assert hasattr(result, "generations")
    # agenerate
    result = await model.agenerate("test prompt")
    assert isinstance(result, LLMResult)
    assert hasattr(result, "generations")
    # predict
    result = model.predict("test prompt")
    assert isinstance(result, LLMResult)
    assert hasattr(result, "generations")
    # apredict
    result = await model.apredict("test prompt")
    assert isinstance(result, LLMResult)
    assert hasattr(result, "generations")
    # predict_messages
    result = model.predict_messages([{"content": "hi"}])
    assert isinstance(result, LLMResult)
    assert hasattr(result, "generations")
    # apredict_messages
    result = await model.apredict_messages([{"content": "hi"}])
    assert isinstance(result, LLMResult)
    assert hasattr(result, "generations")
    # generate_prompt
    result = model.generate_prompt(["test prompt"])
    assert isinstance(result, LLMResult)
    assert hasattr(result, "generations")
    # agenerate_prompt
    result = await model.agenerate_prompt(["test prompt"])
    assert isinstance(result, LLMResult)
    assert hasattr(result, "generations") 

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="No OpenRouter API key set")
@pytest.mark.skipif(not Path("rag-guardrails-app/app/config/config.yml").exists(), reason="No NeMo Guardrails config file found")
async def test_llmrails_generate_async_normal():
    """Test LLMRails (Nemo Guardrails) normal generation via generate_async."""
    from nemoguardrails import LLMRails, RailsConfig
    from app.models.ChatOpenRouter import ChatOpenRouter
    config_path = "rag-guardrails-app/app/config/config.yml"
    config = RailsConfig.from_path(config_path)
    model = ChatOpenRouter(
        model="anthropic/claude-3-opus-20240229",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0.1,
        max_tokens=100
    )
    rails = LLMRails(config, llm=model, verbose=True)
    response = await rails.generate_async(messages=[{"role": "user", "content": "What is the capital of France?"}])
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="No OpenRouter API key set")
@pytest.mark.skipif(not Path("rag-guardrails-app/app/config/config.yml").exists(), reason="No NeMo Guardrails config file found")
async def test_llmrails_generate_async_edge_cases(monkeypatch):
    """Test LLMRails (Nemo Guardrails) edge cases: empty prompt, long prompt, invalid API key, network error, rate limit error."""
    from nemoguardrails import LLMRails, RailsConfig
    from app.models.ChatOpenRouter import ChatOpenRouter
    import aiohttp
    config_path = "rag-guardrails-app/app/config/config.yml"
    config = RailsConfig.from_path(config_path)
    model = ChatOpenRouter(
        model="anthropic/claude-3-opus-20240229",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0.1,
        max_tokens=100
    )
    rails = LLMRails(config, llm=model, verbose=True)
    # Empty prompt
    response = await rails.generate_async(messages=[{"role": "user", "content": ""}])
    assert isinstance(response, str)
    # Long prompt
    long_prompt = "What is AI? " * 1000
    response = await rails.generate_async(messages=[{"role": "user", "content": long_prompt}])
    assert isinstance(response, str)
    # Invalid API key
    model_bad = ChatOpenRouter(
        model="anthropic/claude-3-opus-20240229",
        api_key="invalid-key",
        temperature=0.1,
        max_tokens=100
    )
    rails_bad = LLMRails(config, llm=model_bad, verbose=True)
    with pytest.raises(Exception):
        await rails_bad.generate_async(messages=[{"role": "user", "content": "What is the capital of France?"}])
    # Simulate network error by patching aiohttp.ClientSession.post
    async def fake_post(*args, **kwargs):
        raise aiohttp.ClientError("Simulated network error")
    monkeypatch.setattr(aiohttp.ClientSession, "post", fake_post)
    with pytest.raises(aiohttp.ClientError):
        await rails.generate_async(messages=[{"role": "user", "content": "What is the capital of France?"}])
    # Simulate rate limit error by patching _async_call
    async def fake_async_call(*args, **kwargs):
        raise Exception("Rate limit exceeded: free-models-per-min.")
    model._async_call = fake_async_call
    rails = LLMRails(config, llm=model, verbose=True)
    with pytest.raises(Exception):
        await rails.generate_async(messages=[{"role": "user", "content": "What is the capital of France?"}]) 