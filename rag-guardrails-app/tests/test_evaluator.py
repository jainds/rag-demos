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
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    from langchain_community.embeddings import HuggingFaceEmbeddings
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
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from app.services import evaluator as evaluator_mod
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    called = {}
    orig_init = AnswerRelevancy.__init__
    def custom_init(self, *args, **kwargs):
        # If called from app default metrics, embeddings must be set
        if 'embeddings' in kwargs:
            assert kwargs['embeddings'] is embedder, f"App did not set the correct embedder: {kwargs.get('embeddings')}"
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
    assert called.get('ok'), "App did not set the correct embedder in AnswerRelevancy"

def test_patch_metric_prompt_adds_json_instruction():
    from ragas.metrics import ContextPrecision, ContextRecall
    from app.services.evaluator import patch_metric_prompt
    metric1 = ContextPrecision(llm=None)
    metric2 = ContextRecall(llm=None)
    patched1 = patch_metric_prompt(metric1)
    patched2 = patch_metric_prompt(metric2)
    assert "Return your answer as a JSON object" in patched1.context_precision_prompt.instruction
    assert "Return your answer as a JSON object" in patched2.context_recall_prompt.instruction 