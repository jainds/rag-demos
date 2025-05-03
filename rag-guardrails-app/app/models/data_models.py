from pydantic import BaseModel

class MetricsConfig(BaseModel):
    faithfulness: bool = True
    answer_relevancy: bool = False
    context_precision: bool = True
    context_recall: bool = False
    context_relevance: bool = False

class QueryRequest(BaseModel):
    question: str
    metrics: MetricsConfig = MetricsConfig()