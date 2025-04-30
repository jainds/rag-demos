from app.services.rag import rag_pipeline

def test_rag_pipeline():
    answer = rag_pipeline("What is the Eiffel Tower?")
    assert isinstance(answer, str)
    assert len(answer) > 0
