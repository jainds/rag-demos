from ragas import evaluate
from ragas.test_dataset import TestDataset

def evaluate_response(question, answer):
    # Load test dataset
    # test_data = TestDataset.from_json("data/test_dataset.json")
    test_data = TestDataset.from_dicts([{"question": question, "answer": answer, "ground_truths": ["ground truth"]}]) # placeholder
    
    # Evaluate
    results = evaluate(
        test_data.to_pandas(),
        metrics=["faithfulness", "answer_relevancy", "context_relevancy", "context_precision", "context_recall"]
    )
    return results