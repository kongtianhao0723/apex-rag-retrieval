from apex_rag.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from apex_rag.ingestion.chunking import original_chunks
from apex_rag.models import Document
from apex_rag.retrieval.engine import RetrievalEngine


def test_metrics_and_engine() -> None:
    documents = [
        Document("D1", "Fee", "Fees", "Apex", 2024, "The fee is three percent."),
        Document("D2", "APR", "Interest", "Apex", 2024, "The APR is variable."),
    ]
    engine = RetrievalEngine(original_chunks(documents))
    results = engine.hybrid_search("What is the fee?", final_top_n=2)
    relevant = {"D1"}
    assert recall_at_k(results, relevant, 1) == 1.0
    assert precision_at_k(results, relevant, 1) == 1.0
    assert reciprocal_rank(results, relevant) == 1.0
    assert ndcg_at_k(results, relevant, 1) == 1.0
