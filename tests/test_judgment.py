import pytest

from apex_rag.answering import interpret_judgment


def test_interpret_judgment_all_relevant_and_faithful() -> None:
    raw = """
    {
      "chunks": [
        {"doc_id": "DOC-1", "relevant": true},
        {"doc_id": "DOC-2", "relevant": true}
      ],
      "claims": [
        {"claim": "The fee is 3 percent.", "supported": true},
        {"claim": "The offer lasts 15 cycles.", "supported": true}
      ]
    }
    """
    judgment = interpret_judgment(raw, ["DOC-1", "DOC-2"], "The fee is 3 percent for 15 cycles.")
    assert judgment.context_relevance == 1.0
    assert judgment.faithfulness == 1.0
    assert judgment.error == ""
    assert [item["relevant"] for item in judgment.chunk_verdicts] == [True, True]


def test_interpret_judgment_partial_support() -> None:
    raw = """```json
    {"chunks": [{"doc_id": "DOC-1", "relevant": true}, {"doc_id": "DOC-2", "relevant": false}],
     "claims": [{"claim": "The fee is 3 percent.", "supported": true},
                {"claim": "The annual fee is waived.", "supported": false}]}
    ```"""
    judgment = interpret_judgment(
        raw, ["DOC-1", "DOC-2"], "The fee is 3 percent and the annual fee is waived."
    )
    assert judgment.context_relevance == 0.5
    assert judgment.faithfulness == 0.5
    assert judgment.chunk_verdicts[1]["doc_id"] == "DOC-2"


def test_interpret_judgment_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        interpret_judgment("the context looks relevant", ["DOC-1"], "The fee is 3 percent.")
