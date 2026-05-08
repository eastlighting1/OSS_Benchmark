from __future__ import annotations

import json

from src.datasets.multihoprag import load_multihoprag_documents, load_multihoprag_questions


def test_multihoprag_loader_preserves_gold_document_mapping(tmp_path) -> None:
    corpus = [
        {
            "title": "Alpha Launch",
            "url": "https://example.com/alpha",
            "source": "Example",
            "category": "technology",
            "published_at": "2023-01-01T00:00:00+00:00",
            "body": "Alpha Corp launched the new system.",
        },
        {
            "title": "Beta Funding",
            "url": "https://example.com/beta",
            "source": "Example",
            "category": "business",
            "published_at": "2023-01-02T00:00:00+00:00",
            "body": "Beta Labs raised new funding.",
        },
    ]
    questions = [
        {
            "query": "Which company launched the new system?",
            "answer": "Alpha Corp",
            "question_type": "inference_query",
            "evidence_list": [
                {
                    "title": "Alpha Launch",
                    "url": "https://example.com/alpha",
                    "fact": "Alpha Corp launched the new system.",
                }
            ],
        }
    ]
    corpus_path = tmp_path / "corpus.json"
    questions_path = tmp_path / "MultiHopRAG.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    questions_path.write_text(json.dumps(questions), encoding="utf-8")

    documents = load_multihoprag_documents(corpus_path, questions_path, max_documents=1, max_questions=1)
    loaded_questions = load_multihoprag_questions(questions_path, max_questions=1)

    assert len(documents) == 1
    assert loaded_questions[0].gold_document_ids == (documents[0].document_id,)
    assert loaded_questions[0].gold_facts == ("Alpha Corp launched the new system.",)
