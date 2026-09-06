"""
Tests the CSV output of the SparseEncoder evaluators.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sentence_transformers import SparseEncoder
from sentence_transformers.base.evaluation import BaseEvaluator
from sentence_transformers.sparse_encoder.evaluation import (
    SparseBinaryClassificationEvaluator,
    SparseEmbeddingSimilarityEvaluator,
    SparseInformationRetrievalEvaluator,
    SparseMSEEvaluator,
    SparseRerankingEvaluator,
    SparseTranslationEvaluator,
    SparseTripletEvaluator,
)

SENTENCES1 = ["A person on a horse jumps over a broken down airplane.", "Children smiling and waving at camera"]
SENTENCES2 = ["A person is outdoors, on a horse.", "A person is at a diner, ordering an omelette."]
SENTENCES3 = ["A dog is running through the snow.", "Two men are playing guitar."]


def _build_evaluator(name: str, model: SparseEncoder) -> BaseEvaluator:
    if name == "binary_classification":
        return SparseBinaryClassificationEvaluator(sentences1=SENTENCES1, sentences2=SENTENCES2, labels=[1, 0])
    if name == "embedding_similarity":
        return SparseEmbeddingSimilarityEvaluator(sentences1=SENTENCES1, sentences2=SENTENCES2, scores=[0.9, 0.1])
    if name == "triplet":
        return SparseTripletEvaluator(anchors=SENTENCES1, positives=SENTENCES2, negatives=SENTENCES3)
    if name == "information_retrieval":
        return SparseInformationRetrievalEvaluator(
            queries={"q1": SENTENCES1[0]},
            corpus={"d1": SENTENCES2[0], "d2": SENTENCES3[0]},
            relevant_docs={"q1": {"d1"}},
        )
    if name == "mse":
        return SparseMSEEvaluator(source_sentences=SENTENCES1, target_sentences=SENTENCES2, teacher_model=model)
    if name == "reranking":
        samples = [{"query": SENTENCES1[0], "positive": [SENTENCES2[0]], "negative": [SENTENCES3[0]]}]
        return SparseRerankingEvaluator(samples=samples)
    if name == "translation":
        return SparseTranslationEvaluator(source_sentences=SENTENCES1, target_sentences=SENTENCES2)
    raise ValueError(f"Unknown evaluator {name!r}")


@pytest.mark.parametrize(
    "evaluator_name",
    [
        "binary_classification",
        "embedding_similarity",
        "triplet",
        "information_retrieval",
        "mse",
        "reranking",
        "translation",
    ],
)
def test_evaluator_csv_header_matches_row(
    splade_bert_tiny_model: SparseEncoder, tmp_path: Path, evaluator_name: str
) -> None:
    """The CSV an evaluator writes must have exactly as many header cells as value cells, and must
    not repeat a column name. The sparsity columns are appended to the header by
    ``_append_csv_headers`` and to the row by ``append_to_last_row``, so a header written twice
    silently shifts every value into the wrong column."""
    evaluator = _build_evaluator(evaluator_name, splade_bert_tiny_model)
    evaluator(splade_bert_tiny_model, output_path=str(tmp_path))

    with open(tmp_path / evaluator.csv_file, newline="", encoding="utf-8") as f:
        header, row = list(csv.reader(f))

    assert len(header) == len(set(header)), f"Duplicate columns in the header: {header}"
    assert len(header) == len(row), f"Header has {len(header)} columns but the row has {len(row)}: {header} / {row}"
