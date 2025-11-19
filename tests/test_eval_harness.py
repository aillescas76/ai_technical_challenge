from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

import pytest

from app.evaluation import eval as eval_module
from app.evaluation.eval import ExpectedCitation, EvalExample, MetricAccumulator, _citation_scores, _retrieval_scores
from app.services.prompt import ContextChunk
from app.services.rag import CostBreakdown, RagAnswer, TokenUsage
from app.api.schemas import Citation


def _sample_answer(text: str) -> RagAnswer:
    citation = Citation(
        id="chunk-1",
        airline="SkyFly",
        title="Carry-on Rules",
        source_path="policies/skyfly/baggage.md",
        source_url=None,
        chunk_index=0,
        score=0.9,
        snippet="Carry-on bags must fit under the seat.",
    )
    context = ContextChunk(
        content="Carry-on bags must fit under the seat.",
        airline="SkyFly",
        title="Carry-on Rules",
        source_path="policies/skyfly/baggage.md",
        chunk_id="chunk-1",
        source_url=None,
    )
    return RagAnswer(
        answer=text,
        citations=[citation],
        contexts=[context],
        retrievals=[],
        latency_ms=25.0,
        tokens=TokenUsage(prompt=12, completion=6, embedding=4),
        costs=CostBreakdown(prompt_usd=0.0001, completion_usd=0.0002, embedding_usd=0.00003),
    )


def test_load_eval_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "EX-1",
                "question": "Who charges $40?",
                "expected_answer": "AA charges $40.",
                "expected_citations": [{"airline": "American Airlines", "title": "Checked bag policy"}],
                "airlines": ["American Airlines"],
                "category": "baggage",
                "tags": ["fees"],
                "refusal_expected": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = eval_module.load_eval_dataset(dataset_path)

    assert len(examples) == 1
    assert examples[0].id == "EX-1"
    assert examples[0].expected_citations[0].title == "Checked bag policy"


def test_metric_accumulator_summary() -> None:
    acc = MetricAccumulator()
    acc.add(
        {
            "retrieval_recall": 1.0,
            "retrieval_mrr": 0.5,
            "citation_precision": 0.5,
            "citation_recall": 1.0,
            "refusal_accuracy": 1.0,
            "latency_ms": 100.0,
            "tokens": {"prompt": 10, "completion": 4, "embedding": 2},
            "cost_usd": 0.002,
        }
    )
    summary = acc.summary()
    assert summary["recall_at_k"] == pytest.approx(1.0)
    assert summary["latency_ms"]["p50"] == pytest.approx(100.0)
    assert summary["token_totals"]["prompt"] == 10


def test_retrieval_and_citation_scores() -> None:
    class Obj:
        def __init__(self, airline: str, title: str) -> None:
            self.metadata = {"airline": airline, "title": title}

    retrievals = [Obj("AA", "Checked bag policy"), Obj("Delta", "Pets")]
    expected = [
        ExpectedCitation(airline="American Airlines", title="Checked bag policy"),
        ExpectedCitation(airline="Delta Air Lines", title="Pets"),
    ]
    recall, mrr = _retrieval_scores(retrievals, expected)
    assert recall == pytest.approx(1.0)
    assert mrr == pytest.approx(1.0)

    citations = [
        Citation(
            id="1",
            airline="American Airlines",
            title="Checked bag policy",
            source_path="a",
            source_url=None,
            chunk_index=0,
            score=0.9,
            snippet="",
        )
    ]
    precision, citation_recall = _citation_scores(citations, expected)
    assert precision == pytest.approx(1.0)
    assert citation_recall == pytest.approx(0.5)


def test_eval_runner_creates_results_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    example = EvalExample(
        id="AA-BAG-1",
        question="Who charges $40?",
        expected_answer="AA charges $40",
        expected_citations=[ExpectedCitation("SkyFly", "Carry-on Rules")],
        airlines=["SkyFly"],
        category="baggage",
        tags=["fees"],
        refusal_expected=False,
    )

    class _FakeReporter:
        def is_enabled(self) -> bool:
            return False

        def log_eval(self, **_: object) -> None:
            raise AssertionError("Reporter should not be called")

    class _FakeEngine:
        def __init__(self, store_provider) -> None:
            self.store_provider = store_provider

        def answer(self, request):
            return _sample_answer("AA charges $40.")

    monkeypatch.setattr(eval_module, "LangfuseReporter", lambda: _FakeReporter())
    monkeypatch.setattr(eval_module, "RagEngine", lambda provider: _FakeEngine(provider))
    monkeypatch.setattr(eval_module.VectorStore, "load", lambda path: object())

    output = tmp_path / "results.jsonl"
    runner = eval_module.EvalRunner([example])
    summary = runner.run(output_path=output)

    assert summary["dataset_size"] == 1
    data = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 1
    payload = json.loads(data[0])
    assert payload["answer"]["answer"].startswith("AA charges")
