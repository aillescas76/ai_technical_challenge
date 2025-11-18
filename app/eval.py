from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.airlines import normalize_airline_key
from app.config import DATA_DIR, EVAL_DATASET_PATH, VECTOR_STORE_PATH
from app.rag import RagEngine, RagRequest
from app.telemetry import LangfuseReporter
from app.vector_store import VectorStore


logger = logging.getLogger(__name__)


@dataclass
class ExpectedCitation:
    airline: str
    title: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectedCitation":
        return cls(
            airline=str(data.get("airline", "")).strip(),
            title=str(data.get("title", "")).strip(),
        )


@dataclass
class EvalExample:
    id: str
    question: str
    expected_answer: str
    expected_citations: List[ExpectedCitation]
    airlines: List[str]
    category: str
    tags: List[str]
    refusal_expected: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalExample":
        citations = [
            ExpectedCitation.from_dict(item) for item in data.get("expected_citations", [])
        ]
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            expected_answer=str(data.get("expected_answer", "")),
            expected_citations=citations,
            airlines=[str(name) for name in data.get("airlines", [])],
            category=str(data.get("category", "")),
            tags=[str(tag) for tag in data.get("tags", [])],
            refusal_expected=bool(data.get("refusal_expected", False)),
        )


def load_eval_dataset(path: Path = EVAL_DATASET_PATH) -> List[EvalExample]:
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")
    examples: List[EvalExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            examples.append(EvalExample.from_dict(json.loads(line)))
    return examples


class MetricAccumulator:
    def __init__(self) -> None:
        self.recall: List[float] = []
        self.mrr: List[float] = []
        self.citation_precision: List[float] = []
        self.citation_recall: List[float] = []
        self.refusal: List[float] = []
        self.latencies: List[float] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embedding_tokens = 0
        self.cost_total = 0.0

    def add(self, example_metrics: Dict[str, Any]) -> None:
        self.recall.append(example_metrics["retrieval_recall"])
        self.mrr.append(example_metrics["retrieval_mrr"])
        self.citation_precision.append(example_metrics["citation_precision"])
        self.citation_recall.append(example_metrics["citation_recall"])
        self.refusal.append(example_metrics["refusal_accuracy"])
        self.latencies.append(example_metrics["latency_ms"])
        self.prompt_tokens += example_metrics["tokens"]["prompt"]
        self.completion_tokens += example_metrics["tokens"]["completion"]
        self.embedding_tokens += example_metrics["tokens"]["embedding"]
        self.cost_total += example_metrics["cost_usd"]

    def summary(self) -> Dict[str, Any]:
        latency_p50 = median(self.latencies) if self.latencies else 0.0
        latency_p95 = _percentile(self.latencies, 0.95)
        return {
            "recall_at_k": _mean(self.recall),
            "mrr_at_k": _mean(self.mrr),
            "citation_precision": _mean(self.citation_precision),
            "citation_recall": _mean(self.citation_recall),
            "refusal_accuracy": _mean(self.refusal),
            "latency_ms": {"p50": latency_p50, "p95": latency_p95},
            "token_totals": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "embedding": self.embedding_tokens,
            },
            "costs": {"total_usd": self.cost_total},
        }


class EvalRunner:
    def __init__(
        self,
        dataset: Sequence[EvalExample],
        *,
        reporter: Optional[LangfuseReporter] = None,
    ) -> None:
        self.dataset = list(dataset)
        store = VectorStore.load(VECTOR_STORE_PATH)
        self._engine = RagEngine(lambda: store)
        self._reporter = reporter or LangfuseReporter()

    def run(
        self,
        *,
        limit: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        subset = self.dataset[:limit] if limit else self.dataset
        if not subset:
            raise ValueError("No eval examples to process.")

        output_path = output_path or _default_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        accumulator = MetricAccumulator()

        with output_path.open("w", encoding="utf-8") as sink:
            for example in subset:
                rag_answer = self._engine.answer(
                    RagRequest(question=example.question, top_k=5, airline=None)
                )
                metrics = _evaluate_example(example, rag_answer)
                accumulator.add(metrics)

                record = {
                    "id": example.id,
                    "question": example.question,
                    "expected_citations": [
                        {"airline": cite.airline, "title": cite.title}
                        for cite in example.expected_citations
                    ],
                    "answer": rag_answer.to_record(),
                    "metrics": metrics,
                }
                sink.write(json.dumps(record, ensure_ascii=False))
                sink.write("\n")
                self._emit_trace(example, rag_answer, metrics)

        summary = accumulator.summary()
        summary.update(
            {
                "dataset_size": len(subset),
                "results_path": str(output_path),
            }
        )
        logger.info("Eval run complete: %s", json.dumps(summary))
        return summary

    def _emit_trace(self, example: EvalExample, answer, metrics: Dict[str, Any]) -> None:
        if not self._reporter.is_enabled():
            return
        try:
            self._reporter.log_eval(
                run_name="rag-eval",
                input_payload={"question": example.question, "id": example.id},
                output_payload={
                    "answer": answer.answer,
                    "citations": [citation.model_dump() for citation in answer.citations],
                },
                metrics={
                    **metrics,
                    "expected_citations": [
                        {"airline": cite.airline, "title": cite.title}
                        for cite in example.expected_citations
                    ],
                },
                tags=["eval-harness"] + example.tags,
            )
        except Exception:  # pragma: no cover
            logger.exception("Failed to emit LangFuse trace for %s", example.id)


def _evaluate_example(example: EvalExample, answer) -> Dict[str, Any]:
    recall, mrr = _retrieval_scores(answer.retrievals, example.expected_citations)
    citation_precision, citation_recall = _citation_scores(
        answer.citations, example.expected_citations
    )
    refusal_accuracy = _refusal_score(example.refusal_expected, answer.answer)
    metrics = {
        "retrieval_recall": recall,
        "retrieval_mrr": mrr,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "refusal_accuracy": refusal_accuracy,
        "latency_ms": answer.latency_ms,
        "tokens": {
            "prompt": answer.tokens.prompt,
            "completion": answer.tokens.completion,
            "embedding": answer.tokens.embedding,
        },
        "cost_usd": answer.costs.total,
    }
    return metrics


def _retrieval_scores(
    retrievals: Sequence, expected: Sequence[ExpectedCitation]
) -> Tuple[float, float]:
    if not expected:
        return 1.0, 1.0
    relevant = {_normalize_pair(item.airline, item.title) for item in expected}
    hits: List[int] = []
    seen: set = set()
    for idx, result in enumerate(retrievals):
        pair = _normalize_pair(
            str(result.metadata.get("airline", "")),
            str(result.metadata.get("title", "")),
        )
        if pair in relevant and pair not in seen:
            hits.append(idx)
            seen.add(pair)
    recall = len(seen) / len(relevant) if relevant else 1.0
    mrr = 1.0 / (hits[0] + 1) if hits else 0.0
    return recall, mrr


def _citation_scores(
    citations: Sequence, expected: Sequence[ExpectedCitation]
) -> Tuple[float, float]:
    if not expected:
        return 1.0, 1.0
    expected_pairs = {_normalize_pair(item.airline, item.title) for item in expected}
    cited_pairs = {
        _normalize_pair(citation.airline, citation.title) for citation in citations
    }
    true_positive = len(expected_pairs & cited_pairs)
    precision = true_positive / len(cited_pairs) if cited_pairs else 0.0
    recall = true_positive / len(expected_pairs) if expected_pairs else 1.0
    return precision, recall


def _refusal_score(refusal_expected: bool, answer_text: str) -> float:
    answer_is_refusal = _is_refusal(answer_text)
    if refusal_expected:
        return 1.0 if answer_is_refusal else 0.0
    return 1.0 if not answer_is_refusal else 0.0


def _is_refusal(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return "no answer" in lowered or "not able to find" in lowered


def _normalize_pair(airline: str, title: str) -> Tuple[str, str]:
    return (normalize_airline_key(airline), " ".join(title.lower().split()))


def _percentile(data: Sequence[float], quantile: float) -> float:
    if not data:
        return 0.0
    sorted_values = sorted(data)
    k = int(round((len(sorted_values) - 1) * quantile))
    return sorted_values[min(max(k, 0), len(sorted_values) - 1)]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DATA_DIR / "evals" / f"run-{timestamp}.jsonl"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the airline RAG eval harness.")
    parser.add_argument("--dataset", type=Path, default=EVAL_DATASET_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    dataset = load_eval_dataset(args.dataset)
    runner = EvalRunner(dataset)
    summary = runner.run(limit=args.limit, output_path=args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
