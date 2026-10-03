"""RAGAS evaluation against labeled MS MARCO v1.1 validation contexts."""
import asyncio
import json
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from .config import get_settings
from .dataset import load_eval_queries
from .embeddings import Embedder
from .store import PassageStore
from .retrieval import RetrievalService


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


async def evaluate(count: int = 20, cfg=None) -> dict:
    cfg = cfg or get_settings()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "dataset": cfg.eval_dataset_id,
        "config": cfg.eval_dataset_config, "split": cfg.eval_dataset_split, "requested_query_count": count,
        "embedding_model": cfg.embedding_model, "top_k": cfg.top_k,
        "packages": {"ragas": package_version("ragas"), "datasets": package_version("datasets"),
                     "sentence-transformers": package_version("sentence-transformers")},
        "metrics": {"context_precision": None, "context_recall": None}}
    try:
        from ragas import SingleTurnSample, EvaluationDataset, evaluate as ragas_evaluate
        from ragas.metrics import NonLLMContextPrecisionWithReference

        rows = load_eval_queries(cfg.eval_dataset_id, cfg.eval_dataset_config, cfg.eval_dataset_split, count, cfg.hf_token_value)
        if len(rows) < count:
            raise RuntimeError(f"Need {count} labeled examples with answers and selected contexts; got {len(rows)}")
        embedder = Embedder(cfg.embedding_model, cfg.device, dimension=cfg.embedding_dimension,
            precision=cfg.embedding_precision, query_instruction=cfg.query_instruction,
            token=cfg.hf_token_value, revision=cfg.model_revision)
        report.update({"model_revision": embedder.model_revision, "embedding_dimension": embedder.dimension,
                       "precision": embedder.precision, "device_requested": cfg.device,
                       "device_actual": embedder.device})
        store = PassageStore(cfg.qdrant_path, cfg.collection_name)
        service = RetrievalService(embedder, store, cfg.top_k, cfg.ef_search)
        samples = []
        retrieved_by_query: list[list[str]] = []
        try:
            for row in rows:
                found = service.search(row["query"])["results"]
                retrieved_by_query.append([item["text"] for item in found])
                samples.append(SingleTurnSample(user_input=row["query"], response=row["reference_answer"],
                    reference=row["reference_answer"], reference_contexts=row["reference_contexts"],
                    retrieved_contexts=[item["text"] for item in found]))
        finally:
            store.close()

        # RAGAS computes reference-context precision without an LLM judge.
        # Recall needs semantic judging or aligned passage-ID qrels, neither configured here.
        metric = NonLLMContextPrecisionWithReference()
        result = ragas_evaluate(EvaluationDataset(samples=samples), metrics=[metric])
        frame = result.to_pandas()
        metric_column = getattr(metric, "name", "non_llm_context_precision_with_reference")
        if metric_column not in frame.columns:
            candidates = [column for column in frame.columns if "context_precision" in column]
            if not candidates:
                raise RuntimeError(f"RAGAS result has no context precision column: {list(frame.columns)}")
            metric_column = candidates[0]
        scores = frame[metric_column].dropna().tolist()
        normalize = lambda text: re.sub(r"\s+", " ", text).strip().casefold()
        recall_values, reciprocal_ranks = [], []
        for row, retrieved in zip(rows, retrieved_by_query):
            relevant = {normalize(text) for text in row["reference_contexts"]}
            ranked = [normalize(text) for text in retrieved]
            hits = [rank for rank, text in enumerate(ranked, 1) if text in relevant]
            recall_values.append(len(set(ranked) & relevant) / len(relevant) if relevant else 0.0)
            reciprocal_ranks.append(1.0 / hits[0] if hits else 0.0)
        report.update({"status": "complete", "query_count": len(rows),
            "metrics": {"context_precision": sum(scores) / len(scores) if scores else None,
                        "context_recall": None},
            "context_precision_per_query": scores,
            "retrieval_diagnostics": {"recall_at_5": sum(recall_values) / len(recall_values),
                "mrr_at_5": sum(reciprocal_ranks) / len(reciprocal_ranks),
                "judgment": "exact normalized text match against MS MARCO is_selected reference passages; reported separately from RAGAS"},
            "precision_method": "RAGAS NonLLMContextPrecisionWithReference compares retrieved passages to the selected MS MARCO reference passage texts.",
            "recall_unavailable_reason": "No LLM judge or aligned passage-ID qrels are configured for semantic reference coverage."})
    except Exception as exc:
        detail = str(exc)
        if cfg.hf_token_value:
            detail = detail.replace(cfg.hf_token_value, "[REDACTED]")
        report.update({"status": "unavailable", "failure": f"{type(exc).__name__}: {detail}"})

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(asyncio.run(evaluate()), indent=2))
