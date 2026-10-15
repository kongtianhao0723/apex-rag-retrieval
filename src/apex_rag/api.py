"""FastAPI search endpoint and browser-based RAG workbench."""

from pathlib import Path
from time import perf_counter
from typing import Any

from apex_rag.cli import _build_engine
from apex_rag.config import settings
from apex_rag.answering import build_answerer, extractive_fallback

try:
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[assignment,misc]


if FastAPI is not None:
    app = FastAPI(title="Apex RAG Search API", version="0.1.0")
    engine = _build_engine()
    answerer = build_answerer(settings.deepseek_api_key, settings.deepseek_model, settings.deepseek_base_url) if settings.enable_answering else None
    frontend = Path(__file__).parent / "web" / "index.html"

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        """Serve the interactive RAG search workbench."""
        return FileResponse(frontend)

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        """Report that the API and retrieval engine are ready."""
        return {"status": "ok", "backend": settings.backend, "answering": "enabled" if answerer else "disabled"}

    @app.get("/api/v1/filters")
    def filters() -> dict[str, list[str]]:
        """Return category and card tier values present in the active index."""
        store = getattr(engine, "store", None)
        if store is not None:
            return {
                "categories": store.distinct_values("category"),
                "card_tiers": store.distinct_values("card_tier"),
            }
        chunks = getattr(engine, "chunks", [])
        return {
            "categories": sorted({chunk.category for chunk in chunks if chunk.category}),
            "card_tiers": sorted({chunk.card_tier for chunk in chunks if chunk.card_tier}),
        }

    @app.post("/api/v1/search")
    def search(payload: dict[str, Any]) -> dict[str, Any]:
        """Search the configured engine and return ranked context chunks."""
        query = str(payload.get("query", "")).strip()
        if not query:
            return {"error": "query is required", "results": []}
        mode = str(payload.get("mode", "enhanced"))
        if mode not in {"dense", "hybrid", "enhanced"}:
            return {"error": "mode must be dense, hybrid, or enhanced", "results": []}
        top_k = max(1, min(int(payload.get("top_k", 5)), 20))
        filters: dict[str, str | int] = {}
        for key in ("category", "card_tier"):
            value = payload.get(key)
            if value not in (None, ""):
                filters[key] = str(value)
        candidate_k = top_k if mode == "dense" else 20
        results, latency_ms = engine.timed_search(
            query, mode, top_k=candidate_k, final_top_n=top_k, filters=filters or None
        )
        response: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "filters": filters,
            "latency_ms": latency_ms,
            "results": [r.as_dict() for r in results],
        }
        if answerer is not None:
            try:
                answer_start = perf_counter()
                answer = answerer.answer(query, results)
                response["answer"] = answer
                response["answer_model"] = settings.deepseek_model
                response["answer_latency_ms"] = (perf_counter() - answer_start) * 1000
            except RuntimeError as exc:
                response["answer"] = extractive_fallback(results)
                response["answer_error"] = str(exc)
                response["answer_source"] = "retrieval_fallback"
        else:
            response["answer"] = None
            response["answer_error"] = "Answering is disabled; set DEEPSEEK_API_KEY to enable it."
        return response
else:
    app = None
