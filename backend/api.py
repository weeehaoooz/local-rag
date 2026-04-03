"""
FastAPI server for the KG-RAG Chat system.

Usage:
    python api.py
    # or: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

from ingestion.ingest_manager import AsyncIngestionManager

# ── Global references ─────────────────────────────────────────────────
_engine = None
_ingest_manager: AsyncIngestionManager | None = None


def _get_engine():
    """Lazy-init the HybridEngine (heavy, only created once)."""
    global _engine
    if _engine is None:
        from retrieval.engine import HybridEngine
        _engine = HybridEngine()
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global _ingest_manager

    # Startup: pre-warm the engine
    _get_engine()

    from config import DATA_DIR, STORAGE_DIR

    # Startup: launch the background ingestion manager
    _ingest_manager = AsyncIngestionManager(
        data_dir=os.getenv("INGEST_DATA_DIR", DATA_DIR),
        storage_dir=os.getenv("INGEST_STORAGE_DIR", STORAGE_DIR),
        max_workers=int(os.getenv("INGEST_MAX_WORKERS", "2")),
        scan_interval_seconds=float(os.getenv("INGEST_SCAN_INTERVAL", "0")),
    )
    await _ingest_manager.start()

    yield

    # Shutdown: stop ingestion manager then engine
    if _ingest_manager is not None:
        await _ingest_manager.stop()
    if _engine is not None:
        _engine.close()


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Local KG-RAG Chat API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow Angular dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    mode: Literal["fast", "planning"] = "fast"
    history: list[dict] = []
    system_prompt: str = ""


class Source(BaseModel):
    title: str
    category: str
    file: str = ""


class GraphNode(BaseModel):
    text: str
    source: str


class ChatStats(BaseModel):
    tps: float = 0.0
    context_utilization: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_window: int = 8192


class ChatResponse(BaseModel):
    response: str
    sources: list[Source]
    stats: ChatStats = ChatStats()
    graph_context: list[GraphNode] = []
    query_type: str = "LOCAL"
    suggested_prompts: list[str] = []


class TitleRequest(BaseModel):
    first_message: str


class TitleResponse(BaseModel):
    title: str


class SuggestionResponse(BaseModel):
    suggestions: list[str]


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/suggestions", response_model=SuggestionResponse)
def get_suggestions():
    engine = _get_engine()
    suggs = engine.get_suggestions(limit=4)
    return SuggestionResponse(suggestions=suggs)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    engine = _get_engine()
    result = engine.chat(
        req.message, 
        mode=req.mode, 
        history=req.history,
        system_prompt=req.system_prompt
    )

    return ChatResponse(
        response=result["response"],
        sources=[Source(**s) for s in result["sources"]],
        stats=ChatStats(**result.get("stats", {})),
        graph_context=[GraphNode(text=g[0], source=g[1]) for g in result.get("graph_context", [])],
        query_type=result.get("query_type", "LOCAL"),
        suggested_prompts=result.get("suggested_prompts", []),
    )


@app.post("/api/title", response_model=TitleResponse)
def generate_title(req: TitleRequest):
    """Generate a short AI title for a conversation based on its first message."""
    engine = _get_engine()
    try:
        prompt = (
            "Generate a concise, descriptive title (3-6 words max) for a conversation "
            "that starts with this message. Return ONLY the title, no quotes, no punctuation at the end.\n\n"
            f"Message: {req.first_message[:300]}"
        )
        title = engine.llm.complete(prompt).text.strip().strip('"\'').strip()
        # Trim to a reasonable length
        if len(title) > 60:
            title = title[:60]
        return TitleResponse(title=title)
    except Exception as e:
        print(f"Title generation failed: {e}")
        return TitleResponse(title=req.first_message[:40] + ("..." if len(req.first_message) > 40 else ""))


# ── Ingestion Routes ───────────────────────────────────────────────────

@app.get("/api/ingest/status")
def ingest_status():
    """Return the current ingestion pipeline status."""
    if _ingest_manager is None:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialised.")
    return _ingest_manager.status()


@app.post("/api/ingest/trigger")
async def ingest_trigger():
    """Manually trigger a scan of the data directory for new/changed files."""
    if _ingest_manager is None:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialised.")
    queued = await _ingest_manager.trigger_scan()
    return {"queued": queued, "status": _ingest_manager.status()}


# ── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
