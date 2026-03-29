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

# ── Global engine reference ───────────────────────────────────────────
_engine = None


def _get_engine():
    """Lazy-init the HybridEngine (heavy, only created once)."""
    global _engine
    if _engine is None:
        from engine import HybridEngine
        _engine = HybridEngine()
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Startup: pre-warm the engine
    _get_engine()
    yield
    # Shutdown: clean up
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


class ChatResponse(BaseModel):
    response: str
    sources: list[Source]
    stats: ChatStats = ChatStats()
    graph_context: list[GraphNode] = []
    query_type: str = "LOCAL"


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    engine = _get_engine()
    result = engine.chat(req.message)

    return ChatResponse(
        response=result["response"],
        sources=[Source(**s) for s in result["sources"]],
        stats=ChatStats(**result.get("stats", {})),
        graph_context=[GraphNode(text=g[0], source=g[1]) for g in result.get("graph_context", [])],
        query_type=result.get("query_type", "LOCAL"),
    )


# ── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
