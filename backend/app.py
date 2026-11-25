r"""Athena HTTP API (and the web UI, when frontend/dist exists).

Easiest, from backend/:   ..\.venv\Scripts\python.exe -m athena.serve   (opens the browser, signed in)
Or:                       ..\.venv\Scripts\python.exe -m uvicorn app:app   (prints the sign-in link)

The API can see your documents and the names and sizes of files in your home
folder, so it is locked down (athena/assistant/security.py): it listens on
127.0.0.1 only, every request needs this install's secret token (a cookie set
by the login link, or an X-Athena-Token header), and requests whose Host
header is not localhost are refused.
"""

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from athena import database
from athena.assistant import agent, security
from athena.assistant.api import router as assistant_router
from athena.assistant.router import smart_route
from athena.config import PROJECT_DIR, settings
from athena.document_reader import SUPPORTED_EXTENSIONS
from athena.engine import parse_command, process, run_command
from athena.folder_indexer import index_folder
from athena.indexer import index_document
from athena.llm import LLMError, is_available
from athena.rag import RagAnswer, stream_answer
from athena.vector_store import get_store
from athena.web import FetchError, fetch_to_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("athena.api")

MAX_UPLOAD_BYTES = int(os.getenv("ATHENA_MAX_UPLOAD_MB", "50")) * 1024 * 1024
PORT = int(os.getenv("ATHENA_PORT", "8000"))
CHAT_RATE_PER_MINUTE = int(os.getenv("ATHENA_CHAT_RATE", "20"))
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"


def _warm_up() -> None:
    """Load every model in the background so the first question is not slow:
    the LLM (~15-20 s from disk), the embedder and the reranker."""
    from athena.assistant.router import _semantic
    from athena.llm import warm_up
    from athena.retriever import retrieve

    started = time.perf_counter()
    warm_up()
    store = get_store()
    # Embed the router's example requests now, not on the first message.
    _semantic.classify("warm up")
    if store.count():
        retrieve("warm up", store, k=1)
    logger.info("Models loaded in %.1fs", time.perf_counter() - started)


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.initialize_database()  # at server start, not at import
    logger.info("Open Athena with: %s", login_url())
    if os.getenv("ATHENA_WARM_UP", "true").lower() == "true":
        threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="Athena", version="1.0.0", lifespan=lifespan)

# The Vite dev server (npm run dev) runs on another port; production serves
# the UI from this same origin, so no CORS is needed there.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------- guards

def login_url(port: int = PORT) -> str:
    # The token travels in the #fragment: browsers never send that part to
    # the server, so it doesn't end up in request logs.
    return f"http://127.0.0.1:{port}/#token={security.get_token()}"


@app.middleware("http")
async def localhost_only(request: Request, call_next):
    """Refuse requests addressed to any other host name (DNS rebinding), and
    state-changing requests sent from another website's page."""
    allowed = security.allowed_hosts()
    if security.host_of(request.headers.get("host")) not in allowed:
        return JSONResponse(status_code=403, content={"detail": "Athena only answers on localhost."})
    origin = request.headers.get("origin")
    origin_host = security.host_of(origin.split("://", 1)[-1].split("/", 1)[0]) if origin else ""
    cross_site = bool(origin) and origin != "null" and origin_host not in allowed
    if request.method not in ("GET", "HEAD", "OPTIONS") and cross_site:
        return JSONResponse(status_code=403, content={"detail": "Cross-site request refused."})
    return await call_next(request)


def require_session(request: Request) -> None:
    token = request.cookies.get(security.COOKIE_NAME) or request.headers.get(security.HEADER_NAME)
    if not security.token_matches(token):
        raise HTTPException(401, "Not signed in. Open Athena with the link from `python -m athena.serve`.")


_recent: dict[str, deque] = defaultdict(deque)
_rate_lock = threading.Lock()


def rate_limit(request: Request) -> None:
    """Each chat runs the CPU flat out for ~20 s; a runaway client or script
    must not queue hundreds of them."""
    key = request.client.host if request.client else "local"
    now = time.monotonic()
    with _rate_lock:
        window = _recent[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= CHAT_RATE_PER_MINUTE:
            raise HTTPException(429, f"More than {CHAT_RATE_PER_MINUTE} questions a minute. Slow down a little.")
        window.append(now)


guarded = [Depends(require_session)]
_index_lock = threading.Lock()  # one indexing job at a time


@app.exception_handler(LLMError)
async def llm_error(_: Request, exc: LLMError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ----------------------------------------------------------------- health

# ----------------------------------------------------------------- session

class SessionRequest(BaseModel):
    token: str


@app.get("/api/session")
def session_status(request: Request):
    token = request.cookies.get(security.COOKIE_NAME) or request.headers.get(security.HEADER_NAME)
    return {"authenticated": security.token_matches(token)}


@app.post("/api/session")
def sign_in(body: SessionRequest):
    if not security.token_matches(body.token.strip()):
        raise HTTPException(401, "Wrong token.")
    response = JSONResponse({"authenticated": True})
    response.set_cookie(security.COOKIE_NAME, security.get_token(), max_age=30 * 86400,
                        httponly=True, samesite="strict", path="/")
    return response


@app.delete("/api/session")
def sign_out():
    response = JSONResponse({"authenticated": False})
    response.delete_cookie(security.COOKIE_NAME, path="/")
    return response


app.include_router(assistant_router, dependencies=guarded)


@app.get("/api/health", dependencies=guarded)
def health():
    ok, detail = is_available()
    store = get_store()
    return {
        "status": "ok" if ok else "degraded",
        "llm": {"ok": ok, "detail": detail, "model": settings.llm_model},
        "index": {"chunks": store.count(), "documents": len(store.documents()),
                  "embedding_model": settings.embedding_model},
        "retrieval": {"mode": settings.retrieval_mode, "rerank": settings.rerank,
                      "reranker": settings.reranker_model, "top_k": settings.top_k},
    }


@app.get("/api/metrics", dependencies=guarded)
def metrics():
    return database.query_metrics()


# -------------------------------------------------------------- documents

@app.get("/api/documents", dependencies=guarded)
def list_documents():
    return get_store().documents()


def _safe_name(name: str) -> str:
    name = Path(name).name  # drop any directory part
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9._ +()-]+", "_", stem).strip(" .")[:120] or "document"
    return stem + ext.lower()


@app.post("/api/documents", dependencies=guarded)
async def upload_document(file: UploadFile = File(...)):
    name = _safe_name(file.filename or "")
    if os.path.splitext(name)[1] not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "Empty file")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / name
    path.write_bytes(data)
    return _index_one(path)


class UrlRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)


@app.post("/api/documents/url", dependencies=guarded)
def add_url(request: UrlRequest):
    try:
        path = fetch_to_file(request.url, settings.upload_dir / "web")
    except FetchError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _index_one(path, {"source_url": request.url})


def _index_one(path: Path, extra: dict | None = None) -> dict:
    with _index_lock:
        try:
            result = index_document(str(path), get_store(), extra_metadata=extra)
        except Exception as exc:
            logger.warning("indexing %s failed: %s", path.name, exc)
            raise HTTPException(422, f"Could not index {path.name}: {exc}") from exc
    return {"filename": path.name, "status": result.status.value, "chunks": result.chunks,
            "title": result.title, "ocr_pages": result.ocr_pages, "duplicate_of": result.duplicate_of}


@app.delete("/api/documents/{filename}", dependencies=guarded)
def delete_document(filename: str):
    store = get_store()
    match = next((d for d in store.documents() if d["filename"] == filename), None)
    if match is None:
        raise HTTPException(404, "No such document")
    with _index_lock:
        store.delete_file(match["filepath"])
    # Only files Athena saved itself are deleted from disk; your own files
    # in data/ are just removed from the index.
    path = Path(match["filepath"]).resolve()
    deleted_file = False
    if path.is_relative_to(settings.upload_dir.resolve()) and path.is_file():
        path.unlink()
        deleted_file = True
    return {"filename": filename, "removed_from_index": True, "deleted_file": deleted_file}


_reindex_state = {"running": False, "last": None}


@app.post("/api/reindex", dependencies=guarded)
def reindex():
    """Index new/changed files in the data folder (unchanged files are skipped)."""
    if _reindex_state["running"]:
        return {"started": False, "detail": "Already running"}

    def job():
        with _index_lock:
            _reindex_state["running"] = True
            started = time.perf_counter()
            try:
                report = index_folder(settings.data_dir, get_store())
                _reindex_state["last"] = {"counts": dict(report.counts), "chunks": report.chunks,
                                          "failures": report.failures,
                                          "seconds": round(time.perf_counter() - started, 1)}
            finally:
                _reindex_state["running"] = False

    threading.Thread(target=job, daemon=True).start()
    return {"started": True}


@app.get("/api/reindex", dependencies=guarded)
def reindex_status():
    return _reindex_state


@app.get("/api/documents/{filename}/pages/{page}.png", dependencies=guarded)
def page_image(filename: str, page: int):
    """A PDF page as an image, for the citation preview."""
    import pymupdf

    match = next((d for d in get_store().documents() if d["filename"] == filename), None)
    if match is None or not match["filepath"].lower().endswith(".pdf"):
        raise HTTPException(404, "No such PDF")
    with pymupdf.open(match["filepath"]) as doc:
        if not 1 <= page <= doc.page_count:
            raise HTTPException(404, "No such page")
        png = doc[page - 1].get_pixmap(dpi=110).tobytes("png")
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


# ------------------------------------------------------------------- chat

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=32)
    files: list[str] | None = Field(default=None, max_length=50)  # restrict to these documents
    # A button in a result card ("Dig in", "Open folder"): skips understanding
    # the message. Only read-only tools; paths are checked by policy.py.
    action: dict | None = None


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse, dependencies=[*guarded, Depends(rate_limit)])
def chat_legacy(request: ChatRequest):
    """Milestone 1 endpoint: plain-text answer. The UI uses /api/chat/stream."""
    try:
        return ChatResponse(response=process(request.message))
    except LLMError as exc:
        logger.warning("LLM unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal error") from exc


def _result_payload(result: RagAnswer, total_seconds: float) -> dict:
    g = result.grounding
    return {
        "answer": result.answer,
        "found": result.found,
        "cited": [s.number for s in result.cited],
        "standalone_question": result.standalone_question,
        "abstained_early": result.abstained_early,
        "grounding": None if g is None else {
            "supported": g.supported, "unsupported": g.unsupported, "checked": g.checked,
            "invalid_citations": g.invalid_citations, "uncited_sentences": g.uncited_sentences},
        "timings": {"retrieval_ms": round(result.retrieval_seconds * 1000),
                    "generation_ms": round(result.generation_seconds * 1000),
                    "total_ms": round(total_seconds * 1000)},
        "top_score": result.top_score,
        "llm": result.llm_stats,
    }


@app.post("/api/chat/stream", dependencies=[*guarded, Depends(rate_limit)])
def chat_stream(request: ChatRequest):
    """Server-sent events: rewrite?, sources, token..., done (or error)."""
    conversation_id = request.conversation_id
    if conversation_id and not database.conversation_exists(conversation_id):
        raise HTTPException(404, "No such conversation")
    if not conversation_id:
        conversation_id = database.create_conversation(request.message.strip()[:60])
    history = database.history_pairs(conversation_id)
    # What this chat was last looking at (a folder, a list of files, a file),
    # so follow-ups like "what's in local" or "the second one" make sense.
    context = next((m["meta"]["context"] for m in reversed(database.get_messages(conversation_id))
                    if m["role"] == "assistant" and isinstance((m["meta"] or {}).get("context"), dict)), {})
    database.add_message(conversation_id, "user", request.message)

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def events():
        yield sse("conversation", {"id": conversation_id})
        command = parse_command(request.message)
        if command:
            text = run_command(*command)
            database.add_message(conversation_id, "assistant", text, {"command": command[0]})
            yield sse("done", {"answer": text, "found": True, "command": command[0], "cited": []})
            return

        if not request.files:  # documents ticked in the Library = a question about them
            try:
                documents = False
                for event, data in agent.handle(request.message, context, smart_route, request.action):
                    if event == "documents":
                        documents = True
                        break
                    if event == "done":
                        database.add_message(conversation_id, "assistant", data["answer"], data)
                        yield sse("done", data)
                    elif event in ("token", "replace"):
                        yield sse(event, {"text": data})
                    else:
                        yield sse(event, data)
                if not documents:
                    return
            except LLMError as exc:
                yield sse("error", {"detail": str(exc)})
                return
            except Exception:
                logger.exception("Unhandled error in the assistant")
                yield sse("error", {"detail": "Internal error"})
                return

        started = time.perf_counter()
        sources: list[dict] = []
        try:
            for event in stream_answer(request.message, history=history, files=request.files or None):
                if event["type"] == "rewrite":
                    yield sse("rewrite", {"question": event["question"]})
                elif event["type"] == "sources":
                    sources = event["sources"]
                    yield sse("sources", {"sources": sources})
                elif event["type"] == "token":
                    yield sse("token", {"text": event["text"]})
                elif event["type"] == "done":
                    result: RagAnswer = event["result"]
                    payload = _result_payload(result, time.perf_counter() - started)
                    database.add_message(conversation_id, "assistant", result.answer,
                                         {**payload, "sources": sources})
                    database.log_query(
                        found=result.found, abstained_early=result.abstained_early,
                        grounded=result.grounding.supported if result.grounding else None,
                        retrieval_ms=payload["timings"]["retrieval_ms"],
                        generation_ms=payload["timings"]["generation_ms"],
                        total_ms=payload["timings"]["total_ms"],
                        prompt_tokens=result.llm_stats.get("prompt_tokens"),
                        output_tokens=result.llm_stats.get("output_tokens"),
                        top_score=result.top_score,
                    )
                    yield sse("done", payload)
        except LLMError as exc:
            yield sse("error", {"detail": str(exc)})
        except Exception:
            logger.exception("Unhandled error in /api/chat/stream")
            yield sse("error", {"detail": "Internal error"})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------- conversations

@app.get("/api/conversations", dependencies=guarded)
def conversations():
    return database.list_conversations()


@app.get("/api/conversations/{conversation_id}", dependencies=guarded)
def conversation(conversation_id: str):
    if not database.conversation_exists(conversation_id):
        raise HTTPException(404, "No such conversation")
    return {"id": conversation_id, "messages": database.get_messages(conversation_id)}


@app.delete("/api/conversations/{conversation_id}", dependencies=guarded)
def delete_conversation(conversation_id: str):
    if not database.delete_conversation(conversation_id):
        raise HTTPException(404, "No such conversation")
    return {"deleted": conversation_id}


# -------------------------------------------------------------------- UI

if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        candidate = (FRONTEND_DIST / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(FRONTEND_DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    def home():
        return {"message": "Athena is running. Build the UI with: cd frontend && npm install && npm run build"}
