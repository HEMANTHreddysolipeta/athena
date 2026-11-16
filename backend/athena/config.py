"""Central configuration.

Every tunable value lives here and can be overridden with an environment
variable (or a line in backend/.env). Code imports `settings` instead of
hardcoding paths, model names or limits.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


# BGE models are trained with this instruction in front of queries only.
_QUERY_PREFIXES = {"bge": "Represent this sentence for searching relevant passages: "}


@dataclass(frozen=True)
class Settings:
    # Where documents to index live. A dedicated folder, not all of Downloads.
    data_dir: Path = Path(os.getenv("ATHENA_DATA_DIR", PROJECT_DIR / "data"))

    # Uploaded files and saved web pages go here (inside data_dir, so the
    # file commands can see them too).
    upload_dir: Path = Path(os.getenv("ATHENA_UPLOAD_DIR", Path(os.getenv("ATHENA_DATA_DIR", PROJECT_DIR / "data")) / "uploads"))

    # Vector store. v3 = section-aware chunks with title/heading metadata.
    # (chroma_db = the Aug-2026 index, chroma_db_v2 = Milestone 1; both untouched.)
    chroma_dir: Path = Path(os.getenv("ATHENA_CHROMA_DIR", BACKEND_DIR / "chroma_db_v3"))
    collection: str = os.getenv("ATHENA_COLLECTION", "athena")

    # Embeddings, chosen by the eval (README, "Embedding model"): bge-small
    # beat all-MiniLM-L6-v2 on the demo set. Both truncate at 512/256 tokens.
    embedding_model: str = os.getenv("ATHENA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Chunking (characters). ~800 chars of English is ~200 tokens, which
    # leaves headroom under the embedder's 256-token limit.
    chunk_size: int = _int("ATHENA_CHUNK_SIZE", 800)
    chunk_overlap: int = _int("ATHENA_CHUNK_OVERLAP", 150)

    # Retrieval: "dense" (embeddings), "keyword" (BM25), "hybrid" (both,
    # fused with reciprocal rank fusion). The reranker then re-scores the
    # fused candidates with a cross-encoder that reads query and chunk together.
    retrieval_mode: str = os.getenv("ATHENA_RETRIEVAL_MODE", "hybrid")
    candidates: int = _int("ATHENA_CANDIDATES", 30)  # per retriever, before fusion
    rerank: bool = _bool("ATHENA_RERANK", True)
    rerank_pool: int = _int("ATHENA_RERANK_POOL", 20)  # fused candidates the reranker reads
    reranker_model: str = os.getenv("ATHENA_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Below this reranker score (a logit) the best chunk is judged not to
    # answer the question, and Athena says "not found" without calling the
    # LLM. Tuned on the eval set; see README "Abstaining".
    min_rerank_score: float = _float("ATHENA_MIN_RERANK_SCORE", -6.0)
    top_k: int = _int("ATHENA_TOP_K", 6)  # eval: 6 x 800 chars beat 4 x 1200 at the same context size
    # Rough budget for retrieved text sent to the LLM (chars; ~4 chars/token).
    max_context_chars: int = _int("ATHENA_MAX_CONTEXT_CHARS", 6000)

    # LLM (Ollama)
    ollama_url: str = os.getenv("ATHENA_OLLAMA_URL", "http://localhost:11434")
    # gemma3:4b: measured ~35 output tokens for a one-sentence answer on this
    # CPU vs 265-1065 for qwen3:4b, which keeps reasoning even with think=false.
    llm_model: str = os.getenv("ATHENA_LLM_MODEL", "gemma3:4b")
    llm_num_ctx: int = _int("ATHENA_LLM_NUM_CTX", 4096)
    llm_temperature: float = _float("ATHENA_LLM_TEMPERATURE", 0.2)
    llm_timeout_seconds: int = _int("ATHENA_LLM_TIMEOUT", 300)
    # Keep the model in memory between questions: reloading gemma3:4b from
    # disk costs ~15-20 s on this laptop, more than writing the answer.
    llm_keep_alive: str = os.getenv("ATHENA_LLM_KEEP_ALIVE", "30m")
    # CPU threads for the model (0 = Ollama decides). On an i5-1335U, 8
    # threads read prompts ~25% faster than the default; tune per machine.
    llm_threads: int = _int("ATHENA_LLM_THREADS", 0)


    @property
    def collection_name(self) -> str:
        """One collection per embedding model: vectors from different models
        can't be compared, so switching models must not mix them."""
        return f"{self.collection}-{_slug(self.embedding_model)}"[:60]

    @property
    def embedding_query_prefix(self) -> str:
        override = os.getenv("ATHENA_EMBEDDING_QUERY_PREFIX")
        if override is not None:
            return override
        name = self.embedding_model.lower()
        return next((p for key, p in _QUERY_PREFIXES.items() if key in name), "")


settings = Settings()
