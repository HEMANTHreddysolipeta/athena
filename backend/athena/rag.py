"""Retrieval-Augmented Generation.

    question ─(follow-up? rewrite with chat history)─► standalone question
      ─► hybrid retrieval + rerank ─► confident? ── no ─► "not found" (no LLM call)
      ─► numbered-source prompt ─► local LLM (streamed)
      ─► parse [n] citations ─► grounding check ─► answer

The LLM never sees the whole library, only the few chunks retrieval
picked, so answer quality is capped by retrieval quality. Two guards keep
a small local model honest: it is not asked at all when retrieval found
nothing relevant, and its specific claims (numbers, codes) are checked
against the sources it cited.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from . import llm
from .config import settings
from .grounding import Grounding, check_grounding
from .retriever import Reranker, Retrieval, retrieve, sigmoid
from .vector_store import RetrievedChunk, VectorStore, get_store

logger = logging.getLogger(__name__)

NOT_FOUND_TOKEN = "NOT_FOUND"
NOT_FOUND_MESSAGE = "I couldn't find that information in your documents."

PROMPT_TEMPLATE = """Answer the question using ONLY the numbered sources below.
Rules:
- Cite the source of every fact right after it, like [1] or [2][3].
- Copy numbers, names and terms exactly as the sources write them.
- Be concise: at most 5 sentences or a short list.
- If the sources do not contain the answer, reply with exactly: {not_found}

SOURCES:
{sources}

QUESTION: {question}

ANSWER:"""

REWRITE_TEMPLATE = """Rewrite the follow-up question so it can be understood without the conversation.
Keep it short. Reply with the rewritten question only.

CONVERSATION:
{history}

FOLLOW-UP: {question}

STANDALONE QUESTION:"""

# Words that usually mean a question leans on the previous turn.
_FOLLOW_UP = re.compile(
    r"\b(it|its|it's|this|that|these|those|they|them|their|he|she|his|her|there|same|above|"
    r"previous|earlier|former|latter)\b|^(and|also|what about|how about|why|so|but|then)\b",
    re.IGNORECASE,
)


@dataclass
class Source:
    number: int
    filename: str
    page: int | None
    distance: float
    text: str
    title: str = ""
    section: str | None = None
    relevance: float | None = None  # 0..1 from the reranker, when it ran
    chunk_id: str = ""

    def label(self) -> str:
        page = f", page {self.page}" if self.page is not None else ""
        return f"[{self.number}] {self.filename}{page}"

    def to_dict(self) -> dict:
        return {"number": self.number, "filename": self.filename, "page": self.page,
                "title": self.title, "section": self.section, "relevance": self.relevance,
                "text": self.text, "chunk_id": self.chunk_id}


@dataclass
class RagAnswer:
    question: str
    answer: str
    found: bool
    cited: list[Source] = field(default_factory=list)  # sources the answer referenced
    context: list[Source] = field(default_factory=list)  # everything sent to the LLM
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    standalone_question: str | None = None  # set when a follow-up was rewritten
    grounding: Grounding | None = None
    abstained_early: bool = False  # "not found" decided by retrieval, LLM not called
    top_score: float | None = None
    llm_stats: dict = field(default_factory=dict)


def select_context(chunks: list[RetrievedChunk], max_chars: int) -> list[Source]:
    """Keep chunks in rank order until the character budget is used up."""
    sources, used = [], 0
    for chunk in chunks:
        if sources and used + len(chunk.text) > max_chars:
            break
        sources.append(
            Source(
                number=len(sources) + 1,
                filename=chunk.metadata.get("filename", "unknown"),
                page=chunk.metadata.get("page"),
                distance=chunk.distance,
                text=chunk.text,
                title=chunk.metadata.get("title", ""),
                section=chunk.metadata.get("section"),
                relevance=round(sigmoid(chunk.rerank_score), 3) if chunk.rerank_score is not None else None,
                chunk_id=chunk.id,
            )
        )
        used += len(chunk.text)
    return sources


def build_prompt(question: str, sources: list[Source]) -> str:
    def block(s: Source) -> str:
        where = f" > {s.section}" if s.section else ""
        return f"{s.label()}{where}\n{s.text}"

    return PROMPT_TEMPLATE.format(
        not_found=NOT_FOUND_TOKEN, sources="\n\n".join(block(s) for s in sources), question=question
    )


def parse_citations(answer: str, sources: list[Source]) -> list[Source]:
    # Models cite as [1], [1][2] or [1, 2, 4]; accept all three.
    numbers = {
        int(n)
        for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer)
        for n in group.split(",")
    }
    return [s for s in sources if s.number in numbers]


def needs_rewrite(question: str, history: list[tuple[str, str]] | None) -> bool:
    if not history:
        return False
    return bool(_FOLLOW_UP.search(question.strip())) or len(question.split()) <= 3


def rewrite_question(question: str, history: list[tuple[str, str]], generate: Callable[[str], str]) -> str:
    turns = []
    for q, a in history[-2:]:  # the last two turns are enough context, and keep the prompt short
        turns.append(f"User: {q}\nAssistant: {a[:400]}")
    rewritten = generate(REWRITE_TEMPLATE.format(history="\n".join(turns), question=question)).strip()
    rewritten = rewritten.strip().strip('"').splitlines()[0] if rewritten else ""
    # A runaway or empty rewrite is worse than the original question.
    return rewritten if 3 <= len(rewritten) <= 300 else question


def stream_answer(
    question: str,
    store: VectorStore | None = None,
    generate: Callable[[str], str] | None = None,
    generate_stream: Callable[[str], Iterator[str]] | None = None,
    top_k: int | None = None,
    max_context_chars: int | None = None,
    files: list[str] | None = None,
    history: list[tuple[str, str]] | None = None,
    reranker: Reranker | None = None,
    rerank: bool | None = None,
) -> Iterator[dict]:
    """The RAG pipeline as events, for streaming UIs:

    {"type": "rewrite", "question": ...}        follow-up rewritten (optional)
    {"type": "sources", "sources": [...]}       what the model will read
    {"type": "token", "text": ...}              answer text, as generated
    {"type": "done", "result": RagAnswer}       final answer with checks
    """
    store = store or get_store()
    top_k = top_k or settings.top_k
    max_context_chars = max_context_chars or settings.max_context_chars
    if generate is None and generate_stream is None:
        generate, generate_stream = llm.generate, llm.generate_stream
    elif generate_stream is None:
        def generate_stream(prompt, _generate=generate):
            yield _generate(prompt)
    elif generate is None:
        def generate(prompt, _stream=generate_stream):
            return "".join(_stream(prompt))

    query = question
    standalone = None
    if needs_rewrite(question, history):
        query = standalone = rewrite_question(question, history, generate)
        yield {"type": "rewrite", "question": standalone}

    retrieval: Retrieval = retrieve(query, store, k=top_k, files=files, reranker=reranker, rerank=rerank)
    base = dict(question=question, standalone_question=standalone,
                retrieval_seconds=retrieval.seconds, top_score=retrieval.top_score)

    if not retrieval.chunks:
        yield {"type": "done", "result": RagAnswer(answer=NOT_FOUND_MESSAGE, found=False, **base)}
        return

    sources = select_context(retrieval.chunks, max_context_chars)
    yield {"type": "sources", "sources": [s.to_dict() for s in sources]}

    if not retrieval.confident:
        logger.info("abstaining: best reranker score %.2f < %.2f", retrieval.top_score, settings.min_rerank_score)
        yield {"type": "done", "result": RagAnswer(answer=NOT_FOUND_MESSAGE, found=False, context=sources,
                                                   abstained_early=True, **base)}
        return

    started = time.perf_counter()
    pieces: list[str] = []
    held = True  # hold output while it could still be the NOT_FOUND token
    for piece in generate_stream(build_prompt(query, sources)):
        pieces.append(piece)
        if held:
            so_far = "".join(pieces).lstrip()
            if so_far and not NOT_FOUND_TOKEN.startswith(so_far[: len(NOT_FOUND_TOKEN)]):
                held = False
                yield {"type": "token", "text": so_far}  # flush everything held so far
        else:
            yield {"type": "token", "text": piece}
    raw = "".join(pieces).strip()
    generation_seconds = time.perf_counter() - started
    stats = dict(llm.last_stats)

    if NOT_FOUND_TOKEN in raw and len(raw) < 80:
        yield {"type": "done", "result": RagAnswer(answer=NOT_FOUND_MESSAGE, found=False, context=sources,
                                                   generation_seconds=generation_seconds, llm_stats=stats, **base)}
        return
    raw = raw.replace(NOT_FOUND_TOKEN, "").strip()

    yield {"type": "done", "result": RagAnswer(
        answer=raw, found=True, cited=parse_citations(raw, sources), context=sources,
        grounding=check_grounding(raw, [s.text for s in sources]),
        generation_seconds=generation_seconds, llm_stats=stats, **base,
    )}


def answer_question(question: str, **kwargs) -> RagAnswer:
    """Run the whole pipeline and return the final answer (CLI, eval, tests)."""
    result = None
    for event in stream_answer(question, **kwargs):
        if event["type"] == "done":
            result = event["result"]
    return result


def format_answer(result: RagAnswer) -> str:
    """Human-readable answer with citations, for the CLI and the chat API."""
    if not result.found:
        return result.answer
    cited = result.cited or result.context  # model forgot to cite: show what it saw
    heading = "Sources:" if result.cited else "Sources (not cited explicitly):"
    lines = [result.answer]
    if result.grounding and not result.grounding.supported:
        if result.grounding.unsupported:
            lines.append(f"\n⚠ Not found in the cited sources: {', '.join(result.grounding.unsupported)}"
                         " — check these before relying on them.")
        if result.grounding.invalid_citations:
            lines.append(f"\n⚠ The answer cites sources that don't exist: {result.grounding.invalid_citations}")
    lines += ["", heading] + [s.label() for s in cited]
    return "\n".join(lines)
