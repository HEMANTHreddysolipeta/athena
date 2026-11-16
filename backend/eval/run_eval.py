"""Evaluate Athena on the demo corpus (demo_corpus/ + eval/demo_questions.jsonl).

    python eval/run_eval.py retrieval                       # current settings, seconds
    python eval/run_eval.py retrieval --modes all           # dense / keyword / hybrid / hybrid+rerank
    python eval/run_eval.py retrieval --embeddings all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5
    python eval/run_eval.py answers [--limit 10]            # full RAG with the local LLM (slow on CPU)

The eval builds its own index under eval/.index/ from demo_corpus/, separate
from your personal library, so the numbers are reproducible by anyone.

Retrieval metrics (answerable questions):
  hit@1 / hit@k   an expected file is the top result / anywhere in the top k
  MRR             mean of 1/rank of the first chunk from an expected file
  page@k          a top-k chunk is on an expected page (questions with pages)

Answer metrics:
  answered        answerable questions that did not get "not found"
  correct         every answer-keyword group appears in the answer
  cited right     a cited source is an expected file
  grounded        the grounding check found no unsupported numbers/terms
  refused         unanswerable questions correctly answered "not found"
"""

import argparse
import hashlib
import json
import logging
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.config import PROJECT_DIR, settings  # noqa: E402
from athena.folder_indexer import index_folder  # noqa: E402
from athena.vector_store import SentenceTransformerEmbedder, VectorStore  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = PROJECT_DIR / "demo_corpus"
INDEX_DIR = EVAL_DIR / ".index"
RESULTS_DIR = EVAL_DIR / "results"
MODES = ["dense", "keyword", "hybrid", "hybrid+rerank"]


def load_questions(path: Path = EVAL_DIR / "demo_questions.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_fingerprint() -> str:
    sha = hashlib.sha256()
    for p in sorted(CORPUS_DIR.rglob("*")):
        if p.is_file():
            sha.update(p.name.encode())
            sha.update(str(p.stat().st_size).encode())
    # Chunking settings and the reading/chunking code change the index too.
    sha.update(f"{settings.chunk_size}/{settings.chunk_overlap}".encode())
    for module in ("document_reader.py", "pdf_reader.py", "chunker.py", "indexer.py", "ocr.py", "folder_indexer.py"):
        sha.update((Path(__file__).resolve().parent.parent / "athena" / module).read_bytes())
    return sha.hexdigest()[:16]


def eval_store(embedding_model: str | None = None) -> VectorStore:
    """The demo-corpus index for one embedding model, (re)built when stale."""
    model = embedding_model or settings.embedding_model
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    prefix = "Represent this sentence for searching relevant passages: " if "bge" in model.lower() else ""
    store = VectorStore(INDEX_DIR, f"demo-{slug}"[:60], SentenceTransformerEmbedder(model), query_prefix=prefix)

    stamp = INDEX_DIR / f"demo-{slug}.fingerprint"
    fingerprint = corpus_fingerprint()
    if store.count() == 0 or not stamp.exists() or stamp.read_text() != fingerprint:
        print(f"Indexing demo corpus with {model} ...", flush=True)
        started = time.perf_counter()
        store.reset()
        report = index_folder(CORPUS_DIR, store)
        stamp.write_text(fingerprint)
        print(f"  {store.count()} chunks in {time.perf_counter() - started:.0f}s; "
              f"failures: {report.failures or 'none'}", flush=True)
    return store


def expected_files(q: dict) -> set[str]:
    return {e["file"] for e in q["expected"]}


def expected_pages(q: dict) -> set[tuple[str, int]]:
    return {(e["file"], p) for e in q["expected"] for p in e["pages"]}


def _norm(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKC", text).replace("’", "'")
    return re.sub(r"\s+", " ", re.sub("[‐-–]", "-", text)).lower()


def evidence_rank(q: dict, chunks) -> int | None:
    """Rank of the first chunk that contains the question's evidence quote."""
    evidence = _norm(q.get("evidence", ""))
    if not evidence:
        return None
    # A quote may straddle two chunks; its first 60 chars are enough to locate it.
    probe = evidence[:60]
    return next((i + 1 for i, c in enumerate(chunks) if probe in _norm(c.text)), None)


def keyword_match(answer: str, groups: list[str]) -> bool:
    text = answer.lower()
    return all(any(alt.lower() in text for alt in group.split("|")) for group in groups)


# ----------------------------------------------------------- retrieval ----

def run_retrieval(store: VectorStore, questions: list[dict], mode: str, k: int) -> dict:
    from athena.retriever import retrieve

    retrieval_mode, rerank = (mode.split("+")[0], mode.endswith("+rerank"))
    rows = []
    for q in questions:
        if not q["expected"]:
            # Unanswerable: record how confident retrieval was (for abstention tuning).
            result = retrieve(q["question"], store, k=k, mode=retrieval_mode, rerank=rerank)
            rows.append({"id": q["id"], "answerable": False, "top_score": result.top_score,
                         "seconds": result.seconds})
            continue
        result = retrieve(q["question"], store, k=k, mode=retrieval_mode, rerank=rerank)
        files = [c.metadata["filename"] for c in result.chunks]
        want = expected_files(q)
        rank = next((i + 1 for i, f in enumerate(files) if f in want), None)
        pages = expected_pages(q)
        page_hit = (any((c.metadata["filename"], c.metadata.get("page")) in pages for c in result.chunks)
                    if pages else None)
        ev = evidence_rank(q, result.chunks) if q.get("specific", True) else None
        rows.append({"id": q["id"], "type": q["type"], "answerable": True, "rank": rank, "evidence_rank": ev,
                     "evidence_scored": q.get("specific", True),
                     "page_hit": page_hit, "top_file": files[0] if files else None,
                     "top_score": result.top_score, "seconds": result.seconds})
    return summarize_retrieval(rows, k) | {"rows": rows}


def summarize_retrieval(rows: list[dict], k: int) -> dict:
    ans = [r for r in rows if r["answerable"]]
    ranks = [r["rank"] for r in ans]
    pages = [r["page_hit"] for r in ans if r["page_hit"] is not None]
    scored = [r for r in ans if r.get("evidence_scored")]
    return {
        "ev_n": len(scored),
        "evidence@1": sum(r["evidence_rank"] == 1 for r in scored),
        f"evidence@{k}": sum(r["evidence_rank"] is not None for r in scored),
        "n": len(ans),
        "hit@1": sum(r == 1 for r in ranks),
        f"hit@{k}": sum(r is not None for r in ranks),
        "mrr": round(statistics.mean(1 / r if r else 0 for r in ranks), 3),
        "page_n": len(pages),
        f"page@{k}": sum(pages),
        "median_ms": round(statistics.median(r["seconds"] for r in rows) * 1000),
    }


def cmd_retrieval(args) -> dict:
    questions = load_questions()
    k = args.k
    modes = MODES if args.modes == "all" else args.modes.split(",")
    models = args.embeddings.split(",") if args.embeddings else [settings.embedding_model]
    report = {"kind": "retrieval", "k": k, "results": []}

    for model in models:
        store = eval_store(model)
        for mode in modes:
            if mode != modes[0] or model != models[0]:
                pass
            run_retrieval(store, questions[:1], mode, k)  # warm-up: model loading isn't latency
            result = run_retrieval(store, questions, mode, k)
            result |= {"embedding": model, "mode": mode, "chunks": store.count()}
            report["results"].append(result)

    print(f"\nRETRIEVAL on {report['results'][0]['n']} answerable questions, k={k}")
    header = (f"{'embedding':24} {'mode':14} {'file@1':>6} {'file@' + str(k):>6} {'MRR':>6} {'page@' + str(k):>8}"
              f" {'evid@1':>7} {'evid@' + str(k):>7} {'ms':>6}")
    print(header + "\n" + "-" * len(header))
    for r in report["results"]:
        print(f"{r['embedding'][:24]:24} {r['mode']:14} {r['hit@1']:>3}/{r['n']:<2} {r[f'hit@{k}']:>3}/{r['n']:<2}"
              f" {r['mrr']:>6.3f} {r[f'page@{k}']:>4}/{r['page_n']:<3} {r['evidence@1']:>3}/{r['ev_n']:<3}"
              f" {r[f'evidence@{k}']:>3}/{r['ev_n']:<3} {r['median_ms']:>6}")
    if args.show_misses:
        for r in report["results"]:
            misses = [row for row in r["rows"] if row["answerable"] and row["rank"] != 1]
            print(f"\n{r['embedding']} / {r['mode']} misses: " +
                  ", ".join(f"{m['id']}(rank={m['rank']})" for m in misses))
    return report


# ------------------------------------------------------------- answers ----

def cmd_answers(args) -> dict:
    from athena.llm import LLMError
    from athena.rag import answer_question

    questions = load_questions()
    if args.only:
        questions = [q for q in questions if q["id"] in args.only.split(",")]
    if args.limit:
        questions = questions[: args.limit]
    store = eval_store()
    run_retrieval(store, questions[:1], "hybrid+rerank", settings.top_k)  # warm-up

    rows = []
    for i, q in enumerate(questions, start=1):
        print(f"  [{i}/{len(questions)}] {q['id']} ...", end=" ", flush=True)
        started = time.perf_counter()
        try:
            result = answer_question(q["question"], store=store)
        except LLMError as exc:
            print("ERROR", exc)
            rows.append({"id": q["id"], "error": str(exc)})
            continue
        answerable = bool(q["expected"])
        row = {
            "id": q["id"], "type": q["type"], "answerable": answerable, "found": result.found,
            "correct": keyword_match(result.answer, q["answer_keywords"]) if answerable and result.found else False,
            "cited_right": any(s.filename in expected_files(q) for s in result.cited) if answerable else None,
            "grounded": result.grounding.supported if result.grounding else None,
            "unsupported": result.grounding.unsupported if result.grounding else [],
            "abstained_early": result.abstained_early,
            "seconds": round(time.perf_counter() - started, 1),
            "llm_seconds": round(result.generation_seconds, 1),
            "answer": result.answer,
        }
        rows.append(row)
        verdict = ("refused" if not result.found else
                   ("correct" if row["correct"] else "WRONG?")) if answerable else \
                  ("refused (good)" if not result.found else "ANSWERED (bad)")
        print(f"{verdict} {row['seconds']}s", flush=True)

    ok = [r for r in rows if "error" not in r]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]
    found = [r for r in ans if r["found"]]
    summary = {
        "n_answerable": len(ans), "n_unanswerable": len(unans), "errors": len(rows) - len(ok),
        "answered": sum(r["found"] for r in ans),
        "correct": sum(r["correct"] for r in ans),
        "cited_right": sum(bool(r["cited_right"]) for r in found),
        "grounded": sum(r["grounded"] is True for r in found),
        "refused_unanswerable": sum(not r["found"] for r in unans),
        "median_s": statistics.median(r["seconds"] for r in ok) if ok else None,
        "max_s": max((r["seconds"] for r in ok), default=None),
    }
    print(f"\nANSWERS (model {settings.llm_model}, {settings.retrieval_mode}"
          f"{'+rerank' if settings.rerank else ''}, k={settings.top_k})")
    for key, value in summary.items():
        print(f"  {key:22}: {value}")
    return {"kind": "answers", "model": settings.llm_model, "summary": summary, "rows": rows}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--modes", default="hybrid+rerank", help="comma list, or 'all'")
    r.add_argument("--embeddings", default=None, help="comma list of embedding models")
    r.add_argument("-k", type=int, default=settings.top_k)
    r.add_argument("--show-misses", action="store_true")
    r.add_argument("--min-evidence", type=int, default=None,
                   help="exit 1 if the last configuration finds fewer exact passages in the top k (CI gate)")
    a = sub.add_parser("answers")
    a.add_argument("--limit", type=int)
    a.add_argument("--only", help="comma list of question ids")
    args = parser.parse_args()

    report = cmd_retrieval(args) if args.command == "retrieval" else cmd_answers(args)
    report |= {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "settings": {k: str(v) for k, v in vars(settings).items() if "dir" not in k}}
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.command}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nFull results: {out}")

    if args.command == "retrieval" and args.min_evidence is not None:
        found = report["results"][-1][f"evidence@{args.k}"]
        if found < args.min_evidence:
            print(f"FAIL: evidence@{args.k} = {found}, required >= {args.min_evidence}")
            sys.exit(1)
        print(f"PASS: evidence@{args.k} = {found} >= {args.min_evidence}")


if __name__ == "__main__":
    main()
