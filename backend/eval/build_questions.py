"""Build eval/demo_questions.jsonl from eval/questions_source.py.

Every question carries an evidence quote. This script proves the quote
really is in the expected document and records which pages it is on, so
the eval never relies on a page number typed from memory:

    python -m eval.build_questions        (run from backend/)
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.document_reader import read_any
from eval.questions_source import Q

CORPUS = Path(__file__).resolve().parent.parent.parent / "demo_corpus"
OUT = Path(__file__).resolve().parent / "demo_questions.jsonl"


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub("[‐-–]", "-", text).replace("’", "'")
    return re.sub(r"\s+", " ", text).lower()


def load_pages() -> dict[str, dict[int | None, str]]:
    docs: dict[str, dict[int | None, str]] = {}
    for path in CORPUS.rglob("*.*"):
        if path.name == "SOURCES.md":
            continue
        if path.suffix == ".pdf" and not path.name.startswith("Scanned"):
            with pymupdf.open(path) as doc:
                docs[path.name] = {i + 1: norm(p.get_text().replace("-\n", "")) for i, p in enumerate(doc)}
        elif path.suffix == ".pdf":  # scanned: only OCR text exists
            docs[path.name] = {s.page: norm(s.text) for s in read_any(str(path)).sections}
        else:
            docs[path.name] = {None: norm("\n".join(s.text for s in read_any(str(path)).sections))}
    return docs


def main() -> int:
    docs = load_pages()
    rows, missing = [], 0
    for qid, kind, question, evidence, files, keywords in Q:
        row = {"id": qid, "type": kind, "question": question, "answer_keywords": keywords, "expected": []}
        if evidence:
            for name in files.split("|"):
                pages = [p for p, text in docs[name].items() if norm(evidence) in text]
                if pages:
                    row["expected"].append({"file": name, "pages": [p for p in pages if p is not None]})
            if not row["expected"]:
                print(f"NOT FOUND: {qid}: {evidence!r}")
                missing += 1
                continue
            row["evidence"] = evidence
            # Short quotes (headings, single terms) occur in many chunks, so they
            # don't prove the right passage was found: only file/page are scored.
            row["specific"] = len(evidence) >= 32
        rows.append(row)

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} questions written to {OUT.name}; {missing} evidence quotes not found")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
