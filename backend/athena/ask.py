"""Command line: ask Athena a question about your indexed documents.

    python -m athena.ask "..." --file NIST.SP.800-63B-4_Authentication.pdf   # one document only

    python -m athena.ask "What is technical SEO?"
    python -m athena.ask "..." --show-context     # print the retrieved chunks
"""

import argparse
import logging
import sys

from .llm import LLMError
from .rag import answer_question, format_answer


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; document text contains characters
    # (e.g. zero-width spaces) that crash print() under that encoding.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Ask a question about your documents.")
    parser.add_argument("question")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--file", action="append", help="only search this document (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        result = answer_question(args.question, files=args.file)
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(format_answer(result))
    score = f", best match {result.top_score:.1f}" if result.top_score is not None else ""
    early = " (decided without the LLM)" if result.abstained_early else ""
    print(f"\n(retrieval {result.retrieval_seconds:.2f}s{score}, generation {result.generation_seconds:.1f}s{early})")

    if args.show_context:
        for s in result.context:
            where = f" > {s.section}" if s.section else ""
            relevance = f"  relevance={s.relevance:.2f}" if s.relevance is not None else ""
            print(f"\n----- {s.label()}{where}{relevance}\n{s.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
