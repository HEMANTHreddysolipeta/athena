"""Command line: index a folder of documents.

    python -m athena.index_folder              # indexes settings.data_dir
    python -m athena.index_folder D:\\notes     # indexes another folder
    python -m athena.index_folder --reset      # wipe the collection first

Nothing runs on import; the old version reset the database as soon as it
was imported.
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import settings
from .folder_indexer import index_folder
from .vector_store import get_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index documents (PDF, DOCX, PPTX, MD, TXT, HTML) into Athena.")
    parser.add_argument("folder", nargs="?", type=Path, default=settings.data_dir)
    parser.add_argument("--reset", action="store_true", help="delete all indexed chunks first")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.folder.is_dir():
        print(f"Folder not found: {args.folder}", file=sys.stderr)
        return 1

    store = get_store()
    if args.reset:
        store.reset()

    report = index_folder(args.folder, store)

    print("\n========== INDEXING COMPLETE ==========")
    for status, n in sorted(report.counts.items()):
        print(f"{status:>10}: {n}")
    print(f"new chunks: {report.chunks}")
    print(f"in store  : {store.count()}")
    for name, error in report.failures:
        print(f"  failed: {name}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
