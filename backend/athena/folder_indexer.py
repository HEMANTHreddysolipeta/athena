import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .document_reader import SUPPORTED_EXTENSIONS
from .indexer import IndexStatus, index_document
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class FolderReport:
    counts: Counter = field(default_factory=Counter)
    chunks: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


# Housekeeping files that describe a folder rather than belong in it.
SKIP_NAMES = {"readme.md", "sources.md", "license", "license.md"}


def find_documents(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.startswith("~$")  # Word lock files
        and p.name.lower() not in SKIP_NAMES
    )


def index_folder(folder: Path, store: VectorStore) -> FolderReport:
    report = FolderReport()
    files = find_documents(folder)
    logger.info("Found %d documents in %s", len(files), folder)

    for i, path in enumerate(files, start=1):
        try:
            result = index_document(str(path), store)
        except Exception as exc:  # one bad file must not stop the run
            report.counts["failed"] += 1
            report.failures.append((path.name, str(exc)))
            logger.warning("[%d/%d] FAILED %s: %s", i, len(files), path.name, exc)
            continue

        report.counts[result.status.value] += 1
        report.chunks += result.chunks
        detail = f"{result.chunks} chunks" if result.status == IndexStatus.INDEXED else result.status.value
        logger.info("[%d/%d] %s: %s", i, len(files), path.name, detail)

    return report
