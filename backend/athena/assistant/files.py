"""Find the user's files by name, type, date or size, or by what is inside them.

Only the user's own files are searched: everything under the home folder
except app data, hidden folders and code dependencies (node_modules, .git,
virtualenvs, build output). On a real laptop that is ~20,000 files out of
1.5 million, and listing them takes about a second, so the list is simply
re-read when it is a few minutes old instead of being kept in a database.

"What is inside" works for documents (PDF, Word, PowerPoint, text, Markdown,
HTML): the first ~1,500 characters of each are turned into a vector with the
same local embedding model the document library uses, in a background job.
Only the vectors, paths and dates are saved (in Athena's private folder);
the text itself is not stored.
"""

import difflib
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import audit, policy
from .security import app_home

logger = logging.getLogger(__name__)

SKIP_DIRS = {"appdata", "library", "node_modules", ".git", "venv", ".venv", "env", "site-packages",
             "__pycache__", ".gradle", ".m2", ".cache", "build", "dist", ".next", "target",
             ".dart_tool", ".idea", ".vscode", "bin", "obj", ".expo", "pods", "deriveddata"}

KIND_EXTS = {
    "pdf": {".pdf"},
    "document": {".pdf", ".doc", ".docx", ".odt", ".rtf", ".txt", ".md", ".pages"},
    "spreadsheet": {".xlsx", ".xls", ".csv", ".ods", ".numbers", ".tsv"},
    "presentation": {".pptx", ".ppt", ".key", ".odp"},
    "image": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".bmp", ".svg", ".tif", ".tiff", ".raw"},
    "video": {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".3gp"},
    "audio": {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"},
    "archive": {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2", ".xz"},
    "installer": {".exe", ".msi", ".msix", ".appx", ".dmg", ".pkg", ".deb", ".rpm", ".apk", ".aab"},
    "code": {".py", ".js", ".ts", ".tsx", ".jsx", ".kt", ".java", ".c", ".cpp", ".h", ".cs", ".go",
             ".rs", ".rb", ".php", ".swift", ".dart", ".pde", ".ipynb", ".sql", ".sh", ".ps1", ".html", ".css"},
    "text": {".txt", ".md", ".log", ".json", ".csv", ".xml", ".yaml", ".yml", ".ini", ".toml"},
}
# OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS: OneDrive/iCloud files whose
# content is in the cloud. Reading one would download it.
_CLOUD_ONLY = 0x1000 | 0x40000 | 0x400000
CONTENT_EXTS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown", ".html", ".htm"}

# Words that describe the search rather than the file ("find my", "file").
_STOP = set(["a", "an", "the", "my", "me", "i", "im", "i'm", "you", "your", "we", "our", "is", "are", "was", "were", "be", "been", "do", "did", "does", "have", "has", "had", "can", "could", "would", "should", "will", "please", "pls", "plz", "find", "search", "look", "looking", "locate", "where", "wheres", "where's", "whats", "what's", "what", "which", "who", "show", "list", "give", "get", "open", "see", "check", "tell", "any", "all", "some", "every", "file", "files", "folder", "folders", "thing", "things", "stuff", "one", "ones", "that", "this", "these", "those", "it", "its", "for", "of", "in", "on", "at", "to", "from", "with", "about", "by", "into", "inside", "laptop", "computer", "pc", "system", "drive", "disk", "somewhere", "saved", "save", "put", "kept", "keep", "stored", "got", "named", "called", "name", "related", "regarding", "there", "here", "need", "want", "lost", "latest", "newest", "recent", "recently", "new", "old", "oldest", "biggest", "largest", "big", "huge", "heavy", "heaviest", "large", "small", "last", "past", "this", "week", "weeks", "month", "months", "year", "years", "day", "days", "today", "yesterday", "ago", "downloaded", "download", "made", "created", "sent", "received"])

_KIND_WORDS = {
    "pdf": "pdf", "pdfs": "pdf",
    "doc": "document", "docs": "document", "docx": "document", "word": "document",
    "document": "document", "documents": "document",
    "excel": "spreadsheet", "xlsx": "spreadsheet", "xls": "spreadsheet", "csv": "spreadsheet",
    "spreadsheet": "spreadsheet", "spreadsheets": "spreadsheet", "sheet": "spreadsheet", "sheets": "spreadsheet",
    "ppt": "presentation", "pptx": "presentation", "slides": "presentation", "slide": "presentation",
    "presentation": "presentation", "presentations": "presentation", "deck": "presentation",
    "photo": "image", "photos": "image", "image": "image", "images": "image", "picture": "image",
    "pictures": "image", "pic": "image", "pics": "image", "screenshot": "image", "screenshots": "image",
    "png": "image", "jpg": "image", "jpeg": "image",
    "video": "video", "videos": "video", "movie": "video", "movies": "video", "clip": "video",
    "clips": "video", "mp4": "video", "mov": "video", "recording": "video", "recordings": "video",
    "audio": "audio", "song": "audio", "songs": "audio", "music": "audio", "mp3": "audio",
    "zip": "archive", "zips": "archive", "archive": "archive", "archives": "archive", "rar": "archive",
    "installer": "installer", "installers": "installer", "setup": "installer", "exe": "installer",
    "msi": "installer", "apk": "installer", "apks": "installer",
    "code": "code", "script": "code", "scripts": "code",
}


@dataclass
class FileInfo:
    path: str
    name: str
    size: int
    mtime: float
    ext: str
    rel_parts: tuple[str, ...]
    online_only: bool = False

    def kind(self) -> str:
        for kind, exts in KIND_EXTS.items():
            if kind not in ("document", "text") and self.ext in exts:
                return kind
        return "document" if self.ext in KIND_EXTS["document"] else "other"

    def public(self) -> dict:
        return {"path": self.path, "name": self.name, "bytes": self.size, "modified": self.mtime,
                "kind": self.kind(), "folder": policy.display(os.path.dirname(self.path)),
                "online_only": self.online_only}


def _words(text: str) -> list[str]:
    """'Hemanth_Reddy_Resume.pdf' -> ['hemanth', 'reddy', 'resume', 'pdf'];
    camelCase and digits are split too ('invoices2023' -> 'invoices', '2023')."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [w for w in re.split(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", text.lower()) if w]


# ------------------------------------------------------------- catalogue

class Catalog:
    """The list of the user's own files, re-read when older than max_age."""

    def __init__(self, max_age: float = 300):
        self.max_age = max_age
        self._files: list[FileInfo] = []
        self._vocab: set[str] = set()
        self._df: Counter = Counter()  # how many file names contain each word
        self._read_at = 0.0
        self._root: str | None = None
        self._lock = threading.Lock()

    def files(self) -> list[FileInfo]:
        root = str(policy.home_root())
        with self._lock:
            if root != self._root or time.time() - self._read_at > self.max_age:
                self._files = list(walk_personal(Path(root)))
                self._df = Counter(w for f in self._files for w in set(_words(Path(f.name).stem)))
                self._vocab = set(self._df)
                self._read_at, self._root = time.time(), root
            return self._files

    def vocab(self) -> set[str]:
        self.files()
        return self._vocab

    def weight(self, word: str) -> float:
        """Rare words say more: 'aadhaar' (in 6 names) outweighs 'card' (in 80)."""
        self.files()
        return math.log(1 + len(self._files) / (1 + self._df.get(word, 0)))

    def invalidate(self) -> None:
        self._read_at = 0.0


def info(path: str) -> dict | None:
    """The same summary as a search result, for one known file (None if gone)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    online = bool(getattr(st, "st_file_attributes", 0) & _CLOUD_ONLY)
    return FileInfo(path, os.path.basename(path), st.st_size, st.st_mtime,
                    os.path.splitext(path)[1].lower(), (), online).public()


def walk_personal(root: Path):
    root_s = str(root)
    stack = [(root_s, ())]
    while stack:
        folder, parts = stack.pop()
        try:
            entries = os.scandir(folder)
        except OSError:
            continue
        with entries:
            for e in entries:
                try:
                    if e.is_symlink() or e.is_junction():
                        continue
                    if e.is_dir(follow_symlinks=False):
                        name = e.name.lower()
                        if name in SKIP_DIRS or name.startswith(".") or name.startswith("$"):
                            continue
                        stack.append((e.path, (*parts, e.name)))
                        continue
                    if e.name.startswith((".", "~$")) or e.name.lower() in ("desktop.ini", "thumbs.db"):
                        continue
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                online = bool(getattr(st, "st_file_attributes", 0) & _CLOUD_ONLY)
                yield FileInfo(e.path, e.name, st.st_size, st.st_mtime,
                               os.path.splitext(e.name)[1].lower(), parts, online)


_catalog = Catalog()


def catalog() -> Catalog:
    return _catalog


# ---------------------------------------------------------------- filters

def parse_filters(message: str) -> dict:
    """Type, date and order words in a request: 'pdfs from last week' ->
    kind=pdf, since_days=7. Used together with what the model extracted."""
    words = _words(message)
    text = " ".join(words)
    kind = next((_KIND_WORDS[w] for w in words if w in _KIND_WORDS), "any")
    since = 0
    if m := re.search(r"\b(?:last|past) (\d+) days?\b", text):
        since = int(m.group(1))
    elif "today" in words:
        since = 1
    elif "yesterday" in words:
        since = 2
    elif re.search(r"\b(this|last|past) week\b", text):
        since = 7
    elif re.search(r"\b(this|last|past) month\b", text):
        since = 31
    elif re.search(r"\b(this|last|past) year\b", text):
        since = 365
    if re.search(r"\b(biggest|largest|heaviest|huge|big|large|bigger|space)\b", text):
        sort = "largest"
    elif re.search(r"\b(oldest)\b", text):
        sort = "oldest"
    elif re.search(r"\b(latest|newest|recent|recently|new|last)\b", text) or since:
        sort = "newest"
    else:
        sort = "relevance"
    return {"kind": kind, "since_days": since, "sort": sort}


def query_terms(text: str) -> list[str]:
    """The words that say which file: 'where is my resume pdf' -> ['resume']."""
    return [w for w in _words(text)
            if w not in _STOP and w not in _KIND_WORDS and (len(w) > 1 or w.isdigit())]


# ----------------------------------------------------------------- search

def search(query: str, kind: str = "any", since_days: int = 0, sort: str = "relevance",
           limit: int = 20, _relax: bool = True) -> dict:
    """Ranked matches. Each term scores where it is found: the file name
    (best), a fuzzy name match for typos, the folder path, and, for
    documents, what the document is about (content vectors)."""
    files = catalog().files()
    terms = query_terms(query)
    exts = KIND_EXTS.get(kind)
    cutoff = time.time() - since_days * 86400 if since_days else 0

    pool = [f for f in files if (not exts or f.ext in exts) and f.mtime >= cutoff]

    cat = catalog()
    vocab = cat.vocab()
    near = {t: set(difflib.get_close_matches(t, vocab, n=6, cutoff=0.8)) - {t}
            for t in terms if len(t) >= 4}
    # A typo ('resme') is weighted like the word it was meant to be.
    weights = {t: max([cat.weight(t)] + [cat.weight(n) for n in near.get(t, ())]) if t not in vocab
               else cat.weight(t) for t in terms}
    total_weight = sum(weights.values()) or 1.0

    content = content_index().similarities(query) if terms else {}

    scored = []
    coverage: dict[str, float] = {}  # share of the words found in the file's own name
    for f in pool:
        if not terms:
            scored.append((0.0, f, "filter"))
            continue
        name_words = set(_words(Path(f.name).stem))
        name_lower = f.name.lower()
        path_words = {w for part in f.rel_parts for w in _words(part)}
        score, hits = 0.0, 0
        for t in terms:
            if t in name_words:
                s = 3.0
            elif len(t) >= 3 and t in name_lower:
                s = 2.5
            elif near.get(t) and (near[t] & name_words or any(n in name_lower for n in near[t])):
                s = 2.0
            elif t in path_words:
                s = 1.2
            else:
                s = 0.0
            score += s * weights[t]
            hits += s > 0
        name_score = score / total_weight
        # People mostly look for their own documents, photos and media, not
        # source code that happens to share a word, unless they ask for code.
        if kind != "code" and f.ext in KIND_EXTS["code"]:
            name_score *= 0.6
        sim = content.get(f.path)
        content_score = max(0.0, (sim - 0.62) * 12) if sim is not None else 0.0
        if hits == 0 and content_score <= 0:
            continue
        # Most terms must match unless the content clearly does.
        if hits < (len(terms) + 1) // 2 and content_score < 1.0:
            continue
        how = "name" if content_score < 0.5 else ("content" if hits == 0 else "name+content")
        name_hits = sum(1 for t in terms if t in name_words or (len(t) >= 3 and t in name_lower))
        coverage[f.path] = name_hits / len(terms)
        scored.append((name_score + content_score, f, how))

    if sort == "largest":
        scored.sort(key=lambda x: (x[1].size, x[0]), reverse=True)
    elif sort == "newest" or (sort == "relevance" and not terms):
        scored.sort(key=lambda x: (x[1].mtime, x[0]), reverse=True)
    elif sort == "oldest":
        scored.sort(key=lambda x: (x[1].mtime, -x[0]))
    else:
        scored.sort(key=lambda x: (x[0], x[1].mtime), reverse=True)

    # "the pitch deck" may be a PDF, not a PowerPoint: if the type filter
    # leaves only weak matches but a strong match exists in another type, use it.
    best = max((sc for sc, _, _ in scored), default=0.0)
    if _relax and exts and terms and best < 2.0:
        loose = search(query, "any", since_days, sort, limit, _relax=False)
        if loose["results"] and loose["results"][0]["score"] >= 2.5:
            return loose | {"kind": kind, "kind_relaxed": True}

    results = []
    for score, f, how in scored[:limit]:
        results.append(f.public() | {"score": round(score, 2), "match": how,
                                     "name_coverage": round(coverage.get(f.path, 0.0), 2)})
    status = content_index().status()
    return {"query": query, "terms": terms, "kind": kind, "since_days": since_days, "sort": sort,
            "total": len(scored), "results": results, "searched": len(files),
            "content_index": {"ready": status["state"] == "done", "done": status.get("done", 0),
                              "total": status.get("total", 0)}}


# ---------------------------------------------------------- content index

def _first_text(path: str, ext: str, limit: int = 1500) -> str:
    """Opening text of a document, without OCR (fast)."""
    if ext == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            parts = []
            for page in doc.pages(0, min(3, doc.page_count)):
                parts.append(page.get_text())
                if sum(map(len, parts)) > limit:
                    break
        return re.sub(r"\s+", " ", " ".join(parts))[:limit]
    if ext in (".txt", ".md", ".markdown"):
        with open(path, encoding="utf-8", errors="replace") as f:
            return re.sub(r"\s+", " ", f.read(limit * 2))[:limit]
    from ..document_reader import read_any

    doc = read_any(path)
    return re.sub(r"\s+", " ", " ".join(s.text for s in doc.sections[:20]))[:limit]


class ContentIndex:
    """Vectors of what each document is about, built in the background and
    updated incrementally (only new or changed files are read again)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._paths: list[str] = []
        self._vectors = None  # numpy array, one row per path
        self._meta: dict[str, float] = {}
        self._state = {"state": "idle"}
        self._loaded = False

    def _files(self) -> tuple[Path, Path]:
        return app_home() / "content_index.json", app_home() / "content_index.npy"

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        meta_path, vec_path = self._files()
        try:
            import numpy as np

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("root") != str(policy.home_root()):
                return
            vectors = np.load(vec_path)
            if len(vectors) == len(meta["paths"]):
                self._paths, self._vectors = meta["paths"], vectors
                self._meta = dict(zip(meta["paths"], meta["mtimes"], strict=True))
                self._state = {"state": "done", "done": len(self._paths), "total": len(self._paths),
                               "built_at": meta.get("built_at")}
        except (OSError, ValueError, KeyError):
            pass

    def status(self) -> dict:
        with self._lock:
            self._load()
            return dict(self._state)

    def start(self) -> bool:
        with self._lock:
            self._load()
            if self._thread and self._thread.is_alive():
                return False
            self._thread = threading.Thread(target=self._build, daemon=True, name="content-index")
            self._thread.start()
            return True

    def _build(self) -> None:
        import numpy as np

        from ..vector_store import get_store

        started = time.time()
        docs = [f for f in catalog().files()
                if f.ext in CONTENT_EXTS and 0 < f.size < 40 * 1024**2 and not f.online_only]
        old = {p: i for i, p in enumerate(self._paths)}
        keep_paths, keep_rows, todo = [], [], []
        for f in docs:
            i = old.get(f.path)
            if i is not None and self._meta.get(f.path) == f.mtime:
                keep_paths.append(f.path)
                keep_rows.append(i)
            else:
                todo.append(f)
        self._state = {"state": "building", "done": len(keep_paths), "total": len(docs)}
        new_paths, new_vecs, mtimes = [], [], {p: self._meta[p] for p in keep_paths}
        store = get_store()
        batch_paths, batch_texts = [], []

        def flush():
            if batch_texts:
                new_vecs.extend(store.embed(batch_texts))
                new_paths.extend(batch_paths)
                batch_paths.clear()
                batch_texts.clear()

        try:
            for f in todo:
                try:
                    text = _first_text(f.path, f.ext)
                except Exception:  # unreadable, encrypted, corrupt: name search still works
                    text = ""
                title = re.sub(r"[_+\-.]+", " ", Path(f.name).stem)
                batch_paths.append(f.path)
                batch_texts.append(f"{title}\n{text}")
                mtimes[f.path] = f.mtime
                if len(batch_texts) >= 32:
                    flush()
                    self._state["done"] = len(keep_paths) + len(new_paths)
            flush()
            dim = len(new_vecs[0]) if new_vecs else (self._vectors.shape[1] if self._vectors is not None else 384)
            kept = self._vectors[keep_rows] if keep_rows else np.zeros((0, dim), dtype=np.float32)
            added = np.asarray(new_vecs, dtype=np.float32).reshape(-1, dim)
            vectors = np.vstack([kept, added]).astype(np.float32)
            paths = keep_paths + new_paths
            meta_path, vec_path = self._files()
            np.save(vec_path, vectors)
            meta_path.write_text(json.dumps({"root": str(policy.home_root()), "paths": paths,
                                             "mtimes": [mtimes[p] for p in paths],
                                             "built_at": time.time()}), encoding="utf-8")
            with self._lock:
                self._paths, self._vectors, self._meta = paths, vectors, mtimes
                self._state = {"state": "done", "done": len(paths), "total": len(paths),
                               "built_at": time.time(), "seconds": round(time.time() - started, 1)}
            audit.log("content_index.built", documents=len(paths), new=len(new_paths),
                      seconds=round(time.time() - started, 1))
        except Exception as exc:
            logger.exception("Content index failed")
            self._state = {"state": "error", "error": str(exc)}

    def similarities(self, query: str) -> dict[str, float]:
        with self._lock:
            self._load()
            paths, vectors = self._paths, self._vectors
        if not query.strip() or vectors is None or not len(paths):
            return {}
        import numpy as np

        from ..vector_store import get_store

        q = np.asarray(get_store().embed_query(query), dtype=np.float32)
        sims = vectors @ q  # vectors are normalised by the embedder
        return dict(zip(paths, sims.tolist(), strict=True))


_content = ContentIndex()


def content_index() -> ContentIndex:
    return _content
