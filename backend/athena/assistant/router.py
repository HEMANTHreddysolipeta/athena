"""Understand a chat message: which tool to use, and with what.

The local model reads the message (plus a little context: the folder or
files the conversation was just about) and returns JSON that is forced to
match ROUTE_SCHEMA, so it can only ever pick one of the read-only tools
below; there is no free-form "run this" option. Everything it returns is
then validated again by the tool itself (policy.py decides what paths are
allowed), so a confused or manipulated answer can't do anything unsafe.

Speed: the instructions below are identical for every message, so Ollama
reuses their processed form and only reads the new message (~2-4 s on CPU).
"""

import json
import logging
import os
import re
import threading
import time

import requests

from ..config import settings
from ..llm import LLMError, _options

logger = logging.getLogger(__name__)

TOOLS = ("storage", "folder", "find", "listed", "dig", "system", "security", "cleanup", "documents", "not_yet",
         "chat")
KINDS = ("any", "pdf", "document", "spreadsheet", "presentation", "image", "video", "audio",
         "archive", "installer", "code", "text")
SORTS = ("relevance", "newest", "oldest", "largest")

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": list(TOOLS)},
        "target": {"type": "string"},
        "question": {"type": "string"},
        "kind": {"type": "string", "enum": list(KINDS)},
        "since_days": {"type": "integer"},
        "sort": {"type": "string", "enum": list(SORTS)},
    },
    "required": ["tool", "target", "question", "kind", "since_days", "sort"],
}

SYSTEM = """You route messages for Athena, an assistant that runs on the user's own laptop.
Pick ONE tool and fill the fields. Reply with JSON only.

Tools:
- storage: free space, how full the laptop/drives are, what uses the most space overall.
- folder: what is inside ONE named folder and what uses its space. target = folder name as the user wrote it.
- find: look for files by name, words in the name, type, date or size. target = search words only (no "my", "file", "the").
- dig: open ONE file and read it: summarise it, explain it, answer a question about it. target = the file name, path, or the number/"it" of a file listed earlier. question = what to find out ("summary" if none).
- system: battery, RAM/memory, CPU, which apps use the most memory, uptime, laptop health.
- security: malware, viruses, spyware, hacked, suspicious or hidden programs, is the laptop safe, antivirus status.
- cleanup: what can be cleaned or deleted to free space: caches, duplicates, old installers, recycle bin contents.
- documents: a question whose answer is inside the user's document library (topics like SEO, solar cells, passwords, NIST, LLM security), not about the laptop.
- not_yet: the user wants Athena to CHANGE something: delete, remove, move, rename, copy, empty recycle bin, uninstall, install, close/kill an app. target = what they want done.
- listed: about the files ALREADY listed in this chat ("those", "them", "they", "these files"): show them
  again, open them, or where they are saved. Only when the context lists files.
- chat: greetings, thanks, "what can you do", anything else.

Fields: kind = file type wanted (else "any"). since_days = how recent in days (last week = 7, yesterday = 1, this month = 30, else 0). sort = "largest" for biggest/heaviest, "newest" for latest/recent, "oldest", else "relevance".
If the context says a folder or files are being discussed, follow-ups like "what's in local", "and Google?", "the second one", "it" refer to them.

Examples:
"hows my storage" -> {"tool":"storage","target":"","question":"","kind":"any","since_days":0,"sort":"relevance"}
"whats eating space in downloads" -> {"tool":"folder","target":"downloads","question":"","kind":"any","since_days":0,"sort":"relevance"}
"find my resume" -> {"tool":"find","target":"resume","question":"","kind":"any","since_days":0,"sort":"relevance"}
"biggest videos on my laptop" -> {"tool":"find","target":"","question":"","kind":"video","since_days":0,"sort":"largest"}
"pdfs i downloaded last week" -> {"tool":"find","target":"downloads","question":"","kind":"pdf","since_days":7,"sort":"newest"}
"summarise the fitcheck spec" -> {"tool":"dig","target":"fitcheck spec","question":"summary","kind":"any","since_days":0,"sort":"relevance"}
"what does the 2nd one say about pricing" -> {"tool":"dig","target":"2","question":"what does it say about pricing","kind":"any","since_days":0,"sort":"relevance"}
"how much battery is left" -> {"tool":"system","target":"battery","question":"","kind":"any","since_days":0,"sort":"relevance"}
"is my pc infected" -> {"tool":"security","target":"","question":"","kind":"any","since_days":0,"sort":"relevance"}
"delete the old installers" -> {"tool":"not_yet","target":"delete old installers","question":"","kind":"any","since_days":0,"sort":"relevance"}
"what can i clean up" -> {"tool":"cleanup","target":"","question":"","kind":"any","since_days":0,"sort":"relevance"}
"what is on-page seo" -> {"tool":"documents","target":"","question":"what is on-page seo","kind":"any","since_days":0,"sort":"relevance"}"""


def _context_lines(context: dict) -> str:
    lines = []
    if context.get("folder"):
        lines.append(f"Folder being discussed: {context['folder']}")
    if context.get("file"):
        lines.append(f"File being discussed: {context['file']}")
    if context.get("files"):
        listed = "; ".join(f"{i}. {name}" for i, name in enumerate(context["files"][:10], start=1))
        lines.append(f"Files listed earlier: {listed}")
    return "\n".join(lines)


def route(message: str, context: dict | None = None, timeout: int = 90) -> dict:
    """{"tool": ..., "target": ..., ...} for a chat message. Raises LLMError."""
    ctx = _context_lines(context or {})
    prompt = (f"Context:\n{ctx}\n\n" if ctx else "") + f"Message: {message.strip()}"
    payload = {
        "model": settings.llm_model, "system": SYSTEM, "prompt": prompt, "stream": False,
        "think": False, "format": ROUTE_SCHEMA, "keep_alive": settings.llm_keep_alive,
        "options": {**_options(), "temperature": 0},
    }
    started = time.perf_counter()
    try:
        response = requests.post(f"{settings.ollama_url}/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        result = json.loads(data.get("response", "{}"))
    except (requests.RequestException, ValueError) as exc:
        raise LLMError(f"Couldn't understand the message: {exc}") from exc
    result = _clean(result)
    result["route_ms"] = round((time.perf_counter() - started) * 1000)
    logger.info("route %r -> %s %r (%s ms, %s prompt tokens)", message[:60], result["tool"],
                result["target"], result["route_ms"], data.get("prompt_eval_count"))
    return result


def _clean(raw: dict) -> dict:
    tool = raw.get("tool") if raw.get("tool") in TOOLS else "chat"
    kind = raw.get("kind") if raw.get("kind") in KINDS else "any"
    sort = raw.get("sort") if raw.get("sort") in SORTS else "relevance"
    try:
        since = max(0, min(int(raw.get("since_days") or 0), 3650))
    except (TypeError, ValueError):
        since = 0
    return {"tool": tool, "target": str(raw.get("target") or "").strip()[:200],
            "question": str(raw.get("question") or "").strip()[:500],
            "kind": kind, "since_days": since, "sort": sort}


# ------------------------------------------------------- semantic router
#
# Asking the model takes ~8-25 s per message on a laptop CPU (gemma3 can't
# reuse its processed instructions between different messages). Most
# messages are clear, so they are first compared, by meaning, with the
# example requests below using the small embedding model that is already
# loaded for search (~30 ms). Only when that comparison isn't clearly sure
# does the message go to the model. These examples are NOT the test set:
# eval/assistant_eval.py tests different wording.

EXAMPLES = {
    "storage": [
        "how is my storage", "how much space is left on my laptop", "is my disk full",
        "how much free space do I have", "storage status", "what is taking up space on my computer",
        "my laptop is out of space", "check my drive space", "how many GB are free",
        "why is my C drive full", "how full is my laptop",
    ],
    "folder": [
        "what is in my downloads folder", "what is inside appdata", "show me what's in desktop",
        "how big is my documents folder", "what is using space in this folder",
        "whats in the pictures folder", "open the videos folder and tell me what is big",
        "what is inside the projects folder",
    ],
    "find": [
        "find my resume", "where is my passport scan", "search for invoice", "find the excel file with expenses",
        "locate my offer letter", "show my recent pdfs", "find photos from last month",
        "where did I save the presentation", "biggest files on my laptop", "find large videos",
        "list zip files in downloads", "do I have a file called notes", "find files about internship",
        "largest photos on my pc", "huge files i have", "heaviest videos", "biggest pdfs",
        "where are my exam marksheets", "where did I keep the meeting notes", "look for the budget pdf",
        "the report I wrote last year, where is it", "where's my id card scan", "which folder has my photos of the trip",
    ],
    "dig": [
        "summarise the project report pdf", "read my resume and tell me my skills",
        "what does the agreement say about payment", "open the spec document and explain it",
        "dig into the second file", "what is this file about", "give me the key points of the pitch deck",
        "go through the notes file", "explain the contents of report.docx", "read it",
        "what does it say about deadlines", "open the quarterly report pdf and explain it to me",
        "read my contract and tell me the notice period", "what's written in the lease agreement",
        "check the invoice file and tell me the total",
    ],
    "system": [
        "how much battery is left", "how much ram am i using", "what apps are using the most memory",
        "is my laptop running slow", "cpu usage", "why is my laptop lagging", "how long has my pc been on",
        "check my laptop health", "memory usage", "is my battery charging", "battery", "ram?",
        "cpu", "charge level", "memory left",
    ],
    "cleanup": [
        "what can I clean up", "how do I free up space", "what can I delete safely", "are there duplicate files",
        "show me old installers", "suggest what to remove to free space", "what junk can I get rid of",
        "how much is in the recycle bin", "which caches can be cleared", "clean up my computer",
        "my pc needs a cleanup", "help me make room on the disk",
    ],
    "security": [
        "any viruses on my computer", "is my laptop hacked", "scan for malware", "is anything suspicious running",
        "is my computer safe", "check for spyware", "is someone spying on my pc", "are there hidden programs running",
        "is my antivirus on", "check if my pc is infected",
    ],
    "documents": [
        "what is on-page seo", "how do solar cells work", "explain link building", "what is keyword research",
        "what are network reduction techniques", "what happens in an incubation program",
        "how do I measure seo progress", "what is negative seo", "what are the password rules in nist",
        "what is prompt injection",
    ],
    "not_yet": [
        "delete the old installers", "empty the recycle bin", "delete this file", "move my resume to documents",
        "uninstall docker", "rename the file", "close chrome", "install vlc", "remove duplicate files",
        "clear the temp folder", "kill the process using memory", "wipe the whole hard drive",
        "erase all my files", "email this document to my friend", "copy my photos to a pen drive",
        "share the file on whatsapp",
    ],
    "chat": [
        "hi", "hello", "thanks", "thank you", "what can you do", "who are you", "help", "good morning",
        "how are you",
    ],
}

SEMANTIC_MIN = float(os.getenv("ATHENA_ROUTER_MIN_SIM", "0.80"))
SEMANTIC_MARGIN = float(os.getenv("ATHENA_ROUTER_MARGIN", "0.03"))
ROUTER_MODE = os.getenv("ATHENA_ROUTER", "hybrid")  # hybrid | model | semantic


class SemanticRouter:
    def __init__(self, examples: dict[str, list[str]] = EXAMPLES, embed=None):
        self.examples = examples
        self._embed = embed
        self._labels: list[str] = []
        self._vectors = None
        self._lock = threading.Lock()

    def _ready(self):
        with self._lock:
            if self._vectors is None:
                import numpy as np

                if self._embed is None:
                    from ..vector_store import get_store

                    self._embed = get_store().embed
                texts = [t for tool in self.examples for t in self.examples[tool]]
                self._labels = [tool for tool in self.examples for _ in self.examples[tool]]
                self._vectors = np.asarray(self._embed(texts), dtype=np.float32)
        return self._vectors

    def scores(self, message: str) -> dict[str, float]:
        import numpy as np

        vectors = self._ready()
        q = np.asarray(self._embed([message.strip().lower()])[0], dtype=np.float32)
        sims = vectors @ q
        best: dict[str, float] = {}
        for label, sim in zip(self._labels, sims.tolist(), strict=True):
            best[label] = max(best.get(label, -1.0), sim)
        return best

    def classify(self, message: str) -> tuple[str, float, float]:
        ranked = sorted(self.scores(message).items(), key=lambda kv: kv[1], reverse=True)
        (tool, sim), (_, second) = ranked[0], ranked[1]
        return tool, sim, sim - second


_semantic = SemanticRouter()

_REFERENCE = re.compile(r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last one|number \d+|#\d+|"
                        r"it|this one|that one|this file|that file)\b", re.I)
# The whole list from the last answer: "show me those files", "where are they".
_PLURAL = re.compile(r"\b(those|these|they|them|theirs|the files|the results|the list|all of them|each of them|"
                     r"both)\b", re.I)
WHERE = re.compile(r"\b(where|location|located|locations|which folders?|what folders?|folders?|path|paths|saved|"
                   r"kept|stored)\b", re.I)
_CHANGE = re.compile(r"\b(delete|remove|move|rename|copy|uninstall|send|share|upload|email)\b", re.I)


def smart_route(message: str, context: dict | None = None, mode: str | None = None,
                semantic: SemanticRouter | None = None, model_route=None) -> dict:
    """Route with the cheapest method that is reliable for this message:
    1. follow-ups that name something from the context (a subfolder of the
       folder being discussed, "the second one" after a file list);
    2. the semantic match, when clearly sure;
    3. otherwise the model (with the context), falling back to the best
       semantic guess if the model is unavailable."""
    context = context or {}
    mode = mode or ROUTER_MODE
    semantic = semantic or _semantic
    model_route = model_route or route
    started = time.perf_counter()

    def done(tool: str, via: str, **fields) -> dict:
        out = {"tool": tool, "target": "", "question": "", "kind": "any", "since_days": 0,
               "sort": "relevance", **fields, "via": via}
        out["route_ms"] = round((time.perf_counter() - started) * 1000)
        return out

    words = len(message.split())
    if mode != "model":
        if context.get("folder") and words <= 10:
            from .storage import match_folder

            path, inside = match_folder(message, context["folder"])
            if path and inside:
                return done("folder", "context", target=path)
        has_files = bool(context.get("files") or context.get("file"))
        if has_files and words <= 14 and not _CHANGE.search(message):
            plural = _PLURAL.search(message)
            # "show me those files", "open them", "where are they", "where is it"
            if plural or (WHERE.search(message) and _REFERENCE.search(message)):
                return done("listed", "context", question=message)
            if _REFERENCE.search(message):
                return done("dig", "context", target="", question=message)

    if mode in ("hybrid", "semantic"):
        tool, sim, margin = semantic.classify(message)
        grounded = _ground(message, tool, words)
        if grounded:
            return done(grounded, "semantic+name", question=message if grounded in ("documents", "dig") else "",
                        similarity=round(sim, 3), margin=round(margin, 3))
        if mode == "semantic" or (sim >= SEMANTIC_MIN and margin >= SEMANTIC_MARGIN):
            return done(tool, "semantic", question=message if tool in ("documents", "dig") else "",
                        similarity=round(sim, 3), margin=round(margin, 3))

    try:
        result = model_route(message, context)
        result["via"] = "model"
        return result
    except LLMError as exc:
        logger.warning("Model routing failed (%s); using the closest example instead", exc)
        tool, sim, margin = semantic.classify(message)
        return done(tool, "semantic-fallback", question=message if tool in ("documents", "dig") else "")


_FILE_WORDS = re.compile(r"\b(pdf|docx?|pptx?|xlsx|csv|apk|exe|zip|file|document|deck|spec|agreement|letter|"
                         r"notes?|report|resume|cv)\b", re.I)
# Reading verbs that only mean "read a file" when a file is named ("explain SEO"
# is a document question), and ones that always mean it ("dig into my resume").
_READ_WEAK = re.compile(r"\b(explain|tell me|what does|what's written|whats written|key points|inside)\b", re.I)
_READ_STRONG = re.compile(r"\b(dig|read|open|summar\w*|go through)\b", re.I)
_FIND_WORDS = re.compile(r"\b(where|find|locate|look for|search|which folder)\b", re.I)
_PATH = re.compile(r"[a-zA-Z]:\\[^\s\"']*|~[\\/][^\s\"']*")


def _ground(message: str, tool: str, words: int) -> str | None:
    """Correct the semantic guess when the message names something real on
    this computer: a path, a folder from the storage scan, or a file whose
    name clearly matches. 'whars un app data' names ~\\AppData, so it is
    about a folder even though the words look like a question about the
    system; 'dig into my resume' names a real file, so it means reading it."""
    import os

    from .dig import _DIG_WORDS
    from .files import parse_filters, search
    from .storage import match_folder

    if m := _PATH.search(message):
        path = os.path.expanduser(m.group(0).rstrip(".,?!"))
        if os.path.isfile(path):
            return "dig"
        if os.path.isdir(path):
            return "folder"

    names_a_file = bool(_FILE_WORDS.search(message)) or parse_filters(message)["kind"] != "any"
    if tool in ("storage", "system", "folder") and words <= 10 and not names_a_file:
        path, _ = match_folder(message)
        if path:
            return "folder"

    strong, weak = _READ_STRONG.search(message), _READ_WEAK.search(message)
    finding = _FIND_WORDS.search(message)
    if tool in ("documents", "find", "dig") and (strong or weak or finding):
        hits = search(_DIG_WORDS.sub(" ", message), limit=2)["results"]
        if not hits or hits[0]["score"] < 1.8:
            return None
        if strong or (weak and names_a_file):
            return "dig"
        if finding and tool in ("documents", "dig"):
            return "find"
    return None
