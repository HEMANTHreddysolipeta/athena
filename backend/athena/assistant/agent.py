"""The laptop assistant's turn: understand the message, run ONE read-only
tool, show the result as a card, and let the local model explain it.

    message --router--> tool + target --tool--> card (numbers, lists, buttons)
                                            \\--> local model writes 1-3 sentences
                                                  about the card (checked: every
                                                  number it writes must be in the card)

Nothing here changes a file. Requests to delete/move/uninstall get an
explanation (and a read-only look at what they'd affect), not an action.
"""

import logging
import os
import re
import time

from .. import llm
from ..llm import LLMError
from . import dig, files, policy, storage, system, threats

logger = logging.getLogger(__name__)

LABELS = {"storage": "Checking your storage", "folder": "Looking inside the folder",
          "find": "Searching your files", "dig": "Reading the file", "system": "Checking your laptop",
          "security": "Checking for signs of malware",
          "cleanup": "Looking for things to clean up", "documents": "Searching your documents",
          "not_yet": "Checking what that would affect", "chat": "Thinking"}

human = storage.human


# --------------------------------------------------------------- summaries

COMMENT_PROMPT = """You are Athena, a friendly assistant on the user's laptop.
The user asked: "{message}"
Facts Athena just measured:
{facts}

Reply in 1-3 short sentences: answer their question directly, point out the most
useful thing to know or do. Use only numbers from the facts, written exactly as
they appear. No greeting, no lists, no markdown headings."""

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers_ok(text: str, facts: str) -> bool:
    known = set(_NUMBER.findall(facts))
    return all(n in known for n in _NUMBER.findall(text))


def comment(message: str, facts: str, fallback: str):
    """Stream a short explanation from the local model as ("token", piece)
    events. If it wrote a number that isn't in the facts (or the model is
    down), a ("replace", fallback) event swaps in the plain sentence.
    Ends with ("final", (text, checked))."""
    text = ""
    try:
        for piece in llm.generate_stream(COMMENT_PROMPT.format(message=message, facts=facts)):
            text += piece
            yield "token", piece
    except LLMError as exc:
        logger.warning("Summary skipped: %s", exc)
        text = ""
    text = text.strip()
    if text and _numbers_ok(text, facts):
        yield "final", (text, True)
        return
    if text:
        logger.info("Summary replaced: it had a number that isn't in the facts: %r", text[:200])
    yield "replace", fallback
    yield "final", (fallback, False)


# ----------------------------------------------------------------- tools

def storage_card() -> dict:
    svc = storage.service()
    summary = svc.summary()
    card = {"kind": "storage", "drives": storage.drives(), "recycle_bin": storage.recycle_bin(),
            "scan": svc.status(), "scanned_at": summary and summary["scanned_at"]}
    if summary:
        card |= {"root": summary["root"], "home_bytes": summary["bytes"],
                 "folders": [f | {"label": storage.describe(f["path"])} for f in summary["home"]["folders"][:8]],
                 "space_hogs": summary["space_hogs"][:6]}
    return card


def storage_facts(card: dict) -> tuple[str, str]:
    lines = [f"Drive {d['mount']}: {human(d['free'])} free of {human(d['total'])}, {d['percent']:.0f}% used"
             for d in card["drives"]]
    if card.get("recycle_bin") and card["recycle_bin"]["bytes"]:
        lines.append(f"Recycle Bin: {human(card['recycle_bin']['bytes'])}")
    for f in card.get("folders", [])[:5]:
        lines.append(f"Folder {policy.display(f['path'])}: {human(f['bytes'])}")
    for h in card.get("space_hogs", [])[:4]:
        lines.append(f"Could free: {h['name']} {human(h['bytes'])} ({h['advice']})")
    if not card.get("folders"):
        lines.append("The home folder has not been scanned yet (a scan shows which folders are biggest).")
    d = card["drives"][0] if card["drives"] else None
    fallback = (f"{d['mount']} has {human(d['free'])} free of {human(d['total'])} ({d['percent']:.0f}% used)."
                if d else "Here's your storage.")
    if card.get("space_hogs"):
        h = card["space_hogs"][0]
        fallback += f" The easiest space to win back is {h['name']} ({human(h['bytes'])})."
    return "\n".join(lines), fallback


def folder_card(path: str) -> dict:
    svc = storage.service()
    view = svc.folder(path)
    real = view["path"]
    prefix = os.path.normcase(real) + os.sep
    report = svc.report or {}
    return {"kind": "folder", "path": real, "display": policy.display(real), "bytes": view["bytes"],
            "files": view["total_files"], "label": storage.describe(real), "crumbs": view["crumbs"],
            "folders": [f | {"label": storage.describe(f["path"])} for f in view["folders"][:12]],
            "small_folders": view["small_folders"],
            "largest_files": [f for f in report.get("largest_files", [])
                              if os.path.normcase(f["path"]).startswith(prefix)][:6]
                             or view["files"][:6],
            "space_hogs": [h for h in report.get("space_hogs", [])
                           if os.path.normcase(h["path"]).startswith(prefix)][:5],
            "protected": policy.protected_reason(policy.resolve(real)),
            "scanned_at": report.get("scanned_at")}


def folder_facts(card: dict) -> tuple[str, str]:
    lines = [f"Folder {card['display']}: {human(card['bytes'])} in {card['files']} files"
             + (f" ({card['label']})" if card.get("label") else "")]
    for f in card["folders"][:6]:
        lines.append(f"Inside: {os.path.basename(f['path'])} {human(f['bytes'])}"
                     + (f" ({f['label']})" if f.get("label") else ""))
    for f in card["largest_files"][:3]:
        lines.append(f"Big file: {os.path.basename(f['path'])} {human(f['bytes'])}")
    for h in card["space_hogs"][:3]:
        lines.append(f"Could free: {h['name']} {human(h['bytes'])} ({h['advice']})")
    if card.get("protected"):
        lines.append(f"Athena will not delete things here directly: {card['protected']}.")
    top = card["folders"][0] if card["folders"] else None
    fallback = f"{card['display']} is {human(card['bytes'])}." + (
        f" Most of it is {os.path.basename(top['path'])} ({human(top['bytes'])})." if top else "")
    return "\n".join(lines), fallback


def cleanup_card() -> dict:
    summary = storage.service().summary()
    card = {"kind": "cleanup", "recycle_bin": storage.recycle_bin(), "scanned": bool(summary)}
    if not summary:
        return card
    safe_ids = {"temp", "huggingface", "pip", "npm", "gradle", "chrome", "edge"}
    hogs = summary["space_hogs"]
    card |= {
        "caches": [h for h in hogs if h["id"] in safe_ids],
        "app_managed": [h for h in hogs if h["id"] in ("docker", "ollama", "avd")],
        "old_installers": summary["old_installers"] | {"files": summary["old_installers"]["files"][:8]},
        "duplicates": summary["duplicates"] | {"groups": summary["duplicates"]["groups"][:6]},
        "root": summary["root"],
    }
    rb = (card["recycle_bin"] or {}).get("bytes", 0)
    card["total"] = (rb + sum(h["bytes"] for h in card["caches"]) + card["old_installers"]["bytes"]
                     + card["duplicates"]["savings"])
    return card


def cleanup_facts(card: dict) -> tuple[str, str]:
    if not card["scanned"]:
        return "No storage scan yet, so Athena can't list what to clean.", \
            "Run a storage scan first and I'll list what's safe to clean."
    lines = [f"Total you could free safely: {human(card['total'])}"]
    if card["recycle_bin"] and card["recycle_bin"]["bytes"]:
        lines.append(f"Recycle Bin: {human(card['recycle_bin']['bytes'])}")
    for h in card["caches"]:
        lines.append(f"Cache {h['name']}: {human(h['bytes'])} ({h['advice']})")
    lines.append(f"Old installers in Downloads: {card['old_installers']['count']} files, "
                 f"{human(card['old_installers']['bytes'])}")
    lines.append(f"Duplicate copies: {card['duplicates']['count']} sets, {human(card['duplicates']['savings'])}")
    for h in card["app_managed"]:
        lines.append(f"Managed by its app: {h['name']} {human(h['bytes'])} ({h['advice']})")
    lines.append("Athena can't delete anything yet; that comes in the next step and will always ask first.")
    return "\n".join(lines), f"You could free about {human(card['total'])} without touching your own files."


def system_facts(card: dict) -> tuple[str, str]:
    m, c = card["memory"], card["cpu"]
    lines = [f"Memory: {human(m['used'])} used of {human(m['total'])} ({m['percent']}%)",
             f"CPU: {c['percent']}% busy, {c['cores']} threads"]
    b = card["battery"]
    if b:
        left = f", about {b['minutes_left']} minutes left" if b["minutes_left"] else ""
        lines.append(f"Battery: {b['percent']}%, {'charging' if b['plugged_in'] else 'on battery'}{left}")
    hours = card["uptime_s"] // 3600
    lines.append(f"On for {hours} hours since the last restart")
    for a in card["top_apps"][:5]:
        lines.append(f"App {a['name']}: {human(a['bytes'])} of memory")
    fallback = f"Memory is {m['percent']}% used and the CPU is {c['percent']}% busy."
    if b:
        fallback += f" Battery is at {b['percent']}%."
    return "\n".join(lines), fallback


def security_facts(card: dict) -> tuple[str, str]:
    if card["verdict"] == "unsupported":
        return "The malware check only works on Windows.", "The malware check only works on Windows for now."
    c = card["counts"]
    lines = [f"Checked {c['tasks']} scheduled tasks, {c['autostart']} programs that start with Windows, "
             f"{c['processes']} running programs and {c['connections']} internet connections.",
             f"Microsoft Defender service: {card['defender']}."]
    for f in card["findings"][:6]:
        lines.append(f"{'SUSPICIOUS' if f['severity'] == 'high' else 'Worth checking'}: {f['where']} "
                     f"\"{f['item']}\": {f['reason']}.")
    if not card["findings"]:
        lines.append("Nothing matched the usual signs of malware.")
    lines.append("This is a quick check, not an antivirus scan; a full Microsoft Defender scan is more thorough.")
    if card["verdict"] == "danger":
        fallback = ("Some things look like malware. Disconnect from the internet and look at the items below; "
                    "Athena can't remove them itself.")
    elif card["verdict"] == "check":
        fallback = "Nothing looks clearly malicious, but a few things are unusual. Check that you recognise them."
    else:
        fallback = ("No signs of malware in these checks. For a deeper look, run a full Microsoft Defender scan.")
    return "\n".join(lines), fallback


HELP_CARD = {"kind": "help", "examples": [
    "How's my storage?", "What's in my Downloads?", "Find my resume",
    "Biggest videos on my laptop", "Summarise the RepScroll gym pitch", "How much RAM am I using?",
    "What can I clean up?", "Any malware on my laptop?", "What is on-page SEO?"]}


# ------------------------------------------------------------------ turn

def handle(message: str, context: dict, route_fn, action: dict | None = None):
    """Yield (event, data) for one chat turn. A first event ("documents", q)
    means: answer from the document library (app.py runs the RAG pipeline)."""
    started = time.perf_counter()
    if action:  # a button in a card, e.g. "Dig into" one of several matches
        route = {"tool": action.get("tool", "chat"), "target": action.get("path", ""),
                 "question": action.get("question", ""), "kind": "any", "since_days": 0,
                 "sort": "relevance", "via": "button", "route_ms": 0}
    else:
        route = route_fn(message, context)
    tool = route["tool"]
    yield "route", {"tool": tool, "label": LABELS.get(tool, "Working"), "via": route.get("via"),
                    "route_ms": route.get("route_ms")}

    if tool == "documents":
        yield "documents", route.get("question") or message
        return

    new_context = dict(context)
    card, text, checked = None, "", None
    commentary: tuple[str, str] | None = None  # (facts, fallback) for the model to explain
    extra: dict = {}
    try:
        if tool == "storage":
            card = storage_card()
            commentary = storage_facts(card)
        elif tool == "folder":
            path = route["target"] if action else None
            if not path:
                for candidate in (route["target"], message):
                    path, _ = storage.match_folder(candidate, context.get("folder")) if candidate else (None, False)
                    if path:
                        break
            if not path:
                if not storage.service().report:
                    card = storage_card()
                    text = "I need to scan your home folder once before I can look inside folders. Click Scan below."
                else:
                    text = (f"I couldn't find a folder called \"{route['target'] or message}\" in the last scan. "
                            "Try the folder's exact name, like Downloads or AppData.")
            else:
                card = folder_card(path)
                new_context["folder"] = card["path"]
                commentary = folder_facts(card)
        elif tool == "find":
            parsed = files.parse_filters(message)
            kind = route["kind"] if route.get("kind", "any") != "any" else parsed["kind"]
            since = route.get("since_days") or parsed["since_days"]
            sort = route["sort"] if route.get("sort", "relevance") != "relevance" else parsed["sort"]
            query = route["target"] if files.query_terms(route["target"]) else message
            result = files.search(query, kind=kind, since_days=since, sort=sort, limit=15)
            card = {"kind": "files"} | result
            new_context["files"] = [r["path"] for r in result["results"][:10]]
            new_context["file"] = result["results"][0]["path"] if len(result["results"]) == 1 else None
            n = result["total"]
            if not n:
                text = "I couldn't find any matching files in your folders."
                if not result["content_index"]["ready"]:
                    text += " I'm still reading your documents, so searching by what's inside them will work soon."
            else:
                what = f"{kind} files" if kind != "any" else "files"
                text = (f"Found {n} matching {what}." if n <= 15 else f"Found {n} matching {what}; here are the top 15.")
                if n > 1:
                    text += " Say \"dig into the 2nd one\" (or click Dig in) to read one."
        elif tool == "dig":
            target = route["target"]
            try:
                path = target if action else dig.resolve(target, message, context)
                new_context["file"] = path
                question = route.get("question") or message
                for event, data in dig.answer(path, question):
                    if event == "card":
                        card = data
                        yield "card", card
                    elif event == "token":
                        text += data
                        yield "token", data
                    else:
                        text = data["answer"]
                        extra = {"sources": data["sources"], "grounding": data["grounding"]}
                checked = None
            except dig.NeedsChoice as choice:
                card = {"kind": "choose", "query": choice.query, "question": route.get("question") or message,
                        "results": choice.candidates}
                new_context["files"] = [c["path"] for c in choice.candidates]
                text = "A few files match. Which one should I read?"
            except (dig.NotFound, policy.PolicyError, PermissionError, ValueError, OSError) as exc:
                text = str(exc)
        elif tool == "system":
            card = {"kind": "system"} | system.info()
            commentary = system_facts(card)
        elif tool == "security":
            card = {"kind": "security"} | threats.check()
            commentary = security_facts(card)
        elif tool == "cleanup":
            card = cleanup_card()
            commentary = cleanup_facts(card)
        elif tool == "not_yet":
            what = route["target"] or "that"
            text = (f"I can't {what.lower()} yet. Changing files (delete, move, rename, uninstall) is the next "
                    "step I'm building, and it will always show you exactly what changes and ask first. "
                    "For now, here's a read-only look; use Show to open it in File Explorer.")
            if re.search(r"installer|duplicate|temp|cache|recycle|bin|junk|clean|space", what + " " + message, re.I):
                card = cleanup_card()
            elif files.query_terms(route["target"]):
                result = files.search(route["target"], limit=8)
                if result["total"]:
                    card = {"kind": "files"} | result
                    new_context["files"] = [r["path"] for r in result["results"]]
        else:  # chat
            if re.search(r"\b(thanks|thank you|thx|ty|great|nice|cool|awesome|ok|okay)\b", message, re.I) \
                    and len(message.split()) <= 6:
                text = "Anytime! Ask me about your files, storage or laptop whenever you like."
            else:
                card = HELP_CARD
                text = ("Hi! I'm Athena, running entirely on this laptop. I can check your storage, look inside "
                        "folders, find files by name or by what's inside them, read and summarise a file, check "
                        "battery and memory, and answer questions from your document library. Nothing leaves "
                        "this computer, and I never change files without asking.")
    except policy.PolicyError as exc:
        text = str(exc)

    if card is not None and tool != "dig":
        yield "card", card
    if commentary:
        for event, data in comment(message, *commentary):
            if event == "final":
                text, checked = data
            else:
                yield event, data
    elif text and tool != "dig":
        yield "token", text
    yield "done", {"answer": text, "found": True, "command": tool, "cited": [], "card": card,
                   "context": new_context, "route": route, "summary_checked": checked,
                   "timings": {"total_ms": round((time.perf_counter() - started) * 1000)}} | extra
