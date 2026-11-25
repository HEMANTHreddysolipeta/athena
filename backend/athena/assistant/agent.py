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
          "not_yet": "Checking what that would affect", "chat": "Thinking",
          "listed": "Looking at those files"}

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
# Questions about usage over time: Athena has no history to answer them from.
_USAGE = re.compile(r"\b(how often|how much time|how long (do|have|did) i|screen time|usage history|"
                    r"(use|used|using) (it|them|\w+) (the )?most|most used|do i (use|open))\b", re.I)


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
        for piece in llm.generate_stream(COMMENT_PROMPT.format(message=message, facts=facts), max_tokens=110):
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
            "namesakes": storage.namesakes(real),
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
    others = card.get("namesakes") or []
    for o in others:
        lines.append(f"Another folder with the same name: {o['display']} {human(o['bytes'])}")
    top = card["folders"][0] if card["folders"] else None
    fallback = f"{card['display']} is {human(card['bytes'])}." + (
        f" Most of it is {os.path.basename(top['path'])} ({human(top['bytes'])})." if top else "")
    if others:
        fallback += f" There's also {others[0]['display']} ({human(others[0]['bytes'])})."
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


# Apps the model must not suggest closing (it suggested closing Ollama, which
# is Athena's own model, in the live test).
_APP_NOTES = {"Ollama": " (runs Athena's AI model; closing it stops Athena)",
              "Windows": " (Windows itself; can't be closed)",
              "Python": " (may be Athena itself)"}


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
    keep = []
    for a in card["top_apps"][:6]:
        note = _APP_NOTES.get(a["name"].split(" (")[0])
        if note:
            keep.append(f"{a['name']}{note}")
        else:
            lines.append(f"{a['name']} uses {human(a['bytes'])} of memory")
    if keep:
        lines.append("Keep running, don't suggest closing: " + "; ".join(keep))
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


def _library_has_documents() -> bool:
    from .. import rag  # the same library the document answers come from

    try:
        return rag.get_store().count() > 0
    except Exception:  # the library is optional for the laptop tools
        return False


def _search_words(context: dict) -> str:
    search = context.get("search") or {}
    words = " ".join(files.query_terms(search.get("query", "")))
    kind = search.get("kind", "any")
    return " ".join(w for w in (words, kind if kind != "any" else "") if w)


def listed_card(message: str, context: dict) -> tuple[dict | None, str]:
    """Follow-ups about the files from the last answer: show them again, or
    say where they are (grouped by folder)."""
    from .router import WHERE

    search = context.get("search")
    if search:
        everything = files.search(search["query"], kind=search.get("kind", "any"),
                                  since_days=search.get("since_days", 0), sort=search.get("sort", "relevance"),
                                  limit=200)["results"]
    else:
        everything = []
    shown = [i for p in (context.get("files") or ([context["file"]] if context.get("file") else []))
             if (i := files.info(p))]
    if not shown and not everything:
        return None, "I don't have a list of files from earlier in this chat. Ask me to find something first."
    what = _search_words(context)
    label = f"{what} files" if what else "files"

    if WHERE.search(message):
        pool = everything or shown
        groups: dict[str, dict] = {}
        for f in pool:
            folder = os.path.dirname(f["path"])
            g = groups.setdefault(folder, {"path": folder, "display": policy.display(folder), "count": 0,
                                           "bytes": 0, "files": []})
            g["count"] += 1
            g["bytes"] += f["bytes"]
            if len(g["files"]) < 6:
                g["files"].append({"name": f["name"], "path": f["path"]})
        ordered = sorted(groups.values(), key=lambda g: (-g["count"], g["display"].lower()))
        card = {"kind": "locations", "what": label, "total": len(pool), "folders": ordered}
        if len(pool) == 1:
            text = f"It's in {ordered[0]['display']}. Click Show to open that folder in File Explorer."
        else:
            def short(display: str) -> str:
                # "~\Projects\shop\backend\migrations" -> "Projects\…\migrations"
                parts = [p for p in display.split(os.sep) if p and p != "~"]
                return parts[0] if len(parts) == 1 else (os.sep.join(parts) if len(parts) == 2
                                                         else f"{parts[0]}{os.sep}…{os.sep}{parts[-1]}")

            listing = ", ".join(f"{short(g['display'])} ({g['count']})" for g in ordered[:4])
            more = f" and {len(ordered) - 4} more" if len(ordered) > 4 else ""
            text = (f"Your {len(pool)} {label} are in {len(ordered)} folder{'s' if len(ordered) > 1 else ''}: "
                    f"{listing}{more}. Click Show next to a folder to open it in File Explorer.")
        return card, text

    card = {"kind": "files", "results": shown, "total": len(everything) or len(shown), "searched": None,
            "query": what, "content_index": {"ready": True, "done": 0, "total": 0}, "again": True}
    text = (f"Here are the {len(shown)} {label} again. Show opens one in File Explorer, Dig in reads it for you, "
            "and \"where are they?\" lists their folders.")
    return card, text


HELP_CARD = {"kind": "help", "examples": [
    "How's my storage?", "What's in my Downloads?", "Find my resume",
    "Biggest videos on my laptop", "Summarise my latest report", "How much RAM am I using?",
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
                facts, fallback = folder_facts(card)
                if card["namesakes"]:
                    # Two folders with one name: the model mixed up whose size
                    # was whose (live test), so this answer is written by code.
                    text = fallback + " Click it below to look inside instead."
                else:
                    commentary = facts, fallback
        elif tool == "find":
            parsed = files.parse_filters(message)
            kind = route["kind"] if route.get("kind", "any") != "any" else parsed["kind"]
            since = route.get("since_days") or parsed["since_days"]
            sort = route["sort"] if route.get("sort", "relevance") != "relevance" else parsed["sort"]
            query = route["target"] if files.query_terms(route["target"]) else message
            result = files.search(query, kind=kind, since_days=since, sort=sort, limit=15)
            card = {"kind": "files"} | result
            new_context["files"] = [r["path"] for r in result["results"]]
            new_context["file"] = result["results"][0]["path"] if len(result["results"]) == 1 else None
            # Kept so "where are they?" can cover every match, not only the 15 shown.
            new_context["search"] = {"query": query, "kind": kind, "since_days": since, "sort": sort}
            n = result["total"]
            # "find the password length rule": no file is called that, but the
            # document library may answer it, so ask the documents instead.
            if not n and len(files.query_terms(message)) >= 2 and _library_has_documents():
                yield "documents", message
                return
            if not n:
                text = "I couldn't find any matching files in your folders."
                if not result["content_index"]["ready"]:
                    text += " I'm still reading your documents, so searching by what's inside them will work soon."
            else:
                what = f"{kind} files" if kind != "any" else "files"
                text = (f"Found {n} matching {what}." if n <= 15 else f"Found {n} matching {what}; here are the top 15.")
                if n > 1:
                    text += " Say \"dig into the 2nd one\" (or click Dig in) to read one."
        elif tool == "listed":
            card, text = listed_card(message, context)
            if card is None:
                # No earlier list ("where are the hugging face" right after a
                # cleanup card): look for a folder or files with that name.
                path, _ = storage.match_folder(message, context.get("folder"))
                if path:
                    card = folder_card(path)
                    new_context["folder"] = card["path"]
                    text = f"It's here: {card['display']} ({human(card['bytes'])}). Click Show to open it in File Explorer."
                elif files.query_terms(dig._DIG_WORDS.sub(" ", message)):
                    result = files.search(dig._DIG_WORDS.sub(" ", message), limit=15)
                    if result["total"]:
                        card = {"kind": "files"} | result
                        new_context["files"] = [r["path"] for r in result["results"]]
                        text = f"Found {result['total']} matching files."
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
            if _USAGE.search(message):
                # Athena sees what's running now, not a history of app use.
                text = ("I don't keep track of how often you use apps, so I can't tell you that. "
                        "Here's what's running right now and how much memory each app is using.")
            else:
                commentary = system_facts(card)
        elif tool == "security":
            card = {"kind": "security"} | threats.check()
            commentary = security_facts(card)
        elif tool == "cleanup":
            card = cleanup_card()
            commentary = cleanup_facts(card)
        elif tool == "not_yet":
            from .router import DANGER

            what = f"{route['target']} {message}"
            if DANGER.search(what):
                text = ("That's not something Athena will ever do: wiping or formatting a drive, or deleting "
                        "everything, can't be undone. If you want to free space, ask \"what can I clean up?\".")
            elif re.search(r"\b(send|email|mail|upload|share|post|whatsapp|drive)\b", what, re.I) \
                    and not re.search(r"\b(delete|move|rename)\b", what, re.I):
                text = ("Athena never sends your files anywhere: everything stays on this laptop. "
                        "I can find the file for you; you can then share it yourself.")
            else:
                text = ("I can't make changes like that yet. Deleting, moving and renaming is the next step I'm "
                        "building, and it will always show you exactly what changes and ask first. "
                        "For now, here's a read-only look; use Show to open it in File Explorer.")
                if re.search(r"installer|duplicate|temp|cache|recycle|bin|junk|clean|space", what, re.I):
                    card = cleanup_card()
                else:
                    query = route["target"] if files.query_terms(route["target"]) else dig._DIG_WORDS.sub(" ", message)
                    result = files.search(query, limit=8)
                    # Only show files when the request clearly names them.
                    if result["results"] and result["results"][0].get("name_coverage") == 1.0:
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
