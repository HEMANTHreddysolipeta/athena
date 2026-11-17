"""Chat command router.

Commands need a leading slash: "/find invoice", "/remember color = blue".
The Milestone 1 router matched bare prefixes, so the question "find the
password rules in 800-63B" was treated as a file search. Anything without
a slash is a question about the documents.
"""

import re

from .document_reader import read_document
from .memory import recall, remember
from .rag import answer_question, format_answer
from .search import find_files
from .tools import open_file

HELP = """Ask any question about your documents, or use a command:
/find <name>          list documents whose filename contains <name>
/read <name>          show the text of a document
/open <name>          open a document in its default app
/remember key = value save a note
/recall key           read a saved note
/storage              free space, biggest folders, what you could clean
/help                 this list"""

COMMANDS = ("find", "read", "open", "remember", "recall", "storage", "help")


# Stopgap until the model-based router (assistant step 5): plain questions
# about this computer's storage go to the storage report instead of the
# document search, which can only say "not in your documents".
_STORAGE_WORDS = re.compile(r"\b(stor(age|e)|disk|space|drive|ssd|hdd|gb|full)\b", re.I)
_MY_COMPUTER = re.compile(r"\b(my|laptop|computer|pc|system|c:|c drive)\b", re.I)
_HOWS_MY_COMPUTER = re.compile(r"^\s*(how'?s|how is|check)\s+(my\s+)?(laptop|computer|pc)\b", re.I)


def looks_like_storage_question(text: str) -> bool:
    return bool((_STORAGE_WORDS.search(text) and _MY_COMPUTER.search(text)) or _HOWS_MY_COMPUTER.search(text))


def parse_command(message: str) -> tuple[str, str] | None:
    """('find', 'invoice') for '/find invoice'; None for a normal question."""
    text = message.strip()
    if not text.startswith("/"):
        return ("storage", "") if looks_like_storage_question(text) else None
    name, _, arg = text[1:].partition(" ")
    name = name.lower()
    return (name, arg.strip()) if name in COMMANDS else ("unknown", name)


def run_command(name: str, arg: str) -> str:
    if name == "help":
        return HELP
    if name == "unknown":
        return f"Unknown command /{arg}.\n\n{HELP}"

    if name == "remember":
        if "=" not in arg:
            return "Use: /remember favorite_color = blue"
        key, value = arg.split("=", 1)
        remember(key.strip(), value.strip())
        return "Memory saved."

    if name == "storage":
        from .assistant.storage import chat_summary  # psutil only loads when used
        return chat_summary()

    if name == "recall":
        return recall(arg) or "I don't know that yet."

    if not arg:
        return f"Use: /{name} <part of a filename>"

    files = find_files(arg, limit=10 if name == "find" else 1)
    if not files:
        return "I couldn't find a matching document."

    if name == "find":
        lines = [f"Found {len(files)} matching document(s):"]
        for i, f in enumerate(files, start=1):
            lines.append(f"{i}. {f['name']}  ({f['size'] / 1024:.1f} KB, modified {f['modified']})")
        return "\n".join(lines)

    file = files[0]
    if name == "read":
        try:
            text = read_document(file["path"])
        except ValueError:
            return f"I can't read {file['name']}: unsupported file type."
        return text if len(text) <= 20_000 else text[:20_000] + "\n\n[... truncated ...]"

    # open
    result = open_file(file["path"])
    return f"Opening {file['name']}..." if result is True else result


def process(message: str) -> str:
    command = parse_command(message)
    if command:
        return run_command(*command)
    # There is deliberately no fallback to the bare LLM: an answer without
    # sources is a guess, and a document assistant should say "not found".
    return format_answer(answer_question(message))
