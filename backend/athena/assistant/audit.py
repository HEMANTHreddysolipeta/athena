"""Activity log: one JSON line per assistant action, in Athena's private folder.

Append-only from Athena's side, and the folder is protected from the
assistant's own file actions (policy.py), so the model can't erase it.
"""

import json
import threading
import time

from .security import app_home

_lock = threading.Lock()


def log(action: str, **details) -> None:
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "action": action, **details}
    with _lock, (app_home() / "activity.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def recent(limit: int = 100) -> list[dict]:
    path = app_home() / "activity.log"
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
