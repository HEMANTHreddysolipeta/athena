"""Who may talk to the Athena server, and where Athena keeps its own files.

The server can read the names and sizes of everything in your home folder,
so three things must hold:

1. It listens on this computer only (127.0.0.1; see app.py and serve.py).
2. Every API request carries this install's secret token, either as the
   `athena_session` cookie (the browser) or an `X-Athena-Token` header
   (scripts). The token is random, created on first run, and stored in your
   own profile folder, so another user account or a website cannot know it.
3. Requests whose Host header is not localhost are refused. That blocks DNS
   rebinding, where a web page points its own domain at 127.0.0.1 to reach
   local servers. The cookie is SameSite=Strict, so other sites can't send it.
"""

import hmac
import os
import secrets
import sys
from pathlib import Path

COOKIE_NAME = "athena_session"
HEADER_NAME = "x-athena-token"


def app_home() -> Path:
    """Athena's private folder in the user's profile (token, logs, reports).

    Deliberately outside the project folder, so nothing personal can be
    committed to git by accident.
    """
    override = os.getenv("ATHENA_HOME")
    if override:
        path = Path(override)
    elif sys.platform == "win32":
        path = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Athena"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "Athena"
    else:
        path = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "athena"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _private_write(path: Path, text: str) -> None:
    """Write a file only the current user can read (POSIX); on Windows the
    profile folder's permissions already restrict it to this account."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


_token: str | None = None


def get_token() -> str:
    global _token
    if _token:
        return _token
    env = os.getenv("ATHENA_TOKEN", "").strip()
    if env:
        _token = env
        return _token
    path = app_home() / "token"
    if path.is_file():
        stored = path.read_text(encoding="utf-8").strip()
        if len(stored) >= 32:
            _token = stored
            return _token
    _token = secrets.token_urlsafe(32)
    _private_write(path, _token)
    return _token


def token_matches(candidate: str | None) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate.encode(), get_token().encode())


def allowed_hosts() -> set[str]:
    raw = os.getenv("ATHENA_ALLOWED_HOSTS", "127.0.0.1,localhost,::1")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def host_of(value: str | None) -> str:
    """'127.0.0.1:8000' -> '127.0.0.1', '[::1]:8000' -> '::1'."""
    value = (value or "").strip().lower()
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value
