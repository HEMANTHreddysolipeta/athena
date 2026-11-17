"""The one place that decides which paths the assistant may touch.

Two levels:

* READ (sizes, names, show in folder): anything inside the user's home folder.
* CHANGE (move, rename, delete to the Recycle Bin; used from step 3 on):
  inside the home folder, minus the protected areas below.

Everything outside the home folder (Windows, Program Files, other users,
other drives' system areas) is refused for both. Checks run on the fully
resolved path, so '..', symlinks, junctions and 8.3 short names can't be
used to step outside, and the model never gets a way around this file:
every tool calls these functions before acting.
"""

import os
import sys
from pathlib import Path

from .security import app_home


class PolicyError(PermissionError):
    """The path is outside what the assistant is allowed to touch."""


def home_root() -> Path:
    return Path(os.path.realpath(os.getenv("ATHENA_ASSISTANT_ROOT") or Path.home()))


# Never changed, even inside the home folder. Matched against the first
# folder under home, case-insensitively.
_PROTECTED_TOP = {
    "appdata",            # Windows: every app's settings and data
    "library",            # macOS equivalent
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".config",
    ".local", ".password-store",
}
# The well-known folders themselves can't be deleted or moved, only what is
# inside them ("delete Documents" is never what anyone means).
_KEY_FOLDERS = {
    "desktop", "documents", "downloads", "pictures", "music", "videos",
    "movies", "onedrive", "favorites", "contacts", "saved games", "searches", "links",
}


def _key(path: Path | str) -> str:
    return os.path.normcase(str(path))


def resolve(path: str | Path) -> Path:
    """Absolute, real path of an existing file or folder, or PolicyError."""
    text = str(path).strip().strip('"')
    if not text:
        raise PolicyError("No path given.")
    if text.startswith(("\\\\", "//")):
        raise PolicyError("Network and device paths are not allowed.")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        raise PolicyError("Use a full path.")
    # 'file.txt:stream' (NTFS alternate data streams) and the like.
    if sys.platform == "win32" and ":" in str(candidate)[2:]:
        raise PolicyError("Invalid characters in path.")
    real = Path(os.path.realpath(candidate))
    if not real.exists():
        raise PolicyError("That file or folder doesn't exist.")
    return real


def _relative_parts(real: Path) -> tuple[str, ...] | None:
    root = home_root()
    try:
        return real.relative_to(root).parts
    except ValueError:
        # relative_to is case-sensitive; Windows paths are not.
        if _key(real).startswith(_key(root) + os.sep):
            return Path(str(real)[len(str(root)) + 1:]).parts
        return () if _key(real) == _key(root) else None


def check_read(path: str | Path) -> Path:
    real = resolve(path)
    if _relative_parts(real) is None:
        raise PolicyError("Athena only looks inside your home folder.")
    return real


def protected_reason(real: Path) -> str | None:
    """Why this path may not be changed, or None if it may."""
    parts = _relative_parts(real)
    if parts is None:
        return "it is outside your home folder"
    if not parts:
        return "it is your home folder itself"
    top = parts[0].lower()
    if top in _PROTECTED_TOP or (len(parts) == 1 and top.startswith(".")):
        return "it holds app settings or keys"
    if top.startswith("ntuser"):
        return "it is part of your Windows profile"
    if len(parts) == 1 and top in _KEY_FOLDERS:
        return f"{parts[0]} is a main folder (you can change what is inside it)"
    athena = Path(os.path.realpath(app_home()))
    if _key(real) == _key(athena) or _key(real).startswith(_key(athena) + os.sep):
        return "it is Athena's own data (including its activity log)"
    return None


def check_change(path: str | Path) -> Path:
    real = resolve(path)
    reason = protected_reason(real)
    if reason:
        raise PolicyError(f"Athena won't change {real.name or real}: {reason}.")
    return real


def display(real: Path | str) -> str:
    """'~\\Downloads\\x.zip' style path for messages and the UI."""
    parts = _relative_parts(Path(real))
    if parts is None:
        return str(real)
    return str(Path("~", *parts)) if parts else "~"
