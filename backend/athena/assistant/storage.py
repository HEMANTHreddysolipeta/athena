"""Storage report: what is using the disk, and what could be cleaned up.

READ-ONLY. Nothing in this module moves, changes or deletes a file; it reads
names, sizes and dates, and (for duplicate detection) file contents to hash.

A full scan of a home folder is slow (measured: 1.5 million files, 322 GB,
~4 minutes on an i5 laptop), so it runs in a background thread with live
progress and the result is saved, so the report is there on the next start.
Only folders of at least FOLDER_MIN_BYTES are kept in the saved tree; smaller
ones are summed into "smaller items" and measured live when opened.
"""

import hashlib
import heapq
import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import psutil

from . import audit, policy
from .security import app_home

FOLDER_MIN_BYTES = 20 * 1024**2
LARGEST_FILES = 50
DUPLICATE_MIN_BYTES = 1024**2
DUPLICATE_HASH_BUDGET = int(float(os.getenv("ATHENA_DUPLICATE_BUDGET_GB", "8")) * 1024**3)
OLD_DAYS = 30
INSTALLER_EXTS = {".exe", ".msi", ".msix", ".appx", ".zip", ".7z", ".rar", ".iso",
                  ".dmg", ".pkg", ".deb", ".rpm", ".apk", ".tar", ".gz"}
# Duplicates are only looked for in the user's own files: copies inside
# caches, app folders and code dependencies are normal and not theirs to delete.
_NOT_PERSONAL = {"appdata", "library", "node_modules", ".git", "venv", ".venv",
                 "site-packages", "__pycache__", ".gradle", ".m2", ".cache"}
# OneDrive/iCloud "online-only" files: their size is not on this disk.
_CLOUD_ONLY = 0x1000 | 0x40000 | 0x400000  # OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS


def drives() -> list[dict]:
    out, seen = [], set()
    for part in psutil.disk_partitions(all=False):
        if "cdrom" in part.opts or part.fstype in ("", "squashfs", "tmpfs", "devtmpfs", "overlay"):
            continue
        if part.device in seen:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        seen.add(part.device)
        out.append({"mount": part.mountpoint, "total": usage.total, "used": usage.used,
                    "free": usage.free, "percent": usage.percent})
    return out


def recycle_bin() -> dict | None:
    """Size and item count of the Recycle Bin (all drives), Windows only."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class SHQUERYRBINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("i64Size", ctypes.c_int64),
                    ("i64NumItems", ctypes.c_int64)]

    info = SHQUERYRBINFO(cbSize=ctypes.sizeof(SHQUERYRBINFO))
    if ctypes.windll.shell32.SHQueryRecycleBinW(None, ctypes.byref(info)) != 0:
        return None
    return {"bytes": info.i64Size, "items": info.i64NumItems}


def _space_hog_locations(home: Path) -> list[dict]:
    """Well-known places that grow large, with plain advice. Cleaning comes
    in a later step; for now Athena only measures and explains."""
    local = Path(os.getenv("LOCALAPPDATA") or home / "AppData" / "Local")
    env = os.getenv
    rows = [
        ("temp", "Temporary files", [local / "Temp", home / "Library" / "Caches"],
         "Leftovers from installers and apps. Usually safe to clear; files in use are skipped."),
        ("huggingface", "Hugging Face models",
         [Path(env("HF_HOME")) if env("HF_HOME") else home / ".cache" / "huggingface"],
         "AI models downloaded by Python projects. Safe to delete: they download again when needed."),
        ("ollama", "Ollama models",
         [Path(env("OLLAMA_MODELS")) if env("OLLAMA_MODELS") else home / ".ollama" / "models"],
         "Local AI models. Remove ones you don't use with: ollama rm <name>."),
        ("docker", "Docker Desktop disk", [local / "Docker" / "wsl"],
         "Docker images and containers. Free space with: docker system prune. Don't delete the file itself."),
        ("pip", "pip download cache", [local / "pip" / "Cache", home / ".cache" / "pip"],
         "Python package downloads. Safe to clear: pip cache purge."),
        ("npm", "npm cache", [local / "npm-cache", home / ".npm"],
         "JavaScript package downloads. Safe to clear: npm cache clean --force."),
        ("gradle", "Gradle cache", [home / ".gradle" / "caches"],
         "Android/Java build downloads. Safe to delete; the next build downloads what it needs."),
        ("avd", "Android emulators", [home / ".android" / "avd"],
         "Virtual phones. Delete ones you don't use from Android Studio's Device Manager."),
        ("chrome", "Chrome cache", [local / "Google" / "Chrome" / "User Data" / "Default" / "Cache"],
         "Web page cache. Clear it in Chrome: Settings > Privacy > Delete browsing data."),
        ("edge", "Edge cache", [local / "Microsoft" / "Edge" / "User Data" / "Default" / "Cache"],
         "Web page cache. Clear it in Edge: Settings > Privacy > Clear browsing data."),
        ("downloads", "Downloads folder", [home / "Downloads"],
         "Everything you've downloaded. See old installers and duplicates below."),
    ]
    return [{"id": i, "name": n, "paths": p, "advice": a} for i, n, p, a in rows]


class ScanCancelled(Exception):
    pass


class _Scanner:
    """One pass over the home folder: folder sizes, largest files, old
    installers in Downloads, and same-size files for duplicate checking."""

    def __init__(self, root: Path, cancel: threading.Event):
        self.root = str(root)
        self.cancel = cancel
        self.files = 0
        self.bytes = 0
        self.errors = 0
        self.current = ""
        self.own: dict[str, list] = {}          # key -> [display path, own bytes, own files]
        self.parent: dict[str, str] = {}
        self.order: list[str] = []
        self.largest: list[tuple[int, str]] = []
        self.same_size: dict[int, list[str]] = defaultdict(list)
        self.old_installers: list[dict] = []
        self.downloads = os.path.normcase(str(Path(self.root) / "Downloads"))

    def run(self) -> None:
        cutoff = time.time() - OLD_DAYS * 86400
        stack = [(self.root, True, False)]  # (path, personal, inside Downloads)
        while stack:
            if self.cancel.is_set():
                raise ScanCancelled
            path, personal, in_downloads = stack.pop()
            key = os.path.normcase(path)
            self.current = path
            self.order.append(key)
            own_bytes = own_files = 0
            try:
                entries = os.scandir(path)
            except OSError:
                self.errors += 1
                self.own[key] = [path, 0, 0]
                continue
            with entries:
                for entry in entries:
                    try:
                        if entry.is_symlink() or entry.is_junction():
                            continue  # never follow links out of the folder
                        if entry.is_dir(follow_symlinks=False):
                            child = os.path.normcase(entry.path)
                            self.parent[child] = key
                            name = entry.name.lower()
                            stack.append((entry.path, personal and name not in _NOT_PERSONAL
                                          and not (key == os.path.normcase(self.root) and name.startswith(".")),
                                          in_downloads or child == self.downloads))
                            continue
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        self.errors += 1
                        continue
                    if getattr(st, "st_file_attributes", 0) & _CLOUD_ONLY:
                        continue
                    size = st.st_size
                    own_bytes += size
                    own_files += 1
                    if len(self.largest) < LARGEST_FILES:
                        heapq.heappush(self.largest, (size, entry.path))
                    elif size > self.largest[0][0]:
                        heapq.heapreplace(self.largest, (size, entry.path))
                    if personal and size >= DUPLICATE_MIN_BYTES:
                        self.same_size[size].append(entry.path)
                    if (in_downloads and st.st_mtime < cutoff
                            and Path(entry.name).suffix.lower() in INSTALLER_EXTS):
                        self.old_installers.append({"path": entry.path, "bytes": size,
                                                    "modified": st.st_mtime})
            self.own[key] = [path, own_bytes, own_files]
            self.files += own_files
            self.bytes += own_bytes

    def totals(self) -> dict[str, list]:
        """key -> [display path, total bytes, total files] including subfolders.
        Children are always visited after their parent, so walking the visit
        order backwards adds every folder into its parent exactly once."""
        totals = {k: list(v) for k, v in self.own.items()}
        for key in reversed(self.order):
            parent = self.parent.get(key)
            if parent is not None and key in totals:
                totals[parent][1] += totals[key][1]
                totals[parent][2] += totals[key][2]
        return totals


def _sample_hash(path: str, size: int) -> str:
    """Hash of the first, middle and last 64 KB: cheap, splits most same-size
    files that differ, before paying for a full read."""
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for offset in (0, max(0, size // 2 - 32768), max(0, size - 65536)):
            f.seek(offset)
            h.update(f.read(65536))
    return h.hexdigest()


def _full_hash(path: str) -> str:
    h = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(same_size: dict[int, list[str]], cancel: threading.Event,
                    progress: dict, budget: int = DUPLICATE_HASH_BUDGET) -> tuple[list[dict], bool]:
    """Groups of files with identical content. Biggest possible savings first;
    stops after reading `budget` bytes (returns complete=False)."""
    groups = sorted(((s, p) for s, p in same_size.items() if len(p) > 1),
                    key=lambda g: g[0] * (len(g[1]) - 1), reverse=True)
    read = 0
    found = []
    for size, paths in groups:
        if cancel.is_set():
            raise ScanCancelled
        by_sample = defaultdict(list)
        for p in paths:
            try:
                by_sample[_sample_hash(p, size)].append(p)
            except OSError:
                continue
        for candidates in by_sample.values():
            if len(candidates) < 2:
                continue
            if size > 3 * 65536:  # otherwise the sample already read the whole file
                if read + size * len(candidates) > budget:
                    return found, False
                read += size * len(candidates)
                by_full = defaultdict(list)
                for p in candidates:
                    try:
                        by_full[_full_hash(p)].append(p)
                    except OSError:
                        continue
                    progress["hashed_bytes"] = read
                sets = [c for c in by_full.values() if len(c) > 1]
            else:
                sets = [candidates]
            for same in sets:
                found.append({"bytes": size, "paths": sorted(same), "savings": size * (len(same) - 1)})
    return found, True


def _build_report(scanner: _Scanner, duplicates: list[dict], duplicates_complete: bool,
                  started: float) -> dict:
    totals = scanner.totals()
    root_key = os.path.normcase(scanner.root)
    children = defaultdict(list)
    for key, parent in scanner.parent.items():
        if key in totals:
            children[parent].append(key)

    tree = {}
    for key, (path, size, files) in totals.items():
        if size < FOLDER_MIN_BYTES and key != root_key:
            continue
        kids = sorted(children.get(key, []), key=lambda k: totals[k][1], reverse=True)
        big = [k for k in kids if totals[k][1] >= FOLDER_MIN_BYTES]
        small = [k for k in kids if totals[k][1] < FOLDER_MIN_BYTES]
        tree[key] = {"path": path, "bytes": size, "files": files,
                     "own_bytes": scanner.own[key][1],
                     "folders": [{"key": k, "path": totals[k][0], "bytes": totals[k][1],
                                  "files": totals[k][2]} for k in big],
                     "small_folders": {"count": len(small), "bytes": sum(totals[k][1] for k in small)}}

    hogs = []
    for hog in _space_hog_locations(Path(scanner.root)):
        for p in hog["paths"]:
            key = os.path.normcase(os.path.realpath(p))
            if key in totals and totals[key][1] > 0:
                hogs.append({"id": hog["id"], "name": hog["name"], "path": totals[key][0],
                             "bytes": totals[key][1], "advice": hog["advice"]})
                break
    hogs.sort(key=lambda h: h["bytes"], reverse=True)

    installers = sorted(scanner.old_installers, key=lambda f: f["bytes"], reverse=True)
    return {
        "root": scanner.root,
        "scanned_at": time.time(),
        "duration_s": round(time.time() - started, 1),
        "files": scanner.files,
        "bytes": scanner.bytes,
        "unreadable_folders": scanner.errors,
        "tree": tree,
        "largest_files": [{"path": p, "bytes": s} for s, p in sorted(scanner.largest, reverse=True)],
        "space_hogs": hogs,
        "old_installers": {"days": OLD_DAYS, "bytes": sum(f["bytes"] for f in installers),
                           "files": installers[:100], "count": len(installers)},
        "duplicates": {"groups": duplicates[:100], "count": len(duplicates),
                       "savings": sum(d["savings"] for d in duplicates),
                       "complete": duplicates_complete, "min_bytes": DUPLICATE_MIN_BYTES},
    }


class StorageService:
    """At most one scan at a time; the last report is kept in memory and on disk."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._scanner: _Scanner | None = None
        self.state = "idle"      # idle | scanning | duplicates | done | cancelled | error
        self.error: str | None = None
        self.started: float | None = None
        self.dup_progress: dict = {}
        self.report: dict | None = self._load()

    @staticmethod
    def _path() -> Path:
        return app_home() / "storage_report.json"

    def _load(self) -> dict | None:
        try:
            report = json.loads(self._path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        same_root = os.path.normcase(report.get("root", "")) == os.path.normcase(str(policy.home_root()))
        return report if same_root else None

    def start(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._cancel.clear()
            self.state, self.error, self.started = "scanning", None, time.time()
            self.dup_progress = {}
            self._scanner = _Scanner(policy.home_root(), self._cancel)
            self._thread = threading.Thread(target=self._run, daemon=True, name="storage-scan")
            self._thread.start()
            return True

    def cancel(self) -> None:
        self._cancel.set()

    def _run(self) -> None:
        scanner = self._scanner
        audit.log("storage.scan.start", root=policy.display(scanner.root))
        try:
            scanner.run()
            self.state = "duplicates"
            duplicates, complete = find_duplicates(scanner.same_size, self._cancel, self.dup_progress)
            report = _build_report(scanner, duplicates, complete, self.started)
            self._path().write_text(json.dumps(report), encoding="utf-8")
            self.report = report
            self.state = "done"
            audit.log("storage.scan.done", files=scanner.files, bytes=scanner.bytes,
                      seconds=report["duration_s"])
        except ScanCancelled:
            self.state = "cancelled"
            audit.log("storage.scan.cancelled")
        except Exception as exc:  # report it in the UI instead of dying silently
            self.state, self.error = "error", str(exc)
            audit.log("storage.scan.error", error=str(exc))
        finally:
            self._scanner = None

    def status(self) -> dict:
        scanner = self._scanner
        out = {"state": self.state, "error": self.error,
               "elapsed_s": round(time.time() - self.started, 1) if self.started and scanner else None}
        if scanner:
            out.update(files=scanner.files, bytes=scanner.bytes,
                       current=policy.display(scanner.current) if scanner.current else None,
                       expected_files=(self.report or {}).get("files"),
                       hashed_bytes=self.dup_progress.get("hashed_bytes", 0))
        return out

    def summary(self) -> dict | None:
        """The report without the folder tree (that is served per folder)."""
        if not self.report:
            return None
        r = self.report
        root = r["tree"].get(os.path.normcase(r["root"]), {})
        return {k: v for k, v in r.items() if k != "tree"} | {"home": _public_node(root)}

    def folder(self, path: str) -> dict:
        """One folder: subfolder sizes (from the scan) and its own files (live)."""
        real = policy.check_read(path)
        if not real.is_dir():
            raise policy.PolicyError("That is not a folder.")
        node = (self.report or {}).get("tree", {}).get(os.path.normcase(str(real)))
        if node is None:
            node = _measure_live(real, self._cancel)
        files = []
        try:
            with os.scandir(real) as entries:
                for e in entries:
                    try:
                        if e.is_file(follow_symlinks=False):
                            st = e.stat(follow_symlinks=False)
                            files.append({"path": e.path, "bytes": st.st_size, "modified": st.st_mtime})
                    except OSError:
                        continue
        except OSError:
            pass
        files.sort(key=lambda f: f["bytes"], reverse=True)
        parts = policy._relative_parts(real) or ()
        return _public_node(node) | {"files": files[:200], "file_count": len(files),
                                     "total_files": node.get("files", 0),
                                     "crumbs": [{"name": "~" if i == 0 else parts[i - 1],
                                                 "path": str(Path(policy.home_root(), *parts[:i]))}
                                                for i in range(len(parts) + 1)]}


def _public_node(node: dict) -> dict:
    return {"path": node.get("path"), "bytes": node.get("bytes", 0), "files": node.get("files", 0),
            "own_bytes": node.get("own_bytes", 0),
            "folders": [{"path": f["path"], "bytes": f["bytes"], "files": f["files"]}
                        for f in node.get("folders", [])],
            "small_folders": node.get("small_folders", {"count": 0, "bytes": 0})}


def _measure_live(real: Path, cancel: threading.Event) -> dict:
    """A folder the saved report doesn't cover (small, or no scan yet)."""
    scanner = _Scanner(real, cancel)
    scanner.run()
    totals = scanner.totals()
    key = os.path.normcase(str(real))
    kids = sorted((k for k, p in scanner.parent.items() if p == key), key=lambda k: totals[k][1], reverse=True)
    return {"path": str(real), "bytes": totals[key][1], "files": totals[key][2],
            "own_bytes": scanner.own[key][1],
            "folders": [{"path": totals[k][0], "bytes": totals[k][1], "files": totals[k][2]} for k in kids],
            "small_folders": {"count": 0, "bytes": 0}}


_service: StorageService | None = None


def service() -> StorageService:
    global _service
    if _service is None:
        _service = StorageService()
    return _service


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def chat_summary() -> str:
    """Text answer for the /storage chat command."""
    lines = ["Storage:"]
    for d in drives():
        lines.append(f"  {d['mount']}  {human(d['free'])} free of {human(d['total'])} ({d['percent']:.0f}% used)")
    bin_info = recycle_bin()
    if bin_info and bin_info["bytes"]:
        lines.append(f"  Recycle Bin: {human(bin_info['bytes'])} ({bin_info['items']} items)")
    summary = service().summary()
    if not summary:
        lines.append("\nNo scan of your home folder yet. Open the Storage tab and click Scan.")
        return "\n".join(lines)
    when = time.strftime("%d %b %H:%M", time.localtime(summary["scanned_at"]))
    lines.append(f"\nBiggest folders in your home folder (scanned {when}):")
    for f in summary["home"]["folders"][:6]:
        lines.append(f"  {human(f['bytes']):>9}  {policy.display(f['path'])}")
    if summary["space_hogs"]:
        lines.append("\nSpace you might get back:")
        for h in summary["space_hogs"][:5]:
            lines.append(f"  {human(h['bytes']):>9}  {h['name']}: {h['advice']}")
    dup = summary["duplicates"]
    if dup["count"]:
        lines.append(f"\nDuplicate files: {dup['count']} sets, {human(dup['savings'])} could be freed.")
    lines.append("\nOpen the Storage tab for details. Nothing is deleted from here.")
    return "\n".join(lines)


def _norm(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def _fuzzy_in(name: str, text: str, cutoff: float = 0.85) -> bool:
    """Is something close to `name` inside `text`? Compares windows of
    about the same length ('hufggingface' vs 'huggingface' = 0.96)."""
    from difflib import SequenceMatcher

    n = len(name)
    for size in (n - 1, n, n + 1):
        for start in range(0, max(1, len(text) - size + 1)):
            window = text[start:start + size]
            if window[:1] != name[:1] and window[-1:] != name[-1:]:
                continue  # cheap skip: neither end lines up
            if SequenceMatcher(None, name, window).ratio() >= cutoff:
                return True
    return False


def match_folder(message: str, near: str | None = None) -> tuple[str | None, bool]:
    """The scanned folder a chat message names, and whether it was found
    inside `near` (the folder the conversation was just looking at).

    "what's in app data" -> ~\\AppData. After that, "what's in local" means
    ~\\AppData\\Local, not ~\\.local: folders up to two levels inside `near`
    win over folders one level below home.
    """
    report = service().report
    if not report:
        return None, False
    lowered = message.lower()
    text = _norm(message)
    root = os.path.normcase(report["root"])
    near_key = os.path.normcase(near) if near else None
    # Well-known space hogs (Hugging Face, Docker, Temp...) can be found by
    # name wherever they are, not only one level below home.
    # (and their parent: "docker" means AppData\Local\Docker, home of Docker\wsl)
    hogs = {os.path.normcase(p) for h in report.get("space_hogs", [])
            for p in (h["path"], os.path.dirname(h["path"]))}
    best = None
    for key, node in report["tree"].items():
        if key == root or not key.startswith(root + os.sep):
            continue
        base = os.path.basename(node["path"])
        name = _norm(base)
        if len(name) < 4:
            continue
        inside = bool(near_key) and key.startswith(near_key + os.sep) \
            and key[len(near_key) + 1:].count(os.sep) <= 1
        depth = key[len(root) + 1:].count(os.sep)
        if not inside and depth > 1 and key not in hogs:
            continue
        exact = name in text
        # Typos: "hufgging face" still means huggingface (long names only).
        if not exact and not (len(name) >= 6 and _fuzzy_in(name, text)):
            continue
        # A dot-folder only wins if the dot was typed (".local", not "local").
        typed_dot = not base.startswith(".") or base.lower() in lowered
        rank = (inside, exact, typed_dot, len(name), -depth, node["bytes"])
        if best is None or rank > best[0]:
            best = (rank, node["path"], inside)
    return (best[1], best[2]) if best else (None, False)


# Plain-English labels for folders that are big on most computers.
KNOWN_FOLDERS = {
    "docker": "Docker Desktop: containers and images",
    "wsl": "Linux (WSL) virtual disks",
    "android": "Android SDK and emulator images",
    "google": "Chrome and other Google apps: profiles and cache",
    "microsoft": "Windows apps, Edge, Teams and OneDrive data",
    "temp": "temporary files, usually safe to clear",
    "packages": "Microsoft Store apps' data",
    "programs": "apps installed just for you (VS Code, Python, ...)",
    "npm-cache": "npm package downloads",
    "pip": "pip package downloads",
    "uv": "uv Python package cache",
    "pnpm": "pnpm package store",
    "yarn": "Yarn package cache",
    "nuget": "NuGet (.NET) package cache",
    "jetbrains": "JetBrains IDE caches",
    "ms-playwright": "browsers downloaded for Playwright tests",
    "code": "VS Code settings, extensions and cache",
    "cursor": "Cursor editor data",
    "spotify": "Spotify: offline songs and cache",
    "discord": "Discord app and cache",
    "whatsapp": "WhatsApp media and cache",
    "zoom": "Zoom app data",
    "slack": "Slack app and cache",
    "crashdumps": "crash reports from apps, safe to delete",
    "d3dscache": "graphics shader cache, safe to delete",
    "nvidia": "NVIDIA driver and shader caches",
    "nvidia corporation": "NVIDIA driver and shader caches",
    "adobe": "Adobe apps' cache",
    "blackmagic design": "DaVinci Resolve data",
    "obs-studio": "OBS recordings settings and cache",
    "ollama": "Ollama app (models are in ~\\.ollama)",
    "huggingface": "AI models downloaded by Python projects",
    "local": "data apps keep on this computer only (caches, app installs)",
    "roaming": "app settings that follow your Windows account",
    "locallow": "data from apps that run with low permissions (browsers, games)",
    "sdk": "Android SDK: tools and emulator images",
    "system-images": "Android emulator system images",
    "avd": "Android emulators (virtual phones)",
}


def describe(path: str) -> str | None:
    return KNOWN_FOLDERS.get(os.path.basename(path).lower())


def folder_chat(path: str) -> str:
    """Text answer for "what's in <folder>" (read-only)."""
    svc = service()
    try:
        view = svc.folder(path)
    except policy.PolicyError as exc:
        return str(exc)
    real = view["path"]
    lines = [f"{policy.display(real)}: {human(view['bytes'])} in {view['total_files']:,} files"]
    if describe(real):
        lines.append(f"What it is: {describe(real)}.")
    if view["folders"]:
        lines.append("\nBiggest folders inside:")
        for f in view["folders"][:8]:
            label = describe(f["path"])
            lines.append(f"  {human(f['bytes']):>9}  {os.path.basename(f['path'])}"
                         + (f"  ({label})" if label else ""))
    small = view["small_folders"]
    if small["count"]:
        lines.append(f"  + {small['count']} smaller folders, {human(small['bytes'])}")
    prefix = os.path.normcase(real) + os.sep
    big_files = [f for f in (svc.report or {}).get("largest_files", [])
                 if os.path.normcase(f["path"]).startswith(prefix)][:5]
    if big_files:
        lines.append("\nLargest files inside:")
        for f in big_files:
            lines.append(f"  {human(f['bytes']):>9}  {policy.display(f['path'])}")
    hogs = [h for h in (svc.report or {}).get("space_hogs", [])
            if os.path.normcase(h["path"]).startswith(prefix)]
    if hogs:
        lines.append("\nWhat you could clean here:")
        for h in hogs[:6]:
            lines.append(f"  {human(h['bytes']):>9}  {h['name']}: {h['advice']}")
    reason = policy.protected_reason(Path(real))
    if reason:
        tip = "; the cleanup tips above are the safe way" if hogs else ""
        lines.append(f"\nAthena won't delete things in {policy.display(real)} directly ({reason}){tip}.")
    lines.append("\nSizes are from the last scan. Open the Storage tab to click through folders.")
    return "\n".join(lines)
