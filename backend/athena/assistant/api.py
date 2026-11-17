"""HTTP endpoints for the laptop assistant. Mounted by app.py behind the
session-token guard, so only the person who started Athena can call them."""

import os
import subprocess
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import audit, policy, storage

router = APIRouter(prefix="/api/assistant")


@router.get("/storage")
def storage_overview():
    svc = storage.service()
    return {"drives": storage.drives(), "recycle_bin": storage.recycle_bin(),
            "scan": svc.status(), "report": svc.summary()}


@router.post("/storage/scan")
def start_scan():
    started = storage.service().start()
    return {"started": started, "scan": storage.service().status()}


@router.post("/storage/scan/cancel")
def cancel_scan():
    storage.service().cancel()
    return {"scan": storage.service().status()}


@router.get("/storage/folder")
def storage_folder(path: str):
    svc = storage.service()
    if not svc.report:
        raise HTTPException(409, "Scan your home folder first.")
    try:
        return svc.folder(path)
    except policy.PolicyError as exc:
        raise HTTPException(403, str(exc)) from exc


class PathBody(BaseModel):
    path: str


@router.post("/reveal")
def reveal(body: PathBody):
    """Show a file or folder in File Explorer / Finder. Never opens or runs
    the file itself (an .exe would execute), it only selects it."""
    try:
        real = policy.check_read(body.path)
    except policy.PolicyError as exc:
        raise HTTPException(403, str(exc)) from exc
    if sys.platform == "win32":
        explorer = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "explorer.exe")
        subprocess.Popen([explorer, f"/select,{real}"])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(real)])
    else:
        subprocess.Popen(["xdg-open", str(real if real.is_dir() else real.parent)])
    audit.log("reveal", path=policy.display(real))
    return {"ok": True}


@router.get("/activity")
def activity(limit: int = 100):
    return {"entries": audit.recent(min(max(limit, 1), 500))}
