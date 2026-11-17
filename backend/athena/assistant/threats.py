"""A quick, read-only check for common signs of malware on Windows.

Not an antivirus. It looks for the tricks real infections use to hide and
to keep running, the same ones found on a real infected laptop:

* a script interpreter (pythonw, wscript, mshta, powershell...) running a
  file disguised as something else (update.dll, support.ico);
* a program named like a Windows system process (RuntimeBroker, svchost)
  running from a user folder instead of C:\\Windows;
* hidden scheduled tasks, auto-start entries and WMI event consumers that
  launch programs from user-writable folders (AppData, Temp, Downloads);
* Microsoft Defender switched off, or recent Defender detections.

Data is collected with one PowerShell call (no admin needed) and judged by
the pure functions below, so the rules are unit-tested without Windows.
"""

import json
import logging
import os
import re
import subprocess
import sys
import time

from . import audit

logger = logging.getLogger(__name__)

USER_WRITABLE = re.compile(r"\\(appdata|temp|tmp|downloads|users\\public|programdata)\\", re.I)
SCRIPT_HOSTS = {"pythonw.exe", "python.exe", "wscript.exe", "cscript.exe", "mshta.exe", "powershell.exe",
                "pwsh.exe", "rundll32.exe", "regsvr32.exe", "node.exe", "javaw.exe", "cmd.exe"}
SCRIPT_EXTS = {".py", ".pyw", ".js", ".vbs", ".vbe", ".jse", ".wsf", ".ps1", ".bat", ".cmd", ".hta", ".jar"}
DISGUISE_EXTS = {".dll", ".ico", ".dat", ".txt", ".log", ".jpg", ".jpeg", ".png", ".gif", ".tmp", ".bin",
                 ".cfg", ".ini", ".db", ".pdf", ".docx", ".xml"}
SYSTEM_NAMES = {"runtimebroker", "svchost", "csrss", "lsass", "explorer", "dllhost", "taskhostw", "winlogon",
                "conhost", "services", "smss", "spoolsv", "sihost", "wininit", "dwm", "ctfmon", "searchhost",
                "msedgewebview2", "securityhealthservice", "wuauclt", "taskmgr"}

_COLLECT = r"""
$ErrorActionPreference='SilentlyContinue'
$o=@{}
$o.tasks=@(Get-ScheduledTask | Where-Object { $_.TaskPath -notlike '\Microsoft\*' } | ForEach-Object {
  @{name="$($_.TaskPath)$($_.TaskName)"; hidden=[bool]$_.Settings.Hidden; state="$($_.State)";
    description="$($_.Description)"; actions=@($_.Actions | ForEach-Object { "$($_.Execute) $($_.Arguments)".Trim() })} })
$keys='HKCU:\Software\Microsoft\Windows\CurrentVersion\Run','HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce',
      'HKLM:\Software\Microsoft\Windows\CurrentVersion\Run','HKLM:\Software\Microsoft\Windows\CurrentVersion\RunOnce',
      'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Run'
$o.run=@(foreach($k in $keys){ $p=Get-ItemProperty $k; if($p){ $p.PSObject.Properties |
  Where-Object { $_.Name -notlike 'PS*' } | ForEach-Object { @{key=$k; name=$_.Name; value="$($_.Value)"} } } })
$o.startup=@(Get-ChildItem "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup",
  "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\Startup" -File | Where-Object { $_.Name -ne 'desktop.ini' } |
  ForEach-Object { $_.FullName })
$o.wmi=@(Get-CimInstance -Namespace root\subscription -ClassName CommandLineEventConsumer | ForEach-Object {
  @{name=$_.Name; command="$($_.CommandLineTemplate)"} })
$o.procs=@(Get-CimInstance Win32_Process | ForEach-Object {
  @{pid=$_.ProcessId; name="$($_.Name)"; path="$($_.ExecutablePath)"; cmd="$($_.CommandLine)"} })
$o.defender=@{service="$((Get-Service WinDefend).Status)"}
$o.detections=@(Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-Windows Defender/Operational';
  Id=1116,1117,5001; StartTime=(Get-Date).AddDays(-30)} -MaxEvents 10 | ForEach-Object {
  @{time=$_.TimeCreated.ToString('s'); id=$_.Id;
    text=(($_.Message -split "`r?`n" | Where-Object { $_ -match 'Name:|Path:' } | Select-Object -First 2) -join ' ')} })
$o.conns=@(Get-NetTCPConnection -State Established | Where-Object {
  $_.RemoteAddress -notmatch '^(127\.|::1|0\.0\.0\.0|192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.|fe80|::ffff:127)' } |
  ForEach-Object { @{pid=$_.OwningProcess; remote="$($_.RemoteAddress):$($_.RemotePort)"} })
$o | ConvertTo-Json -Depth 5 -Compress
"""


def collect(timeout: int = 90) -> dict:
    """Raw facts from Windows (read-only queries only)."""
    flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", _COLLECT],
        capture_output=True, text=True, timeout=timeout, creationflags=flags)
    out = result.stdout.strip()
    if not out:
        raise RuntimeError("Windows didn't return any data: " + result.stderr.strip()[:200])
    return json.loads(out)


# ------------------------------------------------------------ the rules

def _split_command(command: str) -> tuple[str, list[str]]:
    """'"C:\\a b\\pythonw.exe" "C:\\x\\update.dll"' -> ('C:\\a b\\pythonw.exe', ['C:\\x\\update.dll'])."""
    parts = re.findall(r'"([^"]*)"|(\S+)', command or "")
    tokens = [a or b for a, b in parts]
    return (tokens[0] if tokens else ""), tokens[1:]


def judge_command(command: str) -> tuple[str, str] | None:
    """(severity, reason) if a command line looks like malware hiding, else None.
    severity: 'high' = typical of malware, 'check' = unusual, make sure you know it."""
    exe, args = _split_command(command)
    name = os.path.basename(exe).lower()
    if name in SCRIPT_HOSTS:
        for arg in args:
            ext = os.path.splitext(arg.strip().lower())[1]
            if ext in DISGUISE_EXTS and (USER_WRITABLE.search(arg) or USER_WRITABLE.search(exe)):
                return "high", f"{name} is running {os.path.basename(arg)}, a file disguised as {ext}"
        if name in ("powershell.exe", "pwsh.exe") and re.search(r"\s-(e|enc|encodedcommand)\s", f" {command} ", re.I):
            return "high", "PowerShell running a hidden (encoded) command"
        if name == "mshta.exe" and re.search(r"https?://|javascript:|vbscript:", command, re.I):
            return "high", "mshta running code from the web"
        if USER_WRITABLE.search(exe) or any(USER_WRITABLE.search(a) and os.path.splitext(a)[1].lower()
                                            in SCRIPT_EXTS for a in args):
            return "check", f"{name} running a script from a user folder"
    stem = os.path.splitext(name)[0]
    folder = os.path.basename(os.path.dirname(exe)).lower()
    if exe and USER_WRITABLE.search(exe) and (stem in SYSTEM_NAMES or folder in SYSTEM_NAMES):
        return "high", f"named like the Windows process \"{stem if stem in SYSTEM_NAMES else folder}\" but runs from a user folder"
    return None


def assess(data: dict) -> dict:
    findings = []

    def add(severity, where, item, reason, detail=""):
        findings.append({"severity": severity, "where": where, "item": item, "reason": reason, "detail": detail})

    for task in data.get("tasks") or []:
        for action in task.get("actions") or []:
            verdict = judge_command(action)
            exe, _ = _split_command(action)
            if verdict:
                sev, why = verdict
                if task.get("hidden") and sev == "check":
                    sev = "high"
                    why += " (and the task is hidden)"
                add(sev, "Scheduled task", task["name"].lstrip("\\"), why, action)
            elif task.get("hidden") and USER_WRITABLE.search(exe):
                add("check", "Scheduled task", task["name"].lstrip("\\"),
                    "hidden task that starts a program from a user folder", action)

    for entry in data.get("run") or []:
        verdict = judge_command(entry.get("value", ""))
        if verdict:
            add(*verdict[:1], "Starts with Windows", entry["name"], verdict[1], entry["value"])

    for path in data.get("startup") or []:
        if os.path.splitext(path)[1].lower() in SCRIPT_EXTS | {".exe", ".scr", ".pif"}:
            add("check", "Startup folder", os.path.basename(path), "a script or program placed directly in Startup", path)

    for consumer in data.get("wmi") or []:
        add("high", "WMI event trigger", consumer.get("name", "?"),
            "WMI command triggers are a rare, stealthy way to keep malware running", consumer.get("command", ""))

    suspicious_pids = set()
    for proc in data.get("procs") or []:
        command = proc.get("cmd") or ""
        path = proc.get("path") or ""
        verdict = judge_command(command or f'"{path}"') if (command or path) else None
        if verdict:
            suspicious_pids.add(proc.get("pid"))
            add(verdict[0], "Running now", proc.get("name") or "?", verdict[1], command or path)

    for conn in data.get("conns") or []:
        if conn.get("pid") in suspicious_pids:
            add("high", "Internet connection", conn["remote"], "a suspicious program is connected to the internet",
                f"process {conn['pid']}")

    defender = (data.get("defender") or {}).get("service", "")
    if defender and defender != "Running":
        add("high", "Microsoft Defender", "WinDefend", f"Defender's service is {defender}, not running")
    for event in data.get("detections") or []:
        label = {1116: "Defender found malware", 1117: "Defender removed malware",
                 5001: "Defender real-time protection was turned off"}.get(event.get("id"), "Defender event")
        add("high" if event.get("id") == 5001 else "check", "Microsoft Defender", label,
            f"on {event.get('time', '?')[:10]}", event.get("text", ""))

    order = {"high": 0, "check": 1}
    findings.sort(key=lambda f: order.get(f["severity"], 2))
    high = sum(f["severity"] == "high" for f in findings)
    return {
        "verdict": "danger" if high else ("check" if findings else "clean"),
        "findings": findings[:30],
        "counts": {"tasks": len(data.get("tasks") or []), "autostart": len(data.get("run") or []) +
                   len(data.get("startup") or []), "processes": len(data.get("procs") or []),
                   "connections": len(data.get("conns") or [])},
        "defender": defender or "unknown",
    }


_cache: dict = {}


def check(max_age: float = 120) -> dict:
    if sys.platform != "win32":
        return {"verdict": "unsupported", "findings": [], "counts": {}, "defender": "n/a"}
    if _cache.get("at", 0) > time.time() - max_age:
        return _cache["result"]
    started = time.time()
    result = assess(collect()) | {"seconds": round(time.time() - started, 1), "checked_at": time.time()}
    _cache.update(at=time.time(), result=result)
    audit.log("security.check", verdict=result["verdict"], findings=len(result["findings"]))
    return result
