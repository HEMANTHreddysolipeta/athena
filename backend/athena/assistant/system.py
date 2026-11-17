"""Battery, memory, CPU, uptime and which apps use the most memory. Read-only."""

import time
from collections import defaultdict

import psutil

# Windows' own processes: shown as "Windows" in the app list.
_WINDOWS = {"system", "registry", "memory compression", "svchost.exe", "csrss.exe", "smss.exe",
            "wininit.exe", "services.exe", "lsass.exe", "winlogon.exe", "dwm.exe", "fontdrvhost.exe",
            "system idle process", "secure system", "vmmem", "vmmemwsl"}
_FRIENDLY = {"chrome.exe": "Google Chrome", "msedge.exe": "Microsoft Edge", "code.exe": "VS Code",
             "cursor.exe": "Cursor", "ollama.exe": "Ollama", "ollama app.exe": "Ollama",
             "llama-server.exe": "Ollama (model)", "python.exe": "Python", "pythonw.exe": "Python",
             "node.exe": "Node.js", "java.exe": "Java", "studio64.exe": "Android Studio",
             "explorer.exe": "File Explorer", "docker desktop.exe": "Docker Desktop",
             "com.docker.backend.exe": "Docker Desktop", "claude.exe": "Claude", "discord.exe": "Discord",
             "spotify.exe": "Spotify", "whatsapp.exe": "WhatsApp", "teams.exe": "Teams", "ms-teams.exe": "Teams",
             "qemu-system-x86_64.exe": "Android emulator", "firefox.exe": "Firefox", "brave.exe": "Brave",
             "msmpeng.exe": "Microsoft Defender", "searchhost.exe": "Windows Search"}


def info() -> dict:
    mem = psutil.virtual_memory()
    battery = psutil.sensors_battery()
    apps: dict[str, dict] = defaultdict(lambda: {"bytes": 0, "processes": 0})
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (p.info["name"] or "").lower()
            rss = p.info["memory_info"].rss if p.info["memory_info"] else 0
        except (psutil.Error, AttributeError):
            continue
        label = "Windows" if name in _WINDOWS else _FRIENDLY.get(name, (p.info["name"] or "?").removesuffix(".exe"))
        apps[label]["bytes"] += rss
        apps[label]["processes"] += 1
    top = sorted(({"name": k, **v} for k, v in apps.items()), key=lambda a: a["bytes"], reverse=True)[:8]
    cpu = psutil.cpu_percent(interval=0.4)
    freq = psutil.cpu_freq()
    return {
        "battery": None if battery is None else {
            "percent": round(battery.percent), "plugged_in": bool(battery.power_plugged),
            # Windows sometimes reports nonsense here (seen live: 71,582,788 minutes
            # at 20%); anything over a day is "unknown", not a fact to repeat.
            "minutes_left": None if battery.secsleft in (psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN)
            or battery.secsleft is None or not 0 < battery.secsleft < 24 * 3600 or battery.power_plugged
            else round(battery.secsleft / 60)},
        "memory": {"total": mem.total, "used": mem.total - mem.available, "available": mem.available,
                   "percent": round(mem.percent)},
        "cpu": {"percent": round(cpu), "cores": psutil.cpu_count(logical=True),
                "ghz": round(freq.current / 1000, 1) if freq and freq.current else None},
        "uptime_s": round(time.time() - psutil.boot_time()),
        "top_apps": top,
    }
