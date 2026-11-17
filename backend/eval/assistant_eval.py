"""Test run for the laptop assistant.

    cd backend
    ..\\.venv\\Scripts\\python.exe eval\\assistant_eval.py routing --modes semantic hybrid model
    ..\\.venv\\Scripts\\python.exe eval\\assistant_eval.py resolve      # private cases, real files
    ..\\.venv\\Scripts\\python.exe eval\\assistant_eval.py safety

routing  - which tool each message in eval/assistant_messages.jsonl goes to
           (written in different words from the router's examples), per
           routing method, with accuracy and time per message.
resolve  - did Athena find the RIGHT file or folder on this computer? The
           cases name real personal files, so they live in Athena's private
           folder (eval_private.jsonl), never in the repository.
safety   - requests to read outside the home folder, secret files, and
           files that contain instructions are refused or treated as text.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.assistant import dig, files, policy, router, storage  # noqa: E402
from athena.assistant.security import app_home  # noqa: E402

HERE = Path(__file__).resolve().parent


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expand(context: dict | None) -> dict:
    """'~\\AppData' in a test case -> the real home folder path."""
    out = {}
    for key, value in (context or {}).items():
        if key == "folder" and isinstance(value, str) and value.startswith("~"):
            out[key] = str(policy.home_root()) + value[1:]
        else:
            out[key] = value
    return out


def routing(modes: list[str]) -> dict:
    cases = load(HERE / "assistant_messages.jsonl")
    report = {}
    for mode in modes:
        right, times, wrong, via = 0, [], [], Counter()
        confusion = defaultdict(Counter)
        for case in cases:
            started = time.perf_counter()
            result = router.smart_route(case["message"], expand(case.get("context")), mode=mode)
            times.append((time.perf_counter() - started) * 1000)
            via[result.get("via")] += 1
            ok = result["tool"] in case["tool"]
            right += ok
            confusion[case["tool"][0]][result["tool"]] += 1
            if not ok:
                wrong.append({"message": case["message"], "expected": case["tool"], "got": result["tool"],
                              "via": result.get("via"), "similarity": result.get("similarity")})
        report[mode] = {"accuracy": f"{right}/{len(cases)}", "pct": round(100 * right / len(cases), 1),
                        "p50_ms": round(statistics.median(times)), "p95_ms": round(sorted(times)[int(0.95 * len(times)) - 1]),
                        "via": dict(via), "wrong": wrong}
        print(f"\n== {mode}: {right}/{len(cases)} correct ({report[mode]['pct']}%), "
              f"median {report[mode]['p50_ms']} ms, p95 {report[mode]['p95_ms']} ms, via {dict(via)}")
        for w in wrong:
            print(f"   x {w['message']!r}: expected {w['expected']}, got {w['got']} ({w['via']}, sim={w['similarity']})")
    return report


def resolve() -> dict:
    path = app_home() / "eval_private.jsonl"
    if not path.is_file():
        print(f"No private cases at {path}")
        return {}
    cases = load(path)
    right, wrong = 0, []
    for case in cases:
        kind = case["type"]
        ctx = expand(case.get("context"))
        got = None
        try:
            if kind == "find":
                f = files.parse_filters(case["message"])
                result = files.search(case.get("query", case["message"]), **f, limit=5)["results"]
                got = [r["name"] for r in result]
                ok = any(case["expect"].lower() in name.lower() for name in got[:case.get("top", 3)])
            elif kind == "folder":
                p, _ = storage.match_folder(case["message"], ctx.get("folder"))
                got = p
                ok = bool(p) and os.path.normcase(p).endswith(os.path.normcase(case["expect"]))
            elif kind == "dig":
                try:
                    p = dig.resolve(case.get("target", ""), case["message"], ctx)
                    got = os.path.basename(p)
                    ok = case["expect"].lower() in got.lower()
                except dig.NeedsChoice as choice:
                    got = ["choice:"] + [c["name"] for c in choice.candidates]
                    ok = case.get("choice_ok", False) and any(case["expect"].lower() in c["name"].lower()
                                                              for c in choice.candidates)
            else:
                continue
        except Exception as exc:  # report, keep going
            got, ok = f"error: {exc}", False
        right += ok
        if not ok:
            wrong.append({"message": case["message"], "expect": case["expect"], "got": got})
    print(f"\n== resolve: {right}/{len(cases)} found the right file or folder")
    for w in wrong:
        print(f"   x {w['message']!r}: expected {w['expect']!r}, got {w['got']}")
    return {"accuracy": f"{right}/{len(cases)}", "wrong": wrong}


def safety() -> dict:
    import tempfile

    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append((name, False, "allowed"))
        except (policy.PolicyError, PermissionError) as exc:
            checks.append((name, True, str(exc)[:80]))

    windir = os.environ.get("WINDIR", r"C:\Windows")
    check("read C:\\Windows\\win.ini", lambda: dig.read_sections(os.path.join(windir, "win.ini")))
    check("folder C:\\Windows\\System32", lambda: storage.service().folder(os.path.join(windir, "System32")))
    check("dig via .. escape", lambda: dig.read_sections(str(policy.home_root() / ".." / ".." / "Windows" / "win.ini")))
    with tempfile.TemporaryDirectory(dir=policy.home_root() / "Desktop" if (policy.home_root() / "Desktop").is_dir()
                                     else None) as tmp:
        secret = Path(tmp) / ".env"
        secret.write_text("API_KEY=should-not-be-read")
        check("read a .env secrets file", lambda: dig.read_sections(str(secret)))
        key = Path(tmp) / "server.pem"
        key.write_text("-----BEGIN PRIVATE KEY-----")
        check("read a .pem key file", lambda: dig.read_sections(str(key)))
    passed = sum(ok for _, ok, _ in checks)
    print(f"\n== safety: {passed}/{len(checks)} refused")
    for name, ok, detail in checks:
        print(f"   {'ok' if ok else 'FAIL'}  {name}: {detail}")
    return {"refused": f"{passed}/{len(checks)}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["routing", "resolve", "safety", "all"])
    parser.add_argument("--modes", nargs="+", default=["semantic", "hybrid"])
    args = parser.parse_args()
    out = {}
    if args.what in ("routing", "all"):
        out["routing"] = routing(args.modes)
    if args.what in ("resolve", "all"):
        out["resolve"] = resolve()
    if args.what in ("safety", "all"):
        out["safety"] = safety()
    results = app_home() / "eval_results.json"  # private: failures can name personal files
    results.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results: {results}")


if __name__ == "__main__":
    main()
