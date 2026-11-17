"""Start Athena and open it in the browser, already signed in.

    cd backend
    ..\\.venv\\Scripts\\python.exe -m athena.serve          [--port 8000] [--no-browser]

Always binds to 127.0.0.1: Athena can see the files in your home folder, so
it must never listen on the network.
"""

import argparse
import os
import threading
import time
import urllib.request
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Athena on this computer.")
    parser.add_argument("--port", type=int, default=int(os.getenv("ATHENA_PORT", "8000")))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    os.environ["ATHENA_PORT"] = str(args.port)

    import uvicorn

    import app as athena_app

    url = athena_app.login_url(args.port)

    def open_when_ready() -> None:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/session", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        print(f"\nAthena is running. If the browser didn't open, go to:\n  {url}\n", flush=True)
        if not args.no_browser:
            webbrowser.open(url)

    threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run(athena_app.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
