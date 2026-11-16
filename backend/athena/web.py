"""Save a web page (or an online PDF) into the library so it can be indexed.

Fetching a user-supplied URL from a server is a classic SSRF risk: the URL
could point at the machine itself (localhost admin pages, Ollama's API)
or the home network. So: http/https only, every address the hostname
resolves to must be public, redirects are followed manually and re-checked,
and the download is capped in size and time.
"""

import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

MAX_BYTES = 15 * 1024 * 1024
TIMEOUT = 20
MAX_REDIRECTS = 4
USER_AGENT = "Athena/1.0 (local document assistant)"


class FetchError(Exception):
    pass


def _check_public(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise FetchError("Only http:// and https:// links can be added.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise FetchError(f"Could not resolve {parsed.hostname}.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise FetchError("Links to local or private network addresses are not allowed.")


def _filename(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{parsed.hostname}{parsed.path}").strip("_.")[:90] or "page"
    stem = re.sub(r"\.(html?|pdf|php|aspx?)$", "", stem, flags=re.IGNORECASE)
    return stem + (".pdf" if "pdf" in content_type else ".html")


def fetch_to_file(url: str, dest_dir: Path, get=requests.get) -> Path:
    """Download `url` into dest_dir; returns the saved file's path."""
    current = url.strip()
    for _ in range(MAX_REDIRECTS + 1):
        _check_public(current)
        try:
            response = get(current, timeout=TIMEOUT, stream=True, allow_redirects=False,
                           headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            raise FetchError(f"Could not download the page: {exc.__class__.__name__}") from exc
        if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            current = urljoin(current, response.headers.get("location", ""))
            response.close()
            continue
        break
    else:
        raise FetchError("Too many redirects.")

    if response.status_code != 200:
        raise FetchError(f"The site answered with HTTP {response.status_code}.")
    content_type = response.headers.get("content-type", "").lower()
    if not any(t in content_type for t in ("text/html", "application/xhtml", "application/pdf", "text/plain")):
        raise FetchError(f"Unsupported content type: {content_type or 'unknown'}")

    body = bytearray()
    for block in response.iter_content(64 * 1024):
        body.extend(block)
        if len(body) > MAX_BYTES:
            response.close()
            raise FetchError("The page is larger than 15 MB.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / _filename(current, content_type)
    path.write_bytes(bytes(body))
    return path
