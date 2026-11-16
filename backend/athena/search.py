from datetime import datetime

from .config import settings

# Security: the chat API can reach these commands, so they only ever see
# Athena's own document folder, never the rest of the laptop.
SEARCH_FOLDERS = [settings.data_dir]


def find_files(query: str, limit: int = 10):
    query = query.lower().strip()
    matches = []

    for folder in SEARCH_FOLDERS:
        if not folder.exists():
            continue

        try:
            for path in folder.rglob("*"):
                try:
                    if not path.is_file():
                        continue

                    if query not in path.name.lower():
                        continue

                    stat = path.stat()

                    matches.append({
                        "name": path.name,
                        "path": str(path),
                        "folder": str(path.parent),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(
                            stat.st_mtime
                        ).strftime("%Y-%m-%d %H:%M")
                    })

                except (PermissionError, OSError):
                    continue

        except (PermissionError, OSError):
            continue

    # Newest files first
    matches.sort(
        key=lambda x: x["modified"],
        reverse=True
    )

    return matches[:limit]