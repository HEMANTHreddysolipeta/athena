"""Load the local models without touching the network.

sentence-transformers asks huggingface.co whether a newer model version
exists every time it loads one, even when the model is already on disk. No
document data is sent, but Athena promises that nothing leaves the computer
after setup, so the cached copy is used with local_files_only=True and the
internet is only used the very first time, when the model isn't there yet.
"""

import logging
import os

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# The HTTP client logs every request at INFO; keep the server log readable.
logging.getLogger("httpx").setLevel(logging.WARNING)


def load(factory, name: str, **kwargs):
    try:
        return factory(name, local_files_only=True, **kwargs)
    except OSError:
        logger.info("%s is not downloaded yet: downloading it once (it then works offline)", name)
        return factory(name, **kwargs)
