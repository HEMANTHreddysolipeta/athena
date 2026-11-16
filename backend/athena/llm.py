"""Ollama client.

Ollama runs the model in a separate local server process; we talk to it
over HTTP. That's why a timeout matters: without one, a slow or stuck
model blocks the caller forever.
"""

import json
import logging
import re
import time

import requests

from .config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Athena, a personal AI assistant running locally on the user's computer.
Always respond in English. Be concise and helpful."""

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    """The model could not produce an answer (server down, timeout, bad reply)."""


def _options() -> dict:
    options = {
        "temperature": settings.llm_temperature,
        # Context window in tokens. Prompts longer than this are silently
        # truncated by Ollama, so rag.py budgets the prompt below it.
        "num_ctx": settings.llm_num_ctx,
    }
    if settings.llm_threads:
        options["num_thread"] = settings.llm_threads
    return options


def generate(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    url = f"{settings.ollama_url}/api/generate"
    payload = {
        "model": settings.llm_model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "think": False,  # reasoning models: skip the slow hidden reasoning pass
        "keep_alive": settings.llm_keep_alive,
        "options": _options(),
    }

    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=settings.llm_timeout_seconds)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise LLMError(
            f"{settings.llm_model} did not answer within {settings.llm_timeout_seconds}s"
        ) from exc
    except requests.ConnectionError as exc:
        raise LLMError(f"Ollama is not reachable at {settings.ollama_url}. Is it running?") from exc
    except requests.HTTPError as exc:
        raise LLMError(f"Ollama returned {response.status_code}: {response.text[:200]}") from exc

    data = response.json()
    last_stats.update(
        model=settings.llm_model, prompt_tokens=data.get("prompt_eval_count"),
        output_tokens=data.get("eval_count"), seconds=round(time.perf_counter() - started, 2),
    )
    logger.info(
        "LLM %s: %s prompt tokens, %s output tokens, %.1fs",
        settings.llm_model,
        data.get("prompt_eval_count"),
        data.get("eval_count"),
        time.perf_counter() - started,
    )
    return _THINK_BLOCK.sub("", data.get("response", "")).strip()


def generate_stream(prompt: str, system: str = SYSTEM_PROMPT):
    """Yield the answer in pieces as the model produces them.

    On a CPU the full answer takes 20-30 s, but the first words arrive in a
    few seconds; streaming makes that wait visible instead of silent.
    The final event carries token counts for the metrics.
    """
    url = f"{settings.ollama_url}/api/generate"
    payload = {
        "model": settings.llm_model,
        "system": system,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "keep_alive": settings.llm_keep_alive,
        "options": _options(),
    }
    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=settings.llm_timeout_seconds, stream=True)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise LLMError(f"{settings.llm_model} did not answer within {settings.llm_timeout_seconds}s") from exc
    except requests.ConnectionError as exc:
        raise LLMError(f"Ollama is not reachable at {settings.ollama_url}. Is it running?") from exc
    except requests.HTTPError as exc:
        raise LLMError(f"Ollama returned {response.status_code}: {response.text[:200]}") from exc

    in_think = False
    try:
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("response", "")
            # Drop <think>...</think> blocks from reasoning models.
            if "<think>" in piece:
                in_think = True
            if not in_think and piece:
                yield piece
            if "</think>" in piece:
                in_think = False
            if data.get("done"):
                last_stats.update(
                    model=settings.llm_model,
                    prompt_tokens=data.get("prompt_eval_count"),
                    output_tokens=data.get("eval_count"),
                    seconds=round(time.perf_counter() - started, 2),
                )
                logger.info("LLM %s (stream): %s prompt tokens, %s output tokens, %.1fs",
                            settings.llm_model, data.get("prompt_eval_count"),
                            data.get("eval_count"), time.perf_counter() - started)
    except requests.RequestException as exc:
        raise LLMError(f"Connection to Ollama dropped: {exc}") from exc
    finally:
        response.close()


def is_available() -> tuple[bool, str]:
    """(ok, detail) for the health endpoint: is Ollama up and the model pulled?"""
    try:
        response = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return False, f"Ollama is not reachable at {settings.ollama_url}"
    names = {m["name"] for m in response.json().get("models", [])}
    if settings.llm_model not in names and f"{settings.llm_model}:latest" not in names:
        return False, f"Model {settings.llm_model} is not pulled. Run: ollama pull {settings.llm_model}"
    return True, settings.llm_model


def warm_up() -> None:
    """Load the model into memory without generating anything."""
    try:
        requests.post(f"{settings.ollama_url}/api/generate", timeout=settings.llm_timeout_seconds,
                      json={"model": settings.llm_model, "keep_alive": settings.llm_keep_alive,
                            "options": _options()})
    except requests.RequestException as exc:
        logger.warning("Could not pre-load %s: %s", settings.llm_model, exc)


# Token counts of the most recent call (read by the metrics in rag.py).
last_stats: dict = {}

# Backwards-compatible name used by engine.py
ask_llm = generate
