"""
LLM access layer (LiteLLM -> Ollama, plus embeddings).

Every agent goes through this module rather than calling urllib directly,
so retries, JSON coaxing, and error formatting live in exactly one place.
"""

import json
import re
import urllib.request
import urllib.error

from . import config, observability


def _litellm():
    """Return the litellm module if the configured backend allows it, else None."""
    if config.LLM_BACKEND == "ollama":
        return None
    try:
        import litellm
        return litellm
    except ImportError:
        if config.LLM_BACKEND == "litellm":
            raise RuntimeError("TUTOR_LLM_BACKEND=litellm but litellm is not installed (pip install litellm).")
        return None


def backend_info() -> dict:
    try:
        lite = _litellm()
    except RuntimeError:
        lite = None
    if lite:
        return {"backend": "litellm", "model": config.LITELLM_MODEL}
    return {"backend": "ollama (direct)", "model": config.MODEL_NAME}


def _complete_litellm(lite, messages: list, timeout: int) -> tuple:
    observability.setup_langfuse()
    try:
        resp = lite.completion(
            model=config.LITELLM_MODEL, messages=messages, api_base=config.LITELLM_API_BASE,
            timeout=timeout, metadata={"trace_id": observability.current_trace_id(),
                                       "generation_name": "tutor-llm-call"},
        )
    except Exception as e:  # LiteLLM raises provider-specific errors; surface one clear type
        raise RuntimeError(f"LLM call failed via LiteLLM ({config.LITELLM_MODEL}): {e}")
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    tokens = {"prompt": getattr(usage, "prompt_tokens", None), "completion": getattr(usage, "completion_tokens", None)}
    return text, tokens


def _complete_ollama(messages: list, timeout: int) -> tuple:
    payload = {"model": config.MODEL_NAME, "messages": messages, "stream": False}
    if "qwen3" in config.MODEL_NAME.lower():
        payload["think"] = False  # skip Qwen3's reasoning phase: faster, and keeps JSON replies clean
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(config.OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {config.OLLAMA_URL}. Is it running? "
            f"(ollama serve / ollama run {config.MODEL_NAME}) Details: {e}"
        )
    tokens = {"prompt": body.get("prompt_eval_count"), "completion": body.get("eval_count")}
    return body.get("message", {}).get("content", ""), tokens


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove Qwen3-style <think>...</think> reasoning blocks (also an unclosed one)."""
    text = _THINK_RE.sub("", text)
    return re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)


def call_llm_messages(messages: list, timeout: int = config.LLM_TIMEOUT_SECONDS) -> str:
    """Send a full messages list to the model and return the assistant text.
    Goes through LiteLLM when available (so the provider can be swapped with an
    env var), otherwise straight to Ollama. Every call is traced."""
    lite = _litellm()
    with observability.span("llm_call", **backend_info()) as meta:
        text, tokens = _complete_litellm(lite, messages, timeout) if lite else _complete_ollama(messages, timeout)
        meta["tokens"] = tokens
        text = strip_thinking(text)
        meta["chars_out"] = len(text)
    if not text.strip():
        raise RuntimeError("Empty response from the model.")
    return text.strip()


def embed(texts: list) -> list:
    """Embed a list of strings -> list of float vectors (one per text).
    Uses the local Ollama embedding model (default nomic-embed-text), through
    LiteLLM when available. Raises RuntimeError if embeddings are unavailable;
    callers (vector_store) then fall back to keyword search."""
    if not texts:
        return []
    lite = _litellm()
    with observability.span("embed", model=config.EMBED_MODEL, n=len(texts)):
        if lite:
            try:
                resp = lite.embedding(model=f"ollama/{config.EMBED_MODEL}", input=texts,
                                      api_base=config.LITELLM_API_BASE)
            except Exception as e:
                raise RuntimeError(f"Embedding call failed via LiteLLM: {e}")
            rows = [d["embedding"] if isinstance(d, dict) else d.embedding for d in resp.data]
        else:
            base = config.OLLAMA_URL.split("/api/")[0]
            req = urllib.request.Request(
                f"{base}/api/embed",
                data=json.dumps({"model": config.EMBED_MODEL, "input": texts}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_SECONDS) as resp:
                    rows = json.loads(resp.read().decode("utf-8")).get("embeddings", [])
            except (urllib.error.URLError, ValueError) as e:
                raise RuntimeError(f"Embedding call failed (is `ollama pull {config.EMBED_MODEL}` done?): {e}")
    if len(rows) != len(texts):
        raise RuntimeError("Embedding service returned the wrong number of vectors.")
    return rows


def call_llm(system: str, user: str, timeout: int = config.LLM_TIMEOUT_SECONDS) -> str:
    """Call the local Ollama chat endpoint with a single system+user turn."""
    return call_llm_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        timeout=timeout,
    )


def extract_json(text: str):
    """Best-effort JSON extraction. Small local models often wrap JSON in
    prose or code fences, or produce near-valid JSON -- this tries fallbacks."""
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError("Could not parse JSON from model output.")


def call_llm_json(system: str, user: str, attempts: int = config.JSON_CALL_ATTEMPTS):
    """Call the LLM expecting JSON back, retrying with a stricter reminder
    if parsing fails -- small local models sometimes need a nudge."""
    last_err = None
    for attempt in range(attempts):
        prompt = user
        if attempt > 0:
            prompt += (
                "\n\nReminder: reply with ONLY the JSON object. "
                "No markdown fences, no explanation, no extra text."
            )
        try:
            raw = call_llm(system, prompt)
            return extract_json(raw)
        except (ValueError, RuntimeError) as e:
            last_err = e
    raise RuntimeError(f"Model did not return valid JSON after {attempts} attempts: {last_err}")
