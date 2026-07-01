"""LLM model factory + sync boundary for pydantic-ai.

Two public entry points:

- `get_llm_model()` — builds a `pydantic_ai.Model` from `SiteConfig`,
  routing to the right provider.
- `run_agent_sync(coro)` — drives a pydantic-ai coroutine to completion
  from sync code, on a dedicated worker thread with a long-lived event
  loop. Used everywhere instead of `Agent.run_sync`.

Why a persistent worker thread (not `Agent.run_sync`, not `asyncio.run`):

- `Agent.run_sync` uses an anyio portal that leaves the caller thread's
  running-loop slot populated. Subsequent sync Playwright calls on the
  daemon thread then raise
  `"using Playwright Sync API inside the asyncio loop"`.
- `asyncio.run` per call closes its loop on exit. The openai / anthropic
  SDKs wrap `httpx.AsyncClient` in a subclass whose `__del__` does
  `get_running_loop().create_task(self.aclose())`. If GC fires the
  wrapper from call N during call N+1's loop, the cleanup task tries to
  close a transport bound to call N's now-closed loop →
  `RuntimeError: Event loop is closed`.

A single long-lived loop on a dedicated thread eliminates both: all HTTP
clients live on the same loop forever, and the runner thread's asyncio
slot stays inside this module — the caller thread is never touched.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Awaitable, Callable, TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Override the SDK default of 2. Each retry uses the SDK's built-in jittered
# exponential backoff and honors `Retry-After`, so 8 attempts ride through
# typical 429/529 capacity blips (~1–2 minutes) instead of failing in ~1.5s.
_MAX_RETRIES = 8


# ── Async runner ─────────────────────────────────────────────────────

class _AgentRunner:
    """Owns one persistent asyncio loop on a dedicated daemon thread.

    Construct lazily via `_get_runner()` so importing this module is free.
    The thread is a daemon, so no explicit shutdown is needed — it ends
    with the process.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()
        threading.Thread(
            target=self._serve, args=(ready,), daemon=True, name="llm-runner",
        ).start()
        ready.wait()

    def _serve(self, ready: threading.Event) -> None:
        asyncio.set_event_loop(self._loop)
        ready.set()
        self._loop.run_forever()

    def run(self, coro: Awaitable[_T]) -> _T:
        """Submit *coro* to the runner loop; block until it completes."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


_runner: _AgentRunner | None = None
_runner_lock = threading.Lock()


def _get_runner() -> _AgentRunner:
    """Return the process-wide runner, creating it on first call."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = _AgentRunner()
    return _runner


def run_agent_sync(coro: Awaitable[_T]) -> _T:
    """Drive *coro* on the dedicated LLM runner thread + loop."""
    return _get_runner().run(coro)


# ── Rate-limit backoff ───────────────────────────────────────────────

# Some providers (notably Gemini's free tier: 5 requests/min) return HTTP 429
# with a `retryDelay` telling us exactly how long to wait. The openai/anthropic
# SDK clients retry 429 internally, but the Google provider does not honor the
# long free-tier delay — so we retry at the application level here.
_BACKOFF_MAX_ATTEMPTS = 6
_BACKOFF_MAX_DELAY_S = 90.0


def _retry_delay_from_429(exc) -> float | None:
    """Parse the server-advised retry delay (seconds) from a 429, or None.

    Handles both the structured ``RetryInfo`` detail (``retryDelay: '46s'``) and
    the human message (``Please retry in 46.7s``).
    """
    body = getattr(exc, "body", None)
    message = ""
    if isinstance(body, dict):
        error = body.get("error", {}) if isinstance(body.get("error"), dict) else {}
        for detail in error.get("details", []) or []:
            if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
                m = re.match(r"([0-9.]+)s", str(detail.get("retryDelay", "")))
                if m:
                    return float(m.group(1))
        message = str(error.get("message", ""))
    else:
        message = str(body or exc)
    m = re.search(r"retry in ([0-9.]+)\s*s", message)
    return float(m.group(1)) if m else None


def run_agent_with_backoff(
    make_coro: Callable[[], Awaitable[_T]],
    *,
    max_attempts: int = _BACKOFF_MAX_ATTEMPTS,
) -> _T:
    """Run an agent coroutine, retrying on HTTP 429 with server-advised backoff.

    *make_coro* is a **factory** (e.g. ``lambda: agent.run(prompt)``), not a
    coroutine, because a coroutine can only be awaited once — each retry needs a
    fresh one. On a 429 we sleep the ``retryDelay`` the API returned (a small
    cushion added), capped at ``_BACKOFF_MAX_DELAY_S``; other errors propagate
    immediately. The final attempt re-raises so a persistent outage still fails.
    """
    from pydantic_ai.exceptions import ModelHTTPError

    for attempt in range(1, max_attempts + 1):
        try:
            return run_agent_sync(make_coro())
        except ModelHTTPError as exc:
            if getattr(exc, "status_code", None) != 429 or attempt == max_attempts:
                raise
            advised = _retry_delay_from_429(exc)
            delay = min((advised + 1.0) if advised is not None else 5.0 * 2 ** (attempt - 1),
                        _BACKOFF_MAX_DELAY_S)
            logger.warning(
                "LLM rate limited (429) — waiting %.0fs, then retry %d/%d",
                delay, attempt + 1, max_attempts,
            )
            time.sleep(delay)


# ── Per-provider builders ────────────────────────────────────────────

def _build_openai(model, api_key, api_base):
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
    client = AsyncOpenAI(api_key=api_key, max_retries=_MAX_RETRIES)
    return OpenAIModel(model, provider=OpenAIProvider(openai_client=client))


def _build_anthropic(model, api_key, api_base):
    from anthropic import AsyncAnthropic
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    client = AsyncAnthropic(api_key=api_key, max_retries=_MAX_RETRIES)
    return AnthropicModel(model, provider=AnthropicProvider(anthropic_client=client))


def _build_google(model, api_key, api_base):
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider
    return GoogleModel(model, provider=GoogleProvider(api_key=api_key))


def _build_groq(model, api_key, api_base):
    from groq import AsyncGroq
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider
    client = AsyncGroq(api_key=api_key, max_retries=_MAX_RETRIES)
    return GroqModel(model, provider=GroqProvider(groq_client=client))


def _build_mistral(model, api_key, api_base):
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider
    return MistralModel(model, provider=MistralProvider(api_key=api_key))


def _build_cohere(model, api_key, api_base):
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider
    return CohereModel(model, provider=CohereProvider(api_key=api_key))


def _build_openai_compatible(model, api_key, api_base):
    if not api_base:
        raise ValueError("LLM_API_BASE is required for the openai_compatible provider.")
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
    return OpenAIModel(model, provider=OpenAIProvider(
        base_url=api_base, api_key=api_key,
    ))


_PROVIDER_BUILDERS: dict[str, Callable] = {
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "google": _build_google,
    "groq": _build_groq,
    "mistral": _build_mistral,
    "cohere": _build_cohere,
    "openai_compatible": _build_openai_compatible,
}

# Bare-model fallbacks: only these prefixes are unambiguous enough to route
# without an explicit `provider:` prefix (mirrors pydantic-ai's own legacy map).
# groq/mistral/cohere/openai_compatible carry no reliable prefix, so they must
# be written `provider:model`.
_LEGACY_MODEL_PREFIXES = {
    "gpt": "openai", "o1": "openai", "o3": "openai",
    "claude": "anthropic", "gemini": "google",
}


def split_model_id(ai_model: str) -> tuple[str, str]:
    """Split a `provider:model` identifier into ``(provider, model)``.

    A bare model name is accepted only when its prefix unambiguously implies a
    provider (see ``_LEGACY_MODEL_PREFIXES``); anything else raises so the
    misconfiguration surfaces instead of silently hitting the wrong API.
    """
    if ":" in ai_model:
        provider, _, model = ai_model.partition(":")
        return provider, model
    for prefix, provider in _LEGACY_MODEL_PREFIXES.items():
        if ai_model.startswith(prefix):
            return provider, ai_model
    raise ValueError(
        f"AI_MODEL {ai_model!r} has no provider prefix. "
        f"Use 'provider:model', e.g. 'anthropic:{ai_model}'."
    )


# ── Model factory ────────────────────────────────────────────────────

def _validated_site_config():
    """Load `SiteConfig` and assert the required LLM fields are populated."""
    from openoutreach.core.models import SiteConfig

    cfg = SiteConfig.load()
    if not cfg.llm_api_key:
        raise ValueError("LLM_API_KEY is not set in Site Configuration.")
    if not cfg.ai_model:
        raise ValueError("AI_MODEL is not set in Site Configuration.")
    return cfg


def build_llm_model(ai_model: str, api_key: str, api_base: str = ""):
    """Build a pydantic-ai `Model` from explicit credentials.

    Shared by `get_llm_model` (saved `SiteConfig`) and `verify_llm_credentials`
    (candidate values, before they are persisted).
    """
    provider, model = split_model_id(ai_model)
    builder = _PROVIDER_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown LLM provider {provider!r} in AI_MODEL {ai_model!r}. "
            f"Use one of: {', '.join(_PROVIDER_BUILDERS)}."
        )
    return builder(model, api_key, api_base)


def get_llm_model():
    """Return a configured pydantic-ai `Model` for the current `SiteConfig`."""
    cfg = _validated_site_config()
    return build_llm_model(cfg.ai_model, cfg.llm_api_key, cfg.llm_api_base)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(max=10), reraise=True)
def _ping_model(ai_model: str, api_key: str, api_base: str) -> None:
    """Send one trivial request to prove the credentials work (or raise)."""
    from pydantic_ai import Agent

    run_agent_sync(Agent(build_llm_model(ai_model, api_key, api_base)).run("ping"))


def verify_llm_credentials(ai_model: str, api_key: str, api_base: str = "") -> str | None:
    """Live ping for onboarding: return ``None`` if the model answers, else the error."""
    try:
        _ping_model(ai_model, api_key, api_base)
        return None
    except Exception as exc:  # noqa: BLE001 — verification reports every failure mode
        return str(exc)
