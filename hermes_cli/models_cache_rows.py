"""Rows of the provider-model disk cache (``$HERMES_HOME/provider_models_cache.json``): validity,
serve tier and which row a ``provider_model_ids`` result writes. Split out of ``hermes_cli.models``,
which keeps the file I/O, the TTL constants (tests patch them there) and the cache readers.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from hermes_cli.models_catalog_static import AuthoritativeModels, CuratedFallbackModels

if TYPE_CHECKING:
    from typing import TypeGuard


def _cache_entry_valid(
    entry: Any, fp: str, *, allow_empty: bool = False) -> TypeGuard[dict[str, Any]]:
    """Well-formed cache row for fingerprint *fp*. Requires a numeric ``at`` so corrupt disk state
    degrades to a cache miss instead of raising; empty model lists are valid only when the caller
    opts into an authoritative empty catalog or the row stores an ``AuthoritativeModels`` answer."""
    return (
        isinstance(entry, dict)
        and entry.get("fp") == fp
        and isinstance(entry.get("models"), list)
        and (allow_empty or bool(entry["models"]) or entry.get("authoritative") is True)
        and isinstance(entry.get("at"), (int, float))
        and not isinstance(entry.get("at"), bool))


def cached_rows(entry: dict[str, Any]) -> list[str]:
    """A valid row's model ids, typed ``AuthoritativeModels`` again when the row stored one, so the
    pickers tell an account's empty catalog from a miss after a restart too."""
    return (AuthoritativeModels if entry.get("authoritative") is True else list)(entry["models"])


def _live_result_entry(fp: str, live: list[str], existing: Any, at: Optional[float] = None) -> Optional[dict]:
    """Row to store for a ``provider_model_ids`` result, or ``None`` to keep *existing*: a curated
    fallback never replaces the account's real catalog for the same credentials, and when it is
    stored it is flagged so it expires on the short fallback TTL."""
    from hermes_cli.models import _cache_entry

    if isinstance(live, AuthoritativeModels):
        return {**_cache_entry(fp, live, at), "authoritative": True}
    if not isinstance(live, CuratedFallbackModels):
        return _cache_entry(fp, live, at)
    if _cache_entry_valid(existing, fp) and not existing.get("fallback"):
        return None
    return {**_cache_entry(fp, live, at), "fallback": True}


def _disk_serve_tier(entry: Any, fp: str, now: float, *, is_ollama: bool,
                     ttl_seconds: Optional[int] = None) -> Optional[str]:
    """How :func:`hermes_cli.models.cached_provider_model_ids` serves *entry* without the network.

    ``"fresh"`` inside the row's TTL (a curated fallback row only for
    ``_PROVIDER_MODELS_FALLBACK_TTL``), ``"stale"`` for a non-empty or authoritative, non-fallback
    row inside ``_PROVIDER_MODELS_STALE_SERVE_MAX`` (served while an SWR thread revalidates), else
    ``None``: the call would block on a live fetch. Empty native catalogs are authoritative only
    inside the short native TTL, never through the stale window."""
    from hermes_cli import models

    if ttl_seconds is None:
        ttl_seconds = models._PROVIDER_MODELS_CACHE_TTL
    if is_ollama:
        ttl_seconds = min(ttl_seconds, models._OLLAMA_LOCAL_MODELS_CACHE_TTL)
    if not _cache_entry_valid(entry, fp, allow_empty=is_ollama):
        return None
    age = now - entry["at"]
    if age < (models._PROVIDER_MODELS_FALLBACK_TTL if entry.get("fallback") else ttl_seconds):
        return "fresh"
    servable = entry["models"] or entry.get("authoritative") is True
    if servable and not entry.get("fallback") and age < models._PROVIDER_MODELS_STALE_SERVE_MAX:
        return "stale"
    return None


def same_identity_authoritative_rows(provider: str, api_key: str, base_url: str) -> Optional[list[str]]:
    """The authoritative disk row, any age, when *api_key* and *base_url* are exactly the key and
    endpoint discovery resolves for *provider* (the row's fingerprint), else ``None``. Setup probes
    with the key it is configuring: a failed probe for the same identity keeps its account list."""
    from hermes_cli import models
    from hermes_cli.models_account_identity import account_request

    normalized = models._normalized_cache_slug(provider)
    request = account_request(normalized)
    probed = (str(api_key or "").strip(), str(base_url or "").strip().rstrip("/"))
    if request is None or not request.api_key or tuple(request) != probed:
        return None
    entry = models._load_provider_models_cache().get(normalized)
    if _cache_entry_valid(entry, models._credential_fingerprint(normalized, request)) and entry.get("authoritative") is True:
        return models._discovered_rows(normalized, entry)
    return None
